from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dubpipeline.config import PipelineConfig
from dubpipeline.text.technical_tokens import TechnicalTokenError, TechnicalTokenProtector
from dubpipeline.text.translation_validation import (
    TranslationValidationError,
    validate_translation_text,
)
from dubpipeline.translation.service import TranslationModelError, TranslatorService
from dubpipeline.utils.logging import error, info, step, warn

_WS_RE = re.compile(r"\s+", re.UNICODE)


def run(cfg: PipelineConfig) -> None:
    try:
        translator = TranslatorService.from_config(cfg)
    except TranslationModelError as exc:
        raise SystemExit(str(exc)) from None

    step(
        f"Translation model: {translator.model_label} "
        f"[{translator.model_id}] via {translator.backend}\n"
    )

    try:
        translate_segments(
            cfg=cfg,
            input_file=cfg.paths.segments_file,
            output_file=cfg.paths.segments_tgt_file,
            translator=translator,
        )
    except TranslationModelError as exc:
        raise SystemExit(str(exc)) from None
    finally:
        release = str(cfg.translate.release_vram).strip().lower()
        if release not in {"0", "false", "no", "off"}:
            translator.release_after_translate(batch_file_count=int(getattr(cfg, "batch_file_count", 1) or 1))


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    return _WS_RE.sub(" ", text)


def _default_cache_path(output_file: str | Path) -> Path:
    out = Path(output_file)
    return out.with_suffix(out.suffix + ".translate_cache.sqlite")


def _open_cache(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS translations (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
        """
    )
    return con


def _make_cache_key(scope: str, text: str) -> str:
    payload = f"{scope}|{_normalize_text(text)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _segment_id(seg: dict, fallback_index: int) -> str:
    value = str(seg.get("id", "")).strip()
    return value or str(fallback_index)


def _placeholders(tokens: dict[str, str]) -> str:
    return ",".join(tokens.keys()) if tokens else "none"


def _raise_translation_failure(
    *,
    segment_id: str,
    source_lang: str,
    target_lang: str,
    reason: str,
) -> None:
    message = (
        "[TRANSLATE] translation_validation_failed "
        f"segment_id={segment_id} source_lang={source_lang} "
        f"target_lang={target_lang} reason={reason}"
    )
    error(message + "\n")
    raise TranslationModelError(message)


def _cache_get_many(con: sqlite3.Connection, keys: List[str]) -> Dict[str, str]:
    if not keys:
        return {}

    out: Dict[str, str] = {}
    chunk = 500
    for offset in range(0, len(keys), chunk):
        key_chunk = keys[offset:offset + chunk]
        placeholders = ",".join("?" for _ in key_chunk)
        rows = con.execute(
            f"SELECT k, v FROM translations WHERE k IN ({placeholders})",  # noqa: S608 - placeholders are safe
            key_chunk,
        ).fetchall()
        out.update({k: v for k, v in rows})
    return out


def _cache_put_many(con: sqlite3.Connection, items: List[Tuple[str, str]]) -> None:
    if not items:
        return
    con.executemany("INSERT OR REPLACE INTO translations(k, v) VALUES(?, ?)", items)
    con.commit()


def _sent_fallback_enabled() -> bool:
    raw = os.getenv("DUBPIPELINE_TRANSLATE_SENT_FALLBACK", "")
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def translate_segments(
    cfg: PipelineConfig,
    input_file: str | Path,
    output_file: str | Path,
    translator: TranslatorService,
) -> None:
    t0 = time.perf_counter()

    input_path = Path(input_file)
    output_path = Path(output_file)
    with input_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    cache_db_cfg = str(cfg.translate.cache_db or "").strip()
    cache_path = Path(cache_db_cfg).expanduser() if cache_db_cfg else _default_cache_path(output_path)

    con = _open_cache(cache_path)
    try:
        texts: list[str] = []
        keys: list[str] = []
        for seg in segments:
            text = seg.get("text", "") or ""
            texts.append(text)
            keys.append(_make_cache_key(translator.cache_scope, text))

        cached = _cache_get_many(con, keys)
        misses: dict[str, str] = {}
        for cache_key, text in zip(keys, texts):
            if not _normalize_text(text):
                continue
            if cache_key not in cached:
                misses.setdefault(cache_key, text)

        info(f"[INFO] Segments: {len(segments)}\n")
        info(f"[INFO] Cache hits: {len(cached)}\n")
        info(f"[INFO] Need translate: {len(misses)} (unique)\n")
        info(f"[CACHE] {cache_path}\n")

        source_lang = (cfg.languages.src or "").strip().lower()
        target_lang = (cfg.languages.tgt or "").strip().lower()
        segment_ids_by_key: dict[str, str] = {}
        for idx, (cache_key, seg) in enumerate(zip(keys, segments)):
            segment_ids_by_key.setdefault(cache_key, _segment_id(seg, idx))

        if misses:
            sent_fallback = _sent_fallback_enabled()
            miss_keys = list(misses.keys())
            protector = TechnicalTokenProtector()
            protected_by_key = {k: protector.protect(misses[k]) for k in miss_keys}
            miss_texts = [protected_by_key[k].text for k in miss_keys]
            translated_texts = translator.translate_texts(miss_texts, sent_fallback=sent_fallback)
            new_items: list[tuple[str, str]] = []
            retry_keys: list[str] = []
            for cache_key, translated_text in zip(miss_keys, translated_texts):
                protected = protected_by_key[cache_key]
                segment_id = segment_ids_by_key.get(cache_key, cache_key)
                try:
                    restored_text = protector.restore(translated_text, protected.tokens)
                except TechnicalTokenError as exc:
                    warn(
                        "[TRANSLATE] technical_token_restore_failed "
                        f"segment_id={segment_id} source_lang={source_lang} "
                        f"target_lang={target_lang} placeholder={_placeholders(protected.tokens)} "
                        f"attempt=1 action=retry reason={exc}\n"
                    )
                    retry_keys.append(cache_key)
                    continue

                try:
                    validate_translation_text(
                        source_text=misses[cache_key],
                        translated_text=restored_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        segment_id=segment_id,
                    )
                except TranslationValidationError as exc:
                    _raise_translation_failure(
                        segment_id=segment_id,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        reason=str(exc),
                    )
                new_items.append((cache_key, restored_text))
                cached[cache_key] = restored_text

            if retry_keys:
                retry_texts = [protected_by_key[k].text for k in retry_keys]
                retry_translated = translator.translate_texts(retry_texts, sent_fallback=False)
                for cache_key, translated_text in zip(retry_keys, retry_translated):
                    protected = protected_by_key[cache_key]
                    segment_id = segment_ids_by_key.get(cache_key, cache_key)
                    try:
                        restored_text = protector.restore(translated_text, protected.tokens)
                    except TechnicalTokenError as exc:
                        _raise_translation_failure(
                            segment_id=segment_id,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            reason=(
                                "technical_token_restore_failed "
                                f"placeholder={_placeholders(protected.tokens)} attempt=2 {exc}"
                            ),
                        )
                    try:
                        validate_translation_text(
                            source_text=misses[cache_key],
                            translated_text=restored_text,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            segment_id=segment_id,
                        )
                    except TranslationValidationError as exc:
                        _raise_translation_failure(
                            segment_id=segment_id,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            reason=str(exc),
                        )
                    info(
                        "[TRANSLATE] technical_token_restore_recovered "
                        f"segment_id={segment_id} source_lang={source_lang} "
                        f"target_lang={target_lang} placeholder={_placeholders(protected.tokens)} "
                        "attempt=2 action=accept\n"
                    )
                    new_items.append((cache_key, restored_text))
                    cached[cache_key] = restored_text

            _cache_put_many(con, new_items)

        translated = []
        for idx, seg in enumerate(segments):
            text = seg.get("text", "") or ""
            cache_key = keys[idx]
            text_tgt = cached.get(cache_key, "") if _normalize_text(text) else ""
            if _normalize_text(text):
                try:
                    validate_translation_text(
                        source_text=text,
                        translated_text=text_tgt,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        segment_id=_segment_id(seg, idx),
                    )
                except TranslationValidationError as exc:
                    _raise_translation_failure(
                        segment_id=_segment_id(seg, idx),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        reason=str(exc),
                    )
            seg_out = {"id": idx, **seg, "text_tgt": text_tgt}
            if target_lang == "ru":
                seg_out["text_ru"] = text_tgt
            translated.append(seg_out)
            if idx < 20:
                info(f"[{idx}] {text} -> {text_tgt}\n")
            elif idx == 20:
                info("... (log truncated)\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        t1 = time.perf_counter()
        info(f"\n[OK] Translated {len(translated)} segments.\n")
        info(f"[SAVE] {output_path}\n")
        info(f"[TIME] total_step={t1 - t0:.2f}s\n")
    finally:
        con.close()
