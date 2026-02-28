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
from dubpipeline.translation.service import TranslationModelError, TranslatorService
from dubpipeline.utils.logging import info, step

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
            output_file=cfg.paths.segments_ru_file,
            translator=translator,
        )
    except TranslationModelError as exc:
        raise SystemExit(str(exc)) from None
    finally:
        release = str(cfg.translate.release_vram).strip().lower()
        if release not in {"0", "false", "no", "off"}:
            translator.release()


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

        if misses:
            sent_fallback = _sent_fallback_enabled()
            miss_keys = list(misses.keys())
            miss_texts = [misses[k] for k in miss_keys]
            ru_texts = translator.translate_texts(miss_texts, sent_fallback=sent_fallback)
            new_items = list(zip(miss_keys, ru_texts))
            _cache_put_many(con, new_items)
            cached.update({k: v for k, v in new_items})

        translated = []
        for idx, seg in enumerate(segments):
            text = seg.get("text", "") or ""
            cache_key = keys[idx]
            text_ru = cached.get(cache_key, "") if _normalize_text(text) else ""
            seg_out = {"id": idx, **seg, "text_ru": text_ru}
            translated.append(seg_out)
            if idx < 20:
                info(f"[{idx}] {text} -> {text_ru}\n")
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
