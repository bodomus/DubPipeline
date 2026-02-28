from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dubpipeline.models.catalog import (
    NOT_INSTALLED_REASON,
    NOT_SUPPORTED_REASON,
    get_model_spec,
    get_model_status,
)

if TYPE_CHECKING:
    from dubpipeline.config import PipelineConfig

_WS_RE = re.compile(r"\s+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+", re.UNICODE)

_NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "uk": "ukr_Cyrl",
    "pl": "pol_Latn",
}


class TranslationModelError(RuntimeError):
    pass


class TranslationModelUnavailableError(TranslationModelError):
    pass


@dataclass(frozen=True)
class ActiveModel:
    model_id: str
    label: str
    backend: str
    model_ref: str


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    return _WS_RE.sub(" ", text)


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    chunks = [part.strip() for part in _SENT_SPLIT_RE.split(normalized) if part.strip()]
    return chunks or [normalized]


def _sent_count(text: str) -> int:
    return len(_split_sentences(text))


def _looks_truncated(src: str, translated: str) -> bool:
    src_norm = _normalize_text(src)
    out_norm = _normalize_text(translated)
    if not src_norm:
        return False
    if not out_norm:
        return True

    src_sentences = _sent_count(src_norm)
    out_sentences = _sent_count(out_norm)
    if src_sentences >= 2 and out_sentences < src_sentences:
        return True

    if len(src_norm) >= 120 and len(out_norm) < int(0.35 * len(src_norm)):
        return True

    return False


def _nllb_lang_code(lang_code: str) -> str:
    code = (lang_code or "").strip()
    if not code:
        return "eng_Latn"
    if "_" in code and len(code) >= 7:
        return code
    return _NLLB_LANG_MAP.get(code.lower(), code)


class TranslatorService:
    _HF_CACHE: dict[tuple[str, str], tuple[object, object]] = {}

    def __init__(self, cfg: "PipelineConfig") -> None:
        self._cfg = cfg
        self._active = self._resolve_active_model(cfg)
        self._hf_cache_key: tuple[str, str] | None = None

    @property
    def model_id(self) -> str:
        return self._active.model_id

    @property
    def model_label(self) -> str:
        return self._active.label

    @property
    def backend(self) -> str:
        return self._active.backend

    @property
    def cache_scope(self) -> str:
        return f"{self.backend}|{self.model_id}"

    @classmethod
    def from_config(cls, cfg: "PipelineConfig") -> "TranslatorService":
        return cls(cfg)

    def _resolve_active_model(self, cfg: "PipelineConfig") -> ActiveModel:
        model_id = (cfg.translation.model_id or "").strip()
        if not model_id:
            raise TranslationModelUnavailableError(
                "Translation model is not configured. Please choose a model in Models..."
            )

        spec = get_model_spec(model_id)
        status = get_model_status(model_id)
        if not status.enabled:
            if status.reason == NOT_SUPPORTED_REASON:
                raise TranslationModelUnavailableError(
                    f"Translation model '{spec.label}' is not supported yet in this build. "
                    "Please choose another model in Models..."
                )
            if status.reason == NOT_INSTALLED_REASON or not status.available:
                raise TranslationModelUnavailableError(
                    f"Translation model '{spec.label}' is not installed locally. "
                    "Please install it or choose another model in Models..."
                )
            raise TranslationModelUnavailableError(
                f"Translation model '{spec.label}' is unavailable ({status.reason}). "
                "Please choose another model in Models..."
            )

        return ActiveModel(
            model_id=spec.id,
            label=spec.label,
            backend=spec.backend,
            model_ref=spec.model_ref,
        )

    def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
        if not texts:
            return []
        if self.backend == "argos":
            return self._translate_argos(texts, sent_fallback=sent_fallback)
        if self.backend in {"nllb", "opus_mt"}:
            return self._translate_hf(texts, sent_fallback=sent_fallback)
        raise TranslationModelUnavailableError(
            f"Translation backend '{self.backend}' is not supported yet. "
            "Please choose another model in Models..."
        )

    def release(self) -> None:
        if self._hf_cache_key is None:
            return
        cached = self._HF_CACHE.pop(self._hf_cache_key, None)
        self._hf_cache_key = None
        if cached is None:
            return

        _, model = cached
        try:
            model.to("cpu")
        except Exception:
            pass

        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _is_gpu_enabled(self) -> bool:
        return bool(getattr(self._cfg, "usegpu", False))

    def _device(self) -> str:
        if not self._is_gpu_enabled():
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_hf(self) -> tuple[object, object, str]:
        model_ref = self._active.model_ref
        device = self._device()
        cache_key = (device, model_ref)
        self._hf_cache_key = cache_key

        if cache_key in self._HF_CACHE:
            tokenizer, model = self._HF_CACHE[cache_key]
            return tokenizer, model, device

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise TranslationModelError(
                "Transformers backend is unavailable. Install dependencies for translation backend."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)

        load_kwargs = {"local_files_only": True}
        if device.startswith("cuda"):
            load_kwargs["torch_dtype"] = torch.float16

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_ref,
                use_safetensors=True,
                **load_kwargs,
            )
        except Exception:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_ref,
                    use_safetensors=False,
                    **load_kwargs,
                )
            except Exception as exc:
                raise TranslationModelUnavailableError(
                    f"Translation model '{self._active.label}' is not installed locally. "
                    "Please install it or choose another model in Models..."
                ) from exc

        model.eval()
        model.to(device)
        self._HF_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model, device

    def _hf_generate(
        self,
        batch: list[str],
        tokenizer: object,
        model: object,
        device: str,
    ) -> list[str]:
        import torch

        src_lang = (self._cfg.languages.src or "en").strip()
        tgt_lang = (self._cfg.languages.tgt or "ru").strip()
        generation_kwargs: dict[str, int] = {}

        if self.backend == "nllb":
            src_code = _nllb_lang_code(src_lang)
            tgt_code = _nllb_lang_code(tgt_lang)
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = src_code
            lang_map = getattr(tokenizer, "lang_code_to_id", {}) or {}
            forced_bos = lang_map.get(tgt_code)
            if forced_bos is not None:
                generation_kwargs["forced_bos_token_id"] = int(forced_bos)

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        max_new_tokens = int(self._cfg.translate.max_new_tokens)
        if device.startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_ids = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs,
                )
        else:
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs,
                )
        return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    def _translate_hf(self, texts: list[str], *, sent_fallback: bool) -> list[str]:
        tokenizer, model, device = self._load_hf()
        batch_size = max(1, int(self._cfg.translate.batch_size))

        order = sorted(range(len(texts)), key=lambda idx: len(texts[idx] or ""))
        sorted_texts = [texts[idx] for idx in order]
        sorted_out: list[str] = []

        for offset in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[offset:offset + batch_size]
            translated = self._hf_generate(batch, tokenizer, model, device)

            if sent_fallback:
                fixed: list[str] = []
                for src, ru in zip(batch, translated):
                    if _looks_truncated(src, ru):
                        parts = _split_sentences(src)
                        if len(parts) > 1:
                            translated_parts: list[str] = []
                            for part_offset in range(0, len(parts), batch_size):
                                part_batch = parts[part_offset:part_offset + batch_size]
                                translated_parts.extend(
                                    self._hf_generate(part_batch, tokenizer, model, device)
                                )
                            ru = " ".join(x.strip() for x in translated_parts if x and x.strip()).strip()
                    fixed.append(ru)
                translated = fixed

            sorted_out.extend(translated)

        out = [""] * len(texts)
        for sorted_idx, original_idx in enumerate(order):
            out[original_idx] = sorted_out[sorted_idx]
        return out

    def _translate_argos(self, texts: list[str], *, sent_fallback: bool) -> list[str]:
        src_lang = (self._cfg.languages.src or "en").strip()
        tgt_lang = (self._cfg.languages.tgt or "ru").strip()

        try:
            from argostranslate import package, translate
        except Exception as exc:
            raise TranslationModelUnavailableError(
                "Argos Translate is not installed locally. "
                "Install Argos package or choose another model in Models..."
            ) from exc

        installed = package.get_installed_packages()
        if not any(
            getattr(pkg, "from_code", None) == src_lang and getattr(pkg, "to_code", None) == tgt_lang
            for pkg in installed
        ):
            raise TranslationModelUnavailableError(
                f"Translation model '{self._active.label}' is not installed locally. "
                "Please install it or choose another model in Models..."
            )

        out: list[str] = []
        for src in texts:
            ru = translate.translate(src, src_lang, tgt_lang)
            if sent_fallback and _looks_truncated(src, ru):
                parts = _split_sentences(src)
                if len(parts) > 1:
                    ru_parts = [translate.translate(part, src_lang, tgt_lang) for part in parts]
                    ru = " ".join(part.strip() for part in ru_parts if part and part.strip()).strip()
            out.append(ru)
        return out
