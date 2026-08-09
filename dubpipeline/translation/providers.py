from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dubpipeline.models.storage import configure_argos_packages_dir, get_argos_packages_dir

_WS_RE = re.compile(r"\s+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+", re.UNICODE)
_TORCH_26_LOAD_ERROR_MARKERS = (
    "upgrade torch to at least v2.6",
    "upgrade torch to at least 2.6",
    "cve-2025-32434",
)

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
    local_dir: str = ""


@dataclass(frozen=True)
class TranslationProviderContext:
    active_model: ActiveModel
    src_lang: str
    tgt_lang: str
    usegpu: bool
    batch_size: int
    max_new_tokens: int


class TranslationProvider(Protocol):
    provider_id: str

    def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
        ...

    def release(self) -> None:
        ...


def model_not_installed_message(model_label: str, src_lang: str, tgt_lang: str) -> str:
    pair = f"{src_lang}->{tgt_lang}"
    return (
        f"Translation model '{model_label}' for {pair} is not installed. "
        "Cannot continue translation. Open Models -> Install to download it."
    )


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


def _parse_torch_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", version or "")
    if not match:
        return (0, 0, 0)
    return tuple(int(part or 0) for part in match.groups())


def _torch_version_is_below_26(version: str) -> bool:
    return _parse_torch_version(version) < (2, 6, 0)


def _is_torch_26_bin_weights_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TORCH_26_LOAD_ERROR_MARKERS)


def _has_bin_weights_without_safetensors(model_ref: str) -> bool:
    model_dir = Path(model_ref)
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    try:
        names = {path.name for path in model_dir.iterdir() if path.is_file()}
    except OSError:
        return False

    has_bin = bool({"pytorch_model.bin", "pytorch_model.bin.index.json"}.intersection(names))
    has_safetensors = bool({"model.safetensors", "model.safetensors.index.json"}.intersection(names))
    return has_bin and not has_safetensors


class BaseTranslationProvider:
    provider_id = ""

    def __init__(self, context: TranslationProviderContext) -> None:
        self.context = context

    @property
    def active_model(self) -> ActiveModel:
        return self.context.active_model

    def release(self) -> None:
        return None

    def _model_not_installed_message(self) -> str:
        return model_not_installed_message(
            self.active_model.label,
            self.context.src_lang,
            self.context.tgt_lang,
        )


class HfSeq2SeqTranslationProvider(BaseTranslationProvider):
    provider_id = "hf"

    _HF_CACHE: dict[tuple[str, str], tuple[object, object]] = {}

    def __init__(self, context: TranslationProviderContext) -> None:
        super().__init__(context)
        self._hf_cache_key: tuple[str, str] | None = None

    def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
        if not texts:
            return []

        tokenizer, model, device = self._load_hf()
        batch_size = max(1, int(self.context.batch_size))

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

    def _device(self) -> str:
        if not self.context.usegpu:
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _torch_bin_weights_message(self, model_ref: str, torch_version: str) -> str:
        pair = f"{self.context.src_lang}->{self.context.tgt_lang}"
        location = f" at {model_ref}" if model_ref else ""
        version = torch_version or "an older version"
        return (
            f"Translation provider '{self.provider_id}' cannot load model "
            f"'{self.active_model.label}' for {pair}. The model is installed{location}, "
            "but cannot be loaded because this environment uses "
            f"PyTorch {version} and the local model has PyTorch .bin weights without safetensors. "
            "Modern Transformers blocks loading these weights with torch < 2.6 due to CVE-2025-32434. "
            "Upgrade torch and torchaudio to >= 2.6, or install a safetensors version of the model."
        )

    def _load_hf(self) -> tuple[object, object, str]:
        model_ref = self.active_model.local_dir or self.active_model.model_ref
        if self.active_model.local_dir and not Path(self.active_model.local_dir).exists():
            raise TranslationModelUnavailableError(self._model_not_installed_message())

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
                "Translation provider 'hf' is unavailable. Install transformers dependencies."
            ) from exc

        torch_version = str(getattr(torch, "__version__", ""))
        if (
            _has_bin_weights_without_safetensors(model_ref)
            and _torch_version_is_below_26(torch_version)
        ):
            raise TranslationModelUnavailableError(
                self._torch_bin_weights_message(model_ref, torch_version)
            )

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
                if _is_torch_26_bin_weights_error(exc):
                    raise TranslationModelUnavailableError(
                        self._torch_bin_weights_message(model_ref, torch_version)
                    ) from exc
                raise TranslationModelUnavailableError(self._model_not_installed_message()) from exc

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

        generation_kwargs: dict[str, int] = {}

        if self.active_model.backend == "nllb":
            src_code = _nllb_lang_code(self.context.src_lang)
            tgt_code = _nllb_lang_code(self.context.tgt_lang)
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

        if device.startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out_ids = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=int(self.context.max_new_tokens),
                    **generation_kwargs,
                )
        else:
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=int(self.context.max_new_tokens),
                    **generation_kwargs,
                )
        return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


class ArgosTranslationProvider(BaseTranslationProvider):
    provider_id = "argos"

    def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
        if not texts:
            return []

        configure_argos_packages_dir(create=True)

        try:
            from argostranslate import package, translate
        except Exception as exc:
            raise TranslationModelUnavailableError(self._model_not_installed_message()) from exc

        installed = package.get_installed_packages(path=get_argos_packages_dir(create=True))
        if not any(
            getattr(pkg, "from_code", None) == self.context.src_lang
            and getattr(pkg, "to_code", None) == self.context.tgt_lang
            for pkg in installed
        ):
            raise TranslationModelUnavailableError(self._model_not_installed_message())

        out: list[str] = []
        for src in texts:
            translated = translate.translate(src, self.context.src_lang, self.context.tgt_lang)
            if sent_fallback and _looks_truncated(src, translated):
                parts = _split_sentences(src)
                if len(parts) > 1:
                    translated_parts = [
                        translate.translate(part, self.context.src_lang, self.context.tgt_lang)
                        for part in parts
                    ]
                    translated = " ".join(
                        part.strip() for part in translated_parts if part and part.strip()
                    ).strip()
            out.append(translated)
        return out


_PROVIDER_BY_ID = {
    "argos": ArgosTranslationProvider,
    "hf": HfSeq2SeqTranslationProvider,
}

PUBLIC_TRANSLATION_PROVIDERS = ("auto", "argos")


def provider_id_for_backend(backend: str) -> str:
    normalized = (backend or "").strip().lower()
    if normalized == "argos":
        return "argos"
    if normalized in {"nllb", "opus_mt", "hf"}:
        return "hf"
    return normalized


def resolve_translation_provider_id(requested_provider: str, backend: str) -> str:
    requested = (requested_provider or "").strip().lower()
    if not requested or requested == "auto":
        requested = provider_id_for_backend(backend)
    if requested not in _PROVIDER_BY_ID:
        raise TranslationModelUnavailableError(
            f"translation provider '{requested}' is not supported"
        )
    return requested


def create_translation_provider(
    provider_id: str,
    context: TranslationProviderContext,
) -> TranslationProvider:
    provider = resolve_translation_provider_id(provider_id, context.active_model.backend)
    provider_type = _PROVIDER_BY_ID[provider]
    return provider_type(context)
