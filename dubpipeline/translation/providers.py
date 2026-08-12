from __future__ import annotations

import gc
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dubpipeline.models.storage import configure_argos_packages_dir, get_argos_packages_dir, get_models_root_dir
from dubpipeline.utils.logging import info, step

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

DEFAULT_QWEN_TRANSLATION_PROMPT = (
    "Translate English speech into natural Russian.\n"
    "Rules:\n"
    "- Preserve the exact meaning.\n"
    "- Write natural spoken Russian.\n"
    "- Do not add explanations.\n"
    "- Do not omit technical information.\n"
    "- Preserve exact placeholders and technical tokens unchanged.\n"
    "- Preserve uppercase abbreviations such as API, GPU, VRAM, TTS, XTTS unchanged.\n"
    "- Preserve numbers, units, versions, paths, timestamps, ids, and code-like terms unchanged unless Russian grammar requires spacing.\n"
    "- Prefer concise phrasing suitable for voice dubbing.\n"
    "- Return only the Russian translation, without labels, notes, markdown, or quotes."
)

_QWEN_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:assistant|answer|translation|translated text|russian translation|перевод|ответ)\s*[:：-]\s*",
    re.IGNORECASE,
)
_QWEN_FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$", re.IGNORECASE)
_QWEN_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_QWEN_THINK_MARKER_RE = re.compile(r"</?think>", re.IGNORECASE)
_QWEN_PROMPT_ECHO_MARKERS = (
    "Translate English speech into natural Russian",
    "Rules:",
    "Text:",
    "Return only the Russian translation",
)
QWEN_BASE_MODEL_REF = "Qwen/Qwen3-8B"
QWEN_FP8_MODEL_REF = "Qwen/Qwen3-8B-FP8"
QWEN_AWQ_MODEL_REF = "Qwen/Qwen3-8B-AWQ"
QWEN_GENERATION_PROFILE = "qwen3-nothink-translation-v2"
QWEN_QUANTIZATION_VALUES = ("auto", "fp8", "none", "awq", "bnb_4bit", "bnb_8bit")


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
    qwen_model: str = ""
    qwen_device: str = "auto"
    qwen_dtype: str = "auto"
    qwen_quantization: str = "auto"
    qwen_max_new_tokens: int = 0
    qwen_prompt: str = ""


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

    def offload(self) -> None:
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


class QwenTranslationProvider(BaseTranslationProvider):
    provider_id = "qwen"

    _QWEN_CACHE: dict[tuple[str, str, str, str], tuple[object, object]] = {}

    def __init__(self, context: TranslationProviderContext) -> None:
        super().__init__(context)
        self._cache_key: tuple[str, str, str, str] | None = None

    def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
        _ = sent_fallback
        if not texts:
            return []

        tokenizer, model, device = self._load_qwen()
        out: list[str] = []
        for text in texts:
            if not _normalize_text(text):
                out.append("")
                continue
            prompt = self.build_prompt(text)
            raw = self._generate(prompt, tokenizer, model, device)
            cleaned = self.clean_translation_output(raw)
            out.append(self.validate_translation_output(text, raw, cleaned))
        return out

    def release(self) -> None:
        if self._cache_key is None:
            return
        cached = self._QWEN_CACHE.pop(self._cache_key, None)
        self._cache_key = None
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
        info("[QWEN] Released cached translation model.\n")

    def offload(self) -> None:
        if self._cache_key is None:
            return
        cached = self._QWEN_CACHE.get(self._cache_key)
        if cached is None:
            return
        _, model = cached
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        info("[QWEN] Offloaded translation model to CPU; cache entry kept for batch reuse.\n")

    @classmethod
    def release_all(cls) -> None:
        keys = list(cls._QWEN_CACHE.keys())
        for key in keys:
            cached = cls._QWEN_CACHE.pop(key, None)
            if cached is None:
                continue
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
        if keys:
            info(f"[QWEN] Released {len(keys)} cached translation model(s).\n")

    def build_prompt(self, text: str) -> str:
        prompt = (self.context.qwen_prompt or DEFAULT_QWEN_TRANSLATION_PROMPT).strip()
        return f"{prompt}\n\nText:\n{text.strip()}"

    @classmethod
    def clean_translation_output(cls, text: str) -> str:
        cleaned = _QWEN_THINK_RE.sub("", text or "")
        cleaned = _QWEN_FENCE_RE.sub("", cleaned).strip()
        cleaned = _QWEN_LABEL_PREFIX_RE.sub("", cleaned).strip()

        lines = [line.strip() for line in cleaned.splitlines()]
        lines = [line for line in lines if line]
        if len(lines) > 1:
            # Keep concise text while dropping obvious assistant boilerplate.
            filtered: list[str] = []
            for line in lines:
                low = line.lower()
                if low.startswith(("note:", "explanation:", "analysis:", "комментарий:")):
                    continue
                filtered.append(_QWEN_LABEL_PREFIX_RE.sub("", line).strip())
            lines = [line for line in filtered if line]
        cleaned = " ".join(lines) if lines else cleaned
        cleaned = cleaned.strip().strip('"').strip("'").strip()
        return _normalize_text(cleaned)

    @classmethod
    def validate_translation_output(cls, source: str, raw: str, cleaned: str) -> str:
        source_norm = _normalize_text(source)
        cleaned_norm = _normalize_text(cleaned)
        raw_text = raw or ""
        if source_norm and not cleaned_norm:
            raise TranslationModelError("Qwen translation output is empty for a non-empty source segment.")
        if _QWEN_THINK_MARKER_RE.search(raw_text) or _QWEN_THINK_MARKER_RE.search(cleaned_norm):
            raise TranslationModelError("Qwen translation output contains <think> markers despite thinking mode being disabled.")
        if "```" in cleaned_norm:
            raise TranslationModelError("Qwen translation output contains markdown fences.")
        if _QWEN_LABEL_PREFIX_RE.match(cleaned_norm):
            raise TranslationModelError("Qwen translation output contains an assistant/translation label prefix.")
        if any(marker in raw_text or marker in cleaned_norm for marker in _QWEN_PROMPT_ECHO_MARKERS):
            raise TranslationModelError("Qwen translation output appears to echo the prompt.")
        return cleaned_norm

    def _model_ref(self, *, device: str | None = None, quantization: str | None = None) -> str:
        effective_device = device or self.context.qwen_device
        effective_quantization = quantization or self.context.qwen_quantization
        return resolve_qwen_model_ref(
            model_ref_override=self.context.qwen_model,
            catalog_model_ref=self.active_model.model_ref,
            quantization=effective_quantization,
            device=effective_device,
            usegpu=self.context.usegpu,
        )

    def _device(self) -> str:
        requested = (self.context.qwen_device or "auto").strip().lower()
        if requested not in {"auto", "cuda", "cpu"}:
            raise TranslationModelUnavailableError(
                f"translation provider 'qwen' device '{requested}' is not supported"
            )
        try:
            import torch
        except Exception as exc:
            raise TranslationModelError(
                "Translation provider 'qwen' is unavailable. Install torch and transformers dependencies."
            ) from exc

        cuda_available = bool(torch.cuda.is_available())
        if requested == "cuda":
            if not cuda_available:
                raise TranslationModelUnavailableError(
                    "translation provider 'qwen' requested CUDA but CUDA is not available"
                )
            return "cuda"
        if requested == "cpu":
            return "cpu"
        if self.context.usegpu and cuda_available:
            return "cuda"
        return "cpu"

    def _torch_dtype(self, torch_module: Any, device: str) -> object:
        requested = (self.context.qwen_dtype or "auto").strip().lower()
        if requested in {"", "auto"}:
            return "auto"
        if requested in {"float16", "fp16"}:
            return torch_module.float16
        if requested in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if requested in {"float32", "fp32"}:
            return torch_module.float32
        raise TranslationModelUnavailableError(
            f"translation provider 'qwen' dtype '{requested}' is not supported"
        )

    def _quantization(self, device: str) -> str:
        return resolve_qwen_quantization(
            self.context.qwen_quantization,
            device=device,
            usegpu=self.context.usegpu,
        )

    def _quantization_config(self, quantization: str) -> object | None:
        if quantization not in {"bnb_4bit", "bnb_8bit"}:
            return None
        if importlib.util.find_spec("bitsandbytes") is None:
            raise TranslationModelUnavailableError(
                f"translation provider 'qwen' quantization='{quantization}' requires bitsandbytes."
            )
        if importlib.util.find_spec("accelerate") is None:
            raise TranslationModelUnavailableError(
                f"translation provider 'qwen' quantization='{quantization}' requires accelerate."
            )
        from transformers import BitsAndBytesConfig

        if quantization == "bnb_4bit":
            return BitsAndBytesConfig(load_in_4bit=True)
        return BitsAndBytesConfig(load_in_8bit=True)

    @staticmethod
    def _configure_triton_cache_dir() -> None:
        import os

        if os.getenv("TRITON_CACHE_DIR"):
            return
        cache_dir = get_models_root_dir(create=True) / "triton_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    @staticmethod
    def _ensure_model_device(model: object, device: str) -> None:
        try:
            first_param = next(model.parameters())
            current = str(getattr(first_param, "device", ""))
        except Exception:
            current = ""
        if current and current.startswith(device):
            return
        try:
            model.to(device)
        except Exception:
            pass

    def _load_qwen(self) -> tuple[object, object, str]:
        device = self._device()
        quantization = self._quantization(device)
        model_ref = self._model_ref(device=device, quantization=quantization)
        dtype_key = (self.context.qwen_dtype or "auto").strip().lower() or "auto"
        cache_key = (model_ref, device, dtype_key, quantization)
        self._cache_key = cache_key

        if cache_key in self._QWEN_CACHE:
            tokenizer, model = self._QWEN_CACHE[cache_key]
            self._ensure_model_device(model, device)
            info(f"[QWEN] Cache hit: model={model_ref} device={device} quantization={quantization}\n")
            return tokenizer, model, device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise TranslationModelError(
                "Translation provider 'qwen' is unavailable. Install transformers dependencies."
            ) from exc

        if self.active_model.local_dir and not Path(self.active_model.local_dir).exists():
            raise TranslationModelUnavailableError(self._model_not_installed_message())

        resolved_ref = self.active_model.local_dir or model_ref
        load_kwargs = {
            "local_files_only": True,
            "torch_dtype": self._torch_dtype(torch, device),
        }
        if quantization == "fp8":
            self._configure_triton_cache_dir()
        quantization_config = self._quantization_config(quantization)
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        step(
            "Loading Qwen translation model: "
            f"model={model_ref} local={resolved_ref} device={device} "
            f"dtype={dtype_key} quantization={quantization}\n"
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_ref, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                resolved_ref,
                **load_kwargs,
            )
        except Exception as exc:
            message = str(exc)
            if "requires accelerate" in message.lower():
                raise TranslationModelUnavailableError(
                    "Translation provider 'qwen' cannot load FP8 model because accelerate is not installed. "
                    "Install/update dependencies from requirements.txt."
                ) from exc
            if "no module named 'triton'" in message.lower() or "no module named triton" in message.lower():
                raise TranslationModelUnavailableError(
                    "Translation provider 'qwen' cannot load FP8 model because Triton is not installed. "
                    "Install/update dependencies from requirements.txt; on Windows this uses triton-windows."
                ) from exc
            if "bitsandbytes" in message.lower():
                raise TranslationModelUnavailableError(
                    "Translation provider 'qwen' cannot load the requested quantization because bitsandbytes is not installed. "
                    "Install an optional bitsandbytes/accelerate runtime or choose quantization=fp8/none."
                ) from exc
            raise TranslationModelUnavailableError(
                "Translation provider 'qwen' cannot load local model "
                f"'{model_ref}'. Install/cache it first with Models -> Install, or choose another model."
            ) from exc

        model.eval()
        if quantization_config is None:
            model.to(device)
        self._QWEN_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model, device

    def _format_chat_prompt(self, prompt: str, tokenizer: object) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.context.qwen_prompt or DEFAULT_QWEN_TRANSLATION_PROMPT},
                {"role": "user", "content": prompt.split("\n\nText:\n", 1)[-1]},
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompt

    def _generate(
        self,
        prompt: str,
        tokenizer: object,
        model: object,
        device: str,
    ) -> str:
        import torch

        formatted_prompt = self._format_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        input_ids = inputs.get("input_ids")
        input_len = int(getattr(input_ids, "shape", [0, 0])[-1]) if input_ids is not None else 0
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)

        max_new_tokens = int(self.context.qwen_max_new_tokens or self.context.max_new_tokens or 512)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                num_beams=1,
                pad_token_id=pad_token_id,
            )

        try:
            generated_ids = out_ids[:, input_len:]
        except Exception:
            generated_ids = out_ids
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def resolve_qwen_quantization(
    requested_quantization: str | None,
    *,
    device: str,
    usegpu: bool,
) -> str:
    requested = (requested_quantization or "auto").strip().lower() or "auto"
    if requested not in QWEN_QUANTIZATION_VALUES:
        raise TranslationModelUnavailableError(
            f"translation provider 'qwen' quantization '{requested}' is not supported"
        )
    if requested != "auto":
        return requested
    if device == "cuda" and usegpu:
        return "fp8"
    return "none"


def resolve_qwen_model_ref(
    *,
    model_ref_override: str,
    catalog_model_ref: str,
    quantization: str,
    device: str,
    usegpu: bool,
) -> str:
    explicit = (model_ref_override or "").strip()
    if explicit:
        return explicit
    effective_quantization = resolve_qwen_quantization(
        quantization,
        device=device if device in {"cuda", "cpu"} else ("cuda" if usegpu else "cpu"),
        usegpu=usegpu,
    )
    if effective_quantization == "fp8":
        return QWEN_FP8_MODEL_REF
    if effective_quantization == "awq":
        return QWEN_AWQ_MODEL_REF
    if effective_quantization == "none":
        return QWEN_BASE_MODEL_REF
    return (catalog_model_ref or QWEN_BASE_MODEL_REF).strip()


_PROVIDER_BY_ID = {
    "argos": ArgosTranslationProvider,
    "hf": HfSeq2SeqTranslationProvider,
    "qwen": QwenTranslationProvider,
}

PUBLIC_TRANSLATION_PROVIDERS = ("auto", "argos", "qwen")


def provider_id_for_backend(backend: str) -> str:
    normalized = (backend or "").strip().lower()
    if normalized == "argos":
        return "argos"
    if normalized in {"nllb", "opus_mt", "hf"}:
        return "hf"
    if normalized in {"llm_qwen", "qwen"}:
        return "qwen"
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
