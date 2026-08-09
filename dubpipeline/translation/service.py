from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dubpipeline.models.catalog import (
    get_model_install_dir,
    get_model_status,
    is_unsupported_pair_reason,
    resolve_model_spec,
)
from dubpipeline.translation.providers import (
    ActiveModel,
    HfSeq2SeqTranslationProvider,
    TranslationModelError,
    TranslationModelUnavailableError,
    TranslationProvider,
    TranslationProviderContext,
    create_translation_provider,
    model_not_installed_message,
    provider_id_for_backend,
    resolve_translation_provider_id,
)

if TYPE_CHECKING:
    from dubpipeline.config import PipelineConfig


@dataclass(frozen=True)
class ResolvedTranslationRuntime:
    active_model: ActiveModel
    provider_id: str


class TranslatorService:
    _HF_CACHE = HfSeq2SeqTranslationProvider._HF_CACHE

    def __init__(self, cfg: "PipelineConfig") -> None:
        self._cfg = cfg
        runtime = self._resolve_runtime(cfg)
        self._active = runtime.active_model
        self._provider_id = runtime.provider_id
        self._provider = self._create_provider()

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
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def cache_scope(self) -> str:
        src_lang = (self._cfg.languages.src or "en").strip().lower() or "en"
        tgt_lang = (self._cfg.languages.tgt or "ru").strip().lower() or "ru"
        return f"{self.backend}|{self.model_id}|{src_lang}->{tgt_lang}"

    @classmethod
    def from_config(cls, cfg: "PipelineConfig") -> "TranslatorService":
        return cls(cfg)

    def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
        return self._provider.translate_texts(texts, sent_fallback=sent_fallback)

    def release(self) -> None:
        self._provider.release()

    def _load_hf(self) -> tuple[object, object, str]:
        if not isinstance(self._provider, HfSeq2SeqTranslationProvider):
            raise TranslationModelUnavailableError(
                f"translation provider '{self._provider_id}' does not use HF model loading"
            )
        return self._provider._load_hf()

    def _resolve_runtime(self, cfg: "PipelineConfig") -> ResolvedTranslationRuntime:
        requested_provider = (getattr(cfg.translation, "provider", "") or "").strip().lower()
        if requested_provider and requested_provider != "auto":
            resolve_translation_provider_id(requested_provider, requested_provider)

        src_lang = (cfg.languages.src or "en").strip().lower() or "en"
        tgt_lang = (cfg.languages.tgt or "ru").strip().lower() or "ru"
        model_id = (cfg.translation.model_id or "").strip()
        if requested_provider == "argos":
            model_id = "argos"
        if not model_id:
            raise TranslationModelUnavailableError(
                "Translation model is not configured. Please choose a model in Models..."
            )

        spec = resolve_model_spec(model_id, src_lang, tgt_lang)
        provider_id = resolve_translation_provider_id(requested_provider, spec.backend)
        if provider_id != provider_id_for_backend(spec.backend):
            raise TranslationModelUnavailableError(
                f"translation provider '{requested_provider}' is not supported for backend '{spec.backend}'"
            )

        status = get_model_status(model_id, src_lang, tgt_lang)
        if not spec.supported:
            raise TranslationModelUnavailableError(
                "Model is planned and not supported yet"
            )
        if is_unsupported_pair_reason(status.reason):
            raise TranslationModelUnavailableError(
                f"Model '{spec.label}' is {status.reason}. Please choose another model in Models..."
            )
        if not status.available:
            raise TranslationModelUnavailableError(
                model_not_installed_message(spec.label, src_lang, tgt_lang)
            )

        local_dir = ""
        model_dir = get_model_install_dir(spec.id, src_lang, tgt_lang)
        if spec.installer == "hf_snapshot" and model_dir is not None and model_dir.exists():
            local_dir = str(model_dir)

        return ResolvedTranslationRuntime(
            active_model=ActiveModel(
                model_id=spec.id,
                label=spec.label,
                backend=spec.backend,
                model_ref=spec.model_ref,
                local_dir=local_dir,
            ),
            provider_id=provider_id,
        )

    def _create_provider(self) -> TranslationProvider:
        src_lang = (self._cfg.languages.src or "en").strip().lower() or "en"
        tgt_lang = (self._cfg.languages.tgt or "ru").strip().lower() or "ru"
        context = TranslationProviderContext(
            active_model=self._active,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            usegpu=bool(getattr(self._cfg, "usegpu", False)),
            batch_size=int(getattr(self._cfg.translate, "batch_size", 64)),
            max_new_tokens=int(getattr(self._cfg.translate, "max_new_tokens", 256)),
        )
        return create_translation_provider(self._provider_id, context)
