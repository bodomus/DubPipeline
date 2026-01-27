from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from dubpipeline.config import PipelineConfig


class Const:
    """
    Lightweight repository for config values.

    Usage:
        from dubpipeline.consts import Const
        Const.bind(cfg)
        max_chars = Const.tts_max_ru_chars()
        backend = Const.translate_backend()
    """

    _cfg: Optional[PipelineConfig] = None

    @classmethod
    def bind(cls, cfg: PipelineConfig) -> None:
        cls._cfg = cfg

    @classmethod
    def cfg(cls) -> PipelineConfig:
        if cls._cfg is None:
            raise RuntimeError("Const is not bound. Call Const.bind(cfg) before using getters.")
        return cls._cfg

    @classmethod
    def get(cls, dotted_path: str, default: Any = None) -> Any:
        obj: Any = cls.cfg()
        for part in dotted_path.split("."):
            if is_dataclass(obj) and hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default
        return obj

    # ---- typed convenience getters (most used) ----

    @classmethod
    def tts_model_name(cls) -> str:
        return str(cls.cfg().tts.model_name)

    @classmethod
    def tts_voice(cls) -> str:
        return str(cls.cfg().tts.voice)

    @classmethod
    def tts_sample_rate(cls) -> int:
        return int(cls.cfg().tts.sample_rate)

    @classmethod
    def tts_warn_limit_chars_ru(cls) -> int:
        return int(cls.cfg().tts.warn_limit_chars_ru)

    @classmethod
    def tts_max_ru_chars(cls) -> int:
        cfg = cls.cfg()
        # Safe clamp (XTTS tends to warn/fail after ~warn_limit_chars_ru for RU)
        try:
            return int(min(cfg.tts.max_ru_chars, max(1, cfg.tts.warn_limit_chars_ru - 2)))
        except Exception:
            return int(cfg.tts.max_ru_chars)

    @classmethod
    def tts_gap_ms(cls) -> int:
        return int(cls.cfg().tts.gap_ms)

    @classmethod
    def tts_breaks(cls) -> list[str]:
        return list(cls.cfg().tts.breaks)

    @classmethod
    def tts_fast_latents(cls) -> bool:
        return bool(cls.cfg().tts.fast_latents)

    @classmethod
    def tts_try_single_call(cls) -> bool:
        return bool(cls.cfg().tts.try_single_call)

    @classmethod
    def tts_try_single_call_max_chars(cls) -> int:
        cfg = cls.cfg()
        lim = int(cfg.tts.try_single_call_max_chars)
        warn_lim = int(cfg.tts.warn_limit_chars_ru)
        return int(min(lim, max(1, warn_lim - 2)))

    @classmethod
    def whisperx_model_name(cls) -> str:
        return str(cls.cfg().whisperx.model_name)

    @classmethod
    def whisperx_batch_size(cls) -> int:
        return int(cls.cfg().whisperx.batch_size)

    @classmethod
    def whisperx_max_gap_between_words(cls) -> float:
        return float(cls.cfg().whisperx.max_gap_between_words)

    @classmethod
    def translate_backend(cls) -> str:
        return str(cls.cfg().translate.backend)

    @classmethod
    def translate_hf_model(cls) -> str:
        return str(cls.cfg().translate.hf_model)
