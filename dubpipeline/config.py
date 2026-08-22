from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

from dubpipeline.models.catalog import (
    get_model_status,
    get_model_spec,
    infer_model_id_from_legacy_translate,
    legacy_translate_backend_for_model,
    resolve_model_spec,
    resolve_default_model_id,
)
from dubpipeline.translation.providers import (
    PUBLIC_TRANSLATION_PROVIDERS,
    QWEN_QUANTIZATION_VALUES,
)
from dubpipeline.utils.logging import info, warn

# =============================================================================
# Config "single source of truth"
# defaults (code) -> pipeline.yaml -> ENV -> CLI args
#
# ENV naming:
#   DB_<GRP>__<KEY>[__<SUBKEY>...]=value
# where <GRP> is 3-letter group code:
#   GEN (root), PTH (paths), FFM (ffmpeg), WHX (whisperx), TRN (translate),
#   TTS (tts), MUX (mux), SEP (source_separation), STP (steps)
#
# Legacy ENV supported (for backward compatibility with older runs):
#   DUBPIPELINE_*
# =============================================================================


# -------------------------
# Dataclasses (typed config)
# -------------------------


@dataclass
class LanguagesConfig:
    src: str = "en"
    tgt: str = "ru"


SUPPORTED_TRANSLATION_LANGUAGES: tuple[str, ...] = ("de", "fr", "es", "ru")
SUPPORTED_SOURCE_LANGUAGES: tuple[str, ...] = (
    "auto",
    "en",
) + SUPPORTED_TRANSLATION_LANGUAGES
LEGACY_TRANSLATION_LANGUAGES: tuple[str, ...] = ("en",)
MUX_LANGUAGE_TAGS: dict[str, str] = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "ru": "rus",
}
TARGET_TRACK_TITLES: dict[str, str] = {
    "de": "German (DubPipeline)",
    "fr": "French (DubPipeline)",
    "es": "Spanish (DubPipeline)",
    "ru": "Russian (DubPipeline)",
}


def normalize_language_code(value: str | None, *, default: str) -> str:
    code = (value or "").strip().lower()
    return code or default


def supported_translation_languages(*, allow_legacy: bool = False) -> tuple[str, ...]:
    if allow_legacy:
        return LEGACY_TRANSLATION_LANGUAGES + SUPPORTED_TRANSLATION_LANGUAGES
    return SUPPORTED_TRANSLATION_LANGUAGES


def validate_translation_language_pair(
    src_lang: str | None,
    tgt_lang: str | None,
    *,
    translate_enabled: bool,
    allow_legacy: bool = False,
) -> str | None:
    src = normalize_language_code(src_lang, default=LanguagesConfig().src)
    tgt = normalize_language_code(tgt_lang, default=LanguagesConfig().tgt)
    source_allowed = SUPPORTED_SOURCE_LANGUAGES
    target_allowed = supported_translation_languages(allow_legacy=allow_legacy)
    source_allowed_text = ", ".join(SUPPORTED_SOURCE_LANGUAGES)
    target_allowed_text = ", ".join(SUPPORTED_TRANSLATION_LANGUAGES)

    if src == "auto" and translate_enabled:
        return "Source language 'auto' cannot be used when the Translate step is enabled. Set --lang-src to a concrete language, for example en."
    if src not in source_allowed:
        return f"Unsupported source language '{src}'. Supported languages: {source_allowed_text}."
    if tgt not in target_allowed:
        return f"Unsupported target language '{tgt}'. Supported languages: {target_allowed_text}."
    if translate_enabled and src == tgt:
        return "Source and target languages must be different when the Translate step is enabled."
    return None


def mux_language_tag(lang_code: str | None, *, default: str) -> str:
    code = normalize_language_code(lang_code, default=default)
    return MUX_LANGUAGE_TAGS.get(code, default)


def default_target_track_title(lang_code: str | None) -> str:
    code = normalize_language_code(lang_code, default=LanguagesConfig().tgt)
    return TARGET_TRACK_TITLES.get(code, f"{code.upper()} (DubPipeline)")


class AudioUpdateMode(str, Enum):
    ADD = "add"
    OVERWRITE = "overwrite"
    OVERWRITE_REORDER = "overwrite_reorder"


_AUDIO_MODE_ALIASES: dict[str, str] = {
    "add": AudioUpdateMode.ADD.value,
    "добавление": AudioUpdateMode.ADD.value,
    "overwrite": AudioUpdateMode.OVERWRITE.value,
    "replace": AudioUpdateMode.OVERWRITE.value,
    "замена": AudioUpdateMode.OVERWRITE.value,
    "overwrite+reorder": AudioUpdateMode.OVERWRITE_REORDER.value,
    "overwrite + reorder": AudioUpdateMode.OVERWRITE_REORDER.value,
    "overwrite_reorder": AudioUpdateMode.OVERWRITE_REORDER.value,
    "rus_first": AudioUpdateMode.OVERWRITE_REORDER.value,
    "изменить порядок": AudioUpdateMode.OVERWRITE_REORDER.value,
    "русская дорожка первой": AudioUpdateMode.OVERWRITE_REORDER.value,
}

_INPUT_MODE_ALIASES: dict[str, str] = {
    "file": "file",
    "single": "file",
    "single_file": "file",
    "dir": "dir",
    "folder": "dir",
    "directory": "dir",
}


def normalize_audio_update_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return AudioUpdateMode.ADD.value
    return _AUDIO_MODE_ALIASES.get(raw, AudioUpdateMode.ADD.value)


def normalize_input_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "file"
    return _INPUT_MODE_ALIASES.get(raw, "file")


@dataclass
class StepsConfig:
    extract_audio: bool = True
    source_separation: bool = True
    asr_whisperx: bool = True
    translate: bool = True
    tts: bool = True
    align: bool = True
    merge: bool = True


@dataclass
class PathsTemplatesConfig:
    audio_wav: str = "{out_dir}/{project_name}.wav"
    separation_dir: str = "{out_dir}/separation/{project_name}"
    separation_vocals_wav: str = "{out_dir}/separation/{project_name}/vocals.wav"
    separation_background_wav: str = (
        "{out_dir}/separation/{project_name}/background.wav"
    )
    separation_metadata_json: str = "{out_dir}/separation/{project_name}/metadata.json"
    segments_json: str = "{out_dir}/{project_name}.segments.json"
    segments_tgt_json: str = "{out_dir}/{project_name}.segments.{target_lang}.json"
    srt_en: str = "{out_dir}/{project_name}.srt"
    tts_segments_dir: str = "{out_dir}/segments/tts_{target_lang}_segments"
    tts_segments_aligned_dir: str = (
        "{out_dir}/segments/tts_{target_lang}_segments_aligned"
    )
    final_video: str = "{out_dir}/{project_name}.{target_lang}.muxed.mp4"

    @property
    def segments_ru_json(self) -> str:
        return self.segments_tgt_json

    @segments_ru_json.setter
    def segments_ru_json(self, value: str) -> None:
        self.segments_tgt_json = value


@dataclass
class PathsConfig:
    # input-ish
    workdir: Path = Path(".")
    out_dir: Path = Path("out")
    input_video: Path = Path()
    input_text: Path | None = None

    # derived/outputs
    audio_wav: Path = Path()
    separation_dir: Path = Path()
    separation_vocals_wav: Path = Path()
    separation_background_wav: Path = Path()
    separation_metadata_json: Path = Path()
    segments_file: Path = Path()
    segments_ru_file: Path = Path()
    srt_file_en: Path = Path()
    tts_segments_dir: Path = Path()
    tts_segments_aligned_dir: Path = Path()
    final_video: Path = Path()
    final_audio: Path | None = None
    voice_input_wav: Path = Path()
    translated_voice_wav: Path = Path()
    background_wav: Path = Path()
    mixed_wav: Path = Path()

    # templates (keep for debugging / printing)
    templates: PathsTemplatesConfig = field(default_factory=PathsTemplatesConfig)

    # ---------------------------------------------------------------------
    # Backward-compat aliases (TEMP)
    #
    # Older step code used:
    #   cfg.paths.segments_path        -> directory with raw TTS segments
    #   cfg.paths.segments_align_path  -> directory with aligned TTS segments
    #
    # New names are:
    #   cfg.paths.tts_segments_dir
    #   cfg.paths.tts_segments_aligned_dir
    #
    # Keep these properties during transition / tests, then remove later.
    # ---------------------------------------------------------------------

    @property
    def segments_tgt_file(self) -> Path:  # noqa: D401
        """Preferred alias for translated segments file."""
        return self.segments_ru_file

    @segments_tgt_file.setter
    def segments_tgt_file(self, value: Path | str) -> None:
        self.segments_ru_file = Path(value)

    @property
    def segments_path(self) -> Path:  # noqa: D401
        """DEPRECATED alias for tts_segments_dir."""
        return self.tts_segments_dir

    @segments_path.setter
    def segments_path(self, value: Path | str) -> None:
        self.tts_segments_dir = Path(value)

    @property
    def segments_align_path(self) -> Path:  # noqa: D401
        """DEPRECATED alias for tts_segments_aligned_dir."""
        return self.tts_segments_aligned_dir

    @segments_align_path.setter
    def segments_align_path(self, value: Path | str) -> None:
        self.tts_segments_aligned_dir = Path(value)


@dataclass
class FfmpegConfig:
    bin: str = "ffmpeg"
    sample_rate: int = 16_000
    channels: int = 1
    audio_codec: str = "pcm_s16le"
    audio_bitrate: str = "128k"


@dataclass
class WhisperxWordMergeConfig:
    max_seg_dur: float = 20.0
    max_seg_chars: int = 350
    min_seg_dur: float = 1.0
    min_seg_chars: int = 25
    merge_max_gap: float = 0.35
    max_seg_dur_post: float = 12.0
    allow_cross_speaker: bool = True


@dataclass
class WhisperxConfig:
    model_name: str = "large-v3"
    batch_size: int = 1
    max_gap_between_words: float = 0.8
    word_merge: WhisperxWordMergeConfig = field(default_factory=WhisperxWordMergeConfig)
    release_vram: bool = True


@dataclass
class TranslateConfig:
    backend: str = "auto"  # auto|argos|hf
    hf_model: str = "Helsinki-NLP/opus-mt-en-ru"
    batch_size: int = 64
    max_new_tokens: int = 256
    cache_db: str = ""
    release_vram: bool = True


@dataclass
class TranslationConfig:
    provider: str = ""
    model: str = ""
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "auto"
    max_new_tokens: int = 0
    prompt: str = ""
    keep_loaded_between_files: bool = True
    offload_after_translate: bool = True
    backend: str = ""
    model_id: str = ""
    model_ref: str = ""


@dataclass
class TtsConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voice: str = ""
    preview_text: str = "Это тестовое воспроизведение выбранного голоса."
    sample_rate: int = 22_050

    speaker_wav: str = ""  # optional reference wav for voice cloning
    warn_limit_chars_ru: int = 182
    max_ru_chars: int = 170
    text_max_chars: int = 400

    gap_ms: int = 80
    breaks: list[str] = field(
        default_factory=lambda: [". ", "! ", "? ", "; ", ": ", " — ", ", "]
    )

    fast_latents: bool = True
    try_single_call: bool = True
    try_single_call_max_chars: int = 1200

    @property
    def max_target_chars(self) -> int:
        return self.max_ru_chars

    @max_target_chars.setter
    def max_target_chars(self, value: int) -> None:
        self.max_ru_chars = int(value)


@dataclass
class MuxConfig:
    ffmpeg_bin: str = "ffmpeg"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    orig_track_title: str = "Original"
    ru_track_title: str = "Russian (DubPipeline)"
    orig_lang: str = "eng"
    ru_lang: str = "rus"

    @property
    def target_track_title(self) -> str:
        return self.ru_track_title

    @target_track_title.setter
    def target_track_title(self, value: str) -> None:
        self.ru_track_title = value

    @property
    def target_lang(self) -> str:
        return self.ru_lang

    @target_lang.setter
    def target_lang(self, value: str) -> None:
        self.ru_lang = value


@dataclass
class OutputConfig:
    move_to_dir: str = ""
    update_existing_file: bool = False
    audio_update_mode: str = AudioUpdateMode.ADD.value


@dataclass
class AudioMergeDuckingConfig:
    enabled: bool = True
    amount_db: float = 10.0
    threshold_db: float = -30.0
    attack_ms: int = 10
    release_ms: int = 250
    ratio: float = 6.0
    knee_db: float = 6.0


@dataclass
class AudioMergeLoudnessConfig:
    enabled: bool = True
    target_i: float = -16.0
    true_peak: float = -1.5


@dataclass
class AudioMergeConfig:
    mode: str = ""
    original_track: str = "auto"
    tts_gain_db: float = 0.0
    original_gain_db: float = 0.0
    ducking: AudioMergeDuckingConfig = field(default_factory=AudioMergeDuckingConfig)
    loudness: AudioMergeLoudnessConfig = field(default_factory=AudioMergeLoudnessConfig)


@dataclass
class SourceSeparationConfig:
    mode: str = "legacy_ducking"
    provider: str = "bs_roformer"
    model_path: str = ""
    command: list[str] = field(default_factory=list)
    fallback_mode: str = "none"
    cache_enabled: bool = True


@dataclass
class PipelineConfig:
    # general
    project_name: str
    project_dir: Path
    mode: str = "Добавление"  # Добавление|Замена
    usegpu: bool = True
    use_existing_subtitles: bool = False
    delete_srt: bool = True
    rebuild: bool = False
    cleanup: bool = False
    keep_temp: bool = False
    batch_file_count: int = 1

    languages: LanguagesConfig = field(default_factory=LanguagesConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    whisperx: WhisperxConfig = field(default_factory=WhisperxConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    mux: MuxConfig = field(default_factory=MuxConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    audio_merge: AudioMergeConfig = field(default_factory=AudioMergeConfig)
    source_separation: SourceSeparationConfig = field(
        default_factory=SourceSeparationConfig
    )

    @property
    def device(self) -> str:
        """return device name."""
        return "cuda" if torch.cuda.is_available() and self.usegpu else "cpu"

    @property
    def compute_type(self) -> str:
        """return compute_type."""

        compute_type = "float16" if self.device == "cuda" else "int8"
        return compute_type


# -------------------------
# Defaults for pipeline.yaml
# -------------------------

DEFAULT_PIPELINE_DICT: Dict[str, Any] = {
    "project_name": "video_sample",
    "input_mode": "file",
    "input_path": "",
    "mode": "Добавление",
    "usegpu": True,
    "use_existing_subtitles": False,
    "delete_srt": True,
    "rebuild": False,
    "cleanup": False,
    "keep_temp": False,
    "batch_file_count": 1,
    "languages": asdict(LanguagesConfig()),
    "steps": asdict(StepsConfig()),
    "paths": {
        "workdir": ".",
        "out_dir": "out",
        "input_video": "{project_name}.mp4",
        "input_text": None,
        "final_audio": None,
        "templates": asdict(PathsTemplatesConfig()),
    },
    "ffmpeg": asdict(FfmpegConfig()),
    "whisperx": asdict(WhisperxConfig()),
    "translate": asdict(TranslateConfig()),
    "translation": asdict(TranslationConfig()),
    "tts": asdict(TtsConfig()),
    "mux": asdict(MuxConfig()),
    "output": {
        "move_to_dir": "",
        "update_existing_file": False,
    },
    "audio_merge": asdict(AudioMergeConfig()),
    "source_separation": asdict(SourceSeparationConfig()),
}


# -------------------------
# Helper functions
# -------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (dicts merged, other types replaced)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(value: str) -> Any:
    s = value.strip()
    if not s:
        return ""
    low = s.lower()
    if low in {"true", "1", "yes", "y", "on"}:
        return True
    if low in {"false", "0", "no", "n", "off"}:
        return False
    if low in {"null", "none"}:
        return None
    # json (lists/dicts)
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            return json.loads(s)
        except Exception:
            return s
    # int / float
    try:
        if re.match(r"^-?\d+$", s):
            return int(s)
        if re.match(r"^-?\d+\.\d+$", s):
            return float(s)
    except Exception:
        pass
    return s


import re  # after helper uses it


def _set_by_path(d: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    cur = d
    parts = list(path)
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def _env_to_overrides(environ: dict[str, str] | None = None) -> Dict[str, Any]:
    env = dict(environ or os.environ)

    grp_map = {
        "GEN": "",  # root
        "PTH": "paths",
        "FFM": "ffmpeg",
        "WHX": "whisperx",
        "TRN": "translate",
        "TLN": "translation",
        "TTS": "tts",
        "MUX": "mux",
        "AMR": "audio_merge",
        "SEP": "source_separation",
        "STP": "steps",
    }

    legacy_map = {
        # TTS
        "DUBPIPELINE_TTS_MAX_RU_CHARS": "tts.max_ru_chars",
        "DUBPIPELINE_TTS_FAST_LATENTS": "tts.fast_latents",
        "DUBPIPELINE_TTS_TRY_SINGLE_CALL": "tts.try_single_call",
        "DUBPIPELINE_TTS_TRY_SINGLE_CALL_MAX_CHARS": "tts.try_single_call_max_chars",
        "DUBPIPELINE_TTS_PREVIEW_TEXT": "tts.preview_text",
        # WhisperX word merge
        "DUBPIPELINE_WORD_MERGE_MAX_SEG_DUR": "whisperx.word_merge.max_seg_dur",
        "DUBPIPELINE_WORD_MERGE_MAX_SEG_CHARS": "whisperx.word_merge.max_seg_chars",
        "DUBPIPELINE_MIN_SEG_DUR": "whisperx.word_merge.min_seg_dur",
        "DUBPIPELINE_MIN_SEG_CHARS": "whisperx.word_merge.min_seg_chars",
        "DUBPIPELINE_MERGE_MAX_GAP": "whisperx.word_merge.merge_max_gap",
        "DUBPIPELINE_MAX_SEG_DUR": "whisperx.word_merge.max_seg_dur_post",
        "DUBPIPELINE_MERGE_ALLOW_CROSS_SPEAKER": "whisperx.word_merge.allow_cross_speaker",
        "DUBPIPELINE_WHISPERX_RELEASE_VRAM": "whisperx.release_vram",
        # Translate
        "DUBPIPELINE_TRANSLATE_BACKEND": "translate.backend",
        "DUBPIPELINE_HF_MODEL": "translate.hf_model",
        "DUBPIPELINE_TRANSLATE_BATCH": "translate.batch_size",
        "DUBPIPELINE_TRANSLATE_MAX_NEW_TOKENS": "translate.max_new_tokens",
        "DUBPIPELINE_CACHE_DB": "translate.cache_db",
        "DUBPIPELINE_TRANSLATE_RELEASE_VRAM": "translate.release_vram",
        # Translation model selector (new unified source of truth)
        "DUBPIPELINE_TRANSLATION_BACKEND": "translation.backend",
        "DUBPIPELINE_TRANSLATION_MODEL": "translation.model",
        "DUBPIPELINE_TRANSLATION_DEVICE": "translation.device",
        "DUBPIPELINE_TRANSLATION_DTYPE": "translation.dtype",
        "DUBPIPELINE_TRANSLATION_QUANTIZATION": "translation.quantization",
        "DUBPIPELINE_TRANSLATION_MAX_NEW_TOKENS": "translation.max_new_tokens",
        "DUBPIPELINE_TRANSLATION_PROMPT": "translation.prompt",
        "DUBPIPELINE_TRANSLATION_KEEP_LOADED_BETWEEN_FILES": "translation.keep_loaded_between_files",
        "DUBPIPELINE_TRANSLATION_OFFLOAD_AFTER_TRANSLATE": "translation.offload_after_translate",
        "DUBPIPELINE_TRANSLATION_MODEL_ID": "translation.model_id",
        "DUBPIPELINE_TRANSLATION_MODEL_REF": "translation.model_ref",
        "DUBPIPELINE_TRANSLATION_PROVIDER": "translation.provider",
        # Output
        "DUBPIPELINE_OUTPUT_MOVE_TO_DIR": "output.move_to_dir",
        "DUBPIPELINE_OUTPUT_UPDATE_EXISTING_FILE": "output.update_existing_file",
        "DUBPIPELINE_OUTPUT_AUDIO_UPDATE_MODE": "output.audio_update_mode",
        # Source separation
        "DUBPIPELINE_SOURCE_SEPARATION_MODE": "source_separation.mode",
        "DUBPIPELINE_SOURCE_SEPARATION_PROVIDER": "source_separation.provider",
        "DUBPIPELINE_SOURCE_SEPARATION_MODEL_PATH": "source_separation.model_path",
        "DUBPIPELINE_SOURCE_SEPARATION_FALLBACK_MODE": "source_separation.fallback_mode",
        "DUBPIPELINE_SOURCE_SEPARATION_CACHE_ENABLED": "source_separation.cache_enabled",
        # Existing subtitles mode
        "DUBPIPELINE_USE_EXISTING_SUBTITLES": "use_existing_subtitles",
    }

    overrides: Dict[str, Any] = {}

    # New-style DB_* keys
    for key, val in env.items():
        if not key.startswith("DB_"):
            continue
        rest = key[3:]  # after DB_
        parts = rest.split("__")
        # support DB_TTS_MAX_RU_CHARS (no __)
        if len(parts) == 1:
            one = parts[0]
            if "_" in one:
                grp, tail = one.split("_", 1)
                parts = [grp, tail]
            else:
                parts = [one]
        grp = parts[0].upper()
        root = grp_map.get(grp)
        if root is None:
            continue
        sub = parts[1:]
        if not sub:
            continue
        path_parts = []
        if root:
            path_parts.append(root)
        path_parts.extend([p.lower() for p in sub])
        _set_by_path(overrides, path_parts, _parse_scalar(val))

    # Legacy DUBPIPELINE_* keys
    for key, path in legacy_map.items():
        if key in env:
            _set_by_path(overrides, path.split("."), _parse_scalar(env[key]))

    return overrides


def _parse_cli_set_args(items: list[str]) -> Dict[str, Any]:
    """Parse --set a.b.c=123 into overrides dict."""
    overrides: Dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            warn(f"[config] Ignoring --set '{raw}': expected KEY=VALUE")
            continue
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            continue
        path = [p.strip() for p in k.split(".") if p.strip()]
        if not path:
            continue
        _set_by_path(overrides, path, _parse_scalar(v))
    return overrides


def _format_all_strings(obj: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _format_all_strings(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_format_all_strings(v, variables) for v in obj]
    if isinstance(obj, str):
        try:
            return obj.format(**variables)
        except Exception:
            return obj
    return obj


def _resolve_paths(
    raw: Dict[str, Any], project_dir: Path, *, create_dirs: bool = True
) -> PathsConfig:
    paths = raw.get("paths", {}) or {}
    tmpl = paths.get("templates", {}) or {}
    # Backward-compat (old keys)
    if "deleteSRT" in raw and "delete_srt" not in raw:
        raw["delete_srt"] = raw.get("deleteSRT")

    workdir = Path(paths.get("workdir", "."))
    if not workdir.is_absolute():
        workdir = (project_dir / workdir).resolve()

    out_dir = Path(paths.get("out_dir", "out"))
    if not out_dir.is_absolute():
        out_dir = (workdir / out_dir).resolve()

    if create_dirs:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    legacy_input_dir = raw.get("input_dir")

    # input_video can be at root/in paths; for new GUI schema also support input_path.
    # Priority keeps backward compatibility with existing YAML.
    input_video_s = (
        raw.get("input_path")
        or paths.get("input_video")
        or raw.get("input_video")
        or legacy_input_dir
        or "{project_name}.mp4"
    )
    languages = raw.get("languages") or {}
    source_lang = (
        str(languages.get("src", LanguagesConfig().src) or LanguagesConfig().src)
        .strip()
        .lower()
        or LanguagesConfig().src
    )
    target_lang = (
        str(languages.get("tgt", LanguagesConfig().tgt) or LanguagesConfig().tgt)
        .strip()
        .lower()
        or LanguagesConfig().tgt
    )
    variables = {
        "project_name": raw.get("project_name", ""),
        "project_dir": str(project_dir),
        "workdir": str(workdir),
        "out_dir": str(out_dir),
        "source_lang": source_lang,
        "src_lang": source_lang,
        "target_lang": target_lang,
        "tgt_lang": target_lang,
    }
    input_video_s = _format_all_strings(input_video_s, variables)
    input_video = Path(input_video_s)
    if not input_video.is_absolute():
        input_video = (workdir / input_video).resolve()

    # templates (with defaults)
    default_tmpl = asdict(PathsTemplatesConfig())
    merged_tmpl = _deep_merge(default_tmpl, tmpl)

    # Backward-compat for older YAML templates:
    # - templates.segments_path        -> templates.tts_segments_dir
    # - templates.segments_align_path  -> templates.tts_segments_aligned_dir
    # If user provided the old key but not the new one, prefer the old key.
    if isinstance(tmpl, dict):
        if "segments_ru_json" in tmpl and (
            "segments_tgt_json" not in tmpl
            or tmpl.get("segments_tgt_json") == default_tmpl.get("segments_tgt_json")
        ):
            merged_tmpl["segments_tgt_json"] = tmpl.get("segments_ru_json")
        if "segments_path" in tmpl and (
            "tts_segments_dir" not in tmpl
            or tmpl.get("tts_segments_dir") == default_tmpl.get("tts_segments_dir")
        ):
            merged_tmpl["tts_segments_dir"] = tmpl.get("segments_path")
        if "segments_align_path" in tmpl and (
            "tts_segments_aligned_dir" not in tmpl
            or tmpl.get("tts_segments_aligned_dir")
            == default_tmpl.get("tts_segments_aligned_dir")
        ):
            merged_tmpl["tts_segments_aligned_dir"] = tmpl.get("segments_align_path")
        if "tts_segments_align_dir" in tmpl and (
            "tts_segments_aligned_dir" not in tmpl
            or tmpl.get("tts_segments_aligned_dir")
            == default_tmpl.get("tts_segments_aligned_dir")
        ):
            merged_tmpl["tts_segments_aligned_dir"] = tmpl.get("tts_segments_align_dir")
        if "srt_file_en" in tmpl and (
            "srt_en" not in tmpl or tmpl.get("srt_en") == default_tmpl.get("srt_en")
        ):
            merged_tmpl["srt_en"] = tmpl.get("srt_file_en")

    merged_tmpl = _format_all_strings(merged_tmpl, variables)

    def _p(s: str) -> Path:
        p = Path(s)
        if not p.is_absolute():
            p = (workdir / p).resolve()
        return p

    template_values = default_tmpl | {
        key: str(value) for key, value in merged_tmpl.items() if key in default_tmpl
    }

    return PathsConfig(
        workdir=workdir,
        out_dir=out_dir,
        input_video=input_video,
        audio_wav=_p(merged_tmpl["audio_wav"]),
        separation_dir=_p(merged_tmpl["separation_dir"]),
        separation_vocals_wav=_p(merged_tmpl["separation_vocals_wav"]),
        separation_background_wav=_p(merged_tmpl["separation_background_wav"]),
        separation_metadata_json=_p(merged_tmpl["separation_metadata_json"]),
        segments_file=_p(merged_tmpl["segments_json"]),
        segments_ru_file=_p(merged_tmpl["segments_tgt_json"]),
        srt_file_en=_p(merged_tmpl["srt_en"]),
        tts_segments_dir=_p(merged_tmpl["tts_segments_dir"]),
        tts_segments_aligned_dir=_p(merged_tmpl["tts_segments_aligned_dir"]),
        final_video=_p(merged_tmpl["final_video"]),
        input_text=(
            Path(paths.get("input_text")).resolve() if paths.get("input_text") else None
        ),
        final_audio=_p(paths.get("final_audio")) if paths.get("final_audio") else None,
        templates=PathsTemplatesConfig(**template_values),
    )


# -------------------------
# Public API
# -------------------------

# default template path (used by GUI helpers)
pipeline_path = Path(__file__).parent / "video.pipeline.yaml"


def get_voice() -> str:
    cfg = load_pipeline_config_ex(pipeline_path)
    return cfg.tts.voice


def load_pipeline_config_ex(
    pipeline_file: Path,
    *,
    cli_set: Optional[list[str]] = None,
    create_dirs: bool = True,
) -> PipelineConfig:
    """Load config with precedence: defaults -> yaml -> env -> cli."""
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Config not found: {pipeline_file}")

    project_dir = pipeline_file.parent.parent.resolve()

    # yaml
    with pipeline_file.open("r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    # normalize some legacy keys in YAML
    if "deleteSRT" in yaml_cfg and "delete_srt" not in yaml_cfg:
        yaml_cfg["delete_srt"] = yaml_cfg.get("deleteSRT")
    if "usegpu" in yaml_cfg and "use_gpu" not in yaml_cfg:
        yaml_cfg["use_gpu"] = yaml_cfg.get("usegpu")

    # merge layers
    merged = _deep_merge(DEFAULT_PIPELINE_DICT, yaml_cfg)
    merged = _deep_merge(merged, _env_to_overrides())
    merged = _deep_merge(merged, _parse_cli_set_args(cli_set or []))

    # resolve paths (needs project_dir)
    paths_cfg = _resolve_paths(merged, project_dir, create_dirs=create_dirs)

    # build nested configs (typed)
    languages = LanguagesConfig(**(merged.get("languages") or {}))
    steps = StepsConfig(**(merged.get("steps") or {}))
    ffmpeg = FfmpegConfig(**(merged.get("ffmpeg") or {}))

    whisperx_raw = merged.get("whisperx") or {}
    wm = WhisperxWordMergeConfig(**(whisperx_raw.get("word_merge") or {}))
    whisperx = WhisperxConfig(
        model_name=str(whisperx_raw.get("model_name", WhisperxConfig().model_name)),
        batch_size=int(whisperx_raw.get("batch_size", WhisperxConfig().batch_size)),
        max_gap_between_words=float(
            whisperx_raw.get(
                "max_gap_between_words", WhisperxConfig().max_gap_between_words
            )
        ),
        word_merge=wm,
        release_vram=bool(whisperx_raw.get("release_vram", True)),
    )

    translate = TranslateConfig(**(merged.get("translate") or {}))
    translation_raw = merged.get("translation") or {}
    translation = TranslationConfig(**translation_raw)
    translation.provider = str(translation.provider or "").strip().lower()
    translation.model = str(translation.model or "").strip()
    translation.device = str(translation.device or "auto").strip().lower() or "auto"
    translation.dtype = str(translation.dtype or "auto").strip().lower() or "auto"
    translation.quantization = (
        str(translation.quantization or "auto").strip().lower() or "auto"
    )
    translation.max_new_tokens = int(translation.max_new_tokens or 0)
    translation.keep_loaded_between_files = bool(translation.keep_loaded_between_files)
    translation.offload_after_translate = bool(translation.offload_after_translate)
    if translation.quantization not in QWEN_QUANTIZATION_VALUES:
        warn(
            f"[config] Unknown translation.quantization='{translation.quantization}'. "
            "Runtime will report a Qwen quantization error."
        )

    translate_defaults = TranslateConfig()
    legacy_model_id = None
    if (
        str(translate.backend).strip().lower() != translate_defaults.backend
        or str(translate.hf_model).strip() != translate_defaults.hf_model
    ):
        legacy_model_id = infer_model_id_from_legacy_translate(
            translate.backend,
            translate.hf_model,
        )
    if translation.provider == "argos":
        translation.model_id = "argos"
    if translation.provider == "qwen":
        translation.model_id = "qwen3_8b"
    if not translation.model_id:
        translation.model_id = legacy_model_id or resolve_default_model_id(
            languages.src, languages.tgt
        )
    if (
        translation.provider
        and translation.provider not in PUBLIC_TRANSLATION_PROVIDERS
    ):
        warn(
            f"[config] Unknown translation.provider='{translation.provider}'. Runtime will report a provider error."
        )

    try:
        model_spec = resolve_model_spec(
            translation.model_id, languages.src, languages.tgt
        )
    except ValueError:
        warn(
            f"[config] Unknown translation.model_id='{translation.model_id}', falling back to default."
        )
        translation.model_id = resolve_default_model_id(languages.src, languages.tgt)
        model_spec = resolve_model_spec(
            translation.model_id, languages.src, languages.tgt
        )

    model_status = get_model_status(translation.model_id, languages.src, languages.tgt)
    if str(model_status.reason or "").startswith("unsupported for "):
        warn(
            f"[config] translation.model_id='{translation.model_id}' is {model_status.reason}; "
            "keeping selection but runtime install/usage will be disabled."
        )

    translation.backend = model_spec.backend
    translation.model_ref = translation.model or model_spec.model_ref

    # Keep legacy translate.* fields synchronized for backward compatibility.
    translate.backend = legacy_translate_backend_for_model(model_spec)
    if translate.backend == "hf":
        translate.hf_model = translation.model_ref

    tts = TtsConfig(**(merged.get("tts") or {}))
    mux = MuxConfig(**(merged.get("mux") or {}))
    output_raw = merged.get("output") or {}
    output = OutputConfig(**output_raw)
    mode_raw = output_raw.get("audio_update_mode", merged.get("mode"))
    output.audio_update_mode = normalize_audio_update_mode(mode_raw)

    audio_merge_raw = dict(merged.get("audio_merge") or {})
    if "bg_gain_db" in audio_merge_raw and "original_gain_db" not in audio_merge_raw:
        audio_merge_raw["original_gain_db"] = audio_merge_raw.get("bg_gain_db")
    ducking_raw = dict(audio_merge_raw.get("ducking") or {})
    loudness_raw = dict(audio_merge_raw.get("loudness") or {})
    ducking_defaults = asdict(AudioMergeDuckingConfig())
    loudness_defaults = asdict(AudioMergeLoudnessConfig())
    audio_merge = AudioMergeConfig(
        mode=str(audio_merge_raw.get("mode", AudioMergeConfig().mode)),
        original_track=str(
            audio_merge_raw.get("original_track", AudioMergeConfig().original_track)
        ),
        tts_gain_db=float(
            audio_merge_raw.get("tts_gain_db", AudioMergeConfig().tts_gain_db)
        ),
        original_gain_db=float(
            audio_merge_raw.get("original_gain_db", AudioMergeConfig().original_gain_db)
        ),
        ducking=AudioMergeDuckingConfig(
            **(
                ducking_defaults
                | {k: v for k, v in ducking_raw.items() if k in ducking_defaults}
            )
        ),
        loudness=AudioMergeLoudnessConfig(
            **(
                loudness_defaults
                | {k: v for k, v in loudness_raw.items() if k in loudness_defaults}
            )
        ),
    )

    source_separation_raw = dict(merged.get("source_separation") or {})
    command_raw = source_separation_raw.get("command", SourceSeparationConfig().command)
    if isinstance(command_raw, str):
        command = [command_raw]
    elif isinstance(command_raw, list):
        command = [str(item) for item in command_raw]
    else:
        command = []
    source_separation = SourceSeparationConfig(
        mode=str(source_separation_raw.get("mode", SourceSeparationConfig().mode))
        .strip()
        .lower(),
        provider=str(
            source_separation_raw.get("provider", SourceSeparationConfig().provider)
        )
        .strip()
        .lower(),
        model_path=str(
            source_separation_raw.get("model_path", SourceSeparationConfig().model_path)
        ).strip(),
        command=command,
        fallback_mode=str(
            source_separation_raw.get(
                "fallback_mode", SourceSeparationConfig().fallback_mode
            )
        )
        .strip()
        .lower(),
        cache_enabled=bool(
            source_separation_raw.get(
                "cache_enabled", SourceSeparationConfig().cache_enabled
            )
        ),
    )

    # inherit language defaults into mux if user didn't override
    default_mux = MuxConfig()
    if not mux.orig_lang or mux.orig_lang == default_mux.orig_lang:
        mux.orig_lang = mux_language_tag(languages.src, default=default_mux.orig_lang)
    if not mux.ru_lang or mux.ru_lang == default_mux.ru_lang:
        mux.ru_lang = mux_language_tag(languages.tgt, default=default_mux.ru_lang)
    if not mux.ru_track_title or mux.ru_track_title == default_mux.ru_track_title:
        mux.ru_track_title = default_target_track_title(languages.tgt)

    # final
    cfg = PipelineConfig(
        project_name=str(merged.get("project_name") or "video_sample"),
        project_dir=project_dir,
        mode=str(merged.get("mode") or "Добавление"),
        usegpu=bool(
            merged.get("usegpu")
            if "usegpu" in merged
            else merged.get("use_gpu", merged.get("usegpu", True))
        ),
        use_existing_subtitles=bool(merged.get("use_existing_subtitles", False)),
        delete_srt=bool(merged.get("delete_srt", merged.get("deleteSRT", True))),
        rebuild=bool(merged.get("rebuild", False)),
        cleanup=bool(merged.get("cleanup", False)),
        keep_temp=bool(merged.get("keep_temp", False)),
        batch_file_count=int(merged.get("batch_file_count", 1) or 1),
        languages=languages,
        steps=steps,
        paths=paths_cfg,
        ffmpeg=ffmpeg,
        whisperx=whisperx,
        translate=translate,
        translation=translation,
        tts=tts,
        mux=mux,
        output=output,
        audio_merge=audio_merge,
        source_separation=source_separation,
    )

    info("[dubpipeline] Config loaded (defaults -> yaml -> env -> cli).\n")
    return cfg


def save_pipeline_yaml(values, pipeline_path: Path) -> Path:
    """
    GUI helper:
    values — dict from window.read()
    pipeline_path — where to save *.pipeline.yaml
    """
    # Start from template if exists, otherwise defaults
    template = Path(__file__).parent / "video.pipeline.yaml"
    if template.exists():
        with template.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(DEFAULT_PIPELINE_DICT, cfg)
    else:
        cfg = copy.deepcopy(DEFAULT_PIPELINE_DICT)

    project_name = values.get("-PROJECT-", "").strip() or cfg.get(
        "project_name", "video_sample"
    )
    out_dir = values.get("-OUT-", "").strip() or (cfg.get("paths", {}) or {}).get(
        "out_dir", "out"
    )
    source_mode = "dir" if bool(values.get("-SRC_DIR-", False)) else "file"
    if "-INPUT_MODE-" in values:
        source_mode = normalize_input_mode(values.get("-INPUT_MODE-"))

    input_path = values.get("-INPUT_PATH-", "").strip()
    if not input_path and source_mode == "dir":
        input_path = values.get("-IN_DIR-", "").strip()
    if not input_path:
        input_path = values.get("-IN-", "").strip()
    if not input_path:
        input_path = (cfg.get("paths", {}) or {}).get(
            "input_video", "{project_name}.mp4"
        )

    selected_mode = values.get("-MODES-", cfg.get("mode", "Добавление"))

    cfg["project_name"] = project_name
    cfg["input_mode"] = source_mode
    cfg["input_path"] = input_path
    cfg["mode"] = selected_mode
    cfg["usegpu"] = bool(values.get("-GPU-", True))
    cfg["use_existing_subtitles"] = bool(
        values.get("-USE_EXISTING_SUBTITLES-", cfg.get("use_existing_subtitles", False))
    )
    cfg["rebuild"] = bool(values.get("-REBUILD-", False))
    cfg["delete_srt"] = bool(values.get("-SRT-", False))
    cfg["cleanup"] = bool(values.get("-CLEANUP-", False))
    steps_values = values.get("-STEPS-")
    if isinstance(steps_values, dict):
        cfg.setdefault("steps", {})
        for key, value in steps_values.items():
            cfg["steps"][key] = bool(value)

    cfg.setdefault("paths", {})
    cfg["paths"]["out_dir"] = out_dir
    cfg["paths"]["input_video"] = input_path
    cfg["input_video"] = input_path
    if source_mode == "dir":
        cfg["input_dir"] = input_path

    # voice
    cfg.setdefault("tts", {})
    cfg["tts"]["voice"] = values.get("-VOICE-", cfg["tts"].get("voice", ""))

    selected_src_lang = normalize_language_code(
        values.get("-LANG_SRC-", (cfg.get("languages") or {}).get("src", "en")),
        default="en",
    )
    selected_tgt_lang = normalize_language_code(
        values.get("-LANG_DST-", (cfg.get("languages") or {}).get("tgt", "ru")),
        default="ru",
    )
    cfg.setdefault("languages", {})
    cfg["languages"]["src"] = selected_src_lang
    cfg["languages"]["tgt"] = selected_tgt_lang

    selected_model_id = (
        values.get("-TRANSLATION_MODEL_ID-", "")
        or (cfg.get("translation", {}) or {}).get("model_id", "")
    ).strip()
    src_lang = selected_src_lang
    tgt_lang = selected_tgt_lang
    if not selected_model_id:
        selected_model_id = resolve_default_model_id(src_lang, tgt_lang)

    try:
        model_spec = resolve_model_spec(selected_model_id, src_lang, tgt_lang)
    except ValueError:
        model_spec = resolve_model_spec(
            resolve_default_model_id(src_lang, tgt_lang), src_lang, tgt_lang
        )

    cfg.setdefault("translation", {})
    cfg["translation"]["provider"] = cfg["translation"].get("provider") or ""
    cfg["translation"]["model"] = cfg["translation"].get("model") or ""
    cfg["translation"]["device"] = cfg["translation"].get("device") or "auto"
    cfg["translation"]["dtype"] = cfg["translation"].get("dtype") or "auto"
    cfg["translation"]["quantization"] = (
        cfg["translation"].get("quantization") or "auto"
    )
    cfg["translation"]["max_new_tokens"] = int(
        cfg["translation"].get("max_new_tokens") or 0
    )
    cfg["translation"]["prompt"] = cfg["translation"].get("prompt") or ""
    cfg["translation"]["keep_loaded_between_files"] = bool(
        cfg["translation"].get("keep_loaded_between_files", True)
    )
    cfg["translation"]["offload_after_translate"] = bool(
        cfg["translation"].get("offload_after_translate", True)
    )
    cfg["translation"]["model_id"] = model_spec.id
    cfg["translation"]["backend"] = model_spec.backend
    cfg["translation"]["model_ref"] = (
        cfg["translation"]["model"] or model_spec.model_ref
    )

    # Keep old keys in sync for compatibility with older scripts/tools.
    cfg.setdefault("translate", {})
    cfg["translate"]["backend"] = legacy_translate_backend_for_model(model_spec)
    if cfg["translate"]["backend"] == "hf":
        cfg["translate"]["hf_model"] = model_spec.model_ref

    cfg.setdefault("output", {})
    cfg["output"]["move_to_dir"] = values.get(
        "-MOVE_TO_DIR-", cfg["output"].get("move_to_dir", "")
    )
    cfg["output"]["update_existing_file"] = bool(
        values.get(
            "-UPDATE_EXISTING_FILE-", cfg["output"].get("update_existing_file", False)
        )
    )
    cfg["output"]["audio_update_mode"] = normalize_audio_update_mode(
        values.get("-MODES-", cfg["output"].get("audio_update_mode", selected_mode))
    )

    pipeline_path = pipeline_path.resolve()
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    with pipeline_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return pipeline_path
