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

from dubpipeline.utils.logging import info, warn


# =============================================================================
# Config "single source of truth"
# defaults (code) -> pipeline.yaml -> ENV -> CLI args
#
# ENV naming:
#   DB_<GRP>__<KEY>[__<SUBKEY>...]=value
# where <GRP> is 3-letter group code:
#   GEN (root), PTH (paths), FFM (ffmpeg), WHX (whisperx), TRN (translate),
#   TTS (tts), MUX (mux), STP (steps)
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
    asr_whisperx: bool = True
    translate: bool = True
    tts: bool = True
    align: bool = True
    merge: bool = True


@dataclass
class PathsTemplatesConfig:
    audio_wav: str = "{out_dir}/{project_name}.wav"
    segments_json: str = "{out_dir}/{project_name}.segments.json"
    segments_ru_json: str = "{out_dir}/{project_name}.segments.ru.json"
    srt_en: str = "{out_dir}/{project_name}.srt"
    tts_segments_dir: str = "{out_dir}/segments/tts_ru_segments"
    tts_segments_aligned_dir: str = "{out_dir}/segments/tts_ru_segments_aligned"
    final_video: str = "{out_dir}/{project_name}.ru.muxed.mp4"


@dataclass
class PathsConfig:
    # input-ish
    workdir: Path = Path(".")
    out_dir: Path = Path("out")
    input_video: Path = Path()

    # derived/outputs
    audio_wav: Path = Path()
    segments_file: Path = Path()
    segments_ru_file: Path = Path()
    srt_file_en: Path = Path()
    tts_segments_dir: Path = Path()
    tts_segments_aligned_dir: Path = Path()
    final_video: Path = Path()

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
class TtsConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voice: str = ""
    preview_text: str = "Это тестовое воспроизведение выбранного голоса."
    sample_rate: int = 22_050

    speaker_wav: str = ""  # optional reference wav for voice cloning
    warn_limit_chars_ru: int = 182
    max_ru_chars: int = 170

    gap_ms: int = 80
    breaks: list[str] = field(default_factory=lambda: [". ", "! ", "? ", "; ", ": ", " — ", ", "])

    fast_latents: bool = True
    try_single_call: bool = True
    try_single_call_max_chars: int = 1200


@dataclass
class MuxConfig:
    ffmpeg_bin: str = "ffmpeg"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    orig_track_title: str = "Original"
    ru_track_title: str = "Russian (DubPipeline)"
    orig_lang: str = "eng"
    ru_lang: str = "rus"


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
class PipelineConfig:
    # general
    project_name: str
    project_dir: Path
    mode: str = "Добавление"  # Добавление|Замена
    usegpu: bool = True
    delete_srt: bool = True
    rebuild: bool = False
    cleanup: bool = False
    keep_temp: bool = False

    languages: LanguagesConfig = field(default_factory=LanguagesConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    whisperx: WhisperxConfig = field(default_factory=WhisperxConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    mux: MuxConfig = field(default_factory=MuxConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    audio_merge: AudioMergeConfig = field(default_factory=AudioMergeConfig)


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
    "delete_srt": True,
    "rebuild": False,
    "cleanup": False,
    "keep_temp": False,
    "languages": asdict(LanguagesConfig()),
    "steps": asdict(StepsConfig()),
    "paths": {
        "workdir": ".",
        "out_dir": "out",
        "input_video": "{project_name}.mp4",
        "templates": asdict(PathsTemplatesConfig()),
    },
    "ffmpeg": asdict(FfmpegConfig()),
    "whisperx": asdict(WhisperxConfig()),
    "translate": asdict(TranslateConfig()),
    "tts": asdict(TtsConfig()),
    "mux": asdict(MuxConfig()),
    "output": {
        "move_to_dir": "",
        "update_existing_file": False,
    },
    "audio_merge": asdict(AudioMergeConfig()),
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
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
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
        "GEN": "",       # root
        "PTH": "paths",
        "FFM": "ffmpeg",
        "WHX": "whisperx",
        "TRN": "translate",
        "TTS": "tts",
        "MUX": "mux",
        "AMR": "audio_merge",
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
        # Output
        "DUBPIPELINE_OUTPUT_MOVE_TO_DIR": "output.move_to_dir",
        "DUBPIPELINE_OUTPUT_UPDATE_EXISTING_FILE": "output.update_existing_file",
        "DUBPIPELINE_OUTPUT_AUDIO_UPDATE_MODE": "output.audio_update_mode",
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


def _resolve_paths(raw: Dict[str, Any], project_dir: Path, *, create_dirs: bool = True) -> PathsConfig:
    paths = raw.get("paths", {}) or {}
    tmpl = (paths.get("templates", {}) or {})
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
    variables = {
        "project_name": raw.get("project_name", ""),
        "project_dir": str(project_dir),
        "workdir": str(workdir),
        "out_dir": str(out_dir),
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
        if "segments_path" in tmpl and "tts_segments_dir" not in tmpl:
            merged_tmpl["tts_segments_dir"] = merged_tmpl.get("segments_path")
        if "segments_align_path" in tmpl and "tts_segments_aligned_dir" not in tmpl:
            merged_tmpl["tts_segments_aligned_dir"] = merged_tmpl.get("segments_align_path")

    merged_tmpl = _format_all_strings(merged_tmpl, variables)

    def _p(s: str) -> Path:
        p = Path(s)
        if not p.is_absolute():
            p = (workdir / p).resolve()
        return p

    return PathsConfig(
        workdir=workdir,
        out_dir=out_dir,
        input_video=input_video,
        audio_wav=_p(merged_tmpl["audio_wav"]),
        segments_file=_p(merged_tmpl["segments_json"]),
        segments_ru_file=_p(merged_tmpl["segments_ru_json"]),
        srt_file_en=_p(merged_tmpl["srt_en"]),
        tts_segments_dir=_p(merged_tmpl["tts_segments_dir"]),
        tts_segments_aligned_dir=_p(merged_tmpl["tts_segments_aligned_dir"]),
        final_video=_p(merged_tmpl["final_video"]),
        templates=PathsTemplatesConfig(**default_tmpl | {k: str(v) for k, v in merged_tmpl.items()}),
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
        max_gap_between_words=float(whisperx_raw.get("max_gap_between_words", WhisperxConfig().max_gap_between_words)),
        word_merge=wm,
        release_vram=bool(whisperx_raw.get("release_vram", True)),
    )

    translate = TranslateConfig(**(merged.get("translate") or {}))
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
        original_track=str(audio_merge_raw.get("original_track", AudioMergeConfig().original_track)),
        tts_gain_db=float(audio_merge_raw.get("tts_gain_db", AudioMergeConfig().tts_gain_db)),
        original_gain_db=float(audio_merge_raw.get("original_gain_db", AudioMergeConfig().original_gain_db)),
        ducking=AudioMergeDuckingConfig(**(ducking_defaults | {k: v for k, v in ducking_raw.items() if k in ducking_defaults})),
        loudness=AudioMergeLoudnessConfig(**(loudness_defaults | {k: v for k, v in loudness_raw.items() if k in loudness_defaults})),
    )

    # inherit language defaults into mux if user didn't override
    if not mux.orig_lang:
        mux.orig_lang = "eng"
    if languages.src and mux.orig_lang == "eng" and languages.src.lower().startswith("en"):
        pass  # ok
    if languages.tgt and mux.ru_lang == "rus" and languages.tgt.lower().startswith("ru"):
        pass  # ok

    # final
    cfg = PipelineConfig(
        project_name=str(merged.get("project_name") or "video_sample"),
        project_dir=project_dir,
        mode=str(merged.get("mode") or "Добавление"),
        usegpu=bool(merged.get("usegpu") if "usegpu" in merged else merged.get("use_gpu", merged.get("usegpu", True))),
        delete_srt=bool(merged.get("delete_srt", merged.get("deleteSRT", True))),
        rebuild=bool(merged.get("rebuild", False)),
        cleanup=bool(merged.get("cleanup", False)),
        keep_temp=bool(merged.get("keep_temp", False)),
        languages=languages,
        steps=steps,
        paths=paths_cfg,
        ffmpeg=ffmpeg,
        whisperx=whisperx,
        translate=translate,
        tts=tts,
        mux=mux,
        output=output,
        audio_merge=audio_merge,
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
    template = (Path(__file__).parent / "video.pipeline.yaml")
    if template.exists():
        with template.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(DEFAULT_PIPELINE_DICT, cfg)
    else:
        cfg = copy.deepcopy(DEFAULT_PIPELINE_DICT)

    project_name = values.get("-PROJECT-", "").strip() or cfg.get("project_name", "video_sample")
    out_dir = values.get("-OUT-", "").strip() or (cfg.get("paths", {}) or {}).get("out_dir", "out")
    source_mode = "dir" if bool(values.get("-SRC_DIR-", False)) else "file"
    if "-INPUT_MODE-" in values:
        source_mode = normalize_input_mode(values.get("-INPUT_MODE-"))

    input_path = values.get("-INPUT_PATH-", "").strip()
    if not input_path and source_mode == "dir":
        input_path = values.get("-IN_DIR-", "").strip()
    if not input_path:
        input_path = values.get("-IN-", "").strip()
    if not input_path:
        input_path = (cfg.get("paths", {}) or {}).get("input_video", "{project_name}.mp4")

    selected_mode = values.get("-MODES-", cfg.get("mode", "Добавление"))

    cfg["project_name"] = project_name
    cfg["input_mode"] = source_mode
    cfg["input_path"] = input_path
    cfg["mode"] = selected_mode
    cfg["usegpu"] = bool(values.get("-GPU-", True))
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

    cfg.setdefault("output", {})
    cfg["output"]["move_to_dir"] = values.get("-MOVE_TO_DIR-", cfg["output"].get("move_to_dir", ""))
    cfg["output"]["update_existing_file"] = bool(
        values.get("-UPDATE_EXISTING_FILE-", cfg["output"].get("update_existing_file", False))
    )
    cfg["output"]["audio_update_mode"] = normalize_audio_update_mode(
        values.get("-MODES-", cfg["output"].get("audio_update_mode", selected_mode))
    )

    pipeline_path = pipeline_path.resolve()
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    with pipeline_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return pipeline_path
