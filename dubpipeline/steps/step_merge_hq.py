from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.logging import debug, error, info


_FFMPEG_ENV_KEY = "DUBPIPELINE_FFMPEG_BIN"
_FFPROBE_ENV_KEY = "DUBPIPELINE_FFPROBE_BIN"
_KEEP_TEMP_ENV_KEY = "DUBPIPELINE_KEEP_TEMP"


@dataclass(frozen=True)
class DuckingConfig:
    enabled: bool
    amount_db: float
    threshold_db: float
    attack_ms: int
    release_ms: int
    ratio: float
    knee_db: float


@dataclass(frozen=True)
class LoudnessConfig:
    enabled: bool
    target_i: float
    true_peak: float


@dataclass(frozen=True)
class MergeHQConfig:
    tts_gain_db: float
    original_gain_db: float
    ducking: DuckingConfig
    loudness: LoudnessConfig


@dataclass(frozen=True)
class AudioStreamInfo:
    stream_index: int
    audio_index: int
    codec_name: str
    language: str
    title: str
    channels: int
    sample_rate: int | None


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _ffmpeg_bin() -> str:
    return str(os.getenv(_FFMPEG_ENV_KEY, "ffmpeg"))


def _ffprobe_bin() -> str:
    return str(os.getenv(_FFPROBE_ENV_KEY, "ffprobe"))


def _keep_temp_enabled() -> bool:
    return _truthy(os.getenv(_KEEP_TEMP_ENV_KEY))


def validate_merge_hq_config(cfg: MergeHQConfig) -> None:
    if cfg.ducking.attack_ms <= 0:
        raise ValueError("ducking.attack_ms must be > 0")
    if cfg.ducking.release_ms <= 0:
        raise ValueError("ducking.release_ms must be > 0")
    if not (0.0 <= cfg.ducking.amount_db <= 24.0):
        raise ValueError("ducking.amount_db must be in [0, 24]")
    if not (-60.0 <= cfg.ducking.threshold_db <= 0.0):
        raise ValueError("ducking.threshold_db must be in [-60, 0]")
    if cfg.ducking.ratio <= 0:
        raise ValueError("ducking.ratio must be > 0")


def probe_audio_streams(video: Path) -> list[AudioStreamInfo]:
    cmd = [
        _ffprobe_bin(),
        "-v",
        "error",
        "-show_streams",
        "-select_streams",
        "a",
        "-print_format",
        "json",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for '{video}': {proc.stderr.strip()}")

    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or []
    out: list[AudioStreamInfo] = []
    for audio_index, stream in enumerate(streams):
        tags = stream.get("tags") or {}
        sample_rate_raw = stream.get("sample_rate")
        sample_rate = None
        if sample_rate_raw is not None:
            try:
                sample_rate = int(sample_rate_raw)
            except Exception:
                sample_rate = None
        out.append(
            AudioStreamInfo(
                stream_index=int(stream.get("index", -1)),
                audio_index=audio_index,
                codec_name=str(stream.get("codec_name", "")),
                language=str(tags.get("language", "")),
                title=str(tags.get("title", "")),
                channels=int(stream.get("channels") or 0),
                sample_rate=sample_rate,
            )
        )
    return out


def _language_matches(requested: str, stream_language: str) -> bool:
    requested_n = requested.strip().lower()
    stream_n = (stream_language or "").strip().lower()
    if not requested_n or not stream_n:
        return False
    return stream_n == requested_n or stream_n.startswith(requested_n) or requested_n.startswith(stream_n)


def select_original_audio_stream(
    streams: list[AudioStreamInfo],
    selector: str,
) -> AudioStreamInfo:
    if not streams:
        raise RuntimeError("Input video has no audio streams")

    normalized = (selector or "auto").strip()
    lowered = normalized.lower()

    if lowered == "auto":
        return streams[0]

    if lowered.startswith("index:"):
        index = int(lowered.split(":", 1)[1])
        if index < 0 or index >= len(streams):
            raise ValueError(f"Audio stream index out of range: {index}")
        return streams[index]

    lang = lowered
    if lowered.startswith("lang:"):
        lang = lowered.split(":", 1)[1].strip()

    for stream in streams:
        if _language_matches(lang, stream.language):
            return stream

    raise ValueError(f"No audio stream matched selector '{selector}'")


def parse_original_track_selector(raw_selector: object) -> str:
    if isinstance(raw_selector, int):
        return f"index:{raw_selector}"
    text = str(raw_selector or "auto").strip()
    if not text:
        return "auto"
    lowered = text.lower()
    if lowered == "auto":
        return "auto"
    if lowered.startswith("index:"):
        return lowered
    if lowered.startswith("lang:"):
        return lowered
    if lowered.isdigit() or (lowered.startswith("-") and lowered[1:].isdigit()):
        return f"index:{lowered}"
    return f"lang:{lowered}"


def build_filtergraph(
    *,
    cfg: MergeHQConfig,
    original_audio_spec: str,
) -> str:
    validate_merge_hq_config(cfg)
    threshold_linear = max(0.000976563, min(1.0, 10 ** (cfg.ducking.threshold_db / 20.0)))

    parts: list[str] = [
        f"[{original_audio_spec}]volume={cfg.original_gain_db:.3f}dB[orig]",
        f"[1:a]volume={cfg.tts_gain_db:.3f}dB[tts]",
    ]

    if cfg.ducking.enabled:
        sidechain_gain = cfg.tts_gain_db + cfg.ducking.amount_db
        parts.extend(
            [
                "[tts]asplit=2[tts_mix][tts_side_src]",
                f"[tts_side_src]volume={sidechain_gain:.3f}dB[tts_side]",
                (
                    "[orig][tts_side]sidechaincompress="
                    f"threshold={threshold_linear:.6f}:"
                    f"ratio={cfg.ducking.ratio:.3f}:"
                    f"attack={cfg.ducking.attack_ms}:"
                    f"release={cfg.ducking.release_ms}:"
                    "makeup=1:"
                    f"knee={cfg.ducking.knee_db:.3f}"
                    "[orig_ducked]"
                ),
                "[orig_ducked][tts_mix]amix=inputs=2:normalize=0[mix]",
            ]
        )
    else:
        parts.append("[orig][tts]amix=inputs=2:normalize=0[mix]")

    if cfg.loudness.enabled:
        parts.append(
            f"[mix]loudnorm=I={cfg.loudness.target_i:.3f}:TP={cfg.loudness.true_peak:.3f}[outa]"
        )
    else:
        parts.append("[mix]anull[outa]")

    return ";".join(parts)


def build_ffmpeg_command(
    *,
    input_video: Path,
    tts_wav: Path,
    output_video: Path,
    filtergraph: str,
    output_container: str | None = None,
) -> list[str]:
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(tts_wav),
        "-filter_complex",
        filtergraph,
        "-map",
        "0:v:0",
        "-map",
        "[outa]",
        "-map",
        "0:s?",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:v",
        "copy",
        "-c:s",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if output_container in {"mp4", "mov"}:
        cmd.extend(["-movflags", "+faststart"])
    if output_container:
        cmd.extend(["-f", output_container])
    cmd.append(str(output_video))
    return cmd


def merge_hq_ducking(
    *,
    input_video: Path,
    tts_wav: Path,
    out_video: Path,
    work_dir: Path,
    cfg: MergeHQConfig,
    original_audio_stream_selector: str = "auto",
) -> Path:
    """Return path to out_video on success. Must use atomic safe-write."""
    validate_merge_hq_config(cfg)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not tts_wav.exists():
        raise FileNotFoundError(f"TTS wav not found: {tts_wav}")

    work_dir.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    temp_out = out_video.with_name(f"{out_video.name}.tmp")
    temp_out.unlink(missing_ok=True)

    info("[merge] mode: hq_ducking")
    info(
        "[merge] ducking: "
        f"enabled={cfg.ducking.enabled}, amount_db={cfg.ducking.amount_db:.2f}, "
        f"threshold_db={cfg.ducking.threshold_db:.2f}, attack_ms={cfg.ducking.attack_ms}, "
        f"release_ms={cfg.ducking.release_ms}, ratio={cfg.ducking.ratio:.2f}, "
        f"knee_db={cfg.ducking.knee_db:.2f}"
    )

    streams = probe_audio_streams(input_video)
    selected = select_original_audio_stream(streams, original_audio_stream_selector)
    info(
        "[merge] selected original audio stream: "
        f"selector={original_audio_stream_selector}, "
        f"audio_index={selected.audio_index}, stream_index={selected.stream_index}, "
        f"lang={selected.language or 'n/a'}, codec={selected.codec_name or 'n/a'}"
    )

    original_spec = f"0:a:{selected.audio_index}"
    filtergraph = build_filtergraph(cfg=cfg, original_audio_spec=original_spec)
    debug(f"[merge] filtergraph: {filtergraph}")

    output_container = None
    suffix = out_video.suffix.lower()
    if suffix == ".mp4":
        output_container = "mp4"
    elif suffix == ".mkv":
        output_container = "matroska"
    elif suffix == ".mov":
        output_container = "mov"

    cmd = build_ffmpeg_command(
        input_video=input_video,
        tts_wav=tts_wav,
        output_video=temp_out,
        filtergraph=filtergraph,
        output_container=output_container,
    )

    info(f"[merge] tmp output path: {temp_out}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            error(proc.stderr)
            raise RuntimeError(f"HQ merge failed with code {proc.returncode}")

        if not temp_out.exists():
            raise RuntimeError(f"Expected temp output is missing: {temp_out}")

        os.replace(temp_out, out_video)
        info(f"[merge] atomic replace completed: {out_video}")
        return out_video
    except Exception:
        if not _keep_temp_enabled():
            temp_out.unlink(missing_ok=True)
        raise


def merge_hq_config_from_pipeline(cfg: PipelineConfig) -> tuple[MergeHQConfig, str]:
    audio_merge_cfg = getattr(cfg, "audio_merge", None)
    if audio_merge_cfg is None:
        raise ValueError("audio_merge config is missing")

    ducking_cfg = getattr(audio_merge_cfg, "ducking", None)
    loudness_cfg = getattr(audio_merge_cfg, "loudness", None)
    if ducking_cfg is None or loudness_cfg is None:
        raise ValueError("audio_merge.ducking and audio_merge.loudness must be configured")

    hq_cfg = MergeHQConfig(
        tts_gain_db=float(getattr(audio_merge_cfg, "tts_gain_db", 0.0)),
        original_gain_db=float(getattr(audio_merge_cfg, "original_gain_db", 0.0)),
        ducking=DuckingConfig(
            enabled=bool(getattr(ducking_cfg, "enabled", True)),
            amount_db=float(getattr(ducking_cfg, "amount_db", 10.0)),
            threshold_db=float(getattr(ducking_cfg, "threshold_db", -30.0)),
            attack_ms=int(getattr(ducking_cfg, "attack_ms", 10)),
            release_ms=int(getattr(ducking_cfg, "release_ms", 250)),
            ratio=float(getattr(ducking_cfg, "ratio", 6.0)),
            knee_db=float(getattr(ducking_cfg, "knee_db", 6.0)),
        ),
        loudness=LoudnessConfig(
            enabled=bool(getattr(loudness_cfg, "enabled", True)),
            target_i=float(getattr(loudness_cfg, "target_i", -16.0)),
            true_peak=float(getattr(loudness_cfg, "true_peak", -1.5)),
        ),
    )
    validate_merge_hq_config(hq_cfg)

    selector = parse_original_track_selector(getattr(audio_merge_cfg, "original_track", "auto"))
    return hq_cfg, selector
