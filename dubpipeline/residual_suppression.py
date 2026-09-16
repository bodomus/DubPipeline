from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dubpipeline.config import PipelineConfig, ResidualSuppressionConfig
from dubpipeline.source_separation import (
    legacy_fallback_enabled,
    resolve_background_audio_for_merge,
    source_separation_enabled,
)
from dubpipeline.utils.logging import info, warn

IMPLEMENTATION_ID = "ffmpeg-sidechaincompress-dry-floor-v3"
METADATA_SCHEMA = 1

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class ResidualSuppressionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResidualSuppressionResult:
    cleaned_background_wav: Path
    cache_hit: bool = False


@dataclass(frozen=True)
class ResidualSuppressionRequest:
    background_wav: Path
    vocals_wav: Path
    cleaned_background_wav: Path
    metadata_json: Path
    ffmpeg_bin: str
    duration_seconds: float
    config: ResidualSuppressionConfig

    @property
    def temp_wav(self) -> Path:
        return self.cleaned_background_wav.with_name(
            f"{self.cleaned_background_wav.stem}.tmp{self.cleaned_background_wav.suffix}"
        )


def residual_suppression_enabled(cfg: PipelineConfig) -> bool:
    step_enabled = bool(
        getattr(getattr(cfg, "steps", None), "residual_suppression", True)
    )
    configured = bool(
        getattr(getattr(cfg, "residual_suppression", None), "enabled", False)
    )
    return step_enabled and configured and source_separation_enabled(cfg)


def validate_config(cfg: ResidualSuppressionConfig) -> None:
    if not (-60.0 <= cfg.threshold_db <= 0.0):
        raise ValueError("residual_suppression.threshold_db must be in [-60, 0]")
    if not (1.0 <= cfg.ratio <= 20.0):
        raise ValueError("residual_suppression.ratio must be in [1, 20]")
    if not (0 < cfg.attack_ms <= 2000):
        raise ValueError("residual_suppression.attack_ms must be in (0, 2000]")
    if not (0 < cfg.release_ms <= 9000):
        raise ValueError("residual_suppression.release_ms must be in (0, 9000]")
    if not (0.0 <= cfg.max_reduction_db <= 24.0):
        raise ValueError(
            "residual_suppression.max_reduction_db must be in [0, 24]"
        )
    if not (-36.0 <= cfg.control_gain_db <= 36.0):
        raise ValueError(
            "residual_suppression.control_gain_db must be in [-36, 36]"
        )
    if not (1.0 <= cfg.knee <= 8.0):
        raise ValueError("residual_suppression.knee must be in [1, 8]")


def build_request(cfg: PipelineConfig) -> ResidualSuppressionRequest:
    suppression_cfg = cfg.residual_suppression
    validate_config(suppression_cfg)
    background_wav = Path(cfg.paths.separation_background_wav)
    return ResidualSuppressionRequest(
        background_wav=background_wav,
        vocals_wav=Path(cfg.paths.separation_vocals_wav),
        cleaned_background_wav=Path(cfg.paths.separation_cleaned_background_wav),
        metadata_json=Path(cfg.paths.residual_suppression_metadata_json),
        ffmpeg_bin=str(getattr(getattr(cfg, "ffmpeg", None), "bin", "ffmpeg")),
        duration_seconds=_audio_duration_seconds(background_wav),
        config=suppression_cfg,
    )


def build_filtergraph(cfg: ResidualSuppressionConfig) -> str:
    validate_config(cfg)
    threshold_linear = max(0.000976563, min(1.0, 10 ** (cfg.threshold_db / 20.0)))
    dry_floor = 10 ** (-cfg.max_reduction_db / 20.0)
    compressed_weight = 1.0 - dry_floor
    return ";".join(
        [
            f"[1:a]apad,volume={cfg.control_gain_db:.3f}dB[control]",
            (
                "[0:a][control]sidechaincompress="
                f"threshold={threshold_linear:.9f}:"
                f"ratio={cfg.ratio:.3f}:"
                f"attack={cfg.attack_ms}:"
                f"release={cfg.release_ms}:"
                "makeup=1:"
                f"knee={cfg.knee:.6f}:"
                "link=maximum:detection=rms:"
                f"mix={compressed_weight:.9f},apad[out]"
            ),
        ]
    )


def build_ffmpeg_command(request: ResidualSuppressionRequest) -> list[str]:
    return [
        request.ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(request.background_wav),
        "-i",
        str(request.vocals_wav),
        "-filter_complex",
        build_filtergraph(request.config),
        "-map",
        "[out]",
        "-c:a",
        "pcm_s16le",
        "-t",
        f"{request.duration_seconds:.9f}",
        str(request.temp_wav),
    ]


def run_residual_suppression(
    cfg: PipelineConfig, *, runner: CommandRunner | None = None
) -> ResidualSuppressionResult | None:
    if not residual_suppression_enabled(cfg):
        info("[residual_suppression] disabled; using separated background unchanged")
        return None

    background = resolve_background_audio_for_merge(cfg)
    if background is None:
        remove_stale_output_paths(
            Path(cfg.paths.separation_cleaned_background_wav),
            Path(cfg.paths.residual_suppression_metadata_json),
        )
        info("[residual_suppression] source separation fell back to legacy ducking")
        return None
    request = build_request(cfg)
    if background != request.background_wav:
        raise ResidualSuppressionError(
            f"Unexpected separated background path: {background}"
        )

    if request.config.cache_enabled:
        cached = read_cached_result(request)
        if cached is not None:
            info(f"[residual_suppression] cache hit: {cached.cleaned_background_wav}")
            return cached

    remove_stale_artifacts(request)
    request.cleaned_background_wav.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(request)
    info(
        "[residual_suppression] enabled "
        f"control={request.vocals_wav} background={request.background_wav} "
        f"output={request.cleaned_background_wav} "
        f"threshold_db={request.config.threshold_db:.2f} "
        f"ratio={request.config.ratio:.2f} "
        f"attack_ms={request.config.attack_ms} "
        f"release_ms={request.config.release_ms} "
        f"max_reduction_db={request.config.max_reduction_db:.2f}"
    )

    started = time.perf_counter()
    command_runner = runner or _run_command
    try:
        proc = command_runner(command)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise ResidualSuppressionError(
                f"FFmpeg residual suppression failed with code {proc.returncode}: {stderr}"
            )
        _validate_output(request.temp_wav, reference=request.background_wav)
        os.replace(request.temp_wav, request.cleaned_background_wav)
        write_metadata(request)
    except Exception as exc:
        remove_stale_artifacts(request)
        if legacy_fallback_enabled(cfg):
            warn(
                "[residual_suppression] failed; using legacy ducking fallback: "
                f"{exc}"
            )
            return None
        if isinstance(exc, ResidualSuppressionError):
            raise
        raise ResidualSuppressionError(f"Residual suppression failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    info(f"[residual_suppression] cache miss; completed in {elapsed:.2f}s")
    return ResidualSuppressionResult(request.cleaned_background_wav, cache_hit=False)


def resolve_background_for_merge(cfg: PipelineConfig) -> Path | None:
    background = resolve_background_audio_for_merge(cfg)
    if background is None or not residual_suppression_enabled(cfg):
        return background

    request = build_request(cfg)
    cached = read_cached_result(request)
    if cached is not None:
        return cached.cleaned_background_wav
    if legacy_fallback_enabled(cfg):
        warn(
            "[residual_suppression] cleaned background is missing or stale; "
            "using legacy original audio"
        )
        return None
    raise FileNotFoundError(
        "Cleaned background is required but missing or stale: "
        f"{request.cleaned_background_wav}"
    )


def read_cached_result(
    request: ResidualSuppressionRequest,
) -> ResidualSuppressionResult | None:
    if not request.metadata_json.is_file():
        return None
    try:
        metadata = json.loads(request.metadata_json.read_text(encoding="utf-8"))
        expected = build_metadata(request)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if metadata != expected:
        return None
    try:
        _validate_output(
            request.cleaned_background_wav, reference=request.background_wav
        )
    except ResidualSuppressionError:
        return None
    return ResidualSuppressionResult(request.cleaned_background_wav, cache_hit=True)


def build_metadata(request: ResidualSuppressionRequest) -> dict[str, Any]:
    return {
        "schema": METADATA_SCHEMA,
        "implementation": IMPLEMENTATION_ID,
        "background": _file_identity(request.background_wav),
        "vocals": _file_identity(request.vocals_wav),
        "config": asdict(request.config),
        "ffmpeg_bin": request.ffmpeg_bin,
        "output": str(request.cleaned_background_wav),
    }


def write_metadata(request: ResidualSuppressionRequest) -> None:
    request.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    temp_metadata = request.metadata_json.with_name(f"{request.metadata_json.name}.tmp")
    temp_metadata.write_text(
        json.dumps(build_metadata(request), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_metadata, request.metadata_json)


def remove_stale_artifacts(request: ResidualSuppressionRequest) -> None:
    remove_stale_output_paths(request.cleaned_background_wav, request.metadata_json)


def remove_stale_output_paths(cleaned_background_wav: Path, metadata_json: Path) -> None:
    for path in (
        cleaned_background_wav,
        cleaned_background_wav.with_name(
            f"{cleaned_background_wav.stem}.tmp{cleaned_background_wav.suffix}"
        ),
        metadata_json,
        metadata_json.with_name(f"{metadata_json.name}.tmp"),
    ):
        path.unlink(missing_ok=True)


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True)


def _validate_output(path: Path, *, reference: Path | None = None) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ResidualSuppressionError(
            f"Residual suppression did not produce a non-empty output: {path}"
        )
    if reference is not None:
        try:
            import soundfile as sf

            output_info = sf.info(str(path))
            reference_info = sf.info(str(reference))
        except Exception as exc:
            raise ResidualSuppressionError(
                f"Could not inspect residual-suppression output: {exc}"
            ) from exc
        if output_info.samplerate != reference_info.samplerate:
            raise ResidualSuppressionError(
                "Residual-suppression output sample rate differs from background"
            )
        if output_info.channels != reference_info.channels:
            raise ResidualSuppressionError(
                "Residual-suppression output channel count differs from background"
            )
        if abs(output_info.frames - reference_info.frames) > 1:
            raise ResidualSuppressionError(
                "Residual-suppression output duration differs from background: "
                f"{output_info.frames} != {reference_info.frames} frames"
            )


def _audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        audio_info = sf.info(str(path))
    except Exception as exc:
        raise ResidualSuppressionError(f"Could not inspect background WAV: {exc}") from exc
    if audio_info.samplerate <= 0 or audio_info.frames <= 0:
        raise ResidualSuppressionError(f"Background WAV is empty or invalid: {path}")
    return audio_info.frames / audio_info.samplerate


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
