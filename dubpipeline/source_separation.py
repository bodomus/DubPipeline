from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.logging import info, warn

SEPARATED_BACKGROUND_MODE = "separated_background"
LEGACY_DUCKING_MODE = "legacy_ducking"
NO_FALLBACK_MODE = "none"
BS_ROFORMER_PROVIDER = "bs_roformer"

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class SourceSeparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceSeparationResult:
    vocals_wav: Path
    background_wav: Path
    cache_hit: bool = False


@dataclass(frozen=True)
class SourceSeparationRequest:
    source_audio: Path
    output_dir: Path
    vocals_wav: Path
    background_wav: Path
    metadata_json: Path
    provider: str
    model_path: str
    command: tuple[str, ...]
    cache_enabled: bool


class AudioBackgroundProvider:
    name = "base"

    def separate(self, request: SourceSeparationRequest) -> SourceSeparationResult:
        raise NotImplementedError


class OriginalAudioBackgroundProvider(AudioBackgroundProvider):
    name = LEGACY_DUCKING_MODE

    def separate(self, request: SourceSeparationRequest) -> SourceSeparationResult:
        return SourceSeparationResult(
            vocals_wav=Path(),
            background_wav=request.source_audio,
            cache_hit=False,
        )


class BsRoformerProvider(AudioBackgroundProvider):
    name = BS_ROFORMER_PROVIDER

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or _run_command

    def separate(self, request: SourceSeparationRequest) -> SourceSeparationResult:
        if not request.model_path:
            raise SourceSeparationError(
                "source_separation.model_path is required for provider 'bs_roformer'"
            )
        if not request.command:
            raise SourceSeparationError(
                "source_separation.command is required for provider 'bs_roformer'"
            )

        request.output_dir.mkdir(parents=True, exist_ok=True)
        command = format_command_template(request)
        info(f"[source_separation] provider=bs_roformer command: {' '.join(command)}")
        proc = self._runner(command)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise SourceSeparationError(
                f"BS Roformer separation failed with code {proc.returncode}: {stderr}"
            )

        validate_stems(request.vocals_wav, request.background_wav)
        return SourceSeparationResult(
            vocals_wav=request.vocals_wav,
            background_wav=request.background_wav,
            cache_hit=False,
        )


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True)


def source_separation_enabled(cfg: PipelineConfig) -> bool:
    mode = (
        str(
            getattr(
                getattr(cfg, "source_separation", None), "mode", LEGACY_DUCKING_MODE
            )
        )
        .strip()
        .lower()
    )
    return mode == SEPARATED_BACKGROUND_MODE


def legacy_fallback_enabled(cfg: PipelineConfig) -> bool:
    fallback = (
        str(
            getattr(
                getattr(cfg, "source_separation", None),
                "fallback_mode",
                NO_FALLBACK_MODE,
            )
        )
        .strip()
        .lower()
    )
    return fallback == LEGACY_DUCKING_MODE


def build_request(cfg: PipelineConfig) -> SourceSeparationRequest:
    sep_cfg = cfg.source_separation
    command = tuple(_coerce_command(getattr(sep_cfg, "command", [])))
    return SourceSeparationRequest(
        source_audio=Path(cfg.paths.audio_wav),
        output_dir=Path(cfg.paths.separation_dir),
        vocals_wav=Path(cfg.paths.separation_vocals_wav),
        background_wav=Path(cfg.paths.separation_background_wav),
        metadata_json=Path(cfg.paths.separation_metadata_json),
        provider=str(sep_cfg.provider or BS_ROFORMER_PROVIDER).strip().lower(),
        model_path=str(sep_cfg.model_path or "").strip(),
        command=command,
        cache_enabled=bool(sep_cfg.cache_enabled),
    )


def _coerce_command(command: object) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list | tuple):
        return [str(item) for item in command]
    return []


def create_provider(
    provider_name: str, *, runner: CommandRunner | None = None
) -> AudioBackgroundProvider:
    normalized = (provider_name or BS_ROFORMER_PROVIDER).strip().lower()
    if normalized == BS_ROFORMER_PROVIDER:
        return BsRoformerProvider(runner=runner)
    if normalized == LEGACY_DUCKING_MODE:
        return OriginalAudioBackgroundProvider()
    raise SourceSeparationError(f"Unknown source separation provider: {provider_name}")


def format_command_template(request: SourceSeparationRequest) -> list[str]:
    mapping = {
        "input_audio": str(request.source_audio),
        "out_dir": str(request.output_dir),
        "vocals_wav": str(request.vocals_wav),
        "background_wav": str(request.background_wav),
        "model_path": request.model_path,
    }
    return [part.format(**mapping) for part in request.command]


def validate_stems(vocals_wav: Path, background_wav: Path) -> None:
    missing = [path for path in (vocals_wav, background_wav) if not path.exists()]
    if missing:
        raise SourceSeparationError(
            "Source separation did not produce expected stems: "
            + ", ".join(map(str, missing))
        )
    empty = [path for path in (vocals_wav, background_wav) if path.stat().st_size <= 0]
    if empty:
        raise SourceSeparationError(
            "Source separation produced empty stems: " + ", ".join(map(str, empty))
        )


def run_source_separation(
    cfg: PipelineConfig,
    *,
    runner: CommandRunner | None = None,
) -> SourceSeparationResult | None:
    if not source_separation_enabled(cfg):
        info("[source_separation] mode=legacy_ducking; skipping source separation")
        return None

    request = build_request(cfg)
    if not request.source_audio.exists():
        raise FileNotFoundError(
            f"Source audio for separation not found: {request.source_audio}"
        )

    if request.cache_enabled:
        cached = read_cached_result(request)
        if cached is not None:
            info(f"[source_separation] cache hit: {request.background_wav}")
            return cached

    provider = create_provider(request.provider, runner=runner)
    try:
        result = provider.separate(request)
    except Exception as exc:
        if legacy_fallback_enabled(cfg):
            warn(f"[source_separation] failed; falling back to legacy ducking: {exc}")
            return None
        raise

    write_metadata(request)
    info(f"[source_separation] vocals: {result.vocals_wav}")
    info(f"[source_separation] background: {result.background_wav}")
    return result


def resolve_background_audio_for_merge(cfg: PipelineConfig) -> Path | None:
    if not source_separation_enabled(cfg):
        return None
    background = Path(cfg.paths.separation_background_wav)
    if background.exists() and background.stat().st_size > 0:
        return background
    if legacy_fallback_enabled(cfg):
        warn(
            "[source_separation] separated background is missing; using legacy original audio"
        )
        return None
    raise FileNotFoundError(
        f"Separated background is required but missing: {background}"
    )


def read_cached_result(
    request: SourceSeparationRequest,
) -> SourceSeparationResult | None:
    if not request.metadata_json.exists():
        return None
    try:
        metadata = json.loads(request.metadata_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata != build_metadata(request):
        return None
    try:
        validate_stems(request.vocals_wav, request.background_wav)
    except SourceSeparationError:
        return None
    return SourceSeparationResult(
        vocals_wav=request.vocals_wav,
        background_wav=request.background_wav,
        cache_hit=True,
    )


def write_metadata(request: SourceSeparationRequest) -> None:
    request.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    request.metadata_json.write_text(
        json.dumps(
            build_metadata(request), ensure_ascii=False, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )


def build_metadata(request: SourceSeparationRequest) -> dict[str, Any]:
    return {
        "schema": 1,
        "source": source_identity(request.source_audio),
        "provider": request.provider,
        "model_path": _stable_model_path(request.model_path),
        "command": list(request.command),
        "outputs": {
            "vocals_wav": str(request.vocals_wav),
            "background_wav": str(request.background_wav),
        },
    }


def source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _stable_model_path(model_path: str) -> str:
    if not model_path:
        return ""
    return str(Path(model_path).expanduser())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
