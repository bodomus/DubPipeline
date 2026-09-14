from __future__ import annotations

import hashlib
import json
import shlex
import shutil
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
AUDIO_SEPARATOR_PROVIDER = "audio_separator"
DEFAULT_AUDIO_SEPARATOR_MODEL = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"

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
    model: str
    model_path: str
    model_file_dir: str
    device: str
    use_gpu: bool
    output_format: str
    sample_rate: int
    command: tuple[str, ...]
    cache_enabled: bool


class AudioBackgroundProvider:
    name = "base"

    def separate(self, request: SourceSeparationRequest) -> SourceSeparationResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


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


class AudioSeparatorProvider(AudioBackgroundProvider):
    name = AUDIO_SEPARATOR_PROVIDER

    def __init__(self, separator_factory: Callable[..., object] | None = None) -> None:
        self._separator_factory = separator_factory
        self._separator: object | None = None
        self._loaded_model_key: tuple[str, str, str] | None = None

    def separate(self, request: SourceSeparationRequest) -> SourceSeparationResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        separator = self._ensure_loaded(request)
        _set_separator_output_dir(separator, request.output_dir)

        info(
            "[source_separation] provider=audio_separator "
            f"model={_audio_separator_model_filename(request)} device={_resolve_device(request)}"
        )
        info(f"[source_separation] separation start: {request.source_audio}")
        try:
            produced = separator.separate(str(request.source_audio))
        except Exception as exc:
            raise SourceSeparationError(
                f"audio-separator separation failed: {exc}"
            ) from exc

        vocals, background = map_audio_separator_outputs(request, produced)
        _copy_stem(vocals, request.vocals_wav)
        _copy_stem(background, request.background_wav)
        validate_stems(request.vocals_wav, request.background_wav)
        info("[source_separation] separation complete")
        return SourceSeparationResult(
            vocals_wav=request.vocals_wav,
            background_wav=request.background_wav,
            cache_hit=False,
        )

    def close(self) -> None:
        self._separator = None
        self._loaded_model_key = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self, request: SourceSeparationRequest) -> object:
        model_filename = _audio_separator_model_filename(request)
        model_file_dir = _audio_separator_model_file_dir(request)
        device = _resolve_device(request)
        model_key = (model_filename, model_file_dir, device)
        if self._separator is not None and self._loaded_model_key == model_key:
            return self._separator

        if self._separator is not None and self._loaded_model_key != model_key:
            self.close()

        separator = self._create_separator(request, device, model_file_dir)
        info(
            "[source_separation] loading audio-separator model "
            f"model={model_filename} device={device}"
        )
        try:
            separator.load_model(model_filename)
        except Exception as exc:
            raise SourceSeparationError(
                f"audio-separator model load failed for '{model_filename}': {exc}"
            ) from exc
        info("[source_separation] audio-separator model load complete")
        self._separator = separator
        self._loaded_model_key = model_key
        return separator

    def _create_separator(
        self, request: SourceSeparationRequest, device: str, model_file_dir: str
    ) -> object:
        factory = self._separator_factory or _load_audio_separator_factory()
        base_kwargs: dict[str, object] = {
            "output_dir": str(request.output_dir),
            "output_format": request.output_format.upper(),
            "sample_rate": int(request.sample_rate),
            "use_autocast": device == "cuda",
        }
        if model_file_dir:
            base_kwargs["model_file_dir"] = model_file_dir

        attempts = [
            base_kwargs | {"use_cuda": device == "cuda"},
            base_kwargs | {"device": device},
            base_kwargs,
        ]
        last_type_error: TypeError | None = None
        for kwargs in attempts:
            try:
                separator = factory(**kwargs)
                _apply_separator_device(separator, device)
                return separator
            except TypeError as exc:
                last_type_error = exc
        raise SourceSeparationError(
            f"audio-separator Separator could not be initialized: {last_type_error}"
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


def validate_model_file_or_fallback(
    cfg: PipelineConfig, request: SourceSeparationRequest
) -> bool:
    try:
        validate_model_file(request)
    except SourceSeparationError as exc:
        if legacy_fallback_enabled(cfg):
            warn(f"[source_separation] model unavailable; using legacy ducking: {exc}")
            return False
        raise
    return True


def build_request(cfg: PipelineConfig) -> SourceSeparationRequest:
    sep_cfg = cfg.source_separation
    command = tuple(_coerce_command(getattr(sep_cfg, "command", [])))
    model_path = _resolve_model_path(str(sep_cfg.model_path or "").strip(), cfg)
    model_file_dir = _resolve_optional_dir(
        str(getattr(sep_cfg, "model_file_dir", "") or "").strip(), cfg
    )
    return SourceSeparationRequest(
        source_audio=Path(cfg.paths.audio_wav),
        output_dir=Path(cfg.paths.separation_dir),
        vocals_wav=Path(cfg.paths.separation_vocals_wav),
        background_wav=Path(cfg.paths.separation_background_wav),
        metadata_json=Path(cfg.paths.separation_metadata_json),
        provider=str(sep_cfg.provider or BS_ROFORMER_PROVIDER).strip().lower(),
        model=str(getattr(sep_cfg, "model", "") or DEFAULT_AUDIO_SEPARATOR_MODEL).strip(),
        model_path=model_path,
        model_file_dir=model_file_dir,
        device=str(getattr(sep_cfg, "device", "auto") or "auto").strip().lower(),
        use_gpu=bool(getattr(cfg, "usegpu", True)),
        output_format=str(getattr(sep_cfg, "output_format", "wav") or "wav")
        .strip()
        .lower(),
        sample_rate=int(getattr(sep_cfg, "sample_rate", 44_100) or 44_100),
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
    provider_name: str,
    *,
    runner: CommandRunner | None = None,
    separator_factory: Callable[..., object] | None = None,
) -> AudioBackgroundProvider:
    normalized = (provider_name or BS_ROFORMER_PROVIDER).strip().lower()
    if normalized == BS_ROFORMER_PROVIDER:
        return BsRoformerProvider(runner=runner)
    if normalized == AUDIO_SEPARATOR_PROVIDER:
        return AudioSeparatorProvider(separator_factory=separator_factory)
    if normalized == LEGACY_DUCKING_MODE:
        return OriginalAudioBackgroundProvider()
    raise SourceSeparationError(f"Unknown source separation provider: {provider_name}")


def create_source_separation_provider(cfg: PipelineConfig) -> AudioBackgroundProvider | None:
    if not source_separation_enabled(cfg):
        return None
    request = build_request(cfg)
    if request.provider == AUDIO_SEPARATOR_PROVIDER:
        return create_provider(request.provider)
    return None


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
    provider: AudioBackgroundProvider | None = None,
) -> SourceSeparationResult | None:
    if not source_separation_enabled(cfg):
        info("[source_separation] mode=legacy_ducking; skipping source separation")
        return None

    request = build_request(cfg)
    if not request.source_audio.exists():
        raise FileNotFoundError(
            f"Source audio for separation not found: {request.source_audio}"
        )
    if not validate_model_file_or_fallback(cfg, request):
        remove_stale_stems(request)
        return None

    if request.cache_enabled:
        cached = read_cached_result(request)
        if cached is not None:
            info(f"[source_separation] cache hit: {request.background_wav}")
            return cached

    remove_stale_stems(request)
    owns_provider = provider is None
    provider = provider or create_provider(request.provider, runner=runner)
    try:
        result = provider.separate(request)
    except Exception as exc:
        if legacy_fallback_enabled(cfg):
            warn(f"[source_separation] failed; falling back to legacy ducking: {exc}")
            return None
        raise
    finally:
        if owns_provider:
            provider.close()

    write_metadata(request)
    info(f"[source_separation] vocals: {result.vocals_wav}")
    info(f"[source_separation] background: {result.background_wav}")
    return result


def resolve_background_audio_for_merge(cfg: PipelineConfig) -> Path | None:
    if not source_separation_enabled(cfg):
        return None
    request = build_request(cfg)
    if not validate_model_file_or_fallback(cfg, request):
        return None
    cached = read_cached_result(request)
    if cached is not None:
        return cached.background_wav
    if legacy_fallback_enabled(cfg):
        warn(
            "[source_separation] separated background is not valid for current input/config/model; "
            "using legacy original audio"
        )
        return None
    raise FileNotFoundError(
        f"Separated background is required but missing or stale: {request.background_wav}"
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
        "schema": 2,
        "source": source_identity(request.source_audio),
        "provider": request.provider,
        "model": model_identity(request),
        "command": list(request.command),
        "parameters": {
            "device": request.device,
            "use_gpu": request.use_gpu,
            "output_format": request.output_format,
            "sample_rate": request.sample_rate,
        },
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


def model_identity(request: SourceSeparationRequest) -> dict[str, Any]:
    if request.provider == AUDIO_SEPARATOR_PROVIDER:
        identity: dict[str, Any] = {
            "model": _audio_separator_model_filename(request),
            "model_file_dir": _stable_optional_dir(
                _audio_separator_model_file_dir(request)
            ),
        }
        if request.model_path:
            model_path = validate_model_file(request)
            stat = model_path.stat()
            identity["local_model"] = {
                "path": str(model_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(model_path),
            }
        return identity
    if request.provider != BS_ROFORMER_PROVIDER:
        return {"path": _stable_model_path(request.model_path)}
    model_path = validate_model_file(request)
    stat = model_path.stat()
    return {
        "path": str(model_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(model_path),
    }


def validate_model_file(request: SourceSeparationRequest) -> Path:
    if request.provider == AUDIO_SEPARATOR_PROVIDER:
        if not request.model_path:
            return Path()
        model_path = Path(request.model_path)
        if not model_path.is_file():
            raise SourceSeparationError(
                f"source_separation.model_path does not exist or is not a file: {model_path}"
            )
        return model_path
    if request.provider != BS_ROFORMER_PROVIDER:
        return Path(request.model_path)
    if not request.model_path:
        raise SourceSeparationError(
            "source_separation.model_path is required for provider 'bs_roformer'"
        )
    model_path = Path(request.model_path)
    if not model_path.is_file():
        raise SourceSeparationError(
            f"source_separation.model_path does not exist or is not a file: {model_path}"
        )
    return model_path


def remove_stale_stems(request: SourceSeparationRequest) -> None:
    for path in (request.vocals_wav, request.background_wav):
        path.unlink(missing_ok=True)


def _resolve_model_path(model_path: str, cfg: PipelineConfig) -> str:
    if not model_path:
        return ""
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path(cfg.paths.workdir) / path
    return str(path.resolve())


def _resolve_optional_dir(path_text: str, cfg: PipelineConfig) -> str:
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path(cfg.paths.workdir) / path
    return str(path.resolve())


def _stable_model_path(model_path: str) -> str:
    if not model_path:
        return ""
    return str(Path(model_path).expanduser().resolve())


def _stable_optional_dir(path_text: str) -> str:
    if not path_text:
        return ""
    return str(Path(path_text).expanduser().resolve())


def _load_audio_separator_factory() -> Callable[..., object]:
    try:
        from audio_separator.separator import Separator
    except Exception as exc:
        raise SourceSeparationError(
            "audio-separator is not installed or cannot be imported. "
            "Install the native provider dependency and retry."
        ) from exc
    return Separator


def _audio_separator_model_filename(request: SourceSeparationRequest) -> str:
    if request.model_path:
        return Path(request.model_path).name
    return request.model or DEFAULT_AUDIO_SEPARATOR_MODEL


def _audio_separator_model_file_dir(request: SourceSeparationRequest) -> str:
    if request.model_path:
        return str(Path(request.model_path).parent)
    return request.model_file_dir


def _resolve_device(request: SourceSeparationRequest) -> str:
    requested = (request.device or "auto").strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise SourceSeparationError(
            "source_separation.device must be one of: auto, cuda, cpu"
        )
    cuda_available = _cuda_available()
    if requested == "cuda":
        if not cuda_available:
            raise SourceSeparationError(
                "source_separation.device=cuda was requested, but CUDA is unavailable"
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if request.use_gpu:
        if not cuda_available:
            raise SourceSeparationError(
                "source_separation.device=auto resolved to CUDA because usegpu=true, "
                "but CUDA is unavailable"
            )
        return "cuda"
    return "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _set_separator_output_dir(separator: object, output_dir: Path) -> None:
    for attr in ("output_dir", "output_directory"):
        if hasattr(separator, attr):
            setattr(separator, attr, str(output_dir))


def _apply_separator_device(separator: object, device: str) -> None:
    try:
        import torch
    except Exception:
        return
    if device == "cpu":
        cpu = torch.device("cpu")
        setattr(separator, "torch_device", cpu)
        setattr(separator, "torch_device_cpu", cpu)
        setattr(separator, "onnx_execution_provider", ["CPUExecutionProvider"])
        return
    if device == "cuda":
        cuda = torch.device("cuda")
        setattr(separator, "torch_device", cuda)
        setattr(separator, "onnx_execution_provider", ["CUDAExecutionProvider"])


def map_audio_separator_outputs(
    request: SourceSeparationRequest, produced: object
) -> tuple[Path, Path]:
    candidates = _output_candidates(request, produced)
    vocals = _find_stem_candidate(candidates, wanted="vocals")
    background = _find_stem_candidate(candidates, wanted="background")
    if vocals is None:
        raise SourceSeparationError("audio-separator did not produce a vocals stem")
    if background is None:
        raise SourceSeparationError(
            "audio-separator did not produce an instrumental/background stem"
        )
    return vocals, background


def _output_candidates(request: SourceSeparationRequest, produced: object) -> list[Path]:
    candidates: list[Path] = []
    if isinstance(produced, (str, Path)):
        candidates.append(Path(produced))
    elif isinstance(produced, Sequence):
        candidates.extend(Path(str(item)) for item in produced)
    resolved = []
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = request.output_dir / candidate
        resolved.append(candidate)
    if resolved:
        return resolved
    return sorted(
        [
            path
            for path in request.output_dir.iterdir()
            if path.is_file() and path.suffix.lower() == f".{request.output_format}"
        ]
    )


def _find_stem_candidate(candidates: Sequence[Path], *, wanted: str) -> Path | None:
    for candidate in candidates:
        normalized = candidate.stem.lower().replace(" ", "_").replace("-", "_")
        if wanted == "vocals":
            if "no_vocal" in normalized or "instrumental" in normalized:
                continue
            if "vocal" in normalized:
                return candidate
        else:
            background_tokens = (
                "instrumental",
                "instrument",
                "background",
                "no_vocal",
                "novocal",
                "karaoke",
                "accompaniment",
                "inst",
            )
            if any(token in normalized for token in background_tokens):
                return candidate
    return None


def _copy_stem(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copyfile(source, target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
