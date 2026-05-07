from __future__ import annotations

import math
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from tqdm.auto import tqdm

from dubpipeline.models.catalog import (
    NOT_INSTALLED_REASON,
    NOT_SUPPORTED_REASON,
    get_model_spec,
    get_model_status,
    is_unsupported_pair_reason,
    resolve_model_spec,
)
from dubpipeline.models.storage import (
    UNKNOWN_SIZE_MIN_FREE_BYTES,
    configure_argos_packages_dir,
    get_argos_packages_dir,
    get_hf_snapshot_dir,
    get_models_root_dir,
    get_partial_model_dir,
)

InstallState = Literal["installed", "not_installed", "downloading", "failed"]
InstallProgressCallback = Callable[["ModelInstallStatus"], None]


class InstallCancelledError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiskSpaceCheckResult:
    ok: bool
    required_bytes: int
    available_bytes: int
    size_unknown: bool


@dataclass(frozen=True)
class ModelInstallStatus:
    model_id: str
    status: InstallState
    progress: float = 0.0
    message: str = ""
    error: str = ""


@dataclass(frozen=True)
class InstallResult:
    model_id: str
    ok: bool
    status: InstallState
    message: str = ""
    error: str = ""
    installed_dir: str = ""
    cancelled: bool = False


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.1f}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress_message(base_message: str, progress: float, elapsed_seconds: float) -> str:
    bounded_progress = max(0.0, min(0.98, float(progress)))
    percent = int(round(bounded_progress * 100))
    elapsed = _format_duration(elapsed_seconds)
    if bounded_progress <= 0:
        return f"{base_message} {percent}% | elapsed {elapsed} | remaining estimating..."

    remaining_seconds = elapsed_seconds * (1.0 - bounded_progress) / bounded_progress
    remaining = _format_duration(remaining_seconds)
    return f"{base_message} {percent}% | elapsed {elapsed} | remaining ~{remaining}"


def _build_install_progress_tqdm(
    *,
    cancel_event: threading.Event,
    progress_hook: Callable[[float], None],
):
    class InstallProgressTqdm(_InstallProgressTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                cancel_event=cancel_event,
                progress_hook=progress_hook,
                **kwargs,
            )

    return InstallProgressTqdm


class _InstallProgressTqdm(tqdm):
    """Small tqdm proxy used by huggingface_hub to report download progress."""

    def __init__(
        self,
        *args,
        cancel_event: threading.Event,
        progress_hook: Callable[[float], None],
        **kwargs,
    ):
        self._cancel_event = cancel_event
        self._progress_hook = progress_hook
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        if self._cancel_event.is_set():
            raise InstallCancelledError("Install cancelled by user.")
        out = super().update(n)
        total = self.total or 0
        if total > 0:
            progress = max(0.0, min(1.0, self.n / total))
            self._progress_hook(progress)
        return out

    def close(self):
        return super().close()


class ModelInstaller:
    def __init__(
        self,
        *,
        snapshot_download_fn: Callable[..., str] | None = None,
        disk_usage_fn: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    ) -> None:
        self._snapshot_download_fn = snapshot_download_fn
        self._disk_usage_fn = disk_usage_fn
        self._lock = threading.Lock()
        self._statuses: dict[str, ModelInstallStatus] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    @staticmethod
    def _status_key(model_id: str, src_lang: str | None = None, tgt_lang: str | None = None) -> str:
        src = (src_lang or "en").strip().lower() or "en"
        tgt = (tgt_lang or "ru").strip().lower() or "ru"
        return f"{model_id}|{src}|{tgt}"

    def get_status(self, model_id: str, *, src_lang: str | None = None, tgt_lang: str | None = None) -> ModelInstallStatus:
        spec = resolve_model_spec(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        status_key = self._status_key(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        with self._lock:
            current = self._statuses.get(status_key)

        if current is not None and current.status == "downloading":
            return current

        if current is not None and current.status == "failed":
            runtime = get_model_status(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
            if not runtime.available:
                return current

        if not spec.supported:
            return ModelInstallStatus(
                model_id=model_id,
                status="not_installed",
                progress=0.0,
                message=NOT_SUPPORTED_REASON,
            )

        runtime = get_model_status(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        if is_unsupported_pair_reason(runtime.reason):
            return ModelInstallStatus(
                model_id=model_id,
                status="not_installed",
                progress=0.0,
                message=runtime.reason,
            )
        if runtime.available:
            return ModelInstallStatus(
                model_id=model_id,
                status="installed",
                progress=1.0,
                message="Installed.",
            )
        return ModelInstallStatus(
            model_id=model_id,
            status="not_installed",
            progress=0.0,
            message=runtime.reason or NOT_INSTALLED_REASON,
        )

    def cancel(self, model_id: str, *, src_lang: str | None = None, tgt_lang: str | None = None) -> None:
        status_key = self._status_key(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        with self._lock:
            cancel_event = self._cancel_events.get(status_key)
            if cancel_event is None:
                return
            cancel_event.set()
            self._statuses[status_key] = ModelInstallStatus(
                model_id=model_id,
                status="downloading",
                progress=self._statuses.get(status_key, ModelInstallStatus(model_id, "downloading")).progress,
                message="Cancelling...",
            )

    def check_free_space(self, model_id: str, *, src_lang: str | None = None, tgt_lang: str | None = None) -> DiskSpaceCheckResult:
        spec = resolve_model_spec(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        models_root = get_models_root_dir(create=True)
        usage = self._disk_usage_fn(models_root)
        if hasattr(usage, "free"):
            free_bytes = int(getattr(usage, "free"))
        else:
            free_bytes = int(usage[2])

        estimated = spec.estimated_size_bytes
        if isinstance(estimated, int) and estimated > 0:
            required = int(math.ceil(estimated * 1.2))
            size_unknown = False
        else:
            required = UNKNOWN_SIZE_MIN_FREE_BYTES
            size_unknown = True

        return DiskSpaceCheckResult(
            ok=free_bytes >= required,
            required_bytes=required,
            available_bytes=free_bytes,
            size_unknown=size_unknown,
        )

    def install(
        self,
        model_id: str,
        *,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        progress_cb: InstallProgressCallback | None = None,
    ) -> InstallResult:
        spec = resolve_model_spec(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        status_key = self._status_key(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        runtime_status = get_model_status(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        if not spec.supported or spec.installer == "none":
            return InstallResult(
                model_id=model_id,
                ok=False,
                status="not_installed",
                message="Model is planned and not supported yet.",
            )
        if is_unsupported_pair_reason(runtime_status.reason):
            return InstallResult(
                model_id=model_id,
                ok=False,
                status="not_installed",
                message=runtime_status.reason,
                error=runtime_status.reason,
            )

        current = self.get_status(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        if current.status == "installed":
            return InstallResult(
                model_id=model_id,
                ok=True,
                status="installed",
                message="Model is already installed.",
            )

        with self._lock:
            if status_key in self._cancel_events:
                return InstallResult(
                    model_id=model_id,
                    ok=False,
                    status="failed",
                    message="Install is already in progress.",
                )
            cancel_event = threading.Event()
            self._cancel_events[status_key] = cancel_event

        free_space = self.check_free_space(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        if not free_space.ok:
            msg = (
                "Not enough disk space. "
                f"Required ~{_format_gib(free_space.required_bytes)} GB, "
                f"available {_format_gib(free_space.available_bytes)} GB."
            )
            if free_space.size_unknown:
                msg += " Model size is unknown; using 10 GB reserve."
            failed = ModelInstallStatus(model_id=model_id, status="failed", progress=0.0, message=msg, error=msg)
            self._set_status(failed, status_key=status_key, progress_cb=progress_cb)
            with self._lock:
                self._cancel_events.pop(status_key, None)
            return InstallResult(model_id=model_id, ok=False, status="failed", message=msg, error=msg)

        initial_msg = "Starting install..."
        if free_space.size_unknown:
            initial_msg = "Starting install (model size unknown, reserving 10 GB)."
        self._set_status(
            ModelInstallStatus(model_id=model_id, status="downloading", progress=0.0, message=initial_msg),
            status_key=status_key,
            progress_cb=progress_cb,
        )

        try:
            if spec.installer == "hf_snapshot":
                installed_dir = self._install_hf_snapshot(
                    spec,
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                    status_key=status_key,
                )
            elif spec.installer == "argos_package":
                installed_dir = self._install_argos_package(
                    spec,
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                    status_key=status_key,
                )
            else:
                raise RuntimeError(f"Unsupported installer kind: {spec.installer}")

            if cancel_event.is_set():
                raise InstallCancelledError("Install cancelled by user.")

            installed_status = ModelInstallStatus(
                model_id=model_id,
                status="installed",
                progress=1.0,
                message="Installed.",
            )
            self._set_status(installed_status, status_key=status_key, progress_cb=progress_cb)
            return InstallResult(
                model_id=model_id,
                ok=True,
                status="installed",
                message="Installed.",
                installed_dir=str(installed_dir),
            )
        except InstallCancelledError as exc:
            status = ModelInstallStatus(
                model_id=model_id,
                status="not_installed",
                progress=0.0,
                message="Install cancelled.",
                error=str(exc),
            )
            self._set_status(status, status_key=status_key, progress_cb=progress_cb)
            return InstallResult(
                model_id=model_id,
                ok=False,
                status="not_installed",
                message="Install cancelled.",
                cancelled=True,
                error=str(exc),
            )
        except Exception as exc:
            msg = str(exc) or "Install failed."
            failed = ModelInstallStatus(model_id=model_id, status="failed", progress=0.0, message=msg, error=msg)
            self._set_status(failed, status_key=status_key, progress_cb=progress_cb)
            return InstallResult(
                model_id=model_id,
                ok=False,
                status="failed",
                message=msg,
                error=msg,
            )
        finally:
            with self._lock:
                self._cancel_events.pop(status_key, None)

    def _set_status(
        self,
        status: ModelInstallStatus,
        *,
        status_key: str | None = None,
        progress_cb: InstallProgressCallback | None,
    ) -> None:
        with self._lock:
            self._statuses[status_key or status.model_id] = status
        if progress_cb is not None:
            progress_cb(status)

    def _install_hf_snapshot(
        self,
        spec,
        *,
        cancel_event: threading.Event,
        progress_cb: InstallProgressCallback | None,
        status_key: str,
    ) -> Path:
        partial_dir = get_partial_model_dir(spec.id, create=True)
        marker_file = partial_dir / "download.inprogress"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text("downloading", encoding="utf-8")

        target_dir = get_hf_snapshot_dir(spec.model_ref, create=True)

        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        snapshot_download = self._snapshot_download_fn
        if snapshot_download is None:
            from huggingface_hub import snapshot_download as _snapshot_download

            snapshot_download = _snapshot_download

        started_at = time.monotonic()

        def _progress_hook(progress: float) -> None:
            bounded_progress = max(0.0, min(0.98, progress))
            self._set_status(
                ModelInstallStatus(
                    model_id=spec.id,
                    status="downloading",
                    progress=bounded_progress,
                    message=_format_progress_message(
                        "Downloading model files...",
                        bounded_progress,
                        time.monotonic() - started_at,
                    ),
                ),
                status_key=status_key,
                progress_cb=progress_cb,
            )

        tqdm_class = _build_install_progress_tqdm(
            cancel_event=cancel_event,
            progress_hook=_progress_hook,
        )

        snapshot_download(
            repo_id=spec.model_ref,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=tqdm_class,
        )
        if cancel_event.is_set():
            raise InstallCancelledError("Install cancelled by user.")

        if marker_file.exists():
            marker_file.unlink()
        shutil.rmtree(partial_dir, ignore_errors=True)
        return target_dir

    def _install_argos_package(
        self,
        spec,
        *,
        cancel_event: threading.Event,
        progress_cb: InstallProgressCallback | None,
        status_key: str,
    ) -> Path:
        partial_dir = get_partial_model_dir(spec.id, create=True)
        partial_dir.mkdir(parents=True, exist_ok=True)
        marker_file = partial_dir / "download.inprogress"
        marker_file.write_text("downloading", encoding="utf-8")

        package_dir = configure_argos_packages_dir(create=True)
        started_at = time.monotonic()

        def _emit(progress: float, message: str) -> None:
            bounded_progress = max(0.0, min(0.98, progress))
            self._set_status(
                ModelInstallStatus(
                    model_id=spec.id,
                    status="downloading",
                    progress=bounded_progress,
                    message=_format_progress_message(
                        message,
                        bounded_progress,
                        time.monotonic() - started_at,
                    ),
                ),
                status_key=status_key,
                progress_cb=progress_cb,
            )

        if cancel_event.is_set():
            raise InstallCancelledError("Install cancelled by user.")
        _emit(0.05, "Updating Argos package index...")

        from packaging.version import Version

        from argostranslate import package

        package.update_package_index()
        if cancel_event.is_set():
            raise InstallCancelledError("Install cancelled by user.")

        src_lang, tgt_lang = _parse_argos_pair(spec.model_ref)
        available = package.get_available_packages()
        candidates = [
            pkg
            for pkg in available
            if getattr(pkg, "from_code", None) == src_lang and getattr(pkg, "to_code", None) == tgt_lang
        ]
        if not candidates:
            raise RuntimeError(f"Argos package not found for {src_lang}->{tgt_lang}.")

        selected = max(candidates, key=lambda pkg: Version(str(getattr(pkg, "package_version", "0") or "0")))

        _emit(0.35, f"Downloading Argos package {src_lang}->{tgt_lang}...")
        download_path = selected.download()
        if cancel_event.is_set():
            raise InstallCancelledError("Install cancelled by user.")

        _emit(0.75, "Installing Argos package...")
        package.install_from_path(download_path)
        try:
            Path(download_path).unlink(missing_ok=True)
        except Exception:
            pass

        installed = package.get_installed_packages(path=get_argos_packages_dir(create=True))
        if not any(
            getattr(pkg, "from_code", None) == src_lang and getattr(pkg, "to_code", None) == tgt_lang
            for pkg in installed
        ):
            raise RuntimeError("Argos package install verification failed.")

        if marker_file.exists():
            marker_file.unlink()
        shutil.rmtree(partial_dir, ignore_errors=True)
        return package_dir


def _parse_argos_pair(model_ref: str) -> tuple[str, str]:
    ref = (model_ref or "").strip().lower()
    if ref.startswith("argos-"):
        parts = ref.split("-")
        if len(parts) >= 3 and parts[1] and parts[2]:
            return parts[1], parts[2]
    return "en", "ru"


_DEFAULT_INSTALLER: ModelInstaller | None = None


def get_model_installer() -> ModelInstaller:
    global _DEFAULT_INSTALLER
    if _DEFAULT_INSTALLER is None:
        _DEFAULT_INSTALLER = ModelInstaller()
    return _DEFAULT_INSTALLER
