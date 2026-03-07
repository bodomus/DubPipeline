from __future__ import annotations

import os
import sys
from pathlib import Path

UNKNOWN_SIZE_MIN_FREE_BYTES = 10 * 1024 * 1024 * 1024


def _resolve_models_root_dir() -> Path:
    override = os.getenv("DUBPIPELINE_MODELS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()

    local_appdata = os.getenv("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata).expanduser() / "DubPipeline" / "models"

    appdata = os.getenv("APPDATA", "").strip()
    if appdata:
        return Path(appdata).expanduser() / "DubPipeline" / "models"

    xdg_data = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg_data:
        return Path(xdg_data).expanduser() / "DubPipeline" / "models"

    return Path.home() / ".local" / "share" / "DubPipeline" / "models"


def get_models_root_dir(*, create: bool = True) -> Path:
    root = _resolve_models_root_dir()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _model_ref_path_parts(model_ref: str) -> list[str]:
    parts: list[str] = []
    for raw in (model_ref or "").replace("\\", "/").split("/"):
        token = raw.strip()
        if token and token != ".":
            parts.append(token)
    return parts


def get_hf_snapshot_dir(model_ref: str, *, create: bool = False) -> Path:
    root = get_models_root_dir(create=create)
    path = root / "hf"
    for part in _model_ref_path_parts(model_ref):
        path = path / part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_argos_packages_dir(*, create: bool = False) -> Path:
    path = get_models_root_dir(create=create) / "argos"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_partial_model_dir(model_id: str, *, create: bool = False) -> Path:
    path = get_models_root_dir(create=create) / ".partial" / model_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configure_argos_packages_dir(*, create: bool = True) -> Path:
    package_dir = get_argos_packages_dir(create=create)
    os.environ["ARGOS_PACKAGES_DIR"] = str(package_dir)
    os.environ["ARGOS_TRANSLATE_PACKAGES_DIR"] = str(package_dir)

    settings_module = sys.modules.get("argostranslate.settings")
    if settings_module is not None:
        settings_module.package_data_dir = package_dir
        settings_module.package_dirs = [package_dir]

    return package_dir
