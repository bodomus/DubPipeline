from __future__ import annotations

from pathlib import Path


def resolve_saved_input_state(raw_cfg: dict) -> tuple[str, str]:
    mode_raw = str(raw_cfg.get("input_mode", "")).strip().lower()
    mode = "dir" if mode_raw == "dir" else "file"

    paths_cfg = raw_cfg.get("paths") or {}
    input_video = str(paths_cfg.get("input_video") or raw_cfg.get("input_video") or "")
    input_path = str(raw_cfg.get("input_path") or "")
    input_dir = str(raw_cfg.get("input_dir") or paths_cfg.get("input_dir") or "")

    if mode_raw not in {"file", "dir"} and input_dir:
        mode = "dir"

    if mode == "dir":
        path_value = input_path or input_dir or input_video
    else:
        path_value = input_path or input_video

    return mode, path_value


def validate_input_path(path_value: str, *, is_dir_mode: bool) -> tuple[bool, str]:
    input_path = Path(path_value.strip()).expanduser()
    if not str(input_path):
        return False, "Укажите путь к входным данным."
    if not input_path.exists():
        return False, "Указанный путь не существует."
    if is_dir_mode and not input_path.is_dir():
        return False, "В режиме 'Папка' нужно указать директорию."
    if not is_dir_mode and not input_path.is_file():
        return False, "В режиме 'Один файл' нужно указать видеофайл."
    return True, ""
