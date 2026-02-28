from __future__ import annotations

from pathlib import Path


def enumerate_input_files(dir_path: str | Path, *, recursive: bool, allowed_exts: set[str]) -> list[Path]:
    root = Path(dir_path)
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        [p for p in iterator if p.is_file() and p.suffix.lower() in allowed_exts],
        key=lambda p: str(p.relative_to(root)).lower(),
    )


def source_mode_disabled_map(*, is_dir: bool) -> dict[str, bool]:
    return {
        "-INPUT_PATH-": False,
        "-BROWSE_INPUT-": False,
        "-RECURSIVE-": not is_dir,
    }
