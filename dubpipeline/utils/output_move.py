from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Sequence

from dubpipeline.utils.logging import info


@dataclass(frozen=True)
class OutputMoveResult:
    source: Path
    destination: Path


class OutputMover:
    def __init__(self, target_dir: str, *, base_dir: Path | None = None) -> None:
        self._raw_target = target_dir
        self._base_dir = base_dir

    def _resolve_target_dir(self) -> Path:
        target = Path(self._raw_target).expanduser()
        if not target.is_absolute() and self._base_dir is not None:
            target = (self._base_dir / target).resolve()
        return target

    @staticmethod
    def _available_destination(target_dir: Path, source: Path) -> tuple[Path, bool]:
        candidate = target_dir / source.name
        if not candidate.exists():
            return candidate, False

        stem = source.stem
        suffix = source.suffix
        idx = 1
        while True:
            candidate = target_dir / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                return candidate, True
            idx += 1

    def move_outputs(self, sources: Sequence[Path]) -> list[OutputMoveResult]:
        move_to_dir = (self._raw_target or "").strip()
        if not move_to_dir:
            info("[dubpipeline] Move output disabled (empty)")
            return []

        target_dir = self._resolve_target_dir()
        info(f"[dubpipeline] Move output enabled: {target_dir}")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to create target dir '{target_dir}': {exc}") from exc

        results: list[OutputMoveResult] = []
        for src in sources:
            if not src.exists():
                raise RuntimeError(f"Output file not found: {src}")

            dst, collided = self._available_destination(target_dir, src)
            if collided:
                info(f"[dubpipeline] Destination exists, using: {dst}")

            info(f"[dubpipeline] Moving file: {src} -> {dst}")
            try:
                shutil.move(str(src), str(dst))
            except Exception as exc:
                raise RuntimeError(f"Failed to move '{src}' to '{dst}': {exc}") from exc

            results.append(OutputMoveResult(source=src, destination=dst))

        return results
