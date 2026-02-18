from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from dubpipeline.utils.logging import info, warn


class AtomicFileReplacer:
    def make_temp_path(self, original: Path) -> Path:
        suffix = original.suffix or ".mp4"
        return original.with_name(f"{original.stem}.tmp.{uuid4().hex}{suffix}")

    def cleanup_temp(self, temp_path: Path) -> None:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception as exc:
            warn(f"[atomic_replace] failed to cleanup temp '{temp_path}': {exc}")

    def replace_with_temp(
        self,
        original: Path,
        temp_path: Path,
        *,
        keep_backup: bool = False,
    ) -> Path | None:
        if not temp_path.exists():
            raise FileNotFoundError(f"Temp file not found: {temp_path}")

        backup_path: Path | None = None
        if keep_backup and original.exists():
            backup_path = original.with_name(f"{original.name}.bak.{uuid4().hex}")
            shutil.copy2(original, backup_path)
            info(f"[atomic_replace] backup created: {backup_path}")

        os.replace(str(temp_path), str(original))
        info(f"[atomic_replace] replaced original file: {original}")
        return backup_path
