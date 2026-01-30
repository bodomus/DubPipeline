from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

from dubpipeline.utils.logging import log


@dataclass(frozen=True)
class GitMeta:
    commit: str = "unknown"
    branch: str = "unknown"
    dirty: bool = False


def _run_git(args: list[str]) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return p.stdout.strip()
    except Exception:
        return None


def get_git_meta() -> GitMeta:
    commit = _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    status = _run_git(["status", "--porcelain"])
    dirty = bool(status)
    return GitMeta(commit=commit, branch=branch, dirty=dirty)


def log_run_header(info_logger, cfg, *, device: str, compute_type: str,
                   asr_model: str, batch_size: int) -> None:
    g = get_git_meta()

    # Вытаскиваем “ключевые параметры” максимально надёжно
    # (что не нашли — не врём, пишем unknown)
    lang = getattr(cfg, "language", None) or getattr(cfg, "lang", None) or "unknown"

    # Если у вас перевод/tts настраиваются env-ами — логируем их все.
    env_dump = {k: v for k, v in os.environ.items() if k.startswith("DUBPIPELINE_")}

    info_logger(f"[RUN_META] git_commit={g.commit} branch={g.branch} dirty={int(g.dirty)}")
    info_logger(f"[RUN_META] device={device} compute_type={compute_type} batch_size={batch_size} asr_model={asr_model}")
    info_logger(f"[RUN_META] python={platform.python_version()} platform={platform.platform()}")

    if env_dump:
        # Не надо простыню на 500 строк — но ключи важны для воспроизводимости
        # Если хотите компактно — ограничьте здесь список.
        for k in sorted(env_dump.keys()):
            info_logger(f"[RUN_META] env {k}={env_dump[k]}")
