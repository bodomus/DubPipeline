from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Union

Arg = Union[str, os.PathLike]

def norm_arg(a: Arg) -> str:
    """PathLike -> str, плюс нормальная строка."""
    return os.fspath(a)

def cmdline_windows(args: Sequence[Arg]) -> str:
    """
    Делает строку команды в стиле Windows CreateProcess (cmd.exe-friendly),
    с корректным quoting по правилам MS CRT.
    Идеально для логов/копипаста.
    """
    args_s = [norm_arg(a) for a in args]
    return subprocess.list2cmdline(args_s)

def cmdline_posix(args: Sequence[Arg]) -> str:
    """
    Для Linux/macOS (если вдруг понадобится).
    """
    import shlex
    return " ".join(shlex.quote(norm_arg(a)) for a in args)

def pretty_cmd(args: Sequence[Arg]) -> str:
    """Авто: Windows -> windows quoting, иначе posix."""
    if os.name == "nt":
        return cmdline_windows(args)
    return cmdline_posix(args)

def run_cmd(
    args: Sequence[Arg],
    *,
    cwd: Arg | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Запуск БЕЗ shell (рекомендуется). Quoting руками не нужен.
    """
    args_s = [norm_arg(a) for a in args]
    return subprocess.run(
        args_s,
        cwd=None if cwd is None else norm_arg(cwd),
        env=env,
        check=check,
        text=True,
        capture_output=capture_output,
    )

# -------------------------
# Пример: ваш ffmpeg mux
# -------------------------

def build_ffmpeg_mux_cmd(video_mp4: Arg, ru_wav: Arg, out_mp4: Arg) -> List[str]:
    return [
        "ffmpeg", "-y",
        "-i", norm_arg(video_mp4),
        "-i", norm_arg(ru_wav),
        "-map", "0:v:0",
        "-map", "0:a:0",   # если может не быть оригинальной дорожки: замените на "0:a?"
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:0", "title=Original",
        "-metadata:s:a:1", "language=rus",
        "-metadata:s:a:1", "title=Russian_Dub",
        norm_arg(out_mp4),
    ]

if __name__ == "__main__":
    video = Path(r"J:\Projects\!!!AI\DubPipeline\tests\data\From Blueprints to C++ in Unreal Engine (Beginner Tutorial).mp4")
    ru    = Path(r"J:\Projects\!!!AI\DubPipeline\tests\out\From Blueprints to C++ in Unreal Engine (Beginner Tutorial).wav")
    out   = Path(r"J:\Projects\!!!AI\DubPipeline\tests\out\From Blueprints to C++ in Unreal Engine (Beginner Tutorial).ru.muxed.mp4")

    cmd = build_ffmpeg_mux_cmd(video, ru, out)

    print("[FFMPEG]", pretty_cmd(cmd))   # можно копипастить в cmd.exe
    run_cmd(cmd, check=True)
