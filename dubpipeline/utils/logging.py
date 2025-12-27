# dubpipeline/utils/logging.py
from __future__ import annotations
import sys
import time
from pathlib import Path

LEVELS = ("DEBUG", "INFO", "STEP", "WARN", "ERROR")

_log_file = None  # сюда положим открытую ручку файла


def init_logger(log_path: str | Path | None) -> None:
    """
    Вызываем один раз в начале работы пайплайна.
    Если log_path = None – пишем только в stdout.
    """
    global _log_file

    if log_path is None:
        _log_file = None
        return

    p = Path(log_path)
    #p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    _log_file = p.open("w", encoding="utf-8")


def _write_line(line: str) -> None:
    """Пишем одновременно в stdout и (опционально) в лог-файл."""
    # 1) stdout – для GUI
    print(line, file=sys.stdout, flush=True)

    # 2) файл
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def log(level: str, msg: str) -> None:
    level = level.upper()
    if level not in LEVELS:
        level = "INFO"

    ts = time.strftime("%H:%M:%S")
    # формат, который потом парсим в GUI:
    # [INFO ] 12:34:56 | сообщение
    line = f"[{level:5}] {ts} | {msg}"
    _write_line(line)


def debug(msg: str) -> None:
    log("DEBUG", msg)


def info(msg: str) -> None:
    log("INFO", msg)


def step(msg: str) -> None:
    log("STEP", msg)


def warn(msg: str) -> None:
    log("WARN", msg)


def error(msg: str) -> None:
    log("ERROR", msg)
