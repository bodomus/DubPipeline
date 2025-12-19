# dubpipeline/utils/logging.py
from __future__ import annotations
import sys
import time

LEVELS = ("DEBUG", "INFO", "STEP", "WARN", "ERROR")


def log(level: str, msg: str) -> None:
    level = level.upper()
    if level not in LEVELS:
        level = "INFO"

    ts = time.strftime("%H:%M:%S")
    # единый формат:
    # [LEVEL] HH:MM:SS | message
    level=level.strip()
    line = f"[{level:5}] {ts} | {msg}"
    print(line, flush=True, file=sys.stdout)


def info(msg: str) -> None:
    log("INFO", msg)


def step(msg: str) -> None:
    log("STEP", msg)


def warn(msg: str) -> None:
    log("WARN", msg)


def error(msg: str) -> None:
    log("ERROR", msg)

def debug(msg: str) -> None:
    log("DEBUG", msg)
