# dubpipeline/utils/timing.py
#
# Utilities for calculate running time of steps
#
#

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import Callable, Dict, Optional, List, Tuple
import threading


LogFn = Callable[[str], None]


def _fmt(seconds: float) -> str:
    # 0.123s / 12.3s / 3m 12s
    if seconds < 10:
        return f"{seconds:.3f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(round(seconds - m * 60))
    return f"{m}m {s:02d}s"


@dataclass
class TimingStat:
    calls: int = 0
    total_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0

    def add(self, dt: float) -> None:
        self.calls += 1
        self.total_s += dt
        if self.calls == 1:
            self.min_s = self.max_s = dt
        else:
            self.min_s = min(self.min_s, dt)
            self.max_s = max(self.max_s, dt)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0


class TimingCollector:
    """
    Сборщик таймингов для одного RUN (одного прогона пайплайна).
    Потокобезопасно (на случай, если шаги внутри используют threads).
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, TimingStat] = {}

    def add(self, name: str, dt: float) -> None:
        with self._lock:
            st = self._stats.get(name)
            if st is None:
                st = TimingStat()
                self._stats[name] = st
            st.add(dt)

    def summary_rows(self) -> List[Tuple[str, TimingStat]]:
        with self._lock:
            rows = list(self._stats.items())
        rows.sort(key=lambda x: x[1].total_s, reverse=True)
        return rows

    def dump_summary(self, log: LogFn, title: str = "Timing summary", top_n: int = 30) -> None:
        rows = self.summary_rows()
        if not rows:
            log(f"[TIMING] {title}: <no data>")
            return

        log(f"[TIMING] {title}:")
        log(f"[TIMING] {'Step':40} | {'Calls':>5} | {'Total':>10} | {'Avg':>10} | {'Min':>10} | {'Max':>10}")
        log(f"[TIMING] {'-'*40}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for i, (name, st) in enumerate(rows[:top_n], start=1):
            log(
                f"[TIMING] {name[:40]:40} | "
                f"{st.calls:5d} | "
                f"{_fmt(st.total_s):>10} | "
                f"{_fmt(st.avg_s):>10} | "
                f"{_fmt(st.min_s):>10} | "
                f"{_fmt(st.max_s):>10}"
            )


# Текущий коллектор прогона (ставим в начале RUN)
_current_collector: Optional[TimingCollector] = None


def set_collector(c: Optional[TimingCollector]) -> None:
    global _current_collector
    _current_collector = c


def get_collector() -> Optional[TimingCollector]:
    return _current_collector


class timed_block:
    """
    Контекст-менеджер для замера куска кода.
    Пример:
        with timed_block("ASR", info):
            ...
    """
    def __init__(self, name: str, log: Optional[LogFn] = None, collector: Optional[TimingCollector] = None) -> None:
        self.name = name
        self.log = log
        self.collector = collector
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = perf_counter() - self._t0
        c = self.collector or get_collector()
        if c is not None:
            c.add(self.name, dt)
        if self.log is not None:
            self.log(f"[TIMING] {self.name}: {_fmt(dt)}")
        # False = не подавлять исключения
        return False


def timed(name: Optional[str] = None, log: Optional[LogFn] = None, collector: Optional[TimingCollector] = None):
    """
    Декоратор для функций (в т.ч. RUN). Замеряет время и пишет в collector, при желании логирует.
    """
    def deco(fn):
        nm = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = perf_counter() - t0
                c = collector or get_collector()
                if c is not None:
                    c.add(nm, dt)
                if log is not None:
                    log(f"[TIMING] {nm}: {_fmt(dt)}")

        return wrapper

    return deco


def timed_run(log: LogFn, run_name: str = "RUN", top_n: int = 30):
    """
    Декоратор именно для верхнего RUN:
    - создаёт TimingCollector
    - ставит его как текущий
    - по выходу печатает общий итог по шагам
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            col = TimingCollector()
            set_collector(col)
            t0 = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = perf_counter() - t0
                col.add(run_name, dt)
                log(f"[TIMING] {run_name} finished in {_fmt(dt)}")
                col.dump_summary(log, title=f"{run_name} timing summary", top_n=top_n)
                set_collector(None)

        return wrapper
    return deco
