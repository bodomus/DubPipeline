from __future__ import annotations


class SegmentProgress:
    def __init__(self, total: int) -> None:
        self._total = max(0, int(total))
        self._last_completed = 0

    def update(self, completed: int) -> None:
        if self._total <= 0:
            percent = 100
            safe_completed = 0
        else:
            safe_completed = max(0, min(int(completed), self._total))
            percent = int((safe_completed / self._total) * 100)
        self._last_completed = safe_completed
        print(f"\rСДЕЛАНО: {safe_completed}/{self._total} ({percent}%)", end="", flush=True)

    def finish(self) -> None:
        if self._total <= 0:
            return
        if self._last_completed != self._total:
            self.update(self._total)
        print()
