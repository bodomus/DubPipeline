# генерация .srt / .vtt

from typing import List
from .transcriber import Segment

class SubtitleFormatter:
    def to_srt(self, segments: List[Segment]) -> str:
        lines: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            start = self._format_time(seg["start"])
            end = self._format_time(seg["end"])
            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"])
            lines.append("")  # пустая строка между субтитрами
        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        # 12.345 -> "00:00:12,345"
        millis = int(round(seconds * 1000))
        s, ms = divmod(millis, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
