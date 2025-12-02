# EN -> RU
from typing import List
from dubpipeline.transcriber import Segment

class Translator:
    def __init__(self):
        # TODO: инициализировать локальный переводчик (например, Argos Translate)
        ...

    def translate_segment_texts(self, segments: List[Segment]) -> List[Segment]:
        """
        Принимает EN-сегменты, возвращает новые сегменты с text на RU.
        """
        translated_segments: List[Segment] = []
        for seg in segments:
            ru_text = self._translate(seg["text"])
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": ru_text,
            })
        return translated_segments

    def _translate(self, text: str) -> str:
        # TODO: реальная реализация локального перевода
        return text  # временный заглушка
