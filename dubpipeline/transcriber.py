# обертка над Whisper/WhisperX

from pathlib import Path
from typing import List, TypedDict

class Segment(TypedDict):
    start: float
    end: float
    text: str

class Transcriber:
    def __init__(self, model_name: str = "large-v2", device: str = "cuda"):
        # TODO: инициализировать Whisper / WhisperX
        self.model_name = model_name
        self.device = device

    def transcribe(self, audio_path: Path) -> List[Segment]:
        """
        Возвращает список сегментов с таймкодами и текстом (EN).
        """
        # TODO: вызвать модель и собрать сегменты
        raise NotImplementedError
