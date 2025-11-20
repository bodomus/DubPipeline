# настройки: пути, модели, устройства, язык

from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    ffmpeg_path: str = "ffmpeg"          # если в PATH, можно так и оставить
    work_dir: Path = Path("./work")
    transcription_model: str = "large-v2"
    translation_source_lang: str = "en"
    translation_target_lang: str = "ru"
    device: str = "cuda"                 # или "cpu"

config = AppConfig()
