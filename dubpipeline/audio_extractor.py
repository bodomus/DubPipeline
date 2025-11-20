# извлечение аудио из видео (ffmpeg)

import subprocess
from pathlib import Path
from .config import config

class AudioExtractor:
    def __init__(self, ffmpeg_path: str | None = None):
        self.ffmpeg_path = ffmpeg_path or config.ffmpeg_path

    def extract_wav(self, video_path: Path, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vn",          # без видео
            "-acodec", "pcm_s16le",
            "-ar", "16000", # 16kHz для ASR
            "-ac", "1",     # mono
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        return out_path
