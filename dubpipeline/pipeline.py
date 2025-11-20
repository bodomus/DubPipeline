# оркестратор шагов

from pathlib import Path
from .config import config
from .audio_extractor import AudioExtractor
from .transcriber import Transcriber
from .translator import Translator
from .subtitles import SubtitleFormatter

class SubtitlesPipeline:
    def __init__(self):
        self.audio_extractor = AudioExtractor()
        self.transcriber = Transcriber(
            model_name=config.transcription_model,
            device=config.device,
        )
        self.translator = Translator()
        self.subtitle_formatter = SubtitleFormatter()

    def run(self, video_path: Path, output_srt_path: Path) -> Path:
        work_audio = config.work_dir / (video_path.stem + ".wav")
        audio_path = self.audio_extractor.extract_wav(video_path, work_audio)
        en_segments = self.transcriber.transcribe(audio_path)
        ru_segments = self.translator.translate_segment_texts(en_segments)
        srt_text = self.subtitle_formatter.to_srt(ru_segments)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)
        output_srt_path.write_text(srt_text, encoding="utf-8")
        return output_srt_path
