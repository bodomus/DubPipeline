from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from dubpipeline.cli import run_pipeline
from dubpipeline.config import PathsConfig, PipelineConfig, StepsConfig
from dubpipeline.steps import audio_mix_step


def _make_cfg(root: Path) -> PipelineConfig:
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(project_name="sample", project_dir=root)
    cfg.paths = PathsConfig(
        workdir=root,
        out_dir=out_dir,
        input_video=root / "input.mp4",
        audio_wav=out_dir / "sample.wav",
        voice_input_wav=out_dir / "voice_input.wav",
        translated_voice_wav=out_dir / "translated_voice.wav",
        background_wav=out_dir / "background.wav",
        mixed_wav=out_dir / "mixed.wav",
        segments_file=out_dir / "sample.segments.json",
        segments_ru_file=out_dir / "sample.segments.ru.json",
        srt_file_en=out_dir / "sample.srt",
        tts_segments_dir=out_dir / "segments" / "tts_ru_segments",
        tts_segments_aligned_dir=out_dir / "segments" / "tts_ru_segments_aligned",
        final_video=out_dir / "sample.ru.muxed.mp4",
    )
    return cfg


class AudioMixStepTests(unittest.TestCase):
    def test_ffmpeg_command_is_built_correctly(self):
        cmd = audio_mix_step.build_ffmpeg_command(
            Path("noise.wav"),
            Path("translated_voice.wav"),
            Path("mixed.tmp.wav"),
            tts_gain_db=0.0,
            bg_gain_db=0.0,
        )
        self.assertEqual(
            cmd,
            [
                "ffmpeg",
                "-y",
                "-i",
                "noise.wav",
                "-i",
                "translated_voice.wav",
                "-filter_complex",
                "[0:a]volume=0.0dB[bg];[1:a]volume=0.0dB[tts];[bg][tts]amix=inputs=2:normalize=0[m]",
                "-map",
                "[m]",
                "-c:a",
                "pcm_s16le",
                "mixed.tmp.wav",
            ],
        )

    def test_missing_background_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_cfg(root)
            cfg.paths.translated_voice_wav.write_text("voice", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                audio_mix_step.run(cfg)

    def test_external_voice_skips_extract_audio_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = _make_cfg(root)
            cfg.steps = StepsConfig(
                extract_audio=True,
                asr_whisperx=False,
                translate=False,
                tts=False,
                align=False,
                merge=False,
            )

            pipeline_path = root / "video.pipeline.yaml"
            pipeline_path.write_text("project_name: sample", encoding="utf-8")

            external = root / "voice.wav"
            external.write_text("voice", encoding="utf-8")
            cfg.external_voice_track = str(external)
            cfg.paths.input_video.write_text("video", encoding="utf-8")

            fake_modules = {
                "dubpipeline.steps.step_align": types.SimpleNamespace(run=lambda _cfg: None),
                "dubpipeline.steps.step_merge_py": types.SimpleNamespace(run=lambda _cfg: None),
                "dubpipeline.steps.step_translate": types.SimpleNamespace(run=lambda _cfg: None),
                "dubpipeline.steps.step_tts": types.SimpleNamespace(run=lambda _cfg: None),
                "dubpipeline.steps.step_whisperx": types.SimpleNamespace(run=lambda _cfg: None),
            }
            with patch.dict("sys.modules", fake_modules):
                with patch("dubpipeline.steps.step_extract_audio.run") as extract_run:
                    run_pipeline(cfg, pipeline_path)
                    extract_run.assert_not_called()

            self.assertTrue(cfg.paths.voice_input_wav.exists())
            self.assertEqual(cfg.paths.audio_wav, cfg.paths.voice_input_wav)


if __name__ == "__main__":
    unittest.main()
