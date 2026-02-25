from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from dubpipeline.config import PathsConfig, PipelineConfig


def _make_cfg(root: Path, *, provider: str) -> PipelineConfig:
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(project_name="sample", project_dir=root)
    cfg.tts.provider = provider
    cfg.paths = PathsConfig(
        workdir=root,
        out_dir=out_dir,
        input_video=root / "input.mp4",
        audio_wav=out_dir / "sample.wav",
        voice_input_wav=out_dir / "voice_input.wav",
        translated_voice_wav=out_dir / "translated_voice.wav",
        background_wav=out_dir / "noise.wav",
        mixed_wav=out_dir / "mixed.wav",
        segments_file=out_dir / "sample.segments.json",
        segments_ru_file=out_dir / "sample.segments.ru.json",
        srt_file_en=out_dir / "sample.srt",
        tts_segments_dir=out_dir / "segments" / "tts_ru_segments",
        tts_segments_aligned_dir=out_dir / "segments" / "tts_ru_segments_aligned",
        final_video=out_dir / "sample.ru.muxed.mp4",
    )
    return cfg


class TtsProviderImportTests(unittest.TestCase):
    def test_step_tts_module_imports_without_tts_package(self):
        mod = importlib.import_module("dubpipeline.steps.step_tts")
        self.assertTrue(hasattr(mod, "run"))

    def test_coqui_provider_without_package_has_clear_error(self):
        step_tts = importlib.import_module("dubpipeline.steps.step_tts")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp), provider="coqui")
            with self.assertRaises(RuntimeError) as ctx:
                step_tts.run(cfg)
            self.assertIn("TTS provider=coqui выбран", str(ctx.exception))
            self.assertIn("pip install TTS", str(ctx.exception))

    def test_non_coqui_provider_does_not_require_tts_import(self):
        step_tts = importlib.import_module("dubpipeline.steps.step_tts")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp), provider="edge-tts")
            with self.assertRaises(RuntimeError) as ctx:
                step_tts.run(cfg)
            self.assertIn("provider=edge-tts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
