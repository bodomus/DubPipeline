from __future__ import annotations

import importlib
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dubpipeline.config import PathsConfig, PipelineConfig
from dubpipeline.steps.step_tts_core import synthesize_segments_to_wavs


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
    @staticmethod
    def _case_dir(prefix: str) -> Path:
        root = Path("tests/.tmp_runtime")
        root.mkdir(parents=True, exist_ok=True)
        case = root / f"{prefix}_{uuid4().hex}"
        case.mkdir(parents=True, exist_ok=True)
        return case

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

    def test_synthesize_preview_text_uses_requested_language(self):
        step_tts = importlib.import_module("dubpipeline.steps.step_tts")
        captured: dict[str, object] = {}

        class FakeTts:
            def tts_to_file(self, **kwargs):
                captured.update(kwargs)

        case = self._case_dir("tts_preview_lang")
        out_file = case / "preview.wav"
        with patch("dubpipeline.steps.step_tts._load_tts", return_value=FakeTts()):
            step_tts.synthesize_preview_text(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                voice_id="speaker",
                preview_text="Bonjour",
                out_file=out_file,
                use_gpu=False,
                lang="fr",
            )

        self.assertEqual(captured["language"], "fr")

    def test_run_uses_target_language_for_segment_synthesis(self):
        step_tts = importlib.import_module("dubpipeline.steps.step_tts")
        case = self._case_dir("tts_target_lang")
        cfg = _make_cfg(case, provider="coqui")
        cfg.languages.tgt = "de"
        cfg.paths.segments_tgt_file.write_text(
            json.dumps([{"id": "1", "start": 0.0, "end": 1.0, "text_tgt": "Hallo"}], ensure_ascii=False),
            encoding="utf-8",
        )

        with (
            patch.object(step_tts, "TTS", object()),
            patch("dubpipeline.steps.step_tts.synthesize_segments_to_wavs", return_value=[] ) as synthesize_segments_to_wavs,
        ):
            step_tts.run(cfg)

        self.assertEqual(synthesize_segments_to_wavs.call_args.kwargs["lang"], "de")
        self.assertEqual(Path(synthesize_segments_to_wavs.call_args.args[2]), cfg.paths.tts_segments_dir)

    def test_segment_synthesis_prefers_translated_text(self):
        case = self._case_dir("tts_prefers_translated_text")
        cfg = _make_cfg(case, provider="coqui")
        cfg.languages.tgt = "ru"
        captured: dict[str, object] = {}

        class FakeTts:
            speakers = ["speaker"]

            def tts_to_file(self, **kwargs):
                captured.update(kwargs)

        segments = [
            {
                "id": "1",
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world",
                "text_tgt": "Привет, мир",
            }
        ]

        with patch("dubpipeline.steps.step_tts_core._load_tts", return_value=FakeTts()):
            wavs = synthesize_segments_to_wavs(segments, cfg, cfg.paths.tts_segments_dir, show_progress=False)

        self.assertEqual(captured["text"], "Привет, мир")
        self.assertEqual(captured["language"], "ru")
        self.assertEqual(wavs, [cfg.paths.tts_segments_dir / "seg_0001.wav"])


if __name__ == "__main__":
    unittest.main()
