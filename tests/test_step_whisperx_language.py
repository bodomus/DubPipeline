from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

from dubpipeline.config import PipelineConfig


if "whisperx" not in sys.modules:
    sys.modules["whisperx"] = types.SimpleNamespace()

from dubpipeline.steps import step_whisperx


class WhisperxLanguageTests(unittest.TestCase):
    def test_transcribe_kwargs_uses_configured_language(self):
        self.assertEqual(step_whisperx._transcribe_kwargs("en"), {"language": "en"})
        self.assertEqual(step_whisperx._transcribe_kwargs("ru"), {"language": "ru"})

    def test_transcribe_kwargs_keeps_auto_detection(self):
        self.assertEqual(step_whisperx._transcribe_kwargs("auto"), {})

    def test_alignment_language_uses_configured_language_when_set(self):
        self.assertEqual(step_whisperx._alignment_language("en", "cy"), "en")

    def test_alignment_language_uses_detected_language_for_auto(self):
        self.assertEqual(step_whisperx._alignment_language("auto", "cy"), "cy")

    def test_configured_source_language_defaults_to_english(self):
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = ""
        self.assertEqual(step_whisperx._configured_source_language(cfg), "en")

    def test_load_align_model_error_suggests_lang_src(self):
        def fail(**_kwargs):
            raise ValueError("missing model")

        original = step_whisperx.whisperx
        step_whisperx.whisperx = types.SimpleNamespace(load_align_model=fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "--lang-src en"):
                step_whisperx._load_align_model("cy", "cpu")
        finally:
            step_whisperx.whisperx = original


if __name__ == "__main__":
    unittest.main()
