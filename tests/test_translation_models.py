from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dubpipeline.config import PipelineConfig
from dubpipeline.models.catalog import (
    ModelStatus,
    build_model_choices,
    get_model_status,
    list_model_specs,
)
from dubpipeline.steps import step_translate


class TranslationModelCatalogTests(unittest.TestCase):
    def test_catalog_has_unique_ids_and_valid_tiers(self):
        specs = list_model_specs()
        ids = [spec.id for spec in specs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(spec.tier in {"A", "B", "C"} for spec in specs))

        required_ids = {
            "nllb_200_1_3b",
            "nllb_200_3_3b",
            "qwen2_5_7b",
            "qwen2_5_14b",
            "mistral_7b",
            "mixtral_8x7b",
            "opus_mt",
            "argos",
        }
        self.assertTrue(required_ids.issubset(set(ids)))


class TranslationModelStatusTests(unittest.TestCase):
    def test_status_is_not_installed_when_model_files_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("dubpipeline.models.catalog._local_model_dirs", return_value=[tmp_path / "models"]):
                with patch("dubpipeline.models.catalog._hf_cache_roots", return_value=[tmp_path / "hf_cache"]):
                    status = get_model_status("nllb_200_1_3b")

        self.assertFalse(status.available)
        self.assertFalse(status.enabled)
        self.assertEqual(status.reason, "not installed")


class TranslationModelChoiceTests(unittest.TestCase):
    def test_choices_contain_enabled_and_disabled_flags_with_not_installed_marker(self):
        def fake_status(model_id: str) -> ModelStatus:
            if model_id == "nllb_200_1_3b":
                return ModelStatus(available=True, enabled=True, reason="")
            if model_id == "opus_mt":
                return ModelStatus(available=False, enabled=False, reason="not installed")
            if model_id in {"qwen2_5_7b", "qwen2_5_14b", "mistral_7b", "mixtral_8x7b"}:
                return ModelStatus(available=False, enabled=False, reason="not supported yet")
            return ModelStatus(available=True, enabled=True, reason="")

        with patch("dubpipeline.models.catalog.get_model_status", side_effect=fake_status):
            choices = build_model_choices()

        nllb_choice = next(choice for choice in choices if choice.model_id == "nllb_200_1_3b")
        self.assertTrue(nllb_choice.enabled)

        opus_choice = next(choice for choice in choices if choice.model_id == "opus_mt")
        self.assertFalse(opus_choice.enabled)
        self.assertIn("not installed", opus_choice.display)

        qwen_choice = next(choice for choice in choices if choice.model_id == "qwen2_5_7b")
        self.assertFalse(qwen_choice.enabled)
        self.assertIn("not supported yet", qwen_choice.display)


class TranslationStepIntegrationTests(unittest.TestCase):
    def test_translate_step_uses_translation_model_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seg_path = root / "segments.json"
            out_path = root / "segments.ru.json"
            seg_path.write_text(
                json.dumps(
                    [
                        {"start": 0.0, "end": 1.0, "text": "Hello"},
                        {"start": 1.0, "end": 2.0, "text": "World"},
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            cfg = PipelineConfig(project_name="sample", project_dir=root)
            cfg.paths.segments_file = seg_path
            cfg.paths.segments_ru_file = out_path
            cfg.translation.model_id = "opus_mt"
            cfg.translation.backend = "opus_mt"
            cfg.translation.model_ref = "Helsinki-NLP/opus-mt-en-ru"
            cfg.translate.release_vram = False

            captured: dict[str, str] = {}

            class FakeTranslator:
                model_label = "OPUS-MT (Helsinki-NLP)"
                model_id = "opus_mt"
                backend = "opus_mt"
                cache_scope = "opus_mt|opus_mt"

                def __init__(self) -> None:
                    self.calls: list[list[str]] = []
                    self.release_called = False

                def translate_texts(self, texts: list[str], *, sent_fallback: bool = True) -> list[str]:
                    self.calls.append(list(texts))
                    return [f"RU:{t}" for t in texts]

                def release(self) -> None:
                    self.release_called = True

            fake_translator = FakeTranslator()

            def fake_from_config(runtime_cfg: PipelineConfig) -> FakeTranslator:
                captured["model_id"] = runtime_cfg.translation.model_id
                return fake_translator

            with patch("dubpipeline.steps.step_translate.TranslatorService.from_config", side_effect=fake_from_config):
                step_translate.run(cfg)

            self.assertEqual(captured.get("model_id"), "opus_mt")
            self.assertEqual(fake_translator.calls, [["Hello", "World"]])

            with out_path.open("r", encoding="utf-8") as f:
                translated = json.load(f)
            self.assertEqual([item["text_ru"] for item in translated], ["RU:Hello", "RU:World"])


if __name__ == "__main__":
    unittest.main()
