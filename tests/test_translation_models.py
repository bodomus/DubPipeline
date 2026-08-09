from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dubpipeline.config import PipelineConfig
from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.models.catalog import (
    NOT_SUPPORTED_REASON,
    ModelStatus,
    build_model_choices,
    get_model_status,
    is_unsupported_pair_reason,
    list_model_specs,
)
from dubpipeline.translation.service import ActiveModel, TranslationModelUnavailableError, TranslatorService
from dubpipeline.translation.providers import (
    ArgosTranslationProvider,
    HfSeq2SeqTranslationProvider,
    TranslationProviderContext,
    create_translation_provider,
    resolve_translation_provider_id,
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

    def test_qwen_and_mistral_specs_are_planned(self):
        specs = {spec.id: spec for spec in list_model_specs()}
        for model_id in ("qwen2_5_7b", "qwen2_5_14b", "mistral_7b", "mixtral_8x7b"):
            with self.subTest(model_id=model_id):
                spec = specs[model_id]
                self.assertFalse(spec.supported)
                self.assertEqual(spec.installer, "none")
                self.assertEqual(spec.status_hint, "planned")


class TranslationModelStatusTests(unittest.TestCase):
    def test_status_is_not_installed_when_model_files_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": str(tmp_path / "app_models")}):
                with patch("dubpipeline.models.catalog._legacy_local_model_dirs", return_value=[tmp_path / "models"]):
                    with patch("dubpipeline.models.catalog._hf_cache_roots", return_value=[tmp_path / "hf_cache"]):
                        status = get_model_status("nllb_200_1_3b")

        self.assertFalse(status.available)
        self.assertFalse(status.enabled)
        self.assertEqual(status.reason, "not installed")

    def test_status_is_not_installed_when_hf_snapshot_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "hf_cache"
            snapshot = (
                cache_root
                / "models--facebook--nllb-200-1.3B"
                / "snapshots"
                / "partial"
            )
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": str(tmp_path / "app_models")}):
                with patch("dubpipeline.models.catalog._legacy_local_model_dirs", return_value=[tmp_path / "models"]):
                    with patch("dubpipeline.models.catalog._hf_cache_roots", return_value=[cache_root]):
                        status = get_model_status("nllb_200_1_3b")

        self.assertFalse(status.available)
        self.assertFalse(status.enabled)
        self.assertEqual(status.reason, "not installed")

    def test_status_is_installed_when_hf_snapshot_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_root = tmp_path / "hf_cache"
            snapshot = (
                cache_root
                / "models--facebook--nllb-200-1.3B"
                / "snapshots"
                / "complete"
            )
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"00")

            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": str(tmp_path / "app_models")}):
                with patch("dubpipeline.models.catalog._legacy_local_model_dirs", return_value=[tmp_path / "models"]):
                    with patch("dubpipeline.models.catalog._hf_cache_roots", return_value=[cache_root]):
                        status = get_model_status("nllb_200_1_3b")

        self.assertTrue(status.available)
        self.assertTrue(status.enabled)
        self.assertEqual(status.reason, "")


class TranslationModelChoiceTests(unittest.TestCase):
    def test_choices_contain_enabled_and_disabled_flags_with_not_installed_marker(self):
        def fake_status(model_id: str, src_lang: str | None = None, tgt_lang: str | None = None) -> ModelStatus:
            if model_id == "nllb_200_1_3b":
                return ModelStatus(available=True, enabled=True, reason="")
            if model_id == "opus_mt":
                return ModelStatus(available=False, enabled=False, reason="not installed")
            if model_id in {"qwen2_5_7b", "qwen2_5_14b", "mistral_7b", "mixtral_8x7b"}:
                return ModelStatus(available=False, enabled=False, reason=NOT_SUPPORTED_REASON)
            return ModelStatus(available=True, enabled=True, reason="")

        with patch("dubpipeline.models.catalog.get_model_status", side_effect=fake_status):
            choices = build_model_choices()

        nllb_choice = next(choice for choice in choices if choice.model_id == "nllb_200_1_3b")
        self.assertTrue(nllb_choice.enabled)

        opus_choice = next(choice for choice in choices if choice.model_id == "opus_mt")
        self.assertFalse(opus_choice.enabled)
        self.assertIn("not installed", opus_choice.display)
        self.assertTrue(opus_choice.supported)
        self.assertEqual(opus_choice.installer, "hf_snapshot")

        qwen_choice = next(choice for choice in choices if choice.model_id == "qwen2_5_7b")
        self.assertFalse(qwen_choice.enabled)
        self.assertIn("planned", qwen_choice.display)
        self.assertFalse(qwen_choice.supported)
        self.assertEqual(qwen_choice.installer, "none")

    def test_pair_specific_model_choice_marks_unsupported_pair(self):
        choices = build_model_choices("it", "ru")

        argos_choice = next(choice for choice in choices if choice.model_id == "argos")
        self.assertFalse(argos_choice.enabled)
        self.assertIn("unsupported for it->ru", argos_choice.display)


class TranslationPairStatusTests(unittest.TestCase):
    def test_pair_specific_model_reports_unsupported_pair(self):
        status = get_model_status("argos", src_lang="it", tgt_lang="ru")

        self.assertFalse(status.available)
        self.assertFalse(status.enabled)
        self.assertTrue(is_unsupported_pair_reason(status.reason))

    def test_translator_service_rejects_unsupported_pair_before_load(self):
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "it"
        cfg.languages.tgt = "ru"
        cfg.translation.model_id = "argos"

        with self.assertRaises(TranslationModelUnavailableError) as ctx:
            TranslatorService(cfg)

        self.assertIn("unsupported for it->ru", str(ctx.exception))

    def test_translator_service_reports_missing_pair_specific_model(self):
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "en"
        cfg.languages.tgt = "de"
        cfg.translation.model_id = "opus_mt"

        with patch(
            "dubpipeline.translation.service.get_model_status",
            return_value=ModelStatus(available=False, enabled=False, reason="not installed"),
        ):
            with self.assertRaises(TranslationModelUnavailableError) as ctx:
                TranslatorService(cfg)

        message = str(ctx.exception)
        self.assertIn("OPUS-MT", message)
        self.assertIn("en->de", message)
        self.assertIn("Open Models -> Install", message)

    def test_translator_cache_scope_includes_language_pair(self):
        def make_service(src_lang: str, tgt_lang: str) -> TranslatorService:
            cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
            cfg.languages.src = src_lang
            cfg.languages.tgt = tgt_lang

            service = TranslatorService.__new__(TranslatorService)
            service._cfg = cfg
            service._active = ActiveModel(
                model_id="nllb_200_1_3b",
                label="Meta NLLB-200 (1.3B)",
                backend="nllb",
                model_ref="facebook/nllb-200-1.3B",
            )
            service._hf_cache_key = None
            return service

        en_ru_scope = make_service("en", "ru").cache_scope
        en_de_scope = make_service("en", "de").cache_scope

        self.assertIn("en->ru", en_ru_scope)
        self.assertIn("en->de", en_de_scope)
        self.assertNotEqual(en_ru_scope, en_de_scope)

    def test_provider_factory_resolves_auto_from_backend(self):
        context = TranslationProviderContext(
            active_model=ActiveModel(
                model_id="opus_mt",
                label="OPUS-MT (Helsinki-NLP)",
                backend="opus_mt",
                model_ref="Helsinki-NLP/opus-mt-en-ru",
            ),
            src_lang="en",
            tgt_lang="ru",
            usegpu=False,
            batch_size=1,
            max_new_tokens=32,
        )

        provider = create_translation_provider("auto", context)

        self.assertIsInstance(provider, HfSeq2SeqTranslationProvider)

    def test_provider_factory_creates_argos_provider(self):
        context = TranslationProviderContext(
            active_model=ActiveModel(
                model_id="argos",
                label="Argos Translate",
                backend="argos",
                model_ref="argos-en-ru",
            ),
            src_lang="en",
            tgt_lang="ru",
            usegpu=False,
            batch_size=1,
            max_new_tokens=32,
        )

        provider = create_translation_provider("argos", context)

        self.assertIsInstance(provider, ArgosTranslationProvider)

    def test_provider_factory_rejects_unknown_provider(self):
        with self.assertRaisesRegex(TranslationModelUnavailableError, "translation provider 'xyz' is not supported"):
            resolve_translation_provider_id("xyz", "argos")

    def test_translator_service_explicit_argos_provider_selects_argos_model(self):
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "en"
        cfg.languages.tgt = "ru"
        cfg.translation.provider = "argos"
        cfg.translation.model_id = "opus_mt"

        with (
            patch(
                "dubpipeline.translation.service.get_model_status",
                return_value=ModelStatus(available=True, enabled=True, reason=""),
            ),
            patch("dubpipeline.translation.service.get_model_install_dir", return_value=None),
        ):
            service = TranslatorService(cfg)

        self.assertEqual(service.provider_id, "argos")
        self.assertEqual(service.model_id, "argos")

    def test_translator_service_rejects_invalid_provider(self):
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.translation.provider = "xyz"
        cfg.translation.model_id = "argos"

        with self.assertRaisesRegex(TranslationModelUnavailableError, "translation provider 'xyz' is not supported"):
            TranslatorService(cfg)


class TranslationStepIntegrationTests(unittest.TestCase):
    @staticmethod
    def _case_dir(prefix: str) -> Path:
        root = Path("tests/.tmp_runtime")
        root.mkdir(parents=True, exist_ok=True)
        case = root / f"{prefix}_{uuid4().hex}"
        case.mkdir(parents=True, exist_ok=True)
        return case

    def test_translate_step_uses_translation_model_from_config(self):
        root = self._case_dir("translation_step")
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
        cfg.paths.segments_tgt_file = out_path
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

        def in_memory_cache(_db_path):
            con = sqlite3.connect(":memory:")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                )
                """
            )
            return con

        with (
            patch("dubpipeline.steps.step_translate.TranslatorService.from_config", side_effect=fake_from_config),
            patch("dubpipeline.steps.step_translate._open_cache", side_effect=in_memory_cache),
        ):
            step_translate.run(cfg)

        self.assertEqual(captured.get("model_id"), "opus_mt")
        self.assertEqual(fake_translator.calls, [["Hello", "World"]])

        with out_path.open("r", encoding="utf-8") as f:
            translated = json.load(f)
        self.assertEqual([item["text_tgt"] for item in translated], ["RU:Hello", "RU:World"])
        self.assertEqual([item["text_ru"] for item in translated], ["RU:Hello", "RU:World"])

    def test_config_load_accepts_explicit_argos_provider(self):
        root = self._case_dir("translation_provider_config")
        pipeline_file = root / "video.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
languages:
  src: en
  tgt: ru
paths:
  workdir: .
  out_dir: out
  input_video: sample.mp4
translation:
  provider: argos
""".strip(),
            encoding="utf-8",
        )

        with patch(
            "dubpipeline.config.get_model_status",
            return_value=ModelStatus(available=True, enabled=True, reason=""),
        ):
            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertEqual(cfg.translation.provider, "argos")
        self.assertEqual(cfg.translation.model_id, "argos")
        self.assertEqual(cfg.translation.backend, "argos")


if __name__ == "__main__":
    unittest.main()
