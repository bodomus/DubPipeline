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
from dubpipeline.translation.service import ActiveModel, TranslationModelError, TranslationModelUnavailableError, TranslatorService
from dubpipeline.translation.providers import (
    ArgosTranslationProvider,
    DEFAULT_QWEN_TRANSLATION_PROMPT,
    HfSeq2SeqTranslationProvider,
    QWEN_FP8_MODEL_REF,
    QWEN_GENERATION_CANDIDATES,
    QWEN_GENERATION_PROFILE,
    QwenTranslationProvider,
    TranslationProviderContext,
    create_translation_provider,
    resolve_qwen_model_ref,
    resolve_qwen_quantization,
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
            "qwen3_8b",
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

    def test_provider_factory_creates_qwen_provider(self):
        context = TranslationProviderContext(
            active_model=ActiveModel(
                model_id="qwen3_8b",
                label="Qwen3 (8B FP8)",
                backend="llm_qwen",
                model_ref=QWEN_FP8_MODEL_REF,
            ),
            src_lang="en",
            tgt_lang="ru",
            usegpu=False,
            batch_size=1,
            max_new_tokens=32,
        )

        provider = create_translation_provider("qwen", context)

        self.assertIsInstance(provider, QwenTranslationProvider)

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

    def test_translator_service_explicit_qwen_provider_selects_default_model(self):
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "en"
        cfg.languages.tgt = "ru"
        cfg.translation.provider = "qwen"

        with (
            patch(
                "dubpipeline.translation.service.get_model_status",
                return_value=ModelStatus(available=True, enabled=True, reason=""),
            ),
            patch("dubpipeline.translation.service.get_model_install_dir", return_value=None),
        ):
            service = TranslatorService(cfg)

        self.assertEqual(service.provider_id, "qwen")
        self.assertEqual(service.model_id, "qwen3_8b")
        self.assertEqual(service.backend, "llm_qwen")
        self.assertEqual(service._active.model_ref, QWEN_FP8_MODEL_REF)

    def test_qwen_cache_scope_includes_prompt_and_model_identity(self):
        def make_service(prompt: str, model_ref: str = QWEN_FP8_MODEL_REF) -> TranslatorService:
            cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
            cfg.languages.src = "en"
            cfg.languages.tgt = "ru"
            cfg.translation.provider = "qwen"
            cfg.translation.prompt = prompt
            cfg.translation.quantization = "fp8"

            service = TranslatorService.__new__(TranslatorService)
            service._cfg = cfg
            service._provider_id = "qwen"
            service._active = ActiveModel(
                model_id="qwen3_8b",
                label="Qwen3 (8B FP8)",
                backend="llm_qwen",
                model_ref=model_ref,
            )
            return service

        default_scope = make_service("").cache_scope
        custom_scope = make_service("Translate only as subtitles.").cache_scope
        other_model_scope = make_service("", "Qwen/Qwen3-8B-AWQ").cache_scope

        self.assertIn("provider=qwen", default_scope)
        self.assertIn(f"model_ref={QWEN_FP8_MODEL_REF}", default_scope)
        self.assertIn("prompt_sha256=", default_scope)
        self.assertIn(f"generation={QWEN_GENERATION_PROFILE}", default_scope)
        self.assertNotEqual(default_scope, custom_scope)
        self.assertNotEqual(default_scope, other_model_scope)


class QwenTranslationProviderTests(unittest.TestCase):
    def setUp(self):
        QwenTranslationProvider._QWEN_CACHE.clear()

    def tearDown(self):
        QwenTranslationProvider._QWEN_CACHE.clear()

    @staticmethod
    def _context(**overrides) -> TranslationProviderContext:
        values = {
            "active_model": ActiveModel(
                model_id="qwen3_8b",
                label="Qwen3 (8B)",
                backend="llm_qwen",
                model_ref="Qwen/Qwen3-8B",
            ),
            "src_lang": "en",
            "tgt_lang": "ru",
            "usegpu": False,
            "batch_size": 1,
            "max_new_tokens": 128,
            "qwen_model": "",
            "qwen_device": "cpu",
            "qwen_dtype": "auto",
            "qwen_quantization": "auto",
            "qwen_max_new_tokens": 64,
            "qwen_prompt": "",
        }
        values.update(overrides)
        return TranslationProviderContext(**values)

    def test_prompt_uses_default_rules_and_current_text(self):
        provider = QwenTranslationProvider(self._context())

        prompt = provider.build_prompt("Feed it into the node.")

        self.assertIn(DEFAULT_QWEN_TRANSLATION_PROMPT, prompt)
        self.assertIn("Feed it into the node.", prompt)
        self.assertIn("потоке данных", prompt)
        self.assertIn("feed, drive, pass, route, plug, connect и sample", prompt)
        self.assertIn("Не меняй местами то, что управляет", prompt)

    def test_selected_generation_profile_is_conservative_sampling(self):
        self.assertEqual(QWEN_GENERATION_PROFILE, "qwen3-nothink-conservative-translation-v3")
        self.assertEqual(
            QWEN_GENERATION_CANDIDATES[QWEN_GENERATION_PROFILE].generate_kwargs(),
            {
                "do_sample": True,
                "num_beams": 1,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
            },
        )

    def test_output_cleanup_removes_labels_markdown_and_thinking(self):
        raw = """
<think>internal notes</think>
```text
Translation: Подайте это в ноду.
```
""".strip()

        cleaned = QwenTranslationProvider.clean_translation_output(raw)

        self.assertEqual(cleaned, "Подайте это в ноду.")

    def test_invalid_device_is_rejected_before_loading_model(self):
        provider = QwenTranslationProvider(self._context(qwen_device="tpu"))

        with self.assertRaisesRegex(TranslationModelUnavailableError, "device 'tpu' is not supported"):
            provider._device()

    def test_cuda_request_without_cuda_fails_clearly(self):
        provider = QwenTranslationProvider(self._context(qwen_device="cuda"))

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})()

        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(TranslationModelUnavailableError, "requested CUDA but CUDA is not available"):
                provider._device()

    def test_invalid_dtype_is_rejected(self):
        provider = QwenTranslationProvider(self._context(qwen_dtype="int4"))
        fake_torch = type(
            "FakeTorch",
            (),
            {"float16": object(), "bfloat16": object(), "float32": object()},
        )()

        with self.assertRaisesRegex(TranslationModelUnavailableError, "dtype 'int4' is not supported"):
            provider._torch_dtype(fake_torch, "cpu")

    def test_qwen_quantization_auto_selects_fp8_for_cuda_and_none_for_cpu(self):
        self.assertEqual(resolve_qwen_quantization("auto", device="cuda", usegpu=True), "fp8")
        self.assertEqual(resolve_qwen_quantization("auto", device="cpu", usegpu=True), "none")
        self.assertEqual(resolve_qwen_model_ref(
            model_ref_override="",
            catalog_model_ref=QWEN_FP8_MODEL_REF,
            quantization="auto",
            device="cuda",
            usegpu=True,
        ), QWEN_FP8_MODEL_REF)

    def test_invalid_quantization_is_rejected(self):
        with self.assertRaisesRegex(TranslationModelUnavailableError, "quantization 'int2' is not supported"):
            resolve_qwen_quantization("int2", device="cuda", usegpu=True)

    def test_output_validation_rejects_empty_and_prompt_echo(self):
        with self.assertRaisesRegex(TranslationModelError, "empty"):
            QwenTranslationProvider.validate_translation_output("Hello", "", "")
        with self.assertRaisesRegex(TranslationModelError, "echo"):
            QwenTranslationProvider.validate_translation_output(
                "Hello",
                DEFAULT_QWEN_TRANSLATION_PROMPT,
                DEFAULT_QWEN_TRANSLATION_PROMPT,
            )

    def test_output_validation_rejects_thinking_fences_and_labels(self):
        invalid_outputs = (
            ("<think>analysis</think>Перевод", "Перевод", "think"),
            ("```text\nПеревод\n```", "```text Перевод ```", "fences"),
            ("Assistant: Перевод", "Assistant: Перевод", "label"),
        )
        for raw, cleaned, message in invalid_outputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TranslationModelError, message):
                    QwenTranslationProvider.validate_translation_output("Hello", raw, cleaned)

    def test_model_is_loaded_once_for_multiple_translation_calls(self):
        provider = QwenTranslationProvider(self._context())
        load_count = {"tokenizer": 0, "model": 0}
        chat_template_kwargs: list[dict[str, object]] = []
        generation_kwargs: list[dict[str, object]] = []

        class FakeTensor:
            shape = (1, 3)

            def to(self, _device):
                return self

        class FakeOutput:
            def __getitem__(self, _item):
                return self

        class FakeTokenizer:
            pad_token_id = None
            eos_token_id = 0

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                load_count["tokenizer"] += 1
                return cls()

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
                self.last_messages = messages
                chat_template_kwargs.append(dict(_kwargs))
                return messages[-1]["content"]

            def __call__(self, *_args, **_kwargs):
                return {"input_ids": FakeTensor()}

            def batch_decode(self, *_args, **_kwargs):
                return ["Translation: Готовый перевод."]

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                load_count["model"] += 1
                return cls()

            def eval(self):
                return self

            def to(self, _device):
                return self

            def generate(self, **_kwargs):
                generation_kwargs.append(dict(_kwargs))
                return FakeOutput()

        class FakeInferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                return None

        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": FakeCuda,
                "float16": object(),
                "bfloat16": object(),
                "float32": object(),
                "inference_mode": lambda self=None: FakeInferenceMode(),
            },
        )()
        fake_transformers = type(
            "FakeTransformers",
            (),
            {
                "AutoTokenizer": FakeTokenizer,
                "AutoModelForCausalLM": FakeModel,
            },
        )()

        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            first = provider.translate_texts(["Feed it into the node."])
            second = provider.translate_texts(["Use the material."])

        self.assertEqual(first, ["Готовый перевод."])
        self.assertEqual(second, ["Готовый перевод."])
        self.assertEqual(load_count, {"tokenizer": 1, "model": 1})
        self.assertTrue(chat_template_kwargs)
        self.assertTrue(all(call.get("enable_thinking") is False for call in chat_template_kwargs))
        self.assertTrue(generation_kwargs)
        self.assertTrue(all(call.get("do_sample") is True for call in generation_kwargs))
        self.assertTrue(all(call.get("temperature") == 0.3 for call in generation_kwargs))

    def test_release_removes_cached_model(self):
        provider = QwenTranslationProvider(self._context())
        key = ("Qwen/Qwen3-8B", "cpu", "auto", "none")
        provider._cache_key = key
        model = type("FakeModel", (), {"to": lambda self, _device: self})()
        QwenTranslationProvider._QWEN_CACHE[key] = (object(), model)

        provider.release()

        self.assertNotIn(key, QwenTranslationProvider._QWEN_CACHE)

    def test_offload_keeps_cache_entry_for_batch_reuse(self):
        calls: list[str] = []
        provider = QwenTranslationProvider(self._context())
        key = ("Qwen/Qwen3-8B", "cpu", "auto", "none")
        provider._cache_key = key
        model = type("FakeModel", (), {"to": lambda self, device: calls.append(device) or self})()
        QwenTranslationProvider._QWEN_CACHE[key] = (object(), model)

        provider.offload()

        self.assertIn(key, QwenTranslationProvider._QWEN_CACHE)
        self.assertEqual(calls, ["cpu"])


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

            def release_after_translate(self, *, batch_file_count: int = 1) -> None:
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

    def test_config_load_accepts_explicit_qwen_provider(self):
        root = self._case_dir("translation_qwen_provider_config")
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
  provider: qwen
  model: Qwen/Qwen3-8B
  device: cuda
  dtype: bfloat16
  max_new_tokens: 512
  quantization: fp8
""".strip(),
            encoding="utf-8",
        )

        with patch(
            "dubpipeline.config.get_model_status",
            return_value=ModelStatus(available=True, enabled=True, reason=""),
        ):
            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertEqual(cfg.translation.provider, "qwen")
        self.assertEqual(cfg.translation.model_id, "qwen3_8b")
        self.assertEqual(cfg.translation.backend, "llm_qwen")
        self.assertEqual(cfg.translation.model_ref, "Qwen/Qwen3-8B")
        self.assertEqual(cfg.translation.device, "cuda")
        self.assertEqual(cfg.translation.dtype, "bfloat16")
        self.assertEqual(cfg.translation.quantization, "fp8")
        self.assertEqual(cfg.translation.max_new_tokens, 512)

    def test_config_qwen_provider_overrides_previous_non_qwen_model_id(self):
        root = self._case_dir("translation_qwen_overrides_old_model")
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
  provider: qwen
  model_id: opus_mt
""".strip(),
            encoding="utf-8",
        )

        with patch(
            "dubpipeline.config.get_model_status",
            return_value=ModelStatus(available=True, enabled=True, reason=""),
        ):
            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertEqual(cfg.translation.model_id, "qwen3_8b")
        self.assertEqual(cfg.translation.backend, "llm_qwen")


if __name__ == "__main__":
    unittest.main()
