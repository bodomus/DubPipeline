from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from dubpipeline.config import PipelineConfig
from dubpipeline.models.catalog import build_model_choices, get_model_status
from dubpipeline.models.installer import ModelInstaller, _format_progress_message
from dubpipeline.models.storage import get_hf_snapshot_dir
from dubpipeline.translation.service import TranslationModelUnavailableError, TranslatorService


class _FakeDiskUsage:
    def __init__(self, free: int) -> None:
        self.total = free
        self.used = 0
        self.free = free


class ModelInstallerTests(unittest.TestCase):
    @staticmethod
    def _case_dir(prefix: str) -> Path:
        root = Path("tests/.tmp_runtime")
        root.mkdir(parents=True, exist_ok=True)
        case = root / f"{prefix}_{uuid4().hex}"
        case.mkdir(parents=True, exist_ok=True)
        return case

    def test_free_space_check_blocks_install_when_space_is_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": tmp}):
                installer = ModelInstaller(disk_usage_fn=lambda _path: _FakeDiskUsage(1024 * 1024))
                result = installer.install("nllb_200_3_3b")

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "failed")
        self.assertIn("Not enough disk space", result.message)

    def test_progress_message_includes_percent_elapsed_and_remaining(self):
        message = _format_progress_message(
            "Downloading model files...",
            0.42,
            78.0,
        )

        self.assertEqual(
            message,
            "Downloading model files... 42% | elapsed 01:18 | remaining ~01:48",
        )

    def test_hf_install_uses_expected_local_dir(self):
        captured: dict[str, object] = {}

        def fake_snapshot_download(**kwargs):
            captured.update(kwargs)
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (local_dir / "model.safetensors").write_bytes(b"00")
            return str(local_dir)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": tmp}):
                with patch("dubpipeline.models.catalog._legacy_local_model_dirs", return_value=[]):
                    with patch("dubpipeline.models.catalog._hf_cache_roots", return_value=[]):
                        installer = ModelInstaller(
                            snapshot_download_fn=fake_snapshot_download,
                            disk_usage_fn=lambda _path: _FakeDiskUsage(100 * 1024 * 1024 * 1024),
                        )
                        result = installer.install("opus_mt")
                        expected = get_hf_snapshot_dir("Helsinki-NLP/opus-mt-en-ru", create=False)

        self.assertTrue(result.ok)
        self.assertEqual(Path(captured["local_dir"]), expected)
        self.assertFalse(bool(captured["local_dir_use_symlinks"]))
        self.assertTrue(bool(captured["resume_download"]))
        self.assertTrue(hasattr(captured["tqdm_class"], "get_lock"))

    def test_pair_specific_hf_install_uses_pair_local_dir(self):
        captured: dict[str, object] = {}

        def fake_install_hf_snapshot(_self, spec, *, cancel_event, progress_cb, status_key):
            del cancel_event, progress_cb
            captured["model_ref"] = spec.model_ref
            captured["status_key"] = status_key
            return get_hf_snapshot_dir(spec.model_ref, create=True)

        tmp = self._case_dir("installer_pair_hf")
        with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": str(tmp)}):
            with patch("dubpipeline.models.catalog._legacy_local_model_dirs", return_value=[]):
                with patch("dubpipeline.models.catalog._hf_cache_roots", return_value=[]):
                    with patch.object(
                        ModelInstaller,
                        "_install_hf_snapshot",
                        autospec=True,
                        side_effect=fake_install_hf_snapshot,
                    ):
                        installer = ModelInstaller(
                            disk_usage_fn=lambda _path: _FakeDiskUsage(100 * 1024 * 1024 * 1024),
                        )
                        result = installer.install("opus_mt", src_lang="fr", tgt_lang="de")
                        expected = get_hf_snapshot_dir("Helsinki-NLP/opus-mt-fr-de", create=False)

        self.assertTrue(result.ok)
        self.assertEqual(Path(result.installed_dir), expected)
        self.assertEqual(captured["model_ref"], "Helsinki-NLP/opus-mt-fr-de")
        self.assertEqual(captured["status_key"], "opus_mt|fr|de")

    def test_install_rejects_unsupported_pair_before_download(self):
        installer = ModelInstaller()

        result = installer.install("argos", src_lang="it", tgt_lang="ru")

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "not_installed")
        self.assertIn("unsupported for it->ru", result.message)

    def test_not_supported_model_is_not_installable_and_not_selectable(self):
        installer = ModelInstaller()
        result = installer.install("qwen2_5_7b")
        status = get_model_status("qwen2_5_7b")
        qwen_choice = next(choice for choice in build_model_choices() if choice.model_id == "qwen2_5_7b")

        self.assertFalse(result.ok)
        self.assertIn("planned", result.message.lower())
        self.assertFalse(status.enabled)
        self.assertFalse(qwen_choice.enabled)
        self.assertIn("planned", qwen_choice.display.lower())


class TranslationOfflineSmokeTests(unittest.TestCase):
    def test_installed_hf_model_is_loaded_from_local_dir_with_local_files_only(self):
        class FakeTokenizer:
            calls: list[tuple[object, dict[str, object]]] = []

            @classmethod
            def from_pretrained(cls, model_ref, **kwargs):
                cls.calls.append((model_ref, kwargs))
                return object()

        class _FakeLoadedModel:
            def eval(self):
                return self

            def to(self, _device):
                return self

        class FakeModel:
            calls: list[tuple[object, dict[str, object]]] = []

            @classmethod
            def from_pretrained(cls, model_ref, **kwargs):
                cls.calls.append((model_ref, kwargs))
                return _FakeLoadedModel()

        fake_torch = types.SimpleNamespace(
            float16="float16",
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            AutoModelForSeq2SeqLM=FakeModel,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": tmp}):
                model_dir = get_hf_snapshot_dir("Helsinki-NLP/opus-mt-en-ru", create=True)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")
                (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
                (model_dir / "model.safetensors").write_bytes(b"00")

                cfg = PipelineConfig(project_name="sample", project_dir=Path(tmp))
                cfg.translation.model_id = "opus_mt"
                cfg.translation.backend = "opus_mt"
                cfg.translation.model_ref = "Helsinki-NLP/opus-mt-en-ru"
                cfg.usegpu = False

                TranslatorService._HF_CACHE.clear()
                with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
                    service = TranslatorService(cfg)
                    service._load_hf()

        tokenizer_ref, tokenizer_kwargs = FakeTokenizer.calls[0]
        model_ref, model_kwargs = FakeModel.calls[0]

        self.assertEqual(Path(tokenizer_ref), model_dir)
        self.assertEqual(Path(model_ref), model_dir)
        self.assertTrue(bool(tokenizer_kwargs.get("local_files_only")))
        self.assertTrue(bool(model_kwargs.get("local_files_only")))

    def test_hf_bin_weights_torch_26_error_is_reported_as_load_diagnostic(self):
        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, _model_ref, **_kwargs):
                return object()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, _model_ref, **kwargs):
                if kwargs.get("use_safetensors"):
                    raise OSError("No safetensors weights found")
                raise ValueError("Due to CVE-2025-32434, upgrade torch to at least v2.6 to use torch.load")

        fake_torch = types.SimpleNamespace(
            __version__="2.5.1+cu124",
            float16="float16",
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            AutoModelForSeq2SeqLM=FakeModel,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": tmp}):
                model_dir = get_hf_snapshot_dir("Helsinki-NLP/opus-mt-en-de", create=True)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")
                (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
                (model_dir / "model.safetensors").write_bytes(b"00")

                cfg = PipelineConfig(project_name="sample", project_dir=Path(tmp))
                cfg.languages.src = "en"
                cfg.languages.tgt = "de"
                cfg.translation.model_id = "opus_mt"
                cfg.usegpu = False

                TranslatorService._HF_CACHE.clear()
                with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
                    service = TranslatorService(cfg)
                    with self.assertRaises(TranslationModelUnavailableError) as ctx:
                        service._load_hf()

        message = str(ctx.exception)
        self.assertIn("installed", message)
        self.assertIn("cannot be loaded", message)
        self.assertIn("torch < 2.6", message)
        self.assertIn("CVE-2025-32434", message)
        self.assertNotIn("not installed", message)

    def test_hf_bin_weights_without_safetensors_preflight_reports_torch_upgrade(self):
        fake_torch = types.SimpleNamespace(
            __version__="2.5.1+cu124",
            float16="float16",
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
            AutoModelForSeq2SeqLM=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DUBPIPELINE_MODELS_ROOT": tmp}):
                model_dir = get_hf_snapshot_dir("Helsinki-NLP/opus-mt-en-de", create=True)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")
                (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
                (model_dir / "pytorch_model.bin").write_bytes(b"00")

                cfg = PipelineConfig(project_name="sample", project_dir=Path(tmp))
                cfg.languages.src = "en"
                cfg.languages.tgt = "de"
                cfg.translation.model_id = "opus_mt"
                cfg.usegpu = False

                TranslatorService._HF_CACHE.clear()
                with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
                    service = TranslatorService(cfg)
                    with self.assertRaises(TranslationModelUnavailableError) as ctx:
                        service._load_hf()

        message = str(ctx.exception)
        self.assertIn(str(model_dir), message)
        self.assertIn("PyTorch 2.5.1+cu124", message)
        self.assertIn(".bin weights without safetensors", message)
        self.assertNotIn("not installed", message)


if __name__ == "__main__":
    unittest.main()
