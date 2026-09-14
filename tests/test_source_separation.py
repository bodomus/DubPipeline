from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dubpipeline.cli import _build_cfg_for_input
from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.source_separation import (
    AudioSeparatorProvider,
    SourceSeparationError,
    build_request,
    create_provider,
    read_cached_result,
    resolve_background_audio_for_merge,
    run_source_separation,
    write_metadata,
)


def _write(path: Path, data: bytes = b"audio") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class SourceSeparationConfigTests(unittest.TestCase):
    def test_yaml_parses_source_separation_config_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir(parents=True, exist_ok=True)
            pipeline_file = root / "sample.pipeline.yaml"
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: sample.mp4
source_separation:
  mode: separated_background
  provider: bs_roformer
  model: model_bs_roformer_ep_368_sdr_12.9628.ckpt
  model_path: C:/models/BS-Roformer-InstVoc2.ckpt
  model_file_dir: C:/models
  device: cuda
  output_format: wav
  sample_rate: 44100
  command:
    - separator
    - --input
    - "{input_audio}"
    - --background
    - "{background_wav}"
  fallback_mode: legacy_ducking
  cache_enabled: true
""".strip(),
                encoding="utf-8",
            )

            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

            self.assertEqual(cfg.source_separation.mode, "separated_background")
            self.assertEqual(cfg.source_separation.provider, "bs_roformer")
            self.assertEqual(
                cfg.source_separation.model,
                "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            )
            self.assertEqual(cfg.source_separation.device, "cuda")
            self.assertEqual(cfg.source_separation.fallback_mode, "legacy_ducking")
            self.assertEqual(
                cfg.paths.separation_dir.parts[-3:], ("out", "separation", "sample")
            )
            self.assertEqual(
                cfg.paths.separation_background_wav.parts[-4:],
                ("out", "separation", "sample", "background.wav"),
            )

    def test_batch_input_rebuilds_separation_paths_per_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir(parents=True, exist_ok=True)
            pipeline_file = root / "sample.pipeline.yaml"
            input_file = root / "clip.mp4"
            input_file.write_text("x", encoding="utf-8")
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: clip.mp4
""".strip(),
                encoding="utf-8",
            )

            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
            run_cfg = _build_cfg_for_input(cfg, input_file)

            self.assertEqual(
                run_cfg.paths.separation_dir.parts[-3:], ("out", "separation", "clip")
            )
            self.assertEqual(
                run_cfg.paths.separation_metadata_json.parts[-4:],
                ("out", "separation", "clip", "metadata.json"),
            )


class SourceSeparationRuntimeTests(unittest.TestCase):
    def _config(self, root: Path, *, fallback: str = "none", device: str = "cpu"):
        project = root / "project"
        project.mkdir(parents=True, exist_ok=True)
        pipeline_file = project / "sample.pipeline.yaml"
        pipeline_file.write_text(
            f"""
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: sample.mp4
source_separation:
  mode: separated_background
  provider: bs_roformer
  model_path: model.ckpt
  command:
    - separator
    - --input
    - "{{input_audio}}"
    - --out-dir
    - "{{out_dir}}"
    - --model
    - "{{model_path}}"
  fallback_mode: {fallback}
  cache_enabled: true
""".strip(),
            encoding="utf-8",
        )
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        _write(cfg.paths.audio_wav)
        _write(Path(cfg.paths.workdir) / "model.ckpt", b"model-v1")
        return cfg

    def test_provider_runner_writes_stems_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            calls: list[list[str]] = []

            def runner(command):
                calls.append(list(command))
                _write(cfg.paths.separation_vocals_wav, b"vocals")
                _write(cfg.paths.separation_background_wav, b"background")
                return subprocess.CompletedProcess(command, 0, "", "")

            result = run_source_separation(cfg, runner=runner)

            self.assertIsNotNone(result)
            self.assertFalse(result.cache_hit)
            self.assertEqual(len(calls), 1)
            self.assertIn(str(cfg.paths.audio_wav), calls[0])
            self.assertTrue(cfg.paths.separation_metadata_json.exists())

    def test_cache_hit_skips_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            request = build_request(cfg)
            _write(request.vocals_wav, b"vocals")
            _write(request.background_wav, b"background")
            write_metadata(request)

            result = run_source_separation(
                cfg,
                runner=lambda command: self.fail("runner should not be called"),
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.cache_hit)

    def test_cache_miss_when_source_identity_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            request = build_request(cfg)
            _write(request.vocals_wav, b"vocals")
            _write(request.background_wav, b"background")
            write_metadata(request)
            _write(request.source_audio, b"changed")

            self.assertIsNone(read_cached_result(request))

    def test_cache_miss_when_model_file_changes_at_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            request = build_request(cfg)
            _write(request.vocals_wav, b"vocals")
            _write(request.background_wav, b"background")
            write_metadata(request)
            _write(Path(request.model_path), b"model-v2")

            self.assertIsNone(read_cached_result(request))

    def test_missing_model_file_fails_before_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            Path(cfg.paths.workdir, "model.ckpt").unlink()

            with self.assertRaisesRegex(SourceSeparationError, "model_path"):
                run_source_separation(
                    cfg,
                    runner=lambda command: self.fail("runner should not be called"),
                )

    def test_missing_model_file_falls_back_before_runner_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")
            Path(cfg.paths.workdir, "model.ckpt").unlink()

            result = run_source_separation(
                cfg,
                runner=lambda command: self.fail("runner should not be called"),
            )

            self.assertIsNone(result)
            self.assertIsNone(resolve_background_audio_for_merge(cfg))

    def test_failure_falls_back_only_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")

            result = run_source_separation(
                cfg,
                runner=lambda command: subprocess.CompletedProcess(
                    command, 2, "", "boom"
                ),
            )

            self.assertIsNone(result)
            self.assertIsNone(resolve_background_audio_for_merge(cfg))

    def test_stale_stems_are_not_used_after_failed_fresh_run_with_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")
            request = build_request(cfg)
            _write(request.vocals_wav, b"stale vocals")
            _write(request.background_wav, b"stale background")
            write_metadata(request)
            _write(request.source_audio, b"changed source means cache miss")

            result = run_source_separation(
                cfg,
                runner=lambda command: subprocess.CompletedProcess(
                    command, 2, "", "boom"
                ),
            )

            self.assertIsNone(result)
            self.assertFalse(request.vocals_wav.exists())
            self.assertFalse(request.background_wav.exists())
            self.assertIsNone(resolve_background_audio_for_merge(cfg))

    def test_failure_without_fallback_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none")

            with self.assertRaises(SourceSeparationError):
                run_source_separation(
                    cfg,
                    runner=lambda command: subprocess.CompletedProcess(
                        command, 2, "", "boom"
                    ),
                )

    def test_merge_uses_separated_background_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none")
            request = build_request(cfg)
            _write(request.vocals_wav, b"vocals")
            _write(request.background_wav, b"background")
            write_metadata(request)

            self.assertEqual(
                resolve_background_audio_for_merge(cfg),
                cfg.paths.separation_background_wav,
            )


class FakeSeparator:
    instances: list["FakeSeparator"] = []
    fail_load = False
    fail_separate = False
    omit_vocals = False
    omit_background = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output_dir = kwargs["output_dir"]
        self.loaded_models: list[str] = []
        self.separated_inputs: list[str] = []
        FakeSeparator.instances.append(self)

    def load_model(self, model_filename):
        if FakeSeparator.fail_load:
            raise RuntimeError("load failed")
        self.loaded_models.append(model_filename)

    def separate(self, input_audio):
        if FakeSeparator.fail_separate:
            raise RuntimeError("separation failed")
        self.separated_inputs.append(input_audio)
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        if not FakeSeparator.omit_background:
            background = out_dir / f"{Path(input_audio).stem}_(Instrumental).wav"
            _write(background, b"background")
            outputs.append(background.name)
        if not FakeSeparator.omit_vocals:
            vocals = out_dir / f"{Path(input_audio).stem}_(Vocals).wav"
            _write(vocals, b"vocals")
            outputs.append(vocals.name)
        return outputs


class AudioSeparatorProviderTests(unittest.TestCase):
    def setUp(self):
        FakeSeparator.instances.clear()
        FakeSeparator.fail_load = False
        FakeSeparator.fail_separate = False
        FakeSeparator.omit_vocals = False
        FakeSeparator.omit_background = False

    def _config(self, root: Path, *, fallback: str = "none", device: str = "cpu"):
        project = root / "project"
        project.mkdir(parents=True, exist_ok=True)
        pipeline_file = project / "sample.pipeline.yaml"
        pipeline_file.write_text(
            f"""
project_name: sample
usegpu: false
paths:
  workdir: .
  out_dir: out
  input_video: sample.mp4
source_separation:
  mode: separated_background
  provider: audio_separator
  model: model_bs_roformer_ep_368_sdr_12.9628.ckpt
  device: {device}
  output_format: wav
  sample_rate: 44100
  fallback_mode: {fallback}
  cache_enabled: true
""".strip(),
            encoding="utf-8",
        )
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        _write(cfg.paths.audio_wav, b"source")
        return cfg

    def test_provider_factory_selects_audio_separator_provider(self):
        provider = create_provider(
            "audio_separator", separator_factory=lambda **kwargs: FakeSeparator(**kwargs)
        )

        self.assertIsInstance(provider, AudioSeparatorProvider)
        self.assertEqual(FakeSeparator.instances, [])

    def test_model_is_loaded_lazily_once_for_multiple_separations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root)
            provider = AudioSeparatorProvider(separator_factory=FakeSeparator)

            first = run_source_separation(cfg, provider=provider)
            self.assertIsNotNone(first)

            second_cfg = self._config(root / "second")
            second = run_source_separation(second_cfg, provider=provider)
            self.assertIsNotNone(second)

            self.assertEqual(len(FakeSeparator.instances), 1)
            self.assertEqual(
                FakeSeparator.instances[0].loaded_models,
                ["model_bs_roformer_ep_368_sdr_12.9628.ckpt"],
            )
            self.assertEqual(len(FakeSeparator.instances[0].separated_inputs), 2)
            provider.close()

    def test_cache_hit_skips_native_model_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            request = build_request(cfg)
            _write(request.vocals_wav, b"vocals")
            _write(request.background_wav, b"background")
            write_metadata(request)

            result = run_source_separation(
                cfg,
                provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
            )

            self.assertIsNotNone(result)
            self.assertTrue(result.cache_hit)
            self.assertEqual(FakeSeparator.instances, [])

    def test_native_output_order_does_not_control_stem_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))

            result = run_source_separation(
                cfg,
                provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
            )

            self.assertIsNotNone(result)
            self.assertEqual(Path(cfg.paths.separation_vocals_wav).read_bytes(), b"vocals")
            self.assertEqual(
                Path(cfg.paths.separation_background_wav).read_bytes(), b"background"
            )

    def test_native_load_failure_honors_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")
            FakeSeparator.fail_load = True

            result = run_source_separation(
                cfg,
                provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
            )

            self.assertIsNone(result)
            self.assertIsNone(resolve_background_audio_for_merge(cfg))

    def test_native_separation_failure_is_explicit_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none")
            FakeSeparator.fail_separate = True

            with self.assertRaisesRegex(SourceSeparationError, "separation failed"):
                run_source_separation(
                    cfg,
                    provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
                )

    def test_missing_native_vocals_output_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none")
            FakeSeparator.omit_vocals = True

            with self.assertRaisesRegex(SourceSeparationError, "vocals stem"):
                run_source_separation(
                    cfg,
                    provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
                )

    def test_missing_native_background_output_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none")
            FakeSeparator.omit_background = True

            with self.assertRaisesRegex(SourceSeparationError, "instrumental/background"):
                run_source_separation(
                    cfg,
                    provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
                )

    def test_native_import_unavailable_honors_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")
            with patch(
                "dubpipeline.source_separation._load_audio_separator_factory",
                side_effect=SourceSeparationError("native dependency unavailable"),
            ):
                result = run_source_separation(cfg)

            self.assertIsNone(result)

    def test_cuda_request_without_cuda_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none", device="cuda")
            with patch("dubpipeline.source_separation._cuda_available", return_value=False):
                with self.assertRaisesRegex(SourceSeparationError, "CUDA is unavailable"):
                    run_source_separation(
                        cfg,
                        provider=AudioSeparatorProvider(separator_factory=FakeSeparator),
                    )

            self.assertEqual(FakeSeparator.instances, [])
