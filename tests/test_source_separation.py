from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from dubpipeline.cli import _build_cfg_for_input
from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.source_separation import (
    SourceSeparationError,
    build_request,
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
  model_path: C:/models/BS-Roformer-InstVoc2.ckpt
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
    def _config(self, root: Path, *, fallback: str = "none"):
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
            _write(cfg.paths.separation_background_wav, b"background")

            self.assertEqual(
                resolve_background_audio_for_merge(cfg),
                cfg.paths.separation_background_wav,
            )
