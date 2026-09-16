from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf

from dubpipeline.cli import _build_cfg_for_input
from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.residual_suppression import (
    ResidualSuppressionError,
    build_filtergraph,
    build_request,
    read_cached_result,
    resolve_background_for_merge,
    run_residual_suppression,
)
from dubpipeline.source_separation import (
    build_request as build_source_request,
    write_metadata as write_source_metadata,
)


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


class ResidualSuppressionTests(unittest.TestCase):
    def _config(
        self,
        root: Path,
        *,
        enabled: bool = True,
        fallback: str = "none",
        threshold_db: float = -35.0,
    ):
        project = root / "project"
        project.mkdir(parents=True, exist_ok=True)
        pipeline = project / "sample.pipeline.yaml"
        pipeline.write_text(
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
  device: cpu
  fallback_mode: {fallback}
  cache_enabled: true
residual_suppression:
  enabled: {str(enabled).lower()}
  threshold_db: {threshold_db}
  ratio: 12.0
  attack_ms: 5
  release_ms: 80
  max_reduction_db: 12.0
  control_gain_db: 0.0
  knee: 2.828427
  cache_enabled: true
""".strip(),
            encoding="utf-8",
        )
        return load_pipeline_config_ex(pipeline, create_dirs=False)

    def _write_cached_stems(
        self, cfg, background: bytes = b"background", vocals: bytes = b"vocals"
    ) -> None:
        cfg.paths.audio_wav.parent.mkdir(parents=True, exist_ok=True)
        cfg.paths.audio_wav.write_bytes(b"source")
        cfg.paths.separation_background_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            cfg.paths.separation_background_wav,
            np.full(800, 0.05, dtype=np.float32),
            8_000,
            subtype="PCM_16",
        )
        sf.write(
            cfg.paths.separation_vocals_wav,
            np.full(800, 0.1, dtype=np.float32),
            8_000,
            subtype="PCM_16",
        )
        write_source_metadata(build_source_request(cfg))

    def test_defaults_are_disabled_and_paths_are_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            pipeline = project / "sample.pipeline.yaml"
            pipeline.write_text(
                "project_name: sample\npaths:\n  workdir: .\n  out_dir: out\n",
                encoding="utf-8",
            )

            cfg = load_pipeline_config_ex(pipeline, create_dirs=False)

            self.assertFalse(cfg.residual_suppression.enabled)
            self.assertEqual(
                cfg.paths.separation_cleaned_background_wav.parts[-4:],
                ("out", "separation", "sample", "background.cleaned.wav"),
            )
            self.assertEqual(
                cfg.paths.residual_suppression_metadata_json.parts[-4:],
                ("out", "separation", "sample", "residual_suppression.json"),
            )

    def test_batch_input_rebuilds_residual_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            input_file = Path(tmp) / "clip.mp4"
            input_file.write_bytes(b"video")

            run_cfg = _build_cfg_for_input(cfg, input_file)

            self.assertEqual(
                run_cfg.paths.separation_cleaned_background_wav.parts[-4:],
                ("out", "separation", "clip", "background.cleaned.wav"),
            )

    def test_environment_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            pipeline = project / "sample.pipeline.yaml"
            pipeline.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
residual_suppression:
  enabled: false
  threshold_db: -35
""".strip(),
                encoding="utf-8",
            )

            with patch.dict(
                "os.environ",
                {
                    "DB_RSP__ENABLED": "true",
                    "DUBPIPELINE_RESIDUAL_SUPPRESSION_THRESHOLD_DB": "-30",
                },
                clear=False,
            ):
                cfg = load_pipeline_config_ex(
                    pipeline,
                    cli_set=["residual_suppression.threshold_db=-25"],
                    create_dirs=False,
                )

            self.assertTrue(cfg.residual_suppression.enabled)
            self.assertEqual(cfg.residual_suppression.threshold_db, -25.0)

    def test_disabled_mode_routes_raw_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), enabled=False)
            self._write_cached_stems(cfg)

            self.assertIsNone(run_residual_suppression(cfg))
            self.assertEqual(
                resolve_background_for_merge(cfg),
                cfg.paths.separation_background_wav,
            )

    def test_filtergraph_uses_vocal_sidechain_and_dry_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp)).residual_suppression

            graph = build_filtergraph(cfg)

            self.assertIn("sidechaincompress=", graph)
            self.assertIn("threshold=0.017782794", graph)
            self.assertIn("mix=0.748811357,apad[out]", graph)

    def test_cache_hit_and_tuning_change_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            self._write_cached_stems(cfg)
            calls = 0

            def runner(command):
                nonlocal calls
                calls += 1
                shutil.copyfile(cfg.paths.separation_background_wav, Path(command[-1]))
                return subprocess.CompletedProcess(command, 0, "", "")

            first = run_residual_suppression(cfg, runner=runner)
            second = run_residual_suppression(cfg, runner=runner)

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(calls, 1)

            cfg.residual_suppression.threshold_db = -30.0
            self.assertIsNone(read_cached_result(build_request(cfg)))

    def test_invalid_metadata_never_reuses_stale_cleaned_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            self._write_cached_stems(cfg)
            request = build_request(cfg)
            request.cleaned_background_wav.write_bytes(b"stale")
            request.metadata_json.write_text("{}", encoding="utf-8")

            self.assertIsNone(read_cached_result(request))

    def test_failure_removes_stale_output_and_uses_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")
            self._write_cached_stems(cfg)
            request = build_request(cfg)
            request.cleaned_background_wav.write_bytes(b"stale")

            def runner(command):
                return subprocess.CompletedProcess(command, 1, "", "failure")

            result = run_residual_suppression(cfg, runner=runner)

            self.assertIsNone(result)
            self.assertFalse(request.cleaned_background_wav.exists())
            self.assertFalse(request.metadata_json.exists())
            self.assertIsNone(resolve_background_for_merge(cfg))

    def test_missing_source_stems_use_legacy_fallback_before_audio_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="legacy_ducking")
            cleaned = cfg.paths.separation_cleaned_background_wav
            metadata = cfg.paths.residual_suppression_metadata_json
            cleaned.parent.mkdir(parents=True, exist_ok=True)
            cleaned.write_bytes(b"stale")
            metadata.write_text("{}", encoding="utf-8")

            result = run_residual_suppression(cfg)

            self.assertIsNone(result)
            self.assertFalse(cleaned.exists())
            self.assertFalse(metadata.exists())

    def test_failure_is_explicit_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp), fallback="none")
            self._write_cached_stems(cfg)

            def runner(command):
                return subprocess.CompletedProcess(command, 1, "", "failure")

            with self.assertRaisesRegex(
                ResidualSuppressionError, "FFmpeg residual suppression failed"
            ):
                run_residual_suppression(cfg, runner=runner)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class ResidualSuppressionIntegrationTests(unittest.TestCase):
    def test_active_audio_is_reduced_and_inactive_audio_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = ResidualSuppressionTests()
            cfg = helper._config(root, threshold_db=-40.0)
            sr = 48_000
            duration = 4.0
            count = int(sr * duration)
            time_axis = np.arange(count, dtype=np.float64) / sr
            background = (0.18 * np.sin(2 * math.pi * 220.0 * time_axis)).astype(
                np.float32
            )
            vocals = np.zeros(count, dtype=np.float32)
            active_start = int(1.0 * sr)
            active_end = int(2.0 * sr)
            vocals[active_start:active_end] = (
                0.9
                * np.sin(
                    2
                    * math.pi
                    * 440.0
                    * time_axis[active_start:active_end]
                )
            ).astype(np.float32)

            cfg.paths.audio_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(cfg.paths.audio_wav, background, sr, subtype="PCM_16")
            cfg.paths.separation_background_wav.parent.mkdir(
                parents=True, exist_ok=True
            )
            sf.write(
                cfg.paths.separation_background_wav,
                background,
                sr,
                subtype="PCM_16",
            )
            sf.write(
                cfg.paths.separation_vocals_wav,
                vocals,
                sr,
                subtype="PCM_16",
            )
            write_source_metadata(build_source_request(cfg))

            result = run_residual_suppression(cfg)

            self.assertIsNotNone(result)
            cleaned, cleaned_sr = sf.read(result.cleaned_background_wav)
            self.assertEqual(cleaned_sr, sr)
            cleaned = np.asarray(cleaned, dtype=np.float32)
            self.assertEqual(cleaned.shape, background.shape)

            active_original = _rms(background[int(1.2 * sr) : int(1.8 * sr)])
            active_cleaned = _rms(cleaned[int(1.2 * sr) : int(1.8 * sr)])
            inactive_original = _rms(background[int(3.0 * sr) : int(3.8 * sr)])
            inactive_cleaned = _rms(cleaned[int(3.0 * sr) : int(3.8 * sr)])
            active_delta_db = 20 * math.log10(
                (active_cleaned + 1e-9) / (active_original + 1e-9)
            )
            inactive_delta_db = 20 * math.log10(
                (inactive_cleaned + 1e-9) / (inactive_original + 1e-9)
            )

            self.assertLess(active_delta_db, -4.0)
            self.assertGreater(active_delta_db, -12.5)
            self.assertLess(abs(inactive_delta_db), 1.0)


if __name__ == "__main__":
    unittest.main()
