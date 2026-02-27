from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.steps.step_merge_hq import (
    DuckingConfig,
    LoudnessConfig,
    MergeHQConfig,
    build_ffmpeg_command,
    build_filtergraph,
    merge_hq_config_from_pipeline,
    merge_hq_ducking,
    validate_merge_hq_config,
)


def _base_hq_cfg() -> MergeHQConfig:
    return MergeHQConfig(
        tts_gain_db=0.0,
        original_gain_db=0.0,
        ducking=DuckingConfig(
            enabled=True,
            amount_db=10.0,
            threshold_db=-30.0,
            attack_ms=10,
            release_ms=250,
            ratio=6.0,
            knee_db=6.0,
        ),
        loudness=LoudnessConfig(enabled=True, target_i=-16.0, true_peak=-1.5),
    )


def _rms(segment: np.ndarray) -> float:
    if segment.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(segment), dtype=np.float64)))


class MergeHQUnitTests(unittest.TestCase):
    def test_yaml_parse_to_merge_hq_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_file = root / "sample.pipeline.yaml"
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: input.mp4
audio_merge:
  mode: hq_ducking
  original_track: lang:en
  tts_gain_db: 1.0
  original_gain_db: -2.0
  ducking:
    enabled: true
    amount_db: 11.0
    threshold_db: -28.0
    attack_ms: 8
    release_ms: 300
    ratio: 7.0
    knee_db: 4.0
  loudness:
    enabled: false
    target_i: -18.0
    true_peak: -2.0
""".strip(),
                encoding="utf-8",
            )

            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
            merge_cfg, selector = merge_hq_config_from_pipeline(cfg)

            self.assertEqual(cfg.audio_merge.mode, "hq_ducking")
            self.assertEqual(selector, "lang:en")
            self.assertEqual(merge_cfg.tts_gain_db, 1.0)
            self.assertEqual(merge_cfg.original_gain_db, -2.0)
            self.assertEqual(merge_cfg.ducking.attack_ms, 8)
            self.assertEqual(merge_cfg.ducking.release_ms, 300)
            self.assertFalse(merge_cfg.loudness.enabled)

    def test_validation_rules(self):
        cfg = _base_hq_cfg()

        with self.assertRaises(ValueError):
            validate_merge_hq_config(
                replace(cfg, ducking=replace(cfg.ducking, attack_ms=0))
            )
        with self.assertRaises(ValueError):
            validate_merge_hq_config(
                replace(cfg, ducking=replace(cfg.ducking, release_ms=0))
            )
        with self.assertRaises(ValueError):
            validate_merge_hq_config(
                replace(cfg, ducking=replace(cfg.ducking, amount_db=30.0))
            )
        with self.assertRaises(ValueError):
            validate_merge_hq_config(
                replace(cfg, ducking=replace(cfg.ducking, threshold_db=-70.0))
            )

    def test_filtergraph_generation(self):
        cfg = _base_hq_cfg()
        graph = build_filtergraph(cfg=cfg, original_audio_spec="0:a:0")
        self.assertIn("aresample=48000", graph)
        self.assertIn("sidechaincompress=", graph)
        self.assertIn("amix=inputs=2:normalize=0", graph)
        self.assertIn("loudnorm=I=-16.000:TP=-1.500", graph)

        graph_no_loud = build_filtergraph(
            cfg=replace(cfg, loudness=replace(cfg.loudness, enabled=False)),
            original_audio_spec="0:a:0",
        )
        self.assertNotIn("loudnorm=", graph_no_loud)
        self.assertIn("[mix]anull[outa]", graph_no_loud)

    def test_ffmpeg_command_generation(self):
        cmd = build_ffmpeg_command(
            input_video=Path("in.mp4"),
            tts_wav=Path("tts.wav"),
            output_video=Path("out.mp4.tmp"),
            filtergraph="[0:a:0]volume=0dB[orig];[1:a]volume=0dB[tts];[orig][tts]amix=inputs=2:normalize=0[outa]",
        )
        self.assertIn("ffmpeg", Path(cmd[0]).name.lower())
        self.assertIn("-filter_complex", cmd)
        self.assertIn("-map", cmd)
        self.assertIn("[outa]", cmd)
        self.assertIn("-ar", cmd)
        self.assertIn("48000", cmd)
        self.assertEqual(cmd[-1], "out.mp4.tmp")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe are required")
class MergeHQIntegrationTests(unittest.TestCase):
    def _run(self, cmd: list[str]) -> None:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")

    def test_hq_ducking_reduces_original_during_tts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 48_000
            dur = 6.0
            n = int(sr * dur)
            t = np.arange(n, dtype=np.float64) / sr

            rng = np.random.default_rng(42)
            original = 0.18 * np.sin(2 * math.pi * 220.0 * t) + 0.04 * rng.standard_normal(n)
            original = np.clip(original, -0.95, 0.95).astype(np.float32)

            tts = np.zeros(n, dtype=np.float32)
            active_start = int(1.0 * sr)
            active_end = int(2.2 * sr)
            pulse_period = int(0.05 * sr)
            pulse_width = int(0.005 * sr)
            for idx in range(active_start, active_end, pulse_period):
                tts[idx : min(idx + pulse_width, active_end)] = 0.95

            original_wav = root / "original.wav"
            tts_wav = root / "tts.wav"
            input_video = root / "input.mp4"
            output_video = root / "output.mp4"
            extracted_wav = root / "output_audio.wav"

            sf.write(original_wav, original, sr, subtype="PCM_16")
            sf.write(tts_wav, tts, sr, subtype="PCM_16")

            self._run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=320x240:r=25:d=6",
                    "-i",
                    str(original_wav),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-metadata:s:a:0",
                    "language=eng",
                    str(input_video),
                ]
            )

            cfg = MergeHQConfig(
                tts_gain_db=-20.0,
                original_gain_db=0.0,
                ducking=DuckingConfig(
                    enabled=True,
                    amount_db=20.0,
                    threshold_db=-36.0,
                    attack_ms=5,
                    release_ms=250,
                    ratio=12.0,
                    knee_db=6.0,
                ),
                loudness=LoudnessConfig(enabled=False, target_i=-16.0, true_peak=-1.5),
            )

            out = merge_hq_ducking(
                input_video=input_video,
                tts_wav=tts_wav,
                out_video=output_video,
                work_dir=root,
                cfg=cfg,
                original_audio_stream_selector="lang:en",
            )
            self.assertEqual(out, output_video)
            self.assertTrue(output_video.exists())
            self.assertFalse((output_video.with_name(f"{output_video.name}.tmp")).exists())

            self._run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(output_video),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(sr),
                    str(extracted_wav),
                ]
            )

            out_audio, out_sr = sf.read(extracted_wav)
            self.assertEqual(out_sr, sr)
            if out_audio.ndim > 1:
                out_audio = out_audio.mean(axis=1)
            out_audio = np.asarray(out_audio, dtype=np.float32)

            def _slice(signal: np.ndarray, t0: float, t1: float) -> np.ndarray:
                return signal[int(t0 * sr) : int(t1 * sr)]

            rms_orig_active = _rms(_slice(original, 1.1, 2.0))
            rms_out_active = _rms(_slice(out_audio, 1.1, 2.0))
            rms_orig_inactive = _rms(_slice(original, 4.0, 5.0))
            rms_out_inactive = _rms(_slice(out_audio, 4.0, 5.0))

            active_delta_db = 20.0 * math.log10((rms_out_active + 1e-9) / (rms_orig_active + 1e-9))
            inactive_delta_db = abs(
                20.0 * math.log10((rms_out_inactive + 1e-9) / (rms_orig_inactive + 1e-9))
            )

            self.assertLessEqual(active_delta_db, -6.0)
            self.assertLessEqual(inactive_delta_db, 1.0)


if __name__ == "__main__":
    unittest.main()
