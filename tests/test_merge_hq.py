from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.steps.step_merge_hq import _assert_video_copy_only, build_hq_ducking_ffmpeg_cmd


def _require_ffmpeg(testcase: unittest.TestCase) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        testcase.skipTest("ffmpeg/ffprobe are not available in test environment")


class MergeHqUnitTests(unittest.TestCase):
    def test_hq_command_has_video_copy_and_forbidden_tokens_absent(self):
        _require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "in.mp4"
            tts = root / "tts.wav"
            out = root / "out.mp4"
            pipeline_file = root / "video.pipeline.yaml"
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: in.mp4
audio_merge:
  mode: hq_ducking
  video:
    copy_stream: true
""".strip(),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=128x72:rate=30",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=44100",
                    "-t",
                    "0.5",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=880:sample_rate=44100",
                    "-t",
                    "0.5",
                    str(tts),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            cfg = load_pipeline_config_ex(
                pipeline_file,
                cli_set=[
                    "paths.audio_wav=tts.wav",
                    "paths.final_video=out.mp4",
                    "audio_merge.loudness.enabled=false",
                ],
                create_dirs=False,
            )
            cmd = build_hq_ducking_ffmpeg_cmd(cfg, video, tts, out)
            joined = " ".join(cmd)
            self.assertIn("-c:v", cmd)
            self.assertIn("copy", cmd)
            self.assertIn("-map", cmd)
            self.assertIn("0:v:0", cmd)
            for token in ["-vf", "-filter:v", "scale=", "fps=", " -r ", "libx264"]:
                self.assertNotIn(token, joined)

    def test_video_copy_validator_rejects_video_encode(self):
        with self.assertRaises(ValueError):
            _assert_video_copy_only(["ffmpeg", "-i", "in.mp4", "-c:v", "libx264", "out.mp4"])

    def test_hq_ducking_requires_video_copy_stream_true(self):
        root = Path(tempfile.mkdtemp())
        pipeline_file = root / "video.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: in.mp4
audio_merge:
  mode: hq_ducking
  video:
    copy_stream: false
""".strip(),
            encoding="utf-8",
        )
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        self.assertFalse(cfg.audio_merge.video.copy_stream)


class MergeHqIntegrationTests(unittest.TestCase):
    def test_hq_merge_keeps_video_stream_properties(self):
        _require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "in_4k.mp4"
            tts = root / "tts.wav"
            out = root / "merged.mp4"
            pipeline_file = root / "video.pipeline.yaml"
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: in_4k.mp4
  templates:
    audio_wav: "{workdir}/tts.wav"
    final_video: "{workdir}/merged.mp4"
audio_merge:
  mode: hq_ducking
  loudness:
    enabled: false
""".strip(),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=3840x2160:rate=30",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=330:sample_rate=44100",
                    "-t",
                    "1.0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(video),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=900:sample_rate=44100",
                    "-t",
                    "1.0",
                    str(tts),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
            from dubpipeline.steps import step_merge_hq

            step_merge_hq.run(cfg)
            self.assertTrue(out.exists())
            self.assertFalse((root / "in_4k.mp4.tmp").exists())

            probe_in = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,avg_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=0",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            probe_out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,avg_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=0",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(probe_in.strip(), probe_out.strip())

            audio_probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,sample_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=0",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("codec_name=aac", audio_probe)
            self.assertIn("sample_rate=44100", audio_probe)


if __name__ == "__main__":
    unittest.main()
