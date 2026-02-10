from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dubpipeline.cli import _build_cli_set, _parse_steps_arg, build_parser
from dubpipeline.config import load_pipeline_config_ex


class CliTests(unittest.TestCase):
    def test_help_contains_new_flags(self):
        help_text = build_parser().format_help()
        for flag in [
            "--recursive",
            "--glob",
            "--out",
            "--lang-src",
            "--lang-dst",
            "--steps",
            "--usegpu",
            "--cpu",
            "--rebuild",
            "--delete-temp",
            "--keep-temp",
            "--plan",
        ]:
            self.assertIn(flag, help_text)

    def test_steps_patch_and_list(self):
        parser = build_parser()
        patch = _parse_steps_arg("+asr,-tts", parser)
        self.assertEqual(patch["asr"], True)
        self.assertEqual(patch["tts"], False)

        listed = _parse_steps_arg("asr,translate,tts,merge", parser)
        self.assertEqual(listed["extract_audio"], False)
        self.assertEqual(listed["asr"], True)

    def test_steps_unknown_is_error(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            _parse_steps_arg("+unknown", parser)

    def test_conflicting_device_flags_are_error(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--usegpu", "--cpu"])
        with self.assertRaises(SystemExit):
            _build_cli_set(args, parser)

    def test_conflicting_temp_flags_are_error(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--delete-temp", "--keep-temp"])
        with self.assertRaises(SystemExit):
            _build_cli_set(args, parser)

    def test_cli_overrides_yaml_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = root / "project"
            proj.mkdir(parents=True, exist_ok=True)
            pipeline_file = proj / "video.pipeline.yaml"
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out_from_yaml
  input_video: source.mp4
""".strip(),
                encoding="utf-8",
            )

            cfg = load_pipeline_config_ex(
                pipeline_file,
                cli_set=["paths.out_dir=out_from_cli"],
                create_dirs=False,
            )
            self.assertTrue(str(cfg.paths.out_dir).endswith("out_from_cli"))


if __name__ == "__main__":
    unittest.main()
