from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from dubpipeline.cli import (
    _build_cli_set,
    _detect_input_source,
    _parse_steps_arg,
    _resolve_input_options,
    build_parser,
)
from dubpipeline.config import load_pipeline_config_ex


class CliTests(unittest.TestCase):
    def test_help_contains_new_flags(self):
        help_text = build_parser().format_help()
        for flag in [
            "--in-file",
            "--in-dir",
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

    def test_in_file_and_in_dir_are_mutually_exclusive(self):
        parser = build_parser()
        cases = [
            ["run", "video.pipeline.yaml", "--in-file", "a.mp4", "--in-dir", "."],
            ["run", "video.pipeline.yaml", "--in-dir", ".", "--in-file", "a.mp4"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(argv)
                self.assertIn("not allowed with argument", stderr.getvalue())

    def test_in_file_overrides_input_video(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "one.mp4"
            source.write_text("x", encoding="utf-8")
            args = parser.parse_args(["run", "video.pipeline.yaml", "--in-file", str(source)])

            cli_set = _build_cli_set(args, parser)
            self.assertIn(f"paths.input_video={source.resolve()}", cli_set)

    def test_in_dir_overrides_input_video(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "videos"
            source_dir.mkdir(parents=True, exist_ok=True)
            args = parser.parse_args(["run", "video.pipeline.yaml", "--in-dir", str(source_dir)])

            cli_set = _build_cli_set(args, parser)
            self.assertIn(f"paths.input_video={source_dir.resolve()}", cli_set)

    def test_in_file_requires_existing_file(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--in-file", "missing.mp4"])
        with self.assertRaises(SystemExit):
            _build_cli_set(args, parser)

    def test_in_dir_requires_existing_dir(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--in-dir", "missing_dir"])
        with self.assertRaises(SystemExit):
            _build_cli_set(args, parser)

    def test_recursive_and_glob_ignored_for_file_input(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir(parents=True, exist_ok=True)
            pipeline_file = project / "video.pipeline.yaml"
            src = root / "src.mp4"
            src.write_text("x", encoding="utf-8")
            pipeline_file.write_text(
                """
project_name: sample
paths:
  workdir: .
  out_dir: out
  input_video: src.mp4
""".strip(),
                encoding="utf-8",
            )
            cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

            recursive, glob_pattern = _resolve_input_options(cfg, parser, recursive=True, glob_pattern="*.mp4")
            self.assertFalse(recursive)
            self.assertIsNone(glob_pattern)

    def test_detect_input_source(self):
        parser = build_parser()
        args_yaml = parser.parse_args(["run", "video.pipeline.yaml"])
        self.assertEqual(_detect_input_source(args_yaml), "YAML/ENV/default")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "one.mp4"
            source.write_text("x", encoding="utf-8")
            args_cli = parser.parse_args(["run", "video.pipeline.yaml", "--in-file", str(source)])
            self.assertEqual(_detect_input_source(args_cli), "CLI")

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
