from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from uuid import uuid4

from dubpipeline.cli import (
    _build_cfg_for_input,
    _build_cli_set,
    _detect_input_source,
    _parse_steps_arg,
    _resolve_input_options,
    _validate_run_language_pair,
    build_parser,
)
from dubpipeline.config import PipelineConfig, load_pipeline_config_ex, save_pipeline_yaml


class CliTests(unittest.TestCase):
    @staticmethod
    def _case_dir(prefix: str) -> Path:
        root = Path("tests/.tmp_runtime")
        root.mkdir(parents=True, exist_ok=True)
        case = root / f"{prefix}_{uuid4().hex}"
        case.mkdir(parents=True, exist_ok=True)
        return case

    def test_help_contains_new_flags(self):
        parser = build_parser()
        run_help = parser._subparsers._group_actions[0].choices["run"].format_help()
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
            "--merge-mode",
            "--tts-gain-db",
            "--original-gain-db",
            "--ducking-amount-db",
            "--ducking-threshold-db",
            "--ducking-attack-ms",
            "--ducking-release-ms",
            "--no-loudnorm",
            "--plan",
        ]:
            self.assertIn(flag, run_help)

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
        args = parser.parse_args(
            ["run", "video.pipeline.yaml", "--delete-temp", "--keep-temp"]
        )
        with self.assertRaises(SystemExit):
            _build_cli_set(args, parser)

    def test_lang_src_accepts_english_for_asr_language(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--lang-src", "en"])
        cli_set = _build_cli_set(args, parser)
        self.assertIn("languages.src=en", cli_set)

    def test_lang_src_accepts_auto_for_asr_language(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--lang-src", "auto"])
        cli_set = _build_cli_set(args, parser)
        self.assertIn("languages.src=auto", cli_set)

    def test_lang_src_rejects_unsupported_language(self):
        parser = build_parser()
        args = parser.parse_args(["run", "video.pipeline.yaml", "--lang-src", "it"])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                _build_cli_set(args, parser)
        self.assertIn("--lang-src must be one of", stderr.getvalue())

    def test_same_language_pair_is_rejected_when_translate_enabled(self):
        parser = build_parser()
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "ru"
        cfg.languages.tgt = "ru"
        cfg.steps.translate = True

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                _validate_run_language_pair(cfg, parser, allow_legacy=False)
        self.assertIn("must be different", stderr.getvalue())

    def test_english_source_is_allowed_when_translate_enabled(self):
        parser = build_parser()
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "en"
        cfg.languages.tgt = "ru"
        cfg.steps.translate = True

        _validate_run_language_pair(cfg, parser, allow_legacy=False)

    def test_auto_source_is_rejected_when_translate_enabled(self):
        parser = build_parser()
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "auto"
        cfg.languages.tgt = "ru"
        cfg.steps.translate = True

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                _validate_run_language_pair(cfg, parser, allow_legacy=False)
        self.assertIn("cannot be used when the Translate step is enabled", stderr.getvalue())

    def test_auto_source_is_allowed_when_translate_disabled(self):
        parser = build_parser()
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "auto"
        cfg.languages.tgt = "ru"
        cfg.steps.translate = False

        _validate_run_language_pair(cfg, parser, allow_legacy=False)

    def test_same_language_pair_is_allowed_when_translate_disabled(self):
        parser = build_parser()
        cfg = PipelineConfig(project_name="sample", project_dir=Path("."))
        cfg.languages.src = "ru"
        cfg.languages.tgt = "ru"
        cfg.steps.translate = False

        _validate_run_language_pair(cfg, parser, allow_legacy=False)

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
            args = parser.parse_args(
                ["run", "video.pipeline.yaml", "--in-file", str(source)]
            )

            cli_set = _build_cli_set(args, parser)
            self.assertIn(f"paths.input_video={source.resolve()}", cli_set)

    def test_in_dir_overrides_input_video(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "videos"
            source_dir.mkdir(parents=True, exist_ok=True)
            args = parser.parse_args(
                ["run", "video.pipeline.yaml", "--in-dir", str(source_dir)]
            )

            cli_set = _build_cli_set(args, parser)
            self.assertIn(f"paths.input_video={source_dir.resolve()}", cli_set)

    def test_in_file_requires_existing_file(self):
        parser = build_parser()
        args = parser.parse_args(
            ["run", "video.pipeline.yaml", "--in-file", "missing.mp4"]
        )
        with self.assertRaises(SystemExit):
            _build_cli_set(args, parser)

    def test_in_dir_requires_existing_dir(self):
        parser = build_parser()
        args = parser.parse_args(
            ["run", "video.pipeline.yaml", "--in-dir", "missing_dir"]
        )
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

            recursive, glob_pattern = _resolve_input_options(
                cfg, parser, recursive=True, glob_pattern="*.mp4"
            )
            self.assertFalse(recursive)
            self.assertIsNone(glob_pattern)

    def test_detect_input_source(self):
        parser = build_parser()
        args_yaml = parser.parse_args(["run", "video.pipeline.yaml"])
        self.assertEqual(_detect_input_source(args_yaml), "YAML/ENV/default")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "one.mp4"
            source.write_text("x", encoding="utf-8")
            args_cli = parser.parse_args(
                ["run", "video.pipeline.yaml", "--in-file", str(source)]
            )
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

    def test_cli_merge_overrides_are_translated_to_set(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "video.pipeline.yaml",
                "--merge-mode",
                "hq_ducking",
                "--tts-gain-db",
                "1.5",
                "--original-gain-db",
                "-2",
                "--ducking-amount-db",
                "12",
                "--ducking-threshold-db",
                "-28",
                "--ducking-attack-ms",
                "8",
                "--ducking-release-ms",
                "320",
                "--no-loudnorm",
            ]
        )

        cli_set = _build_cli_set(args, parser)
        self.assertIn("audio_merge.mode=hq_ducking", cli_set)
        self.assertIn("audio_merge.tts_gain_db=1.5", cli_set)
        self.assertIn("audio_merge.original_gain_db=-2.0", cli_set)
        self.assertIn("audio_merge.ducking.amount_db=12.0", cli_set)
        self.assertIn("audio_merge.ducking.threshold_db=-28.0", cli_set)
        self.assertIn("audio_merge.ducking.attack_ms=8", cli_set)
        self.assertIn("audio_merge.ducking.release_ms=320", cli_set)
        self.assertIn("audio_merge.loudness.enabled=false", cli_set)

    def test_default_artifacts_follow_target_language(self):
        root = self._case_dir("cli_target_default_artifacts")
        proj = root / "project"
        proj.mkdir(parents=True, exist_ok=True)
        pipeline_file = proj / "video.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
languages:
  src: en
  tgt: de
paths:
  workdir: .
  out_dir: out
  input_video: source.mp4
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertTrue(str(cfg.paths.segments_tgt_file).endswith("sample.segments.de.json"))
        self.assertTrue(str(cfg.paths.segments_ru_file).endswith("sample.segments.de.json"))
        self.assertTrue(str(cfg.paths.tts_segments_dir).endswith("tts_de_segments"))
        self.assertTrue(str(cfg.paths.tts_segments_aligned_dir).endswith("tts_de_segments_aligned"))
        self.assertTrue(str(cfg.paths.final_video).endswith("sample.de.muxed.mp4"))

    def test_legacy_segments_ru_template_key_maps_to_target_file(self):
        root = self._case_dir("cli_target_legacy_template")
        proj = root / "project"
        proj.mkdir(parents=True, exist_ok=True)
        pipeline_file = proj / "video.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
languages:
  src: en
  tgt: fr
paths:
  workdir: .
  out_dir: out
  input_video: source.mp4
  templates:
    segments_ru_json: "{out_dir}/{project_name}.legacy.fr.json"
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertTrue(str(cfg.paths.segments_tgt_file).endswith("sample.legacy.fr.json"))

    def test_build_cfg_for_input_uses_target_language_artifact_names(self):
        root = self._case_dir("cli_target_artifacts")
        pipeline_file = root / "video.pipeline.yaml"
        source = root / "clip.mp4"
        source.write_text("x", encoding="utf-8")
        pipeline_file.write_text(
            """
project_name: sample
languages:
  src: en
  tgt: es
paths:
  workdir: .
  out_dir: out
  input_video: clip.mp4
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        run_cfg = _build_cfg_for_input(cfg, source)

        self.assertTrue(str(run_cfg.paths.segments_tgt_file).endswith("clip.segments.es.json"))
        self.assertTrue(str(run_cfg.paths.tts_segments_dir).endswith("tts_es_segments"))
        self.assertTrue(str(run_cfg.paths.final_video).endswith("clip.es.muxed.mp4"))

    def test_save_pipeline_yaml_persists_update_existing_settings(self):
        root = self._case_dir("cli_save_update_existing")
        pipeline_file = root / "sample.pipeline.yaml"
        values = {
            "-PROJECT-": "sample",
            "-OUT-": "out",
            "-IN-": "sample.mp4",
            "-MODES-": "Overwrite+Reorder",
            "-GPU-": True,
            "-REBUILD-": False,
            "-SRT-": False,
            "-CLEANUP-": False,
            "-MOVE_TO_DIR-": "",
            "-UPDATE_EXISTING_FILE-": True,
        }

        save_pipeline_yaml(values, pipeline_file)
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertTrue(cfg.output.update_existing_file)
        self.assertEqual(cfg.output.audio_update_mode, "overwrite_reorder")

    def test_save_pipeline_yaml_persists_use_existing_subtitles(self):
        root = self._case_dir("cli_save_existing_subtitles")
        pipeline_file = root / "sample.pipeline.yaml"
        values = {
            "-PROJECT-": "sample",
            "-OUT-": "out",
            "-IN-": "sample.mp4",
            "-MODES-": "Add",
            "-GPU-": True,
            "-USE_EXISTING_SUBTITLES-": True,
            "-REBUILD-": False,
            "-SRT-": False,
            "-CLEANUP-": False,
            "-MOVE_TO_DIR-": "",
            "-UPDATE_EXISTING_FILE-": False,
        }

        save_pipeline_yaml(values, pipeline_file)
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        self.assertTrue(cfg.use_existing_subtitles)

    def test_save_pipeline_yaml_persists_language_pair_and_pair_model_ref(self):
        root = self._case_dir("cli_save_languages")
        pipeline_file = root / "sample.pipeline.yaml"
        values = {
            "-PROJECT-": "sample",
            "-OUT-": "out",
            "-IN-": "sample.mp4",
            "-LANG_SRC-": "fr",
            "-LANG_DST-": "de",
            "-TRANSLATION_MODEL_ID-": "opus_mt",
            "-MODES-": "Add",
            "-GPU-": True,
            "-REBUILD-": False,
            "-SRT-": False,
            "-CLEANUP-": False,
            "-MOVE_TO_DIR-": "",
            "-UPDATE_EXISTING_FILE-": False,
        }

        save_pipeline_yaml(values, pipeline_file)
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertEqual(cfg.languages.src, "fr")
        self.assertEqual(cfg.languages.tgt, "de")
        self.assertEqual(cfg.translation.model_id, "opus_mt")
        self.assertEqual(cfg.translation.model_ref, "Helsinki-NLP/opus-mt-fr-de")

    def test_load_pipeline_config_sets_target_mux_metadata_defaults(self):
        root = self._case_dir("cli_mux_target_defaults")
        pipeline_file = root / "sample.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
languages:
  src: fr
  tgt: de
paths:
  workdir: .
  out_dir: out
  input_video: sample.mp4
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)

        self.assertEqual(cfg.mux.orig_lang, "fra")
        self.assertEqual(cfg.mux.target_lang, "deu")
        self.assertEqual(cfg.mux.target_track_title, "German (DubPipeline)")

    def test_save_pipeline_yaml_persists_unified_input_model(self):
        root = self._case_dir("cli_save_input_model")
        pipeline_file = root / "sample.pipeline.yaml"
        values = {
            "-PROJECT-": "sample",
            "-OUT-": "out",
            "-INPUT_MODE-": "dir",
            "-INPUT_PATH-": "tests/data",
            "-SRC_DIR-": True,
            "-MODES-": "Add",
            "-GPU-": True,
            "-REBUILD-": False,
            "-SRT-": False,
            "-CLEANUP-": False,
            "-MOVE_TO_DIR-": "",
            "-UPDATE_EXISTING_FILE-": False,
        }

        save_pipeline_yaml(values, pipeline_file)
        with pipeline_file.open("r", encoding="utf-8") as f:
            raw = f.read()

        self.assertIn("input_mode: dir", raw)
        self.assertIn("input_path: tests/data", raw)
        self.assertIn("input_video: tests/data", raw)

    def test_load_config_uses_input_path_when_paths_input_video_is_missing(self):
        root = self._case_dir("cli_load_input_path")
        pipeline_file = root / "sample.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
input_mode: file
input_path: from-input-path.mp4
paths:
  workdir: .
  out_dir: out
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        self.assertTrue(str(cfg.paths.input_video).endswith("from-input-path.mp4"))

    def test_legacy_mode_is_normalized_to_audio_update_mode(self):
        root = self._case_dir("cli_legacy_mode")
        pipeline_file = root / "video.pipeline.yaml"
        pipeline_file.write_text(
            """
project_name: sample
mode: Русская дорожка первой
paths:
  workdir: .
  out_dir: out
  input_video: source.mp4
""".strip(),
            encoding="utf-8",
        )

        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=False)
        self.assertEqual(cfg.output.audio_update_mode, "overwrite_reorder")


if __name__ == "__main__":
    unittest.main()
