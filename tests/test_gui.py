from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from dubpipeline.cli import _build_cfg_for_input, _discover_input_files
from dubpipeline.config import save_pipeline_yaml
from dubpipeline.config import load_pipeline_config_ex
from dubpipeline.input_mode import resolve_saved_input_state, validate_input_path
from dubpipeline.steps import step_extract_audio
from dubpipeline.input_discovery import enumerate_input_files, source_mode_disabled_map


FFMPEG = shutil.which("ffmpeg")
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"


class GuiInputEnumerationTests(unittest.TestCase):
    def test_non_recursive_returns_only_top_level_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "b.mp4").write_text("x", encoding="utf-8")

            files = enumerate_input_files(root, recursive=False, allowed_exts={".mp4"})

            self.assertEqual([p.name for p in files], ["a.mp4"])

    def test_recursive_returns_files_from_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "b.mp4").write_text("x", encoding="utf-8")

            files = enumerate_input_files(root, recursive=True, allowed_exts={".mp4"})

            self.assertEqual({p.relative_to(root).as_posix() for p in files}, {"a.mp4", "nested/b.mp4"})

    def test_filter_is_preserved_for_recursive_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "b.txt").write_text("x", encoding="utf-8")

            files = enumerate_input_files(root, recursive=True, allowed_exts={".mp4"})

            self.assertEqual([str(p.relative_to(root)) for p in files], ["a.mp4"])

    def test_empty_folder_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = enumerate_input_files(root, recursive=True, allowed_exts={".mp4"})
            self.assertEqual(files, [])

    def test_recursive_checkbox_state_depends_on_input_mode(self):
        file_mode = source_mode_disabled_map(is_dir=False)
        dir_mode = source_mode_disabled_map(is_dir=True)

        self.assertFalse(file_mode["-INPUT_PATH-"])
        self.assertFalse(file_mode["-BROWSE_INPUT-"])
        self.assertTrue(file_mode["-RECURSIVE-"])

        self.assertFalse(dir_mode["-INPUT_PATH-"])
        self.assertFalse(dir_mode["-BROWSE_INPUT-"])
        self.assertFalse(dir_mode["-RECURSIVE-"])


class GuiUnifiedInputTests(unittest.TestCase):
    @staticmethod
    def _case_dir(prefix: str) -> Path:
        root = REPO_ROOT / "tests" / ".tmp_runtime"
        root.mkdir(parents=True, exist_ok=True)
        case = root / f"{prefix}_{uuid4().hex}"
        case.mkdir(parents=True, exist_ok=True)
        return case

    @staticmethod
    def _run_extract_audio(pipeline_file: Path) -> list[Path]:
        cfg = load_pipeline_config_ex(pipeline_file, create_dirs=True)
        files = _discover_input_files(cfg, recursive=False, glob_pattern="*")
        for src in files:
            run_cfg = _build_cfg_for_input(cfg, src)
            step_extract_audio.run(run_cfg)
        return files

    @unittest.skipUnless(FFMPEG, "ffmpeg is required")
    def test_single_file_mode_creates_output_file(self):
        case = self._case_dir("gui_input_file_mode")
        out_dir = case / "out"
        source_video = TEST_DATA_DIR / "dub_1.mp4"
        pipeline_file = case / "single.pipeline.yaml"

        save_pipeline_yaml(
            {
                "-PROJECT-": "single",
                "-OUT-": str(out_dir),
                "-INPUT_MODE-": "file",
                "-INPUT_PATH-": str(source_video),
                "-SRC_DIR-": False,
                "-MODES-": "Add",
                "-GPU-": False,
                "-REBUILD-": False,
                "-SRT-": False,
                "-CLEANUP-": False,
                "-MOVE_TO_DIR-": "",
                "-UPDATE_EXISTING_FILE-": False,
            },
            pipeline_file,
        )

        files = self._run_extract_audio(pipeline_file)
        self.assertEqual(len(files), 1)
        self.assertTrue((out_dir / "dub_1.wav").exists())

    @unittest.skipUnless(FFMPEG, "ffmpeg is required")
    def test_folder_mode_creates_at_least_one_output_file(self):
        case = self._case_dir("gui_input_dir_mode")
        out_dir = case / "out"
        pipeline_file = case / "folder.pipeline.yaml"

        save_pipeline_yaml(
            {
                "-PROJECT-": "folder",
                "-OUT-": str(out_dir),
                "-INPUT_MODE-": "dir",
                "-INPUT_PATH-": str(TEST_DATA_DIR),
                "-SRC_DIR-": True,
                "-MODES-": "Add",
                "-GPU-": False,
                "-REBUILD-": False,
                "-SRT-": False,
                "-CLEANUP-": False,
                "-MOVE_TO_DIR-": "",
                "-UPDATE_EXISTING_FILE-": False,
            },
            pipeline_file,
        )

        files = self._run_extract_audio(pipeline_file)
        self.assertGreaterEqual(len(files), 1)
        self.assertGreaterEqual(len(list(out_dir.glob("*.wav"))), 1)

    def test_validation_rejects_folder_in_file_mode(self):
        out_dir = self._case_dir("gui_invalid_file_mode") / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, msg = validate_input_path(str(TEST_DATA_DIR), is_dir_mode=False)
        self.assertFalse(ok)
        self.assertIn("Один файл", msg)
        self.assertFalse(any(out_dir.glob("*")))

    def test_validation_rejects_file_in_folder_mode(self):
        source_video = TEST_DATA_DIR / "dub_1.mp4"
        out_dir = self._case_dir("gui_invalid_dir_mode") / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        ok, msg = validate_input_path(str(source_video), is_dir_mode=True)
        self.assertFalse(ok)
        self.assertIn("Папка", msg)
        self.assertFalse(any(out_dir.glob("*")))

    def test_legacy_input_fields_are_mapped_for_ui_mode(self):
        mode_file, path_file = resolve_saved_input_state(
            {
                "input_video": "legacy-file.mp4",
                "input_mode": "file",
            }
        )
        self.assertEqual(mode_file, "file")
        self.assertEqual(path_file, "legacy-file.mp4")

        mode_dir, path_dir = resolve_saved_input_state(
            {
                "input_video": "legacy-file.mp4",
                "input_dir": "legacy-dir",
                "input_mode": "dir",
            }
        )
        self.assertEqual(mode_dir, "dir")
        self.assertEqual(path_dir, "legacy-dir")


if __name__ == "__main__":
    unittest.main()
