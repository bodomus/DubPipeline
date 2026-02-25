from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dubpipeline.input_discovery import enumerate_input_files, source_mode_disabled_map


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

            self.assertEqual({str(p.relative_to(root)) for p in files}, {"a.mp4", "nested/b.mp4"})

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
        self.assertTrue(source_mode_disabled_map(is_dir=False)["-RECURSIVE-"])
        self.assertFalse(source_mode_disabled_map(is_dir=True)["-RECURSIVE-"])


if __name__ == "__main__":
    unittest.main()
