from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dubpipeline.runtime import (
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD,
    configure_pytorch_checkpoint_loading,
)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_pytorch_checkpoint_compatibility_is_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            value = configure_pytorch_checkpoint_loading()

            self.assertEqual(value, "1")
            self.assertEqual(os.environ.get(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD), "1")

    def test_pytorch_checkpoint_compatibility_preserves_explicit_value(self):
        with patch.dict(
            os.environ, {TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: "0"}, clear=True
        ):
            value = configure_pytorch_checkpoint_loading()

            self.assertEqual(value, "0")
            self.assertEqual(os.environ.get(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD), "0")

    def test_gui_configures_runtime_before_ml_adjacent_imports(self):
        gui_source = (
            Path(__file__).resolve().parents[1] / "dubpipeline" / "gui.py"
        ).read_text(encoding="utf-8")

        bootstrap_at = gui_source.index("configure_pytorch_checkpoint_loading()")

        for import_text in (
            "from dubpipeline.cli import",
            "from dubpipeline.config import",
            "from dubpipeline.steps.step_tts import",
        ):
            self.assertLess(bootstrap_at, gui_source.index(import_text))


if __name__ == "__main__":
    unittest.main()
