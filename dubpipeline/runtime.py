from __future__ import annotations

import os


TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


def configure_pytorch_checkpoint_loading() -> str:
    """Configure PyTorch 2.6 checkpoint compatibility for trusted local models."""
    # PyTorch 2.6 defaults torch.load to weights_only=True. Some trusted local
    # checkpoints used by WhisperX/Coqui contain pickled config objects.
    os.environ.setdefault(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, "1")
    return os.environ[TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD]
