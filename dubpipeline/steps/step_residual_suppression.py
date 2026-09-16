from __future__ import annotations

from dubpipeline.config import PipelineConfig
from dubpipeline.residual_suppression import run_residual_suppression
from dubpipeline.utils.logging import info
from dubpipeline.utils.timing import timed


@timed("residual_suppression", log=info)
def run(cfg: PipelineConfig) -> None:
    run_residual_suppression(cfg)
