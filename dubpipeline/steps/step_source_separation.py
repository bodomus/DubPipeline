from __future__ import annotations

from dubpipeline.config import PipelineConfig
from dubpipeline.source_separation import run_source_separation
from dubpipeline.utils.logging import info
from dubpipeline.utils.timing import timed


@timed("source_separation", log=info)
def run(cfg: PipelineConfig) -> None:
    run_source_separation(cfg)
