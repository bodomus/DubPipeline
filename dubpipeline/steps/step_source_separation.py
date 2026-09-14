from __future__ import annotations

from dubpipeline.config import PipelineConfig
from dubpipeline.source_separation import AudioBackgroundProvider, run_source_separation
from dubpipeline.utils.logging import info
from dubpipeline.utils.timing import timed


@timed("source_separation", log=info)
def run(cfg: PipelineConfig, *, provider: AudioBackgroundProvider | None = None) -> None:
    run_source_separation(cfg, provider=provider)
