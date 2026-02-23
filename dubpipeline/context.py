from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelinePaths:
    out_dir: str
    voice_input_wav: str
    background_wav: str
    translated_voice_wav: str
    mixed_wav: str


@dataclass
class PipelineFlags:
    external_voice_mode: bool = False
    background_mode: bool = False
    mix_executed: bool = False


@dataclass
class PipelineContext:
    paths: PipelinePaths
    flags: PipelineFlags = field(default_factory=PipelineFlags)
