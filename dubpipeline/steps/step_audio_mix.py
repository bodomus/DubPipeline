from __future__ import annotations

import shutil
from pathlib import Path

from dubpipeline.context import PipelineContext
from dubpipeline.utils.audio_process import run_ffmpeg
from dubpipeline.utils.logging import info


class AudioMixStep:
    def run(self, *, translated_voice: Path, background_track: Path, ctx: PipelineContext) -> None:
        if not translated_voice.exists():
            raise FileNotFoundError(f"Translated voice not found: {translated_voice}")
        if not background_track.exists():
            raise FileNotFoundError(f"Background audio not found: {background_track}")

        background_wav_path = Path(ctx.paths.background_wav)
        background_wav_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(background_track, background_wav_path)

        mixed_path = Path(ctx.paths.mixed_wav)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(background_wav_path),
            "-i",
            str(translated_voice),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[aout]",
            "-map",
            "[aout]",
            str(mixed_path),
        ]
        info(f"[mix] Running AudioMixStep -> {mixed_path}")
        run_ffmpeg(cmd)
