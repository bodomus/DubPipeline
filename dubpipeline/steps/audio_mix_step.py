from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.logging import error, info


def build_ffmpeg_command(
    noise_wav: Path,
    translated_voice_wav: Path,
    mixed_tmp: Path,
    *,
    tts_gain_db: float,
    bg_gain_db: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    filter_complex = (
        f"[0:a]volume={bg_gain_db}dB[bg];"
        f"[1:a]volume={tts_gain_db}dB[tts];"
        "[bg][tts]amix=inputs=2:normalize=0[m]"
    )
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(noise_wav),
        "-i",
        str(translated_voice_wav),
        "-filter_complex",
        filter_complex,
        "-map",
        "[m]",
        "-c:a",
        "pcm_s16le",
        str(mixed_tmp),
    ]


def run(cfg: PipelineConfig) -> None:
    out_dir = Path(cfg.paths.out_dir)
    translated_voice = Path(getattr(cfg.paths, "translated_voice_wav", out_dir / "translated_voice.wav"))
    background = Path(getattr(cfg.paths, "background_wav", out_dir / "noise.wav"))
    mixed_wav = Path(getattr(cfg.paths, "mixed_wav", out_dir / "mixed.wav"))
    mixed_tmp = mixed_wav.with_name("mixed.tmp.wav")

    if not translated_voice.exists():
        raise FileNotFoundError(f"Translated voice file not found: {translated_voice}")
    if not background.exists():
        raise FileNotFoundError(f"Background file not found: {background}")

    merge_cfg = getattr(cfg, "audio_merge", None)
    tts_gain_db = float(getattr(merge_cfg, "tts_gain_db", 0.0))
    bg_gain_db = float(getattr(merge_cfg, "bg_gain_db", 0.0))
    ffmpeg_bin = str(getattr(getattr(cfg, "ffmpeg", None), "bin", "ffmpeg"))

    mixed_wav.parent.mkdir(parents=True, exist_ok=True)
    mixed_tmp.unlink(missing_ok=True)

    cmd = build_ffmpeg_command(
        background,
        translated_voice,
        mixed_tmp,
        tts_gain_db=tts_gain_db,
        bg_gain_db=bg_gain_db,
        ffmpeg_bin=ffmpeg_bin,
    )
    info(f"[audio_mix] cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            mixed_tmp.unlink(missing_ok=True)
            error(proc.stderr)
            raise RuntimeError(f"Audio mix step failed with code {proc.returncode}")
        if not mixed_tmp.exists():
            raise RuntimeError(f"Audio mix temp output was not created: {mixed_tmp}")
        os.replace(mixed_tmp, mixed_wav)
        info(f"[audio_mix] mixed track saved: {mixed_wav}")
    except Exception:
        mixed_tmp.unlink(missing_ok=True)
        raise
