from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dubpipeline.steps.step_merge_hq import (
    DuckingConfig,
    LoudnessConfig,
    MergeHQConfig,
    render_hq_mix_audio,
)
from dubpipeline.utils.audio_process import MuxMode, mux_smart


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")


def _ffprobe_json(path: Path) -> dict:
    proc = subprocess.run(
        [
            FFPROBE or "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    return json.loads(proc.stdout or "{}")


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe required")
def test_hq_mix_m4a_is_aac_48k(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    tts_wav = tmp_path / "tts.wav"
    out_audio = tmp_path / "hq_mix.m4a"

    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-metadata:s:a:0",
            "language=eng",
            str(input_video),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=2",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "44100",
            str(tts_wav),
        ]
    )

    cfg = MergeHQConfig(
        tts_gain_db=0.0,
        original_gain_db=0.0,
        ducking=DuckingConfig(
            enabled=True,
            amount_db=10.0,
            threshold_db=-30.0,
            attack_ms=10,
            release_ms=250,
            ratio=6.0,
            knee_db=6.0,
        ),
        loudness=LoudnessConfig(enabled=False, target_i=-16.0, true_peak=-1.5),
    )

    render_hq_mix_audio(
        input_video=input_video,
        tts_wav=tts_wav,
        out_audio=out_audio,
        work_dir=tmp_path,
        cfg=cfg,
        original_audio_stream_selector="lang:en",
    )

    info = _ffprobe_json(out_audio)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    assert len(audio_streams) == 1
    assert audio_streams[0].get("codec_name") == "aac"
    assert audio_streams[0].get("sample_rate") == "48000"


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="ffmpeg/ffprobe required")
def test_muxed_mp4_has_two_audio_tracks_and_languages(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    ru_audio = tmp_path / "ru.m4a"
    out_video = tmp_path / "out.mp4"

    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-metadata:s:a:0",
            "language=eng",
            str(input_video),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(ru_audio),
        ]
    )

    mux_smart(
        input_video,
        ru_audio,
        out_video,
        mode=MuxMode.ADD,
        orig_lang="eng",
        ru_lang="rus",
        ru_title="Russian (DubPipeline)",
    )

    info = _ffprobe_json(out_video)
    video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    assert len(video_streams) == 1
    assert len(audio_streams) == 2
    assert audio_streams[0].get("tags", {}).get("language") == "eng"
    assert audio_streams[1].get("tags", {}).get("language") == "rus"
