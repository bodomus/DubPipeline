from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.atomic_replace import AtomicFileReplacer
from dubpipeline.utils.audio_process import run_ffmpeg
from dubpipeline.utils.logging import debug, error, info
from dubpipeline.utils.quote_pretty_run import norm_arg


_FORBIDDEN_VIDEO_TOKENS = {
    "-vf",
    "-filter:v",
    "scale=",
    "fps=",
    "setsar=",
    "setdar=",
    "libx264",
    "libx265",
    "hevc",
}


def _ffprobe_json(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout or "{}")


def _first_stream(info: dict[str, Any], codec_type: str) -> dict[str, Any] | None:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _select_original_track_spec(info: dict[str, Any], original_track: str) -> str:
    if original_track == "auto":
        return "0:a:0"
    if original_track.isdigit():
        return f"0:a:{int(original_track)}"
    if original_track.startswith("index:"):
        return f"0:a:{int(original_track.split(':', 1)[1])}"
    if original_track.startswith("lang_tag:"):
        wanted = original_track.split(":", 1)[1].lower()
        idx = 0
        for stream in info.get("streams", []):
            if stream.get("codec_type") != "audio":
                continue
            tags = stream.get("tags") or {}
            lang = str(tags.get("language", "")).lower()
            if lang == wanted:
                return f"0:a:{idx}"
            idx += 1
    raise ValueError(f"Unsupported original_track='{original_track}'")


def _build_audio_filter(cfg: PipelineConfig, input_audio_spec: str) -> tuple[str, str]:
    m = cfg.audio_merge
    d = m.ducking
    loudness_enabled = bool(m.loudness.enabled)

    if d.enabled:
        comp = (
            f"[orig][tts]sidechaincompress="
            f"threshold={d.threshold_db}dB:ratio={d.ratio}:attack={d.attack_ms}:"
            f"release={d.release_ms}:makeup=0:knee={d.knee_db}dB[orig_ducked];"
        )
    else:
        comp = "[orig]anull[orig_ducked];"

    loud = ""
    out_label = "mix"
    if loudness_enabled:
        loud = f"[mix]loudnorm=I={m.loudness.target_i}:TP={m.loudness.true_peak}[outa]"
        out_label = "outa"

    graph = (
        f"[{input_audio_spec}]volume={m.original_gain_db}dB[orig];"
        f"[1:a]volume={m.tts_gain_db}dB[tts];"
        f"{comp}"
        f"[orig_ducked][tts]amix=inputs=2:weights=1 1:normalize=0[mix]"
    )
    if loud:
        graph = f"{graph};{loud}"
    return graph, out_label


def _resolve_sample_rate(cfg: PipelineConfig, video_info: dict[str, Any], ffmpeg_args: list[str]) -> None:
    sample_rate = str(cfg.audio_merge.audio_out.sample_rate)
    if sample_rate != "auto":
        ffmpeg_args += ["-ar", sample_rate]
        return

    src_spec = _select_original_track_spec(video_info, cfg.audio_merge.original_track)
    src_index = int(src_spec.split(":")[-1])
    audio_index = 0
    for stream in video_info.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        if audio_index == src_index:
            rate = stream.get("sample_rate")
            if rate:
                ffmpeg_args += ["-ar", str(rate)]
            return
        audio_index += 1


def _assert_video_copy_only(cmd: list[str]) -> None:
    if "-c:v" not in cmd or "copy" not in cmd:
        error("[merge_hq] Video re-encode detected in command")
        raise ValueError("HQ ducking must use -c:v copy")
    joined = " ".join(cmd)
    for token in _FORBIDDEN_VIDEO_TOKENS:
        if token in joined:
            error(f"[merge_hq] Forbidden video token found in ffmpeg cmd: {token}")
            raise ValueError(f"Forbidden video token in HQ ducking command: {token}")
    if " -r " in f" {joined} ":
        error("[merge_hq] Forbidden video fps override (-r) detected")
        raise ValueError("HQ ducking command contains forbidden -r")


def build_hq_ducking_ffmpeg_cmd(cfg: PipelineConfig, video: Path, tts_audio: Path, output: Path) -> list[str]:
    video_info = _ffprobe_json(video)
    input_audio_spec = _select_original_track_spec(video_info, cfg.audio_merge.original_track)
    filter_graph, out_label = _build_audio_filter(cfg, input_audio_spec)

    audio_codec = cfg.audio_merge.audio_out.codec
    audio_bitrate = f"{int(cfg.audio_merge.audio_out.bitrate_kbps)}k"
    cmd = [
        cfg.ffmpeg.bin,
        "-y",
        "-i",
        norm_arg(str(video)),
        "-i",
        norm_arg(str(tts_audio)),
        "-filter_complex",
        filter_graph,
        "-map",
        "0:v:0",
        "-map",
        f"[{out_label}]",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
    ]
    _resolve_sample_rate(cfg, video_info, cmd)
    cmd += ["-movflags", "+faststart", norm_arg(str(output))]
    _assert_video_copy_only(cmd)
    return cmd


def run(cfg: PipelineConfig) -> None:
    video = Path(cfg.paths.input_video)
    tts_audio = Path(cfg.paths.audio_wav)
    output = Path(video if cfg.output.update_existing_file else cfg.paths.final_video)

    if not video.exists():
        raise SystemExit(f"Video file not found: {video}")
    if not tts_audio.exists():
        raise SystemExit(f"TTS WAV file not found: {tts_audio}")

    if not cfg.audio_merge.video.copy_stream:
        raise ValueError("audio_merge.video.copy_stream must be true for hq_ducking")

    video_info = _ffprobe_json(video)
    v_stream = _first_stream(video_info, "video")
    if not v_stream:
        raise ValueError("Input video has no video stream")

    info(
        "[merge_hq] detected video stream: "
        f"codec={v_stream.get('codec_name')} size={v_stream.get('width')}x{v_stream.get('height')} "
        f"fps={v_stream.get('avg_frame_rate')}"
    )
    info("[merge_hq] policy: video copy stream = true")

    if cfg.output.update_existing_file:
        replacer = AtomicFileReplacer()
        temp_out = replacer.make_temp_path(video)
        cmd_with_out = build_hq_ducking_ffmpeg_cmd(cfg, video, tts_audio, temp_out)
        debug(f"[merge_hq] ffmpeg cmd: {' '.join(cmd_with_out)}")
        try:
            run_ffmpeg(cmd_with_out)
            replacer.replace_with_temp(video, temp_out, keep_backup=False)
        except Exception:
            replacer.cleanup_temp(temp_out)
            raise
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd_with_out = build_hq_ducking_ffmpeg_cmd(cfg, video, tts_audio, output)
    debug(f"[merge_hq] ffmpeg cmd: {' '.join(cmd_with_out)}")
    run_ffmpeg(cmd_with_out)
