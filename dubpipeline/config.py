from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from .yaml_parser import load_config
from rich import print
import yaml

@dataclass
class StepsConfig:
    extract_audio: bool = True
    # Заготовки под будущие шаги:
    asr_whisperx: bool = False
    translate: bool = False
    tts: bool = False
    merge: bool = False

@dataclass
class PathsConfig:
    workdir: Path #From which run app
    input_video: Path #Full qualificated path to input video
    out_dir: Path #Full qualificated path to output directory
    audio_wav: Path #Full qualificated path to output audio from video one
    segments_path: Path #Full qualificated path to output segments in English
    segments_align_path: Path #Full qualificated path to output segments in Russian
    segments_file:Path #Full qualificated path to output segments in English
    segments_ru_file: Path #Full qualificated path to output segments in Russian
    final_video:Path #Full qualificated path to output final video
    srt_file_en:Path #Full qualificated path to output srt file in English

@dataclass
class FfmpegConfig:
    sample_rate: int = 16_000
    channels: int = 1
    audio_codec: str = "pcm_s16le"

@dataclass
class PipelineConfig:
    # Общие сведения
    project_name: str

    # Директория проекта (папка, где лежит pipeline.yaml)
    project_dir: Path

    # Пути
    paths: PathsConfig

    # Настройки шагов
    steps: StepsConfig

    # Настройки ffmpeg/аудио
    ffmpeg: FfmpegConfig
    # режим добавления\вставки аудио
    mode:str
    languages:str
    usegpu:bool #usegpu

def load_pipeline_config_ex(pipeline_file: Path) -> PipelineConfig:
    project_dir = pipeline_file.parent

    config = load_config(pipeline_file)
    print("[bold green]Load config_ex success...[/bold green]")
    return apply_config(config, project_dir)

def load_pipeline_config(pipeline_file: Path) -> PipelineConfig:
    """Загрузить и провалидировать *.pipeline.yaml, вернуть PipelineConfig."""
    #sd = load_config(pipeline_file)
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Не найден файл конфига: {pipeline_file}")

    project_dir = pipeline_file.parent

    with pipeline_file.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    return apply_config(raw_cfg, project_dir)

def apply_config(raw_cfg: Dict, project_dir:str):
    use_gpu = raw_cfg.get("usegpu", "true")

    mode = raw_cfg.get("mode")
    if not mode:
        mode="add"
    project_name = raw_cfg.get("project_name")

    # --- Steps ---
    steps_dict: Dict[str, Any] = raw_cfg.get("steps", {}) or {}
    steps = StepsConfig(
        extract_audio=bool(steps_dict.get("extract_audio", True)),
        asr_whisperx=bool(steps_dict.get("asr_whisperx", False)),
        translate=bool(steps_dict.get("translate", False)),
        tts=bool(steps_dict.get("tts", False)),
        merge=bool(steps_dict.get("merge", False)),
    )

    languages_dict:Dict[str, Any] = raw_cfg.get("languages", {}) or {}
    languages: str = languages_dict.get("tgt", "ru")

    # --- Paths ---
    paths_dict: Dict[str, Any] = raw_cfg.get("paths", {}) or {}

    workdir_str: str = paths_dict.get("workdir", ".")
    workdir = (project_dir / workdir_str).resolve()

    input_video_str: Optional[str] = paths_dict.get("input_video")
    if not input_video_str:
        # По умолчанию считаем, что видео лежит рядом с yaml и называется project_name + .mp4
        input_video = (project_dir / f"{project_name}.mp4").resolve()
    else:
        input_video = (workdir / input_video_str).resolve()

    out_dir_str: str = paths_dict.get("out_dir", "out")
    out_dir = (workdir / out_dir_str).resolve()

    audio_wav_str: Optional[str] = paths_dict.get("audio_wav")
    if not audio_wav_str:
        # По умолчанию out/<project_name>.wav
        audio_wav = (out_dir / f"{project_name}.wav").resolve()
    else:
        audio_wav = (workdir / audio_wav_str).resolve()

    tts_segments_dir: Optional[str] = paths_dict.get("tts_segments_dir")
    if not tts_segments_dir:
        # По умолчанию out/<project_name>.wav
        tts_segments_dir = out_dir

    tts_segments_align_dir: Optional[str] = paths_dict.get("tts_segments_align_dir")
    if not tts_segments_align_dir:
        # По умолчанию out/<project_name>.wav
        tts_segments_align_dir = out_dir


    final_video_str: Optional[str] = paths_dict.get("final_video")

    srt_file_en: Optional[str] = paths_dict.get("srt_file_en")
    if not srt_file_en:
        # По умолчанию считаем,
        srt_file_en = (project_dir / f"{project_name}.srt").resolve()
    else:
        srt_file_en = (workdir / srt_file_en).resolve()

    paths = PathsConfig(
        workdir=workdir,
        input_video=input_video,
        out_dir=out_dir,
        audio_wav=audio_wav,
        segments_path=tts_segments_dir,
        segments_align_path=tts_segments_align_dir,
        segments_file=tts_segments_dir,
        segments_ru_file=tts_segments_dir,
        final_video=final_video_str,
        srt_file_en = srt_file_en,
    )

    # --- FFMPEG ---
    ffmpeg_dict: Dict[str, Any] = raw_cfg.get("ffmpeg", {}) or {}
    ffmpeg = FfmpegConfig(
        sample_rate=int(ffmpeg_dict.get("sample_rate", 16_000)),
        channels=int(ffmpeg_dict.get("channels", 1)),
        audio_codec=str(ffmpeg_dict.get("audio_codec", "pcm_s16le")),
    )

#    params = load_config(pipeline_file)

    return PipelineConfig(
        project_name=project_name,
        project_dir=project_dir,
        paths=paths,
        steps=steps,
        ffmpeg=ffmpeg,
        mode=mode,
        usegpu=use_gpu,
        languages='ru',
    )


