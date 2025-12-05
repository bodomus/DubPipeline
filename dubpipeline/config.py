from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
#from yaml_parser import load_config

import yaml

from dubpipeline.load_config import load_config


#from dubpipeline.load_config import load_config


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
    workdir: Path
    input_video: Path
    out_dir: Path
    audio_wav: Path
    segments_path: Path
    segments_ru_path: Path
    words_path:Path
    segments_file:Path
    segments_ru_file: Path
    final_video:Path


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



def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_pipeline_config(pipeline_file: Path) -> PipelineConfig:
    """Загрузить и провалидировать *.pipeline.yaml, вернуть PipelineConfig."""
    #sd = load_config(pipeline_file)
    if not pipeline_file.exists():
        raise FileNotFoundError(f"Не найден файл конфига: {pipeline_file}")

    project_dir = pipeline_file.parent

    with pipeline_file.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    mode = raw_cfg.get("mode")
    if not mode:
        mode="add"
    project_name = raw_cfg.get("project_name")
    if not project_name:
        # Если явно не задано, возьмём имя файла без суффикса
        project_name = pipeline_file.stem

    # --- Steps ---
    steps_dict: Dict[str, Any] = raw_cfg.get("steps", {}) or {}
    steps = StepsConfig(
        extract_audio=bool(steps_dict.get("extract_audio", True)),
        asr_whisperx=bool(steps_dict.get("asr_whisperx", False)),
        translate=bool(steps_dict.get("translate", False)),
        tts=bool(steps_dict.get("tts", False)),
        merge=bool(steps_dict.get("merge", False)),
    )

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

    final_video_str: Optional[str] = paths_dict.get("final_video")

    paths = PathsConfig(
        workdir=workdir,
        input_video=input_video,
        out_dir=out_dir,
        audio_wav=audio_wav,
        segments_path=out_dir,
        segments_ru_path=out_dir,
        words_path=out_dir,
        segments_file=out_dir,
        segments_ru_file=out_dir,
        final_video=final_video_str,
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
    )


