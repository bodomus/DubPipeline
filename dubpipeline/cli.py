from __future__ import annotations
from dubpipeline.utils.logging import info, step, warn, error, debug, init_logger
import argparse
from pathlib import Path
import os
import shutil

from .config import load_pipeline_config, load_pipeline_config_ex
from dubpipeline.steps import step_whisperx, step_translate, step_tts, step_align, step_mux_audio, step_merge_py
from .steps import step_extract_audio


def main() -> None:
    """
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        prog="dubpipeline",
        description="DubPipeline: локальный пайплайн дубляжа видео",
    )
    parser.add_argument(
        "command",
        choices=["run"],
        help="Команда запуска. Сейчас поддерживается только 'run'.",
    )
    parser.add_argument(
        "pipeline_file",
        help="Путь к файлу *.pipeline.yaml",
    )

    args = parser.parse_args()

    pipeline_path = Path(args.pipeline_file).expanduser().resolve()
    cfg = load_pipeline_config_ex(pipeline_path)
    # Например: out/<project_name>.log
    log_path = Path(cfg.paths.audio_wav).parent / f"{cfg.project_name}.log"
    init_logger(log_path)

    if args.command == "run":
        if cfg.rebuild:
            if os.path.exists(cfg.paths.srt_file_en):
                os.remove(cfg.paths.srt_file_en)
            if os.path.exists(Path(cfg.paths.segments_path)):
                shutil.rmtree(Path(cfg.paths.segments_path))
            if os.path.exists(Path(cfg.paths.segments_align_path)):
                shutil.rmtree(Path(cfg.paths.segments_align_path))

        # Пока реализован только шаг extract_audio.
        if cfg.steps.extract_audio:
            step_extract_audio.run(cfg)
        if cfg.steps.asr_whisperx:
            step_whisperx.run(cfg)
        if cfg.steps.translate:
            step_translate.run(cfg)
        if cfg.steps.tts:
            step_tts.run(cfg)
            step_align.run(cfg)
        if cfg.steps.merge:
            step_merge_py.run(cfg)
        else:
            info("[dubpipeline] Шаг extract_audio отключён в конфиге.")
        if cfg.deleteSRT and os.path.exists(Path(cfg.paths.srt_file_en)):
            os.remove(cfg.paths.srt_file_en)

if __name__ == "__main__":
    main()
