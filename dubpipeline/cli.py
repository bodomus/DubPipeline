from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_pipeline_config
from dubpipeline.steps import step_whisperx, step_translate, step_tts, step_align, step_mux_audio, step_merge_py
from .steps import step_extract_audio


def main() -> None:
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
    cfg = load_pipeline_config(pipeline_path)

    if args.command == "run":
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
            step_mux_audio.run(cfg)
            step_merge_py.run(cfg)
        else:
            print("[dubpipeline] Шаг extract_audio отключён в конфиге.")


if __name__ == "__main__":
    main()
