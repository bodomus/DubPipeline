from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from typing import Callable
import logging

import torch

from dubpipeline.steps import step_whisperx, step_tts, step_align, step_merge_py, step_translate
from .steps import step_extract_audio
from .config import load_pipeline_config_ex
from dubpipeline.utils.logging import info, init_logger
from dubpipeline.utils.timing import timed_run, timed_block
from dubpipeline.consts import Const


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Точечный override на этот запуск. Пример: --set tts.max_ru_chars=260",
    )
    return parser


def rebuild_cleanup_safe(cfg) -> None:
    """Cleanup без if-лесенок."""
    # файл
    Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    # папки
    shutil.rmtree(Path(cfg.paths.tts_segments_dir), ignore_errors=True)
    shutil.rmtree(Path(cfg.paths.tts_segments_aligned_dir), ignore_errors=True)


def cleanup_garbage(cfg, pipeline_path: Path) -> None:
    """
    Удаляет временные файлы ПОСЛЕ успешного завершения пайплайна.
    Оставляет финальный *.ru.muxed.mp4.
    """
    # На Windows нельзя удалить открытый лог — закрываем handlers
    logging.shutdown()

    out_dir = Path(cfg.paths.out_dir)
    if out_dir.exists():
        # wav/json/log по префиксу проекта
        patterns = [
            f"{cfg.project_name}*.wav",
            f"{cfg.project_name}*.json",
            f"{cfg.project_name}*.log",
        ]
        if cfg.delete_srt:
            patterns.append(f"{cfg.project_name}*.srt")  # или точное имя файла(ов)

        for pat in patterns:
            for f in out_dir.glob(pat):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
                except PermissionError:
                    pass

        # pipeline.yaml (в папке проекта)
        try:
            (out_dir / f"{cfg.project_name}.pipeline.yaml").unlink(missing_ok=True)
        except Exception:
            pass

    # конкретный pipeline_path, который передали в CLI
    try:
        pipeline_path.unlink(missing_ok=True)
    except Exception:
        pass

    # сегменты целиком
    shutil.rmtree(out_dir / "segments", ignore_errors=True)

# важно: info должна быть уже доступна в момент объявления функции
# (декоратор выполняется при импорте модуля)
@timed_run(log=info, run_name="RUN", top_n=50)
def run_pipeline(cfg, pipeline_path: Path) -> None:
    # Гарантия инициализации статического репозитория конфига (на случай вызова run_pipeline не из CLI main)
    Const.bind(cfg)
    device = cfg.device  # уже вычисляется property
    compute_type = cfg.compute_type

    log_run_header(
        info, cfg,
        device=device,
        compute_type=compute_type,
        asr_model=cfg.whisperx.model_name,
        batch_size=cfg.whisperx.batch_size,
    )

    success = False

    if cfg.rebuild:
        with timed_block("00_rebuild_cleanup", log=info):
            rebuild_cleanup_safe(cfg)

    def tts_and_align(c) -> None:
        # хотите видеть отдельно tts и align — делаем два блока
        with timed_block("04a_tts", log=info):
            step_tts.run(c)
        with timed_block("04b_align", log=info):
            step_align.run(c)

    steps: list[tuple[str, bool, Callable]] = [
        ("01_extract_audio", cfg.steps.extract_audio, step_extract_audio.run),
        ("02_asr_whisperx", cfg.steps.asr_whisperx, step_whisperx.run),
        ("03_translate",     cfg.steps.translate,     step_translate.run),
        ("04_tts+align",     cfg.steps.tts,           tts_and_align),
        ("05_merge",         cfg.steps.merge,         step_merge_py.run),
    ]

    for name, enabled, fn in steps:
        if not enabled:
            info(f"[dubpipeline] Шаг {name} отключён в конфиге.")
            continue

        # вот это и есть “замер всех наших шагов”
        with timed_block(name, log=info):
            fn(cfg)

    if cfg.delete_srt:
        with timed_block("99_delete_srt", log=info):
            Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    success = True

    if success and getattr(cfg, "cleanup", False):
        with timed_block("99_cleanup_garbage", log=info):
            cleanup_garbage(cfg, pipeline_path)


def rebuild_cleanup(cfg):
    # файл
    Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    # папки
    shutil.rmtree(Path(cfg.paths.tts_segments_dir), ignore_errors=True)
    shutil.rmtree(Path(cfg.paths.tts_segments_aligned_dir), ignore_errors=True)


def main() -> None:

    print("DUB-20 https://bodomus.youtrack.cloud/issue/DUB-20")
    parser = build_parser()
    args = parser.parse_args()

    pipeline_path = Path(args.pipeline_file).expanduser().resolve()
    cfg = load_pipeline_config_ex(pipeline_path, cli_set=args.set)
    Const.bind(cfg)

    log_path = Path(cfg.paths.out_dir) / f"{cfg.project_name}.log"
    init_logger(log_path)

    commands: dict[str, Callable] = {
        "run": lambda c: run_pipeline(c, pipeline_path),
    }
    commands[args.command](cfg)


if __name__ == "__main__":
    main()
