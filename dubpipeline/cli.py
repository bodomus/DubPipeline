from __future__ import annotations
from dubpipeline.utils.logging import info, init_logger
import argparse
from pathlib import Path
import shutil
from typing import Callable
import logging
from .config import load_pipeline_config_ex
from dubpipeline.steps import step_whisperx, step_translate, step_tts, step_align, step_merge_py
from .steps import step_extract_audio


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
    return parser


def rebuild_cleanup_safe(cfg) -> None:
    """Cleanup без if-лесенок."""
    # файл
    Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    # папки
    shutil.rmtree(Path(cfg.paths.segments_path), ignore_errors=True)
    shutil.rmtree(Path(cfg.paths.segments_align_path), ignore_errors=True)


def cleanup_garbage(cfg, pipeline_path: Path) -> None:
    """
    Удаляет временные файлы ПОСЛЕ успешного завершения пайплайна.
    Оставляет финальный *.ru.muxed.mp4.
    """
    # На Windows нельзя удалить открытый лог — закрываем handlers
    logging.shutdown()

    out_dir = Path(cfg.paths.out_dir)
    out_dir = Path(out_dir).parent
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


def run_pipeline(cfg, pipeline_path: Path) -> None:
    success = False
    if cfg.rebuild:
        # если у вас уже есть rebuild_cleanup(cfg) — используйте её
        # rebuild_cleanup(cfg)
        rebuild_cleanup_safe(cfg)

    def tts_and_align(c) -> None:
        step_tts.run(c)
        step_align.run(c)

    steps: list[tuple[str, bool, Callable]] = [
        ("extract_audio", cfg.steps.extract_audio, step_extract_audio.run),
        ("asr_whisperx", cfg.steps.asr_whisperx, step_whisperx.run),
        ("translate", cfg.steps.translate, step_translate.run),
        ("tts", cfg.steps.tts, tts_and_align),
        ("merge", cfg.steps.merge, step_merge_py.run),
    ]

    for name, enabled, fn in steps:
        if not enabled:
            info(f"[dubpipeline] Шаг {name} отключён в конфиге.")
            continue
        fn(cfg)

    if cfg.delete_srt:
        Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    success = True
    if success and getattr(cfg, "cleanup", False):
        cleanup_garbage(cfg, pipeline_path)


def rebuild_cleanup(cfg):
    # файл
    Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    # папки
    shutil.rmtree(Path(cfg.paths.segments_path), ignore_errors=True)
    shutil.rmtree(Path(cfg.paths.segments_align_path), ignore_errors=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline_path = Path(args.pipeline_file).expanduser().resolve()
    cfg = load_pipeline_config_ex(pipeline_path)

    log_path = Path(cfg.paths.audio_wav).parent / f"{cfg.project_name}.log"
    init_logger(log_path)

    commands: dict[str, Callable] = {
        "run": lambda c: run_pipeline(c, pipeline_path),
    }
    commands[args.command](cfg)


if __name__ == "__main__":
    main()
