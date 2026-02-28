from __future__ import annotations

import argparse
import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from dubpipeline.consts import Const
from dubpipeline.utils.build_info import get_build_info
from dubpipeline.utils.logging import info, init_logger, warn
from dubpipeline.utils.output_move import OutputMover
from dubpipeline.utils.run_meta import log_run_header
from dubpipeline.utils.timing import timed_run, timed_block
from .config import PipelineConfig, load_pipeline_config_ex

STEP_ID_TO_CFG_FIELD = {
    "extract_audio": "extract_audio",
    "asr": "asr_whisperx",
    "translate": "translate",
    "tts": "tts",
    "merge": "merge",
}

STEP_ID_TO_INTERNAL = {
    "extract_audio": "01_extract_audio",
    "asr": "02_asr_whisperx",
    "translate": "03_translate",
    "tts": "04_tts+align",
    "merge": "05_merge",
}


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
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--in-file", default=None, metavar="PATH", help="Входной видеофайл.")
    input_group.add_argument("--in-dir", default=None, metavar="PATH", help="Входная директория с видеофайлами.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Точечный override на этот запуск. Пример: --set tts.max_ru_chars=260",
    )
    parser.add_argument(
        "--move-to-dir",
        default=None,
        help="Переместить выходные файлы в указанную директорию (переопределяет YAML/ENV).",
    )
    parser.add_argument("--recursive", action="store_true", help="Рекурсивный обход входной папки.")
    parser.add_argument("--glob", default=None, metavar="PATTERN", help="Фильтр файлов по glob-шаблону.")
    parser.add_argument("--out", default=None, metavar="DIR", help="Директория временных файлов (paths.out_dir).")
    parser.add_argument("--lang-src", default=None, metavar="LANG", help="Язык источника.")
    parser.add_argument("--lang-dst", default=None, metavar="LANG", help="Язык назначения.")
    parser.add_argument(
        "--steps",
        default=None,
        metavar="LIST",
        help="Шаги: patch-форма (+asr,-tts) или list-форма (asr,translate,tts,merge).",
    )
    parser.add_argument("--usegpu", action="store_true", help="Принудительно использовать GPU.")
    parser.add_argument("--cpu", action="store_true", help="Принудительно использовать CPU.")
    parser.add_argument("--rebuild", action="store_true", help="Принудительно пересоздать артефакты шагов.")
    parser.add_argument("--delete-temp", action="store_true", help="Удалять temp/work файлы по завершении.")
    parser.add_argument("--keep-temp", action="store_true", help="Не удалять temp/work файлы по завершении.")
    parser.add_argument("--merge-mode", default=None, metavar="MODE", help="Режим финального мержа (например, hq_ducking).")
    parser.add_argument("--tts-gain-db", type=float, default=None, metavar="DB", help="Усиление TTS дорожки в dB.")
    parser.add_argument("--original-gain-db", type=float, default=None, metavar="DB", help="Усиление оригинальной дорожки в dB.")
    parser.add_argument("--ducking-amount-db", type=float, default=None, metavar="DB", help="Глубина ducking в dB.")
    parser.add_argument("--ducking-threshold-db", type=float, default=None, metavar="DB", help="Порог sidechain compressor в dB.")
    parser.add_argument("--ducking-attack-ms", type=int, default=None, metavar="MS", help="Attack sidechain compressor в ms.")
    parser.add_argument("--ducking-release-ms", type=int, default=None, metavar="MS", help="Release sidechain compressor в ms.")
    parser.add_argument("--no-loudnorm", action="store_true", help="Отключить loudnorm в режиме hq_ducking.")
    parser.add_argument("--plan", action="store_true", help="Dry-run: показать план и завершить без выполнения.")
    return parser


def _parse_steps_arg(raw_steps: str, parser: argparse.ArgumentParser) -> dict[str, bool]:
    tokens = [t.strip() for t in raw_steps.split(",") if t.strip()]
    if not tokens:
        parser.error("--steps не должен быть пустым")

    is_patch_mode = all(t[0] in "+-" for t in tokens)
    if not is_patch_mode and any(t[0] in "+-" for t in tokens):
        parser.error("--steps: нельзя смешивать patch- и list-формы")

    allowed = sorted(STEP_ID_TO_CFG_FIELD.keys())

    def _validate(step_id: str) -> None:
        if step_id not in STEP_ID_TO_CFG_FIELD:
            parser.error(f"Неизвестный шаг '{step_id}' в --steps. Допустимые: {', '.join(allowed)}")

    parsed: dict[str, bool] = {}
    if is_patch_mode:
        for token in tokens:
            sign, step_id = token[0], token[1:]
            _validate(step_id)
            parsed[step_id] = sign == "+"
        return parsed

    enabled = set()
    for token in tokens:
        _validate(token)
        enabled.add(token)
    for step_id in STEP_ID_TO_CFG_FIELD:
        parsed[step_id] = step_id in enabled
    return parsed


def _build_cli_set(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.usegpu and args.cpu:
        parser.error("Нельзя одновременно указывать --usegpu и --cpu")
    if args.delete_temp and args.keep_temp:
        parser.error("Нельзя одновременно указывать --delete-temp и --keep-temp")

    cli_set = list(args.set)
    if args.move_to_dir is not None:
        cli_set.append(f"output.move_to_dir={args.move_to_dir}")
    if args.out is not None:
        cli_set.append(f"paths.out_dir={args.out}")
    if args.lang_src is not None:
        cli_set.append(f"languages.src={args.lang_src}")
    if args.lang_dst is not None:
        cli_set.append(f"languages.tgt={args.lang_dst}")
    if args.usegpu:
        cli_set.append("usegpu=true")
    if args.cpu:
        cli_set.append("usegpu=false")
    if args.rebuild:
        cli_set.append("rebuild=true")
    if args.delete_temp:
        cli_set.append("cleanup=true")
        cli_set.append("keep_temp=false")
    if args.keep_temp:
        cli_set.append("cleanup=false")
        cli_set.append("keep_temp=true")

    if args.merge_mode is not None:
        cli_set.append(f"audio_merge.mode={args.merge_mode}")
    if args.tts_gain_db is not None:
        cli_set.append(f"audio_merge.tts_gain_db={args.tts_gain_db}")
    if args.original_gain_db is not None:
        cli_set.append(f"audio_merge.original_gain_db={args.original_gain_db}")
    if args.ducking_amount_db is not None:
        cli_set.append(f"audio_merge.ducking.amount_db={args.ducking_amount_db}")
    if args.ducking_threshold_db is not None:
        cli_set.append(f"audio_merge.ducking.threshold_db={args.ducking_threshold_db}")
    if args.ducking_attack_ms is not None:
        cli_set.append(f"audio_merge.ducking.attack_ms={args.ducking_attack_ms}")
    if args.ducking_release_ms is not None:
        cli_set.append(f"audio_merge.ducking.release_ms={args.ducking_release_ms}")
    if args.no_loudnorm:
        cli_set.append("audio_merge.loudness.enabled=false")

    in_file = getattr(args, "in_file", None)
    in_dir = getattr(args, "in_dir", None)

    if in_file is not None:
        in_file_path = Path(in_file).expanduser()
        if not in_file_path.is_file():
            parser.error(f"--in-file должен указывать на существующий файл: '{in_file_path}'")
        cli_set.append(f"paths.input_video={in_file_path.resolve()}")

    if in_dir is not None:
        in_dir_path = Path(in_dir).expanduser()
        if not in_dir_path.is_dir():
            parser.error(f"--in-dir должен указывать на существующую директорию: '{in_dir_path}'")
        cli_set.append(f"paths.input_video={in_dir_path.resolve()}")

    if args.steps is not None:
        parsed_steps = _parse_steps_arg(args.steps, parser)
        patch_mode = all(tok.strip().startswith(("+", "-")) for tok in args.steps.split(",") if tok.strip())
        if patch_mode:
            for step_id, enabled in parsed_steps.items():
                field = STEP_ID_TO_CFG_FIELD[step_id]
                cli_set.append(f"steps.{field}={'true' if enabled else 'false'}")
        else:
            for step_id, enabled in parsed_steps.items():
                field = STEP_ID_TO_CFG_FIELD[step_id]
                cli_set.append(f"steps.{field}={'true' if enabled else 'false'}")

    return cli_set


def _discover_input_files(cfg: PipelineConfig, *, recursive: bool, glob_pattern: str | None) -> list[Path]:
    source = Path(cfg.paths.input_video)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        pattern = glob_pattern or "*"
        iterator = source.rglob(pattern) if recursive else source.glob(pattern)
        files = sorted([p for p in iterator if p.is_file()])
    else:
        files = []

    if source.is_file() and glob_pattern:
        files = [p for p in files if p.match(glob_pattern)]
    return files


def _format_steps(cfg: PipelineConfig) -> list[str]:
    rows: list[str] = []
    for step_id, field in STEP_ID_TO_CFG_FIELD.items():
        enabled = bool(getattr(cfg.steps, field))
        status = "enabled" if enabled else "disabled"
        rows.append(f"  - {step_id} ({STEP_ID_TO_INTERNAL[step_id]}): {status}")
    return rows


def _print_effective_summary(cfg: PipelineConfig, files: list[Path], *, plan_mode: bool) -> None:
    mode = "PLAN" if plan_mode else "RUN"
    print(f"[dubpipeline] Effective config summary ({mode})")
    print(f"  project_name: {cfg.project_name}")
    print(f"  input_video: {cfg.paths.input_video}")
    print(f"  out_dir: {cfg.paths.out_dir}")
    print(f"  lang: {cfg.languages.src} -> {cfg.languages.tgt}")
    print(f"  device: {'gpu' if cfg.usegpu else 'cpu'}")
    print(f"  rebuild: {cfg.rebuild}")
    print(f"  cleanup_temp: {cfg.cleanup}")
    print(f"  update_existing_file: {cfg.output.update_existing_file}")
    print(f"  audio_update_mode: {cfg.output.audio_update_mode}")
    print(f"  audio_merge_mode: {cfg.audio_merge.mode}")
    print("  steps:")
    for row in _format_steps(cfg):
        print(row)
    print(f"  input_files_count: {len(files)}")
    for path in files:
        print(f"    * {path}")


def _resolve_input_options(
    cfg: PipelineConfig,
    parser: argparse.ArgumentParser,
    *,
    recursive: bool,
    glob_pattern: str | None,
) -> tuple[bool, str | None]:
    input_path = Path(cfg.paths.input_video)
    recursive_enabled = recursive
    effective_glob = glob_pattern

    if input_path.is_file() and recursive:
        warn("[dubpipeline] --recursive проигнорирован: вход указан как файл.")
        recursive_enabled = False

    if input_path.is_file() and glob_pattern:
        warn("[dubpipeline] --glob проигнорирован: вход указан как файл.")
        effective_glob = None

    if input_path.is_dir() and glob_pattern is None:
        effective_glob = "*"

    if not input_path.exists():
        parser.error(f"Входной путь не найден: '{input_path}'")

    return recursive_enabled, effective_glob


def _detect_input_source(args: argparse.Namespace) -> str:
    return "CLI" if args.in_file or args.in_dir else "YAML/ENV/default"


def _build_cfg_for_input(base_cfg: PipelineConfig, input_file: Path) -> PipelineConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.project_name = input_file.stem
    cfg.paths.input_video = input_file.resolve()

    out_dir = Path(cfg.paths.out_dir)
    cfg.paths.audio_wav = out_dir / f"{cfg.project_name}.wav"
    cfg.paths.segments_file = out_dir / f"{cfg.project_name}.segments.json"
    cfg.paths.segments_ru_file = out_dir / f"{cfg.project_name}.segments.ru.json"
    cfg.paths.srt_file_en = out_dir / f"{cfg.project_name}.srt"
    cfg.paths.tts_segments_dir = out_dir / "segments" / "tts_ru_segments"
    cfg.paths.tts_segments_aligned_dir = out_dir / "segments" / "tts_ru_segments_aligned"
    cfg.paths.final_video = out_dir / f"{cfg.project_name}.ru.muxed.mp4"
    return cfg


def rebuild_cleanup_safe(cfg) -> None:
    Path(cfg.paths.srt_file_en).unlink(missing_ok=True)
    shutil.rmtree(Path(cfg.paths.tts_segments_dir), ignore_errors=True)
    shutil.rmtree(Path(cfg.paths.tts_segments_aligned_dir), ignore_errors=True)


def cleanup_garbage(cfg, pipeline_path: Path) -> None:
    logging.shutdown()

    out_dir = Path(cfg.paths.out_dir)
    if out_dir.exists():
        patterns = [
            f"{cfg.project_name}*.wav",
            f"{cfg.project_name}*.json",
            f"{cfg.project_name}*.log",
        ]
        if cfg.delete_srt:
            patterns.append(f"{cfg.project_name}*.srt")

        for pat in patterns:
            for f in out_dir.glob(pat):
                try:
                    f.unlink()
                except (FileNotFoundError, PermissionError):
                    pass

        try:
            (out_dir / f"{cfg.project_name}.pipeline.yaml").unlink(missing_ok=True)
        except Exception:
            pass

    try:
        pipeline_path.unlink(missing_ok=True)
    except Exception:
        pass

    shutil.rmtree(out_dir / "segments", ignore_errors=True)


@timed_run(log=info, run_name="RUN", top_n=50)
def run_pipeline(cfg, pipeline_path: Path) -> None:
    from dubpipeline.steps import step_align, step_merge_py, step_translate, step_tts, step_whisperx
    from .steps import step_extract_audio

    Const.bind(cfg)
    device = cfg.device
    compute_type = cfg.compute_type

    log_run_header(
        info,
        cfg,
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
        with timed_block("04a_tts", log=info):
            step_tts.run(c)
        with timed_block("04b_align", log=info):
            step_align.run(c)

    steps: list[tuple[str, bool, Callable]] = [
        ("01_extract_audio", cfg.steps.extract_audio, step_extract_audio.run),
        ("02_asr_whisperx", cfg.steps.asr_whisperx, step_whisperx.run),
        ("03_translate", cfg.steps.translate, step_translate.run),
        ("04_tts+align", cfg.steps.tts, tts_and_align),
        ("05_merge", cfg.steps.merge, step_merge_py.run),
    ]

    for name, enabled, fn in steps:
        if not enabled:
            info(f"[dubpipeline] Шаг {name} отключён в конфиге.")
            continue
        with timed_block(name, log=info):
            fn(cfg)

    if cfg.delete_srt:
        with timed_block("99_delete_srt", log=info):
            Path(cfg.paths.srt_file_en).unlink(missing_ok=True)

    if cfg.output.update_existing_file:
        info("[dubpipeline] Move output skipped: update_existing_file=true")
    else:
        mover = OutputMover(cfg.output.move_to_dir, base_dir=cfg.paths.workdir)
        mover.move_outputs([Path(cfg.paths.final_video)])

    success = True

    if success and getattr(cfg, "cleanup", False):
        with timed_block("99_cleanup_garbage", log=info):
            cleanup_garbage(cfg, pipeline_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline_path = Path(args.pipeline_file).expanduser().resolve()
    os.environ["DUBPIPELINE_KEEP_TEMP"] = "1" if args.keep_temp else "0"
    cli_set = _build_cli_set(args, parser)

    cfg = load_pipeline_config_ex(pipeline_path, cli_set=cli_set, create_dirs=not args.plan)
    if not args.plan:
        Path(cfg.paths.out_dir).mkdir(parents=True, exist_ok=True)

    effective_recursive, effective_glob = _resolve_input_options(
        cfg,
        parser,
        recursive=args.recursive,
        glob_pattern=args.glob,
    )
    input_source = _detect_input_source(args)
    print(f"[dubpipeline] input source: {input_source}")

    files = _discover_input_files(cfg, recursive=effective_recursive, glob_pattern=effective_glob)
    _print_effective_summary(cfg, files, plan_mode=args.plan)

    if args.plan:
        return

    if not files:
        parser.error(f"Не найдено входных файлов для '{cfg.paths.input_video}'")

    log_path = Path(cfg.paths.out_dir) / f"{cfg.project_name}.log"
    init_logger(log_path)
    info(f"[dubpipeline] build: {get_build_info()}")

    for input_file in files:
        run_cfg = _build_cfg_for_input(cfg, input_file)
        Const.bind(run_cfg)
        run_pipeline(run_cfg, pipeline_path)


if __name__ == "__main__":
    main()
