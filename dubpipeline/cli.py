from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from dubpipeline.consts import Const
from dubpipeline.steps.step_text_input import load_text, normalize_text, save_segments_json, split_to_segments
from dubpipeline.steps.step_tts_core import synthesize_segments_to_wavs
from dubpipeline.translation.providers import PUBLIC_TRANSLATION_PROVIDERS
from dubpipeline.utils.concat_wavs import concat_wavs
from dubpipeline.utils.logging import info, init_logger, warn
from dubpipeline.utils.output_move import OutputMover
from dubpipeline.utils.run_meta import log_run_header
from dubpipeline.utils.timing import timed_block, timed_run
from .config import (
    PipelineConfig,
    SUPPORTED_SOURCE_LANGUAGES,
    SUPPORTED_TRANSLATION_LANGUAGES,
    load_pipeline_config_ex,
    normalize_language_code,
    pipeline_path,
    validate_translation_language_pair,
)

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
        description="DubPipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    supported_langs_text = ", ".join(SUPPORTED_TRANSLATION_LANGUAGES)
    supported_source_langs_text = ", ".join(SUPPORTED_SOURCE_LANGUAGES)

    run_parser = subparsers.add_parser("run", help="Run the video pipeline.")
    run_parser.add_argument("pipeline_file", help="Path to *.pipeline.yaml")
    input_group = run_parser.add_mutually_exclusive_group()
    input_group.add_argument("--in-file", default=None, metavar="PATH", help="Input video file.")
    input_group.add_argument("--in-dir", default=None, metavar="PATH", help="Input directory with video files.")
    run_parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override config value.")
    run_parser.add_argument("--move-to-dir", default=None, help="Move outputs to the target directory.")
    run_parser.add_argument("--recursive", action="store_true", help="Recursive scan for --in-dir.")
    run_parser.add_argument("--glob", default=None, metavar="PATTERN", help="Glob filter for input files.")
    run_parser.add_argument("--out", default=None, metavar="DIR", help="Override paths.out_dir.")
    run_parser.add_argument(
        "--lang-src",
        default=None,
        metavar="LANG",
        help=f"Source language. Supported values: {supported_source_langs_text}.",
    )
    run_parser.add_argument(
        "--lang-dst",
        default=None,
        metavar="LANG",
        help=f"Target language. Supported values: {supported_langs_text}.",
    )
    run_parser.add_argument(
        "--translation-provider",
        default=None,
        choices=PUBLIC_TRANSLATION_PROVIDERS,
        metavar="PROVIDER",
        help=f"Translation provider. Supported values: {', '.join(PUBLIC_TRANSLATION_PROVIDERS)}.",
    )
    run_parser.add_argument("--steps", default=None, metavar="LIST", help="Pipeline steps to enable or patch.")
    run_parser.add_argument("--usegpu", action="store_true", help="Force GPU.")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    run_parser.add_argument("--rebuild", action="store_true", help="Rebuild intermediate artifacts.")
    run_parser.add_argument("--delete-temp", action="store_true", help="Delete temp files after completion.")
    run_parser.add_argument("--keep-temp", action="store_true", help="Keep temp files after completion.")
    run_parser.add_argument("--merge-mode", default=None, metavar="MODE", help="Final merge mode.")
    run_parser.add_argument("--tts-gain-db", type=float, default=None, metavar="DB", help="TTS gain in dB.")
    run_parser.add_argument("--original-gain-db", type=float, default=None, metavar="DB", help="Original gain in dB.")
    run_parser.add_argument("--ducking-amount-db", type=float, default=None, metavar="DB", help="Ducking amount in dB.")
    run_parser.add_argument("--ducking-threshold-db", type=float, default=None, metavar="DB", help="Ducking threshold in dB.")
    run_parser.add_argument("--ducking-attack-ms", type=int, default=None, metavar="MS", help="Ducking attack in ms.")
    run_parser.add_argument("--ducking-release-ms", type=int, default=None, metavar="MS", help="Ducking release in ms.")
    run_parser.add_argument("--no-loudnorm", action="store_true", help="Disable loudnorm.")
    run_parser.add_argument("--plan", action="store_true", help="Dry-run mode.")

    speak_parser = subparsers.add_parser("speak", help="Synthesize WAV from text.")
    speak_input = speak_parser.add_mutually_exclusive_group(required=True)
    speak_input.add_argument("--text", default=None, help="Inline text for synthesis.")
    speak_input.add_argument("--text-file", default=None, metavar="PATH", help="Path to a text file.")
    speak_parser.add_argument("--out-audio", required=True, metavar="PATH", help="Output WAV file.")
    speak_parser.add_argument("--voice", default=None, metavar="VOICE", help="XTTS speaker id.")
    speak_parser.add_argument("--speaker-wav", default=None, metavar="PATH", help="Reference WAV for voice cloning.")
    speak_parser.add_argument("--lang", default="ru", metavar="LANG", help="Synthesis language.")
    speak_parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override config value.")
    speak_parser.add_argument("--usegpu", action="store_true", help="Force GPU.")
    speak_parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    speak_parser.add_argument("--plan", action="store_true", help="Dry-run mode.")
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
    supported_langs_text = ", ".join(SUPPORTED_TRANSLATION_LANGUAGES)
    supported_source_langs_text = ", ".join(SUPPORTED_SOURCE_LANGUAGES)
    if args.move_to_dir is not None:
        cli_set.append(f"output.move_to_dir={args.move_to_dir}")
    if args.out is not None:
        cli_set.append(f"paths.out_dir={args.out}")
    if args.lang_src is not None:
        src_lang = normalize_language_code(args.lang_src, default="")
        if src_lang not in SUPPORTED_SOURCE_LANGUAGES:
            parser.error(f"--lang-src must be one of: {supported_source_langs_text}")
        cli_set.append(f"languages.src={src_lang}")
    if args.lang_dst is not None:
        tgt_lang = normalize_language_code(args.lang_dst, default="")
        if tgt_lang not in SUPPORTED_TRANSLATION_LANGUAGES:
            parser.error(f"--lang-dst must be one of: {supported_langs_text}")
        cli_set.append(f"languages.tgt={tgt_lang}")
    if args.translation_provider is not None:
        cli_set.append(f"translation.provider={args.translation_provider}")
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


def _build_speak_cli_set(args: argparse.Namespace, parser: argparse.ArgumentParser, out_audio: Path) -> list[str]:
    if args.usegpu and args.cpu:
        parser.error("Cannot use --usegpu and --cpu together for speak")

    cli_set = list(args.set)
    cli_set.append(f"project_name={out_audio.stem}")
    cli_set.append(f"paths.out_dir={out_audio.parent}")

    if args.voice:
        cli_set.append(f"tts.voice={args.voice}")
    if args.usegpu:
        cli_set.append("usegpu=true")
    if args.cpu:
        cli_set.append("usegpu=false")
    if args.speaker_wav:
        speaker_wav = Path(args.speaker_wav).expanduser()
        if not speaker_wav.is_file():
            parser.error(f"--speaker-wav must point to an existing file: '{speaker_wav}'")
        cli_set.append(f"tts.speaker_wav={speaker_wav.resolve()}")

    return cli_set


def synthesize_text_to_wav(
    *,
    out_audio: Path,
    text: str | None = None,
    text_file: Path | None = None,
    voice: str | None = None,
    speaker_wav: Path | None = None,
    lang: str = "ru",
    use_gpu: bool | None = None,
    cli_set: list[str] | None = None,
    plan: bool = False,
) -> list[Path]:
    out_audio = out_audio.expanduser().resolve()
    if not plan:
        out_audio.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_pipeline_config_ex(pipeline_path, cli_set=cli_set or [], create_dirs=not plan)
    cfg.project_name = out_audio.stem
    cfg.paths.out_dir = out_audio.parent
    cfg.paths.final_audio = out_audio
    if voice:
        cfg.tts.voice = voice
    if speaker_wav:
        cfg.tts.speaker_wav = str(speaker_wav.expanduser().resolve())
    if use_gpu is not None:
        cfg.usegpu = bool(use_gpu)

    raw_text = load_text(text=text, text_file=text_file.expanduser().resolve() if text_file is not None else None)
    normalized = normalize_text(raw_text)
    segments = split_to_segments(normalized, max_chars=cfg.tts.text_max_chars)
    if not segments:
        raise ValueError("Input text is empty after normalization")

    segments_dir = out_audio.parent / f"{out_audio.stem}_segments"
    wavs = synthesize_segments_to_wavs(
        segments,
        cfg,
        segments_dir,
        voice=voice,
        lang=lang,
        speaker_wav=speaker_wav.expanduser().resolve() if speaker_wav is not None else None,
        plan=plan,
        show_progress=not plan,
    )

    if plan:
        print(f"[dubpipeline][speak][plan] segments={len(wavs)} out_audio={out_audio}")
        return wavs

    if not wavs:
        raise RuntimeError("TTS synthesis produced no audio segments")

    concat_wavs(wavs, out_audio, gap_ms=cfg.tts.gap_ms, subtype="PCM_16")
    info(f"[dubpipeline][speak] Output audio written to: {out_audio}")
    return wavs


def _run_speak(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    if parser is None:
        parser = build_parser()
    out_audio = Path(args.out_audio).expanduser().resolve()
    cli_set = _build_speak_cli_set(args, parser, out_audio)
    speaker_wav = Path(args.speaker_wav).expanduser().resolve() if args.speaker_wav else None
    text_file = Path(args.text_file).expanduser().resolve() if args.text_file else None

    synthesize_text_to_wav(
        out_audio=out_audio,
        text=args.text,
        text_file=text_file,
        voice=args.voice,
        speaker_wav=speaker_wav,
        lang=args.lang,
        use_gpu=(False if args.cpu else True if args.usegpu else None),
        cli_set=cli_set,
        plan=bool(args.plan),
    )


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


def _validate_run_language_pair(
    cfg: PipelineConfig,
    parser: argparse.ArgumentParser,
    *,
    allow_legacy: bool,
) -> None:
    message = validate_translation_language_pair(
        cfg.languages.src,
        cfg.languages.tgt,
        translate_enabled=bool(cfg.steps.translate),
        allow_legacy=allow_legacy,
    )
    if message:
        parser.error(message)


def _build_cfg_for_input(base_cfg: PipelineConfig, input_file: Path) -> PipelineConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.project_name = input_file.stem
    cfg.paths.input_video = input_file.resolve()
    target_lang = (cfg.languages.tgt or "ru").strip().lower() or "ru"

    out_dir = Path(cfg.paths.out_dir)
    cfg.paths.audio_wav = out_dir / f"{cfg.project_name}.wav"
    cfg.paths.segments_file = out_dir / f"{cfg.project_name}.segments.json"
    cfg.paths.segments_tgt_file = out_dir / f"{cfg.project_name}.segments.{target_lang}.json"
    cfg.paths.srt_file_en = out_dir / f"{cfg.project_name}.srt"
    cfg.paths.tts_segments_dir = out_dir / "segments" / f"tts_{target_lang}_segments"
    cfg.paths.tts_segments_aligned_dir = out_dir / "segments" / f"tts_{target_lang}_segments_aligned"
    cfg.paths.final_video = out_dir / f"{cfg.project_name}.{target_lang}.muxed.mp4"
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
    from dubpipeline.translation.service import TranslationModelError, TranslatorService
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

    if cfg.steps.translate:
        try:
            TranslatorService.from_config(cfg)
        except TranslationModelError as exc:
            raise SystemExit(str(exc)) from None

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

    if args.command == "speak":
        _run_speak(args, parser)
        return

    pipeline_path = Path(args.pipeline_file).expanduser().resolve()
    os.environ["DUBPIPELINE_KEEP_TEMP"] = "1" if args.keep_temp else "0"
    cli_set = _build_cli_set(args, parser)

    cfg = load_pipeline_config_ex(pipeline_path, cli_set=cli_set, create_dirs=not args.plan)
    _validate_run_language_pair(
        cfg,
        parser,
        allow_legacy=(args.lang_src is None and args.lang_dst is None),
    )
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

    cfg.batch_file_count = len(files)
    log_path = Path(cfg.paths.out_dir) / f"{cfg.project_name}.log"
    init_logger(log_path)

    try:
        for input_file in files:
            run_cfg = _build_cfg_for_input(cfg, input_file)
            Const.bind(run_cfg)
            run_pipeline(run_cfg, pipeline_path)
    finally:
        if len(files) > 1:
            from dubpipeline.translation.service import TranslatorService

            TranslatorService.release_shared_cache(cfg)


if __name__ == "__main__":
    main()
