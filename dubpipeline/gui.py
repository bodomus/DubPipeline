import threading
import subprocess
import sys
import os
import tempfile
from multiprocessing import Process, Queue
from pathlib import Path

import FreeSimpleGUI as sg
import yaml
import re
import time
import shutil

from dubpipeline.cli import synthesize_text_to_wav
from dubpipeline.config import (
    SUPPORTED_TRANSLATION_LANGUAGES,
    get_voice,
    load_pipeline_config_ex,
    normalize_audio_update_mode,
    normalize_language_code,
    pipeline_path,
    save_pipeline_yaml,
    validate_translation_language_pair,
)
from dubpipeline.models.catalog import (
    NOT_SUPPORTED_REASON,
    build_model_choices,
    get_model_spec,
    get_model_status,
    is_unsupported_pair_reason,
    legacy_translate_backend_for_model,
    resolve_model_spec,
)
from dubpipeline.models.installer import (
    ModelInstallStatus,
    get_model_installer,
)
from dubpipeline.translation.service import model_not_installed_message
from dubpipeline.steps.step_tts import list_voices, synthesize_preview_text

from dubpipeline.utils.build_info import get_build_info
from dubpipeline.utils.logging import info, error
from dubpipeline.input_discovery import enumerate_input_files, source_mode_disabled_map
from dubpipeline.input_mode import build_audio_output_path, resolve_saved_input_state, validate_input_path, validate_text_file_path
from dubpipeline.external_subtitles import find_external_subtitle_for_video, missing_subtitles_error


def _preview_worker_target(
    q: Queue,
    *,
    model_name: str,
    voice_id: str,
    preview_text: str,
    out_file: str,
    use_gpu: bool,
    lang: str,
) -> None:
    """Worker for TTS preview. Must be top-level for Windows multiprocessing (spawn/pickle)."""
    try:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        synthesize_preview_text(
            model_name=model_name,
            voice_id=voice_id,
            preview_text=preview_text,
            out_file=out_path,
            use_gpu=use_gpu,
            lang=lang,
        )
        q.put({"ok": True, "file": str(out_path)})
    except Exception as ex:
        q.put({"ok": False, "error": str(ex)})


def _audio_play_worker_target(
    q: Queue,
    *,
    voice_id: str,
    text_file: str,
    out_file: str,
    use_gpu: bool,
) -> None:
    try:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        synthesize_text_to_wav(
            out_audio=out_path,
            text_file=Path(text_file),
            voice=voice_id,
            use_gpu=use_gpu,
        )
        q.put({"ok": True, "file": str(out_path)})
    except Exception as ex:
        q.put({"ok": False, "error": str(ex)})

STEP_LABELS = {
    "extract_audio": "Extract audio",
    "asr_whisperx": "ASR (WhisperX)",
    "translate": "Translate",
    "tts": "TTS",
    "align": "Align",
    "merge": "Merge video",
}

DEFAULT_STEPS = {
    "extract_audio": True,
    "asr_whisperx": True,
    "translate": True,
    "tts": True,
    "align": True,
    "merge": True,
}


MODE_DISPLAY_BY_VALUE = {
    "add": "Add",
    "overwrite": "Overwrite",
    "overwrite_reorder": "Overwrite+Reorder",
}

MODE_VALUE_BY_DISPLAY = {v: k for k, v in MODE_DISPLAY_BY_VALUE.items()}
TAB_VIDEO = "-TAB_VIDEO-"
TAB_AUDIO = "-TAB_AUDIO-"

UPDATE_EXISTING_WARNING = (
    "The original file will be replaced on success. "
    "If an error occurs, the original file will remain unchanged."
)

AUDIO_MODE_TOOLTIP = (
    "Add: keeps the original tracks and appends the dub track at the end.\n"
    "Overwrite: keeps only the new dub track.\n"
    "Overwrite+Reorder: puts the dub track first, followed by the remaining tracks."
)


def _mode_to_display(value: str) -> str:
    return MODE_DISPLAY_BY_VALUE.get(normalize_audio_update_mode(value), "Add")

def show_app_constants():
    print(f'### DUBPIPELINE_TTS_MAX_RU_CHARS: {os.getenv("DUBPIPELINE_TTS_MAX_RU_CHARS")}')
    print(f'###DUBPIPELINE_TTS_TRY_SINGLE_CALL: {os.getenv("DUBPIPELINE_TTS_TRY_SINGLE_CALL")}')
    print(f'### DUBPIPELINE_TTS_TRY_SINGLE_CALL_MAX_CHARS: {os.getenv("DUBPIPELINE_TTS_TRY_SINGLE_CALL_MAX_CHARS")}')
    print(f'### DUBPIPELINE_MIN_SEG_DUR: {os.getenv("DUBPIPELINE_MIN_SEG_DUR")}')
    print(f'### DUBPIPELINE_MIN_SEG_CHARS: {os.getenv("DUBPIPELINE_MIN_SEG_CHARS")}')
    print(f'### DUBPIPELINE_MERGE_MAX_GAP: {os.getenv("DUBPIPELINE_MERGE_MAX_GAP")}')
    print(f'### DUBPIPELINE_MAX_SEG_DUR: {os.getenv("DUBPIPELINE_MAX_SEG_DUR")}')
    print(f'### DUBPIPELINE_MERGE_ALLOW_CROSS_SPEAKER: {os.getenv("DUBPIPELINE_MERGE_ALLOW_CROSS_SPEAKER")}')


USE_SUBPROCESS = True

TEMPLATE_PATH = Path(__file__).with_name("video.pipeline.yaml")
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    BASE_CFG = yaml.safe_load(f)

LOG_LINE_RE = re.compile(
    r"^\[(?P<level>\w+\s*)]\s*"                # [LEVEL]
    r"(?:(?P<time>\d{2}:\d{2}:\d{2})\s*\|\s*)?"  # необязательное "HH:MM:SS | "
    r"(?P<msg>.*)$"                          # остальное — сообщение
)

LEVEL_COLORS = {
    "DEBUG": "darkblue",          # по умолчанию
    "INFO":  "green",
    "STEP":  "blue",
    "WARN":  "orange",
    "ERROR": "red",
}

def print_parsed_log(window, line: str) -> None:
    ml = window["-LOGBOX-"]

    # 1) пробуем распарсить наш формат [LEVEL] HH:MM:SS | msg
    m = LOG_LINE_RE.match(line)
    if m:
        level = m.group("level").upper().strip()
        ts = m.group("time")
        msg = m.group("msg")

        color = LEVEL_COLORS.get(level, None)
        text = f"[{ts}][{level}] {msg}"
        ml.print(text, text_color=color)
        return

    # 2) если формат не наш — всё равно попробуем подсветить явные ошибки
    lower = line.lower()
    if "traceback" in lower or "error" in lower or "exception" in lower:
        ml.print(line, text_color="red")
    else:
        ml.print(line)


class PreviewController:
    def __init__(self) -> None:
        self._worker: Process | None = None
        self._queue: Queue | None = None
        self._player: subprocess.Popen | None = None

    def is_active(self) -> bool:
        worker_alive = self._worker is not None and self._worker.is_alive()
        player_alive = self._player is not None and self._player.poll() is None
        return worker_alive or player_alive

    def stop(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=1)
        self._worker = None
        self._queue = None

        if self._player is not None and self._player.poll() is None:
            self._player.terminate()
            try:
                self._player.wait(timeout=1)
            except Exception:
                self._player.kill()
        self._player = None

    def start_preview(
        self,
        *,
        model_name: str,
        voice_id: str,
        preview_text: str,
        use_gpu: bool,
        lang: str,
        window,
    ) -> None:
        self.stop()
        out_dir = Path(tempfile.gettempdir()) / "dubpipeline_preview"
        out_file = out_dir / f"preview_{int(time.time() * 1000)}.wav"
        out_dir.mkdir(parents=True, exist_ok=True)

        queue: Queue = Queue()

        proc = Process(
            target=_preview_worker_target,
            kwargs={
                "q": queue,
                "model_name": model_name,
                "voice_id": voice_id,
                "preview_text": preview_text,
                "out_file": str(out_file),
                "use_gpu": use_gpu,
                "lang": lang,
            },
            daemon=True,
        )
        proc.start()

        self._worker = proc
        self._queue = queue

        def _monitor() -> None:
            proc.join()
            payload = {"ok": False, "error": "Preview synthesis cancelled"}
            if queue is not None:
                try:
                    payload = queue.get_nowait()
                except Exception:
                    pass
            window.write_event_value("-PREVIEW_READY-", payload)

        threading.Thread(target=_monitor, daemon=True).start()

    def play_file(self, audio_path: str) -> None:
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            audio_path,
        ]
        self._player = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def _wait_playback() -> None:
            if self._player is not None:
                self._player.wait()

        threading.Thread(target=_wait_playback, daemon=True).start()


class AudioPlaybackController:
    def __init__(self) -> None:
        self._worker: Process | None = None
        self._queue: Queue | None = None
        self._player: subprocess.Popen | None = None

    def is_active(self) -> bool:
        worker_alive = self._worker is not None and self._worker.is_alive()
        player_alive = self._player is not None and self._player.poll() is None
        return worker_alive or player_alive

    def stop(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=1)
        self._worker = None
        self._queue = None

        if self._player is not None and self._player.poll() is None:
            self._player.terminate()
            try:
                self._player.wait(timeout=1)
            except Exception:
                self._player.kill()
        self._player = None

    def start(self, *, voice_id: str, text_file: str, use_gpu: bool, window) -> None:
        self.stop()
        out_dir = Path(tempfile.gettempdir()) / "dubpipeline_audio_play"
        out_file = out_dir / f"play_{int(time.time() * 1000)}.wav"
        out_dir.mkdir(parents=True, exist_ok=True)

        queue: Queue = Queue()
        proc = Process(
            target=_audio_play_worker_target,
            kwargs={
                "q": queue,
                "voice_id": voice_id,
                "text_file": text_file,
                "out_file": str(out_file),
                "use_gpu": use_gpu,
            },
            daemon=True,
        )
        proc.start()

        self._worker = proc
        self._queue = queue

        def _monitor() -> None:
            proc.join()
            payload = {"ok": False, "error": "Audio synthesis cancelled"}
            if queue is not None:
                try:
                    payload = queue.get_nowait()
                except Exception:
                    pass
            window.write_event_value("-AUDIO_PLAY_READY-", payload)

        threading.Thread(target=_monitor, daemon=True).start()

    def play_file(self, audio_path: str) -> None:
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            audio_path,
        ]
        self._player = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def _wait_playback() -> None:
            if self._player is not None:
                self._player.wait()

        threading.Thread(target=_wait_playback, daemon=True).start()


def run_pipeline(args_list, window):
    """
    args_list, for example: ["run", "D:/Projects/DubPipeline/out/myproj.pipeline.yaml"]
    Runs in a background thread and sends -LOG- and -DONE- events to the window.
    """
    try:
        cmd = [sys.executable, "-u", "-m", "dubpipeline.cli"] + args_list

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            window.write_event_value("-LOG-", line)

        process.wait()
        exit_code = process.returncode
        window.write_event_value("-DONE-", exit_code)

    except Exception as e:
        window.write_event_value("-LOG-", f"[ERROR] {e}\n")
        window.write_event_value("-DONE-", 1)


def _emit_info(window, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    window.write_event_value("-LOG-", f"[INFO ] {ts} | {msg}\n")


def run_pipeline_sequence(run_items, window):
    """Runs several pipelines one after another in a single thread.

    run_items: list[tuple[label:str, args_list:list[str]]]
    Sends -LOG- events during execution and a single -DONE- event at the end.
    """
    try:
        total = len(run_items)

        for idx, (label, args_list) in enumerate(run_items, start=1):
            # <<< ДОБАВИТЬ: сообщаем GUI какой файл сейчас пойдёт
            window.write_event_value("-FILE-", {"idx": idx, "total": total, "name": label})

            _emit_info(window, f"=== ({idx}/{total}) {label} ===")

            cmd = [sys.executable, "-u", "-m", "dubpipeline.cli"] + args_list
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )

            for line in process.stdout:
                window.write_event_value("-LOG-", line)

            process.wait()
            exit_code = process.returncode
            if exit_code != 0:
                _emit_info(window, f"Stopped: pipeline exited with code {exit_code}")
                window.write_event_value("-DONE-", exit_code)
                return

        window.write_event_value("-DONE-", 0)
    except Exception as e:
        window.write_event_value("-LOG-", f"[ERROR] {e}\n")
        window.write_event_value("-DONE-", 1)


def normalize_steps(steps_dict: dict | None) -> dict:
    normalized = dict(DEFAULT_STEPS)
    if isinstance(steps_dict, dict):
        for key in normalized:
            if key in steps_dict:
                normalized[key] = bool(steps_dict[key])
    return normalized


def steps_summary(steps_dict: dict) -> str:
    enabled = [label for key, label in STEP_LABELS.items() if steps_dict.get(key)]
    if not enabled:
        return "Steps: none selected"
    return f"Steps: {', '.join(enabled)}"


def _persist_base_cfg() -> None:
    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(BASE_CFG, f, allow_unicode=True, sort_keys=False)


def _current_base_languages() -> tuple[str, str]:
    languages = BASE_CFG.get("languages") or {}
    src_lang = normalize_language_code(languages.get("src", "en"), default="en")
    tgt_lang = normalize_language_code(languages.get("tgt", "ru"), default="ru")
    return src_lang, tgt_lang


def _language_choices(current_lang: str) -> list[str]:
    choices = list(SUPPORTED_TRANSLATION_LANGUAGES)
    if current_lang and current_lang not in choices:
        choices.insert(0, current_lang)
    return choices


def _language_pair_summary(src_lang: str, tgt_lang: str, *, translate_enabled: bool) -> tuple[str, str]:
    validation_error = validate_translation_language_pair(
        src_lang,
        tgt_lang,
        translate_enabled=translate_enabled,
        allow_legacy=True,
    )
    if validation_error:
        return f"Language pair: {src_lang} -> {tgt_lang} ({validation_error})", "firebrick"

    legacy_codes = [
        code
        for code in (src_lang, tgt_lang)
        if code not in SUPPORTED_TRANSLATION_LANGUAGES
    ]
    if legacy_codes:
        return (
            f"Language pair: {src_lang} -> {tgt_lang} (legacy config value; new UI supports "
            f"{', '.join(SUPPORTED_TRANSLATION_LANGUAGES)})",
            "darkorange3",
        )
    return f"Language pair: {src_lang} -> {tgt_lang}", "gray35"


def show_steps_modal(parent, current_steps: dict) -> dict:
    steps = normalize_steps(current_steps)
    layout = [[sg.Text("Select generation steps:")]]
    for key in STEP_LABELS:
        layout.append([sg.Checkbox(STEP_LABELS[key], key=f"-STEP-{key}-", default=steps[key])])
    layout.append([sg.Button("Save", key="-SAVE-"), sg.Button("Cancel", key="-CANCEL-")])

    window = sg.Window(
        "Generation Steps",
        layout,
        modal=True,
        finalize=True,
        keep_on_top=True,
        location=parent.current_location() if parent else None,
    )

    result = steps
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-CANCEL-"):
            result = steps
            break
        if event == "-SAVE-":
            result = {key: bool(values.get(f"-STEP-{key}-")) for key in STEP_LABELS}
            break

    window.close()
    return result


def _translation_summary(model_id: str, src_lang: str | None = None, tgt_lang: str | None = None) -> tuple[str, str]:
    if src_lang is None or tgt_lang is None:
        src_lang, tgt_lang = _current_base_languages()
    try:
        spec = resolve_model_spec(model_id, src_lang, tgt_lang)
        status = get_model_status(model_id, src_lang, tgt_lang)
        install_status = get_model_installer().get_status(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
    except Exception:
        return "Machine Translation: unknown model", "firebrick"

    if not spec.supported or status.reason == NOT_SUPPORTED_REASON:
        return f"Machine Translation: {spec.label} (planned)", "gray45"
    if is_unsupported_pair_reason(status.reason):
        return f"Machine Translation: {spec.label} ({status.reason})", "gray45"
    if install_status.status == "failed":
        reason = (install_status.error or install_status.message or "install failed").strip()
        if len(reason) > 64:
            reason = f"{reason[:61]}..."
        return f"Machine Translation: {spec.label} (failed: {reason})", "firebrick"
    if status.enabled:
        return f"Machine Translation: {spec.label}", "darkgreen"
    return f"Machine Translation: {spec.label} (not installed)", "gray35"


def ensure_translation_model_ready_for_start(
    window,
    model_id: str,
    *,
    src_lang: str,
    tgt_lang: str,
) -> str | None:
    model_id = (model_id or "").strip()
    if not model_id:
        msg = "Translation model is not configured. Open Models and choose an installed model."
        sg.popup_error(msg, title="Translation model required", keep_on_top=True)
        _emit_info(window, msg)
        return None

    try:
        spec = resolve_model_spec(model_id, src_lang, tgt_lang)
        status = get_model_status(model_id, src_lang, tgt_lang)
    except Exception as exc:
        msg = f"Translation model cannot be checked: {exc}"
        sg.popup_error(msg, title="Translation model required", keep_on_top=True)
        _emit_info(window, msg)
        return None

    if status.enabled:
        return model_id

    if not spec.supported or status.reason == NOT_SUPPORTED_REASON:
        msg = f"Translation model '{spec.label}' is planned and not supported yet. Choose another model in Models."
        sg.popup_error(msg, title="Translation model required", keep_on_top=True)
        _emit_info(window, msg)
        return None

    if is_unsupported_pair_reason(status.reason):
        msg = f"Translation model '{spec.label}' is {status.reason}. Choose another model in Models."
        sg.popup_error(msg, title="Translation model required", keep_on_top=True)
        _emit_info(window, msg)
        return None

    msg = model_not_installed_message(spec.label, src_lang, tgt_lang)
    _emit_info(window, msg)
    answer = sg.popup_yes_no(
        f"{msg}\n\nOpen Models to install it now?",
        title="Translation model required",
        keep_on_top=True,
    )
    if answer != "Yes":
        return None

    selected_model_id = show_models_modal(
        window,
        model_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    if not selected_model_id:
        return None

    try:
        selected_spec = resolve_model_spec(selected_model_id, src_lang, tgt_lang)
        selected_status = get_model_status(selected_model_id, src_lang, tgt_lang)
    except Exception as exc:
        msg = f"Translation model cannot be checked after install: {exc}"
        sg.popup_error(msg, title="Translation model required", keep_on_top=True)
        _emit_info(window, msg)
        return None

    if not selected_status.enabled:
        msg = model_not_installed_message(selected_spec.label, src_lang, tgt_lang)
        sg.popup_error(msg, title="Translation model required", keep_on_top=True)
        _emit_info(window, msg)
        return None

    persist_languages(src_lang, tgt_lang)
    persist_translation_model(selected_model_id, src_lang=src_lang, tgt_lang=tgt_lang)
    window["-TRANSLATION_MODEL_ID-"].update(selected_model_id)
    summary_text, summary_color = _translation_summary(selected_model_id, src_lang, tgt_lang)
    window["-MODEL_SUMMARY-"].update(summary_text, text_color=summary_color)
    _emit_info(
        window,
        f"Translation model selected: {selected_spec.label} [{selected_spec.id}] "
        f"for {src_lang} -> {tgt_lang}: {selected_spec.model_ref}",
    )
    return selected_model_id


def sync_translation_start_values(
    values: dict,
    model_id: str,
    *,
    src_lang: str,
    tgt_lang: str,
) -> tuple[bool, str]:
    src = normalize_language_code(src_lang, default="en")
    tgt = normalize_language_code(tgt_lang, default="ru")
    selected_model_id = (model_id or "").strip()
    if not selected_model_id:
        return False, "Translation model is not configured. Open Models and choose an installed model."

    spec = resolve_model_spec(selected_model_id, src, tgt)
    status = get_model_status(selected_model_id, src, tgt)
    if not status.enabled:
        return False, model_not_installed_message(spec.label, src, tgt)

    values["-TRANSLATION_MODEL_ID-"] = spec.id
    values["-LANG_SRC-"] = src
    values["-LANG_DST-"] = tgt
    return True, spec.model_ref


def persist_languages(src_lang: str, tgt_lang: str) -> None:
    BASE_CFG.setdefault("languages", {})
    BASE_CFG["languages"]["src"] = normalize_language_code(src_lang, default="en")
    BASE_CFG["languages"]["tgt"] = normalize_language_code(tgt_lang, default="ru")
    _persist_base_cfg()


def persist_translation_model(model_id: str, src_lang: str | None = None, tgt_lang: str | None = None) -> None:
    if src_lang is None or tgt_lang is None:
        src_lang, tgt_lang = _current_base_languages()
    spec = resolve_model_spec(model_id, src_lang, tgt_lang)
    BASE_CFG.setdefault("translation", {})
    BASE_CFG["translation"]["model_id"] = spec.id
    BASE_CFG["translation"]["backend"] = spec.backend
    BASE_CFG["translation"]["model_ref"] = spec.model_ref

    # Keep legacy keys for compatibility with older scripts/tools.
    BASE_CFG.setdefault("translate", {})
    BASE_CFG["translate"]["backend"] = legacy_translate_backend_for_model(spec)
    if BASE_CFG["translate"]["backend"] == "hf":
        BASE_CFG["translate"]["hf_model"] = spec.model_ref
    _persist_base_cfg()


def show_models_modal(parent, current_model_id: str, *, src_lang: str, tgt_lang: str) -> str | None:
    installer = get_model_installer()
    choices = build_model_choices(src_lang, tgt_lang)
    displays = [item.display for item in choices]
    by_display = {item.display: item for item in choices}

    selected_display = ""
    for item in choices:
        if item.model_id == current_model_id:
            selected_display = item.display
            break
    if not selected_display:
        for item in choices:
            if item.model_id and item.enabled:
                selected_display = item.display
                break
    if not selected_display and displays:
        selected_display = displays[0]

    layout = [
        [sg.Text("Machine Translation (Text -> Text)", font=("Segoe UI", 11, "bold"))],
        [
            sg.Text("Model:", size=(8, 1)),
            sg.Combo(
                values=displays,
                key="-MODELS_MT-",
                readonly=True,
                default_value=selected_display,
                enable_events=True,
                size=(52, 1),
            ),
        ],
        [
            sg.ProgressBar(
                max_value=100,
                orientation="h",
                size=(52, 14),
                key="-MODELS_PROGRESS-",
                expand_x=True,
            )
        ],
        [
            sg.Text(
                "",
                key="-MODELS_STATUS-",
                text_color="black",
                size=(72, 3),
                expand_x=True,
            )
        ],
        [
            sg.Button("Install", key="-MODELS_INSTALL-"),
            sg.Button("Cancel Download", key="-MODELS_CANCEL_INSTALL-", disabled=True),
            sg.Button("Apply", key="-MODELS_APPLY-"),
            sg.Button("Close", key="-MODELS_CANCEL-"),
        ],
    ]

    window = sg.Window(
        "Models",
        layout,
        modal=True,
        finalize=True,
        keep_on_top=True,
        location=parent.current_location() if parent else None,
    )

    selected_model_id: str | None = None
    active_download_model_id: str | None = None

    def _update_state(display_value: str) -> None:
        nonlocal selected_model_id
        choice = by_display.get(display_value)
        if choice is None:
            window["-MODELS_STATUS-"].update(
                "Unknown selection. Please choose a model.",
                text_color="firebrick",
            )
            window["-MODELS_PROGRESS-"].update_bar(0)
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        try:
            window["-MODELS_MT-"].Widget.configure(foreground=choice.color)
        except Exception:
            pass

        if choice.is_group_header:
            window["-MODELS_STATUS-"].update(
                "Select a model entry below the tier header.",
                text_color="darkorange",
            )
            window["-MODELS_PROGRESS-"].update_bar(0)
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        if not choice.model_id:
            window["-MODELS_STATUS-"].update(
                "Unknown model entry.",
                text_color="firebrick",
            )
            window["-MODELS_PROGRESS-"].update_bar(0)
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        spec = resolve_model_spec(choice.model_id, src_lang, tgt_lang)
        install_status = installer.get_status(choice.model_id, src_lang=src_lang, tgt_lang=tgt_lang)

        if not spec.supported:
            window["-MODELS_STATUS-"].update(
                "Not supported yet (planned). Install is unavailable.",
                text_color="gray45",
            )
            window["-MODELS_PROGRESS-"].update_bar(0)
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        status = get_model_status(choice.model_id, src_lang, tgt_lang)
        if is_unsupported_pair_reason(status.reason):
            window["-MODELS_STATUS-"].update(
                f"Model is {status.reason}. Install and apply are unavailable for this language pair.",
                text_color="gray45",
            )
            window["-MODELS_PROGRESS-"].update_bar(0)
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        if install_status.status == "downloading":
            msg = install_status.message or "Downloading..."
            window["-MODELS_STATUS-"].update(msg, text_color="blue")
            window["-MODELS_PROGRESS-"].update_bar(int(max(0.0, min(1.0, install_status.progress)) * 100))
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=False)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        if install_status.status == "failed":
            error_text = (install_status.error or install_status.message or "install failed").strip()
            if len(error_text) > 80:
                error_text = f"{error_text[:77]}..."
            window["-MODELS_STATUS-"].update(
                f"Install failed: {error_text}",
                text_color="firebrick",
            )
            window["-MODELS_PROGRESS-"].update_bar(0)
            window["-MODELS_INSTALL-"].update(disabled=False)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        if status.enabled and install_status.status == "installed":
            window["-MODELS_STATUS-"].update("Model is installed and ready.", text_color="darkgreen")
            window["-MODELS_PROGRESS-"].update_bar(100)
            window["-MODELS_INSTALL-"].update(disabled=True)
            window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
            window["-MODELS_APPLY-"].update(disabled=False)
            selected_model_id = choice.model_id
            return

        install_hint = "Model is not installed locally. Click Install."
        if spec.estimated_size_bytes is None:
            install_hint += " Model size is unknown."
        window["-MODELS_STATUS-"].update(install_hint, text_color="gray35")
        window["-MODELS_PROGRESS-"].update_bar(0)
        window["-MODELS_INSTALL-"].update(disabled=False)
        window["-MODELS_CANCEL_INSTALL-"].update(disabled=True)
        window["-MODELS_APPLY-"].update(disabled=True)
        selected_model_id = None

    def _refresh_choices(target_model_id: str | None = None) -> None:
        nonlocal choices, displays, by_display
        choices = build_model_choices(src_lang, tgt_lang)
        displays = [item.display for item in choices]
        by_display = {item.display: item for item in choices}
        window["-MODELS_MT-"].update(values=displays)

        next_display = ""
        if target_model_id:
            for item in choices:
                if item.model_id == target_model_id:
                    next_display = item.display
                    break

        if not next_display:
            current_display = window["-MODELS_MT-"].get()
            if current_display in by_display:
                next_display = current_display

        if not next_display and displays:
            next_display = displays[0]

        window["-MODELS_MT-"].update(value=next_display)
        _update_state(next_display)

    _update_state(selected_display)

    result: str | None = None
    while True:
        event, values = window.read(timeout=200)
        if event in (sg.WIN_CLOSED, "-MODELS_CANCEL-"):
            result = None
            break
        if event == "-MODELS_MT-":
            _update_state(values.get("-MODELS_MT-", ""))
        if event == "__TIMEOUT__":
            if active_download_model_id:
                status = installer.get_status(
                    active_download_model_id,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
                if status.status != "downloading":
                    active_download_model_id = None
                _update_state(values.get("-MODELS_MT-", ""))
            continue
        if event == "-MODELS_INSTALL-":
            choice = by_display.get(values.get("-MODELS_MT-", ""))
            if choice is None or not choice.model_id:
                continue

            model_id = choice.model_id
            active_download_model_id = model_id

            def _progress_callback(status: ModelInstallStatus) -> None:
                window.write_event_value("-MODELS_INSTALL_PROGRESS-", status)

            def _worker() -> None:
                install_result = installer.install(
                    model_id,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    progress_cb=_progress_callback,
                )
                window.write_event_value("-MODELS_INSTALL_DONE-", install_result)

            threading.Thread(target=_worker, daemon=True).start()
            _update_state(values.get("-MODELS_MT-", ""))
        if event == "-MODELS_CANCEL_INSTALL-":
            model_id = active_download_model_id
            if model_id is None:
                choice = by_display.get(values.get("-MODELS_MT-", ""))
                if choice and choice.model_id:
                    model_id = choice.model_id
            if model_id:
                installer.cancel(model_id, src_lang=src_lang, tgt_lang=tgt_lang)
        if event == "-MODELS_INSTALL_PROGRESS-":
            progress = values.get("-MODELS_INSTALL_PROGRESS-")
            if isinstance(progress, ModelInstallStatus):
                selected_choice = by_display.get(values.get("-MODELS_MT-", ""))
                if selected_choice and selected_choice.model_id == progress.model_id:
                    progress_pct = int(max(0.0, min(1.0, progress.progress)) * 100)
                    window["-MODELS_PROGRESS-"].update_bar(progress_pct)
                    color = "blue"
                    if progress.status == "failed":
                        color = "firebrick"
                    window["-MODELS_STATUS-"].update(progress.message or "", text_color=color)
                    if progress.status == "downloading":
                        window["-MODELS_INSTALL-"].update(disabled=True)
                        window["-MODELS_CANCEL_INSTALL-"].update(disabled=False)
                        window["-MODELS_APPLY-"].update(disabled=True)
        if event == "-MODELS_INSTALL_DONE-":
            install_result = values.get("-MODELS_INSTALL_DONE-")
            if install_result is not None and getattr(install_result, "model_id", None) == active_download_model_id:
                active_download_model_id = None
            target_model_id = getattr(install_result, "model_id", None)
            _refresh_choices(target_model_id=target_model_id)
        if event == "-MODELS_APPLY-":
            if selected_model_id:
                result = selected_model_id
                break

    window.close()
    return result


def persist_move_to_dir(path: str) -> None:
    BASE_CFG.setdefault("output", {})
    BASE_CFG["output"]["move_to_dir"] = path
    _persist_base_cfg()


def set_source_mode(window, is_dir: bool) -> None:
    for key, disabled in source_mode_disabled_map(is_dir=is_dir).items():
        window[key].update(disabled=disabled)


def browse_input_path(*, is_dir_mode: bool, video_file_types):
    if is_dir_mode:
        return sg.popup_get_folder("Select folder with videos", no_window=True)
    return sg.popup_get_file("Select video file", file_types=video_file_types, no_window=True)




def browse_text_file():
    return sg.popup_get_file(
        "Select text file",
        file_types=(("Text files", "*.txt;*.md;*.text"), ("All files", "*.*")),
        no_window=True,
    )


def validate_text_file_path(path_value: str) -> tuple[bool, str]:
    text_path = Path(path_value.strip()).expanduser()
    if not str(text_path):
        return False, "Specify a path to a text file."
    if not text_path.exists():
        return False, "The specified text file does not exist."
    if not text_path.is_file():
        return False, "Audio mode requires a file path."
    return True, ""


def resolve_audio_output_path(values, window) -> Path:
    out_path, project_name, out_dir = build_audio_output_path(
        values.get("-TEXT_PATH-", ""),
        values.get("-OUT-", ""),
        values.get("-PROJECT-", ""),
    )
    if out_dir:
        window["-OUT-"].update(out_dir)
    if project_name:
        window["-PROJECT-"].update(project_name)
    return out_path


def resolve_saved_tts_wav_path(values, window) -> Path:
    text_path = Path(values.get("-TEXT_PATH-", "").strip()).expanduser()
    out_dir = values.get("-OUT-", "").strip()
    if not out_dir:
        out_dir = str(text_path.parent)
        window["-OUT-"].update(out_dir)
    return Path(out_dir).expanduser() / f"{text_path.stem}.wav"


def sync_update_existing_controls(window, values) -> None:
    update_existing = bool(values.get("-UPDATE_EXISTING_FILE-", False))
    window["-OUT-"].update(disabled=update_existing)
    window["-BROWSE_OUT-"].update(disabled=update_existing)
    window["-UPDATE_WARNING-"].update(visible=update_existing)

    if not update_existing:
        return

    is_dir_mode = bool(values.get("-SRC_DIR-", False))
    if is_dir_mode:
        candidate = values.get("-INPUT_PATH-", "").strip()
    else:
        in_path = values.get("-INPUT_PATH-", "").strip()
        candidate = str(Path(in_path).expanduser().parent) if in_path else ""

    if candidate:
        window["-OUT-"].update(candidate)


def _validate_existing_subtitles_for_files(files: list[Path]) -> tuple[bool, str]:
    for video_file in files:
        if find_external_subtitle_for_video(video_file) is None:
            return False, missing_subtitles_error(video_file)
    return True, ""


def format_done_status(done_count: int, total_count: int) -> str:
    total = max(0, int(total_count or 0))
    done = max(0, min(int(done_count or 0), total))
    percent = 0 if total == 0 else round(done * 100 / total)
    return f"СДЕЛАНО: {done}/{total} ({percent}%)"


def handle_file_event(window, values, base_title: str, progress_state: dict) -> None:
    data = values["-FILE-"] or {}
    idx = data.get("idx", 0)
    total = data.get("total", 0)
    name = data.get("name", "")
    if total and idx:
        progress_state["done"] = max(0, int(idx) - 1)
        progress_state["total"] = int(total)
        window["-STATUS-"].update(format_done_status(progress_state["done"], progress_state["total"]))
        window["-PROJECT-"].update(name)
        window["-INPUT_PATH-"].update(name)
    else:
        window["-STATUS-"].update(format_done_status(progress_state.get("done", 0), progress_state.get("total", 0)))
    try:
        if total and idx:
            window.TKroot.title(f"{base_title} — {idx}/{total}: {name}")
        else:
            window.TKroot.title(f"{base_title} — {name}")
    except Exception:
        pass

def _prepare_folder_run(values, current_steps, video_exts, window):
    in_dir = values.get("-INPUT_PATH-", "").strip()
    ok, err = validate_input_path(in_dir, is_dir_mode=True)
    if not ok:
        sg.popup_error(err)
        return None, 0

    in_dir_p = Path(in_dir)
    recursive = bool(values.get("-RECURSIVE-", False))
    files = enumerate_input_files(in_dir_p, recursive=recursive, allowed_exts=video_exts)
    _emit_info(window, f"Directory scan: recursive={recursive}")
    _emit_info(window, f"Found {len(files)} files")
    if not files:
        sg.popup_error("No video files were found in the selected folder (*.mp4, *.mkv, *.mov, *.avi).")
        return None, 0

    if bool(values.get("-USE_EXISTING_SUBTITLES-", False)):
        ok_subs, err_subs = _validate_existing_subtitles_for_files(files)
        if not ok_subs:
            sg.popup_error(err_subs)
            return None, 0

    base_out = values["-OUT-"].strip()
    if not base_out:
        base_out = str(in_dir_p)
        window["-OUT-"].update(base_out)

    run_items = []
    for p in files:
        project_name = p.stem
        out_dir = str(Path(base_out) / project_name)

        v = dict(values)
        v["-INPUT_PATH-"] = str(p)
        v["-INPUT_MODE-"] = "file"
        v["-IN-"] = str(p)
        v["-PROJECT-"] = project_name
        v["-OUT-"] = out_dir
        v["-STEPS-"] = dict(current_steps)

        pipeline_file = Path(out_dir) / f"{project_name}.pipeline.yaml"
        pipeline_file.parent.mkdir(parents=True, exist_ok=True)
        if not pipeline_file.exists():
            shutil.copy2(TEMPLATE_PATH, pipeline_file)

        pipeline_path = save_pipeline_yaml(v, pipeline_file)
        display_name = str(p.relative_to(in_dir_p))
        run_items.append((display_name, ["run", str(pipeline_path)]))

    return run_items, len(run_items)


def _prepare_single_file_run(values, current_steps, window):
    input_path = values.get("-INPUT_PATH-", "").strip()
    ok, err = validate_input_path(input_path, is_dir_mode=False)
    if not ok:
        sg.popup_error(err)
        return None, 0

    if bool(values.get("-USE_EXISTING_SUBTITLES-", False)):
        ok_subs, err_subs = _validate_existing_subtitles_for_files([Path(input_path)])
        if not ok_subs:
            sg.popup_error(err_subs)
            return None, 0

    base_out = values["-OUT-"].strip()
    if not base_out:
        base_out = str(Path(input_path).parent)
        window["-OUT-"].update(base_out)

    base_project = values["-PROJECT-"].strip()
    project_name = base_project or Path(input_path).stem
    values["-PROJECT-"] = project_name
    out_dir = str(Path(base_out) / project_name)
    values["-OUT-"] = out_dir
    values["-STEPS-"] = dict(current_steps)
    values["-INPUT_MODE-"] = "file"
    values["-IN-"] = input_path

    pipeline_file = Path(out_dir) / f"{project_name}.pipeline.yaml"
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    if not pipeline_file.exists():
        shutil.copy2(TEMPLATE_PATH, pipeline_file)

    pipeline_path = save_pipeline_yaml(values, pipeline_file)
    args = ["run", str(pipeline_path)]
    return [(Path(input_path).name, args)], 1


def _run_audio_synthesis(values, voice_id: str, out_audio: str, saved_wav: str, window) -> None:
    text_file = Path(values.get("-TEXT_PATH-", "").strip()).expanduser().resolve()
    try:
        synthesize_text_to_wav(
            out_audio=Path(out_audio),
            text_file=text_file,
            voice=voice_id,
            use_gpu=bool(values.get("-GPU-", True)),
        )
        saved_file = ""
        if saved_wav:
            saved_path = Path(saved_wav).expanduser().resolve()
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            out_audio_path = Path(out_audio).expanduser().resolve()
            if saved_path != out_audio_path:
                shutil.copy2(out_audio_path, saved_path)
            saved_file = str(saved_path)
        window.write_event_value("-AUDIO_DONE-", {"ok": True, "file": out_audio, "saved_file": saved_file})
    except Exception as ex:
        window.write_event_value("-AUDIO_DONE-", {"ok": False, "error": str(ex)})


def handle_audio_start_event(values, window, voice_id: str) -> bool:
    text_path = values.get("-TEXT_PATH-", "").strip()
    ok, err = validate_text_file_path(text_path)
    if not ok:
        sg.popup_error(err)
        return False

    if not voice_id:
        msg = "No voice selected for audio synthesis."
        window["-AUDIO_ERROR-"].update(msg)
        sg.popup_error(msg)
        return False

    out_audio = resolve_audio_output_path(values, window)
    saved_wav = ""
    if bool(values.get("-SAVE_AUDIO_WAV-", False)):
        saved_wav = str(resolve_saved_tts_wav_path(values, window).resolve())
    window["-AUDIO_ERROR-"].update("")
    window["-LOGBOX-"].update("")
    window["-STATUS-"].update("Status: running audio synthesis")
    _emit_info(window, f"Audio synthesis started: {Path(text_path).name} -> {out_audio.name}")
    if saved_wav:
        _emit_info(window, f"Saved WAV will be: {Path(saved_wav).name}")
    threading.Thread(
        target=_run_audio_synthesis,
        args=(dict(values), voice_id, str(out_audio.resolve()), saved_wav, window),
        daemon=True,
    ).start()
    return True


def _validate_gui_language_pair(values, current_steps: dict) -> tuple[bool, str]:
    src_lang = normalize_language_code(values.get("-LANG_SRC-", "en"), default="en")
    tgt_lang = normalize_language_code(values.get("-LANG_DST-", "ru"), default="ru")
    validation_error = validate_translation_language_pair(
        src_lang,
        tgt_lang,
        translate_enabled=bool(current_steps.get("translate", False)),
        allow_legacy=True,
    )
    if validation_error:
        return False, validation_error
    return True, ""


def handle_start_event(values, current_steps, video_exts, window, voice_id_by_display, progress_state: dict):
    active_tab = values.get("-MAIN_TABS-", TAB_VIDEO)
    if active_tab == TAB_AUDIO:
        selected_display = values.get("-VOICE-", "")
        selected_voice_id = voice_id_by_display.get(selected_display, selected_display)
        return 1 if handle_audio_start_event(values, window, selected_voice_id) else 0

    lang_ok, lang_error = _validate_gui_language_pair(values, current_steps)
    if not lang_ok:
        sg.popup_error(lang_error)
        return 0

    if bool(values.get("-SRC_DIR-")):
        run_items, run_count = _prepare_folder_run(values, current_steps, video_exts, window)
        if not run_items:
            return 0
        window["-LOGBOX-"].update("")
        progress_state["done"] = 0
        progress_state["total"] = run_count
        window["-STATUS-"].update(format_done_status(0, run_count))
        threading.Thread(target=run_pipeline_sequence, args=(run_items, window), daemon=True).start()
        return run_count

    run_items, run_count = _prepare_single_file_run(values, current_steps, window)
    if not run_items:
        return 0
    window["-LOGBOX-"].update("")
    progress_state["done"] = 0
    progress_state["total"] = run_count
    window["-STATUS-"].update(format_done_status(0, run_count))
    threading.Thread(target=run_pipeline_sequence, args=(run_items, window), daemon=True).start()
    return run_count


def handle_log_event(window, values) -> None:
    raw = values["-LOG-"]
    if raw is not None:
        raw = raw.replace("\r", "\n")
        for line in raw.splitlines():
            if line:
                print_parsed_log(window, line)


def handle_done_event(window, values, base_title: str, last_run_count: int, progress_state: dict) -> None:
    exit_code = values["-DONE-"]
    try:
        window.TKroot.title(base_title)
    except Exception:
        pass

    if exit_code == 0 and last_run_count > 0:
        progress_state["done"] = last_run_count
        progress_state["total"] = last_run_count
        window["-STATUS-"].update(format_done_status(last_run_count, last_run_count))
    else:
        done = progress_state.get("done", 0)
        total = progress_state.get("total", last_run_count)
        window["-STATUS-"].update(f"{format_done_status(done, total)} | ERROR ({exit_code})")


def main():
    show_app_constants()
    sg.theme("SystemDefault")
    audio_mode_labels = tuple(MODE_VALUE_BY_DISPLAY.keys())
    cfg = None
    voices_load_error = ""
    try:
        cfg = load_pipeline_config_ex(pipeline_path)
        voice_infos = list_voices(cfg)
    except Exception as ex:
        error(f"[GUI] Voices loading error: {ex}\n")
        voice_infos = []
        voices_load_error = str(ex)

    voices = [v.display_name or v.id for v in voice_infos]
    voice_id_by_display = {v.display_name or v.id: v.id for v in voice_infos}
    current_voice = get_voice() if pipeline_path.exists() else ""
    current_translation_model_id = ""
    if cfg is not None:
        current_translation_model_id = str(cfg.translation.model_id or "").strip()
    if not current_translation_model_id:
        current_translation_model_id = str(
            ((BASE_CFG.get("translation") or {}).get("model_id", ""))
        ).strip()
    if not current_translation_model_id:
        base_src_lang, base_tgt_lang = _current_base_languages()
        for choice in build_model_choices(base_src_lang, base_tgt_lang):
            if choice.model_id and choice.enabled:
                current_translation_model_id = choice.model_id
                break

    current_steps = normalize_steps(BASE_CFG.get("steps"))
    current_src_lang, current_tgt_lang = _current_base_languages()
    src_language_choices = _language_choices(current_src_lang)
    tgt_language_choices = _language_choices(current_tgt_lang)
    base_output_cfg = BASE_CFG.get("output") or {}
    base_paths_cfg = BASE_CFG.get("paths") or {}
    default_input_mode, default_input_path = resolve_saved_input_state(BASE_CFG)
    default_is_dir = default_input_mode == "dir"
    default_update_existing = bool(base_output_cfg.get("update_existing_file", False))
    default_use_existing_subtitles = bool(BASE_CFG.get("use_existing_subtitles", False))
    default_mode_display = _mode_to_display(base_output_cfg.get("audio_update_mode", BASE_CFG.get("mode", "add")))
    default_out_dir = str(base_paths_cfg.get("out_dir") or "")
    video_exts = {".mp4", ".mkv", ".mov", ".avi"}
    video_file_types = (("Video files", "*.mp4;*.mkv;*.mov;*.avi"),)

    video_tab_layout = [
        [sg.Text("Source:"),
         sg.Radio("Single file", "SRCMODE", key="-SRC_FILE-", default=not default_is_dir, enable_events=True),
         sg.Radio("Folder", "SRCMODE", key="-SRC_DIR-", default=default_is_dir, enable_events=True)],
        [sg.Text("Input path:"),
         sg.Input(key="-INPUT_PATH-", expand_x=True, enable_events=True),
         sg.Button("Browse...", key="-BROWSE_INPUT-")],
        [sg.Checkbox(
            "Recursive (include subfolders)",
            key="-RECURSIVE-",
            default=False,
            disabled=True,
        )],
        [sg.Checkbox(
            "Update existing video file",
            key="-UPDATE_EXISTING_FILE-",
            default=default_update_existing,
            enable_events=True,
        )],
        [sg.Text(
            UPDATE_EXISTING_WARNING,
            key="-UPDATE_WARNING-",
            text_color="orange",
            visible=default_update_existing,
            expand_x=True,
        )],
        [sg.Text("Move to folder:"),
         sg.Input(key="-MOVE_TO_DIR-", expand_x=True, enable_events=True),
         sg.FolderBrowse("...", key="-BROWSE_MOVE_DIR-", target="-MOVE_TO_DIR-")],
        [sg.Checkbox("Use existing subtitles file", key="-USE_EXISTING_SUBTITLES-", default=default_use_existing_subtitles)],
        [sg.Text("Audio track update mode:"),
         sg.Combo(
             values=audio_mode_labels,
             key="-MODES-",
             readonly=True,
             enable_events=True,
             default_value=default_mode_display,
             tooltip=AUDIO_MODE_TOOLTIP,
             size=(40, 1),
         )],
    ]

    audio_tab_layout = [
        [sg.Text("Text file:"),
         sg.Input(key="-TEXT_PATH-", expand_x=True, enable_events=True),
         sg.Button("Browse...", key="-BROWSE_TEXT-")],
        [sg.Checkbox("Save TTS to WAV file", key="-SAVE_AUDIO_WAV-", default=False)],
        [sg.Button("> Play", key="-PLAY_AUDIO-", disabled=(len(voices) == 0)),
         sg.Text("", key="-AUDIO_ERROR-", text_color="red", size=(50, 1), expand_x=True)],
    ]

    layout = [
        [sg.Text("Project name:"),
         sg.Input(key="-PROJECT-", expand_x=True)],
        [sg.Text("Language pair:"),
         sg.Combo(
             values=src_language_choices,
             key="-LANG_SRC-",
             readonly=True,
             enable_events=True,
             default_value=current_src_lang,
             size=(8, 1),
         ),
         sg.Text("->"),
         sg.Combo(
             values=tgt_language_choices,
             key="-LANG_DST-",
             readonly=True,
             enable_events=True,
             default_value=current_tgt_lang,
             size=(8, 1),
         ),
         sg.Text("", key="-LANG_SUMMARY-", expand_x=True)],
        [sg.TabGroup(
            [[
                sg.Tab("Video", video_tab_layout, key=TAB_VIDEO),
                sg.Tab("Audio", audio_tab_layout, key=TAB_AUDIO),
            ]],
            key="-MAIN_TABS-",
            enable_events=True,
            expand_x=True,
        )],
        [sg.Text("TTS voice:"),
         sg.Combo(
             values=voices,
             key="-VOICE-",
             readonly=True,
             enable_events=True,
             default_value=(current_voice if current_voice in voices else (voices[0] if voices else "")),
             disabled=(len(voices) == 0),
             size=(40, 1),
         ),
         sg.Button("> Preview", key="-PREVIEW-", disabled=(len(voices) == 0)),
         sg.Text("", key="-VOICE_ERROR-", text_color="red", size=(40, 1))],
        [sg.Text("Output folder:"),
         sg.Input(key="-OUT-", expand_x=True),
         sg.FolderBrowse("...", key="-BROWSE_OUT-")],
        [sg.Checkbox("Use GPU?", key="-GPU-")],
        [sg.Checkbox("Delete subtitles?", key="-SRT-")],
        [sg.Checkbox("Regenerate all steps (ignore cache)", key="-REBUILD-")],
        [sg.Checkbox("Cleanup temp files after success", key="-CLEANUP-")],
        [sg.Button("Steps...", key="-STEPS-"),
         sg.Text(steps_summary(current_steps), key="-STEPS_SUMMARY-", expand_x=True)],
        [sg.Button("Models...", key="-MODELS-"),
         sg.Text("", key="-MODEL_SUMMARY-", expand_x=True)],
        [sg.Text("Extra CLI arguments (optional):")],
        [sg.Input(key="-EXTRA-", expand_x=True)],
        [sg.Multiline(
            key="-LOGBOX-",
            size=(80, 20),
            autoscroll=True,
            disabled=True,
            font=("Consolas", 9),
            expand_x=True,
            expand_y=True,
        )],
        [sg.Input(key="-TRANSLATION_MODEL_ID-", visible=False, enable_events=False)],
        [sg.Button("Start", key="-START-"),
         sg.Button("Exit", key="-EXIT-"),
         sg.Text("СДЕЛАНО: 0/0 (0%)", key="-STATUS-")],
    ]

    window = sg.Window(
        "DubPipeline GUI",
        layout,
        resizable=True,
        finalize=True,
    )

    build_info = get_build_info()
    BASE_TITLE = f"DubPipeline GUI ({build_info})"
    preview = PreviewController()
    audio_play = AudioPlaybackController()
    progress_state = {"done": 0, "total": 0}

    # на всякий случай ещё раз говорим, что лог должен тянуться
    window["-LOGBOX-"].expand(expand_x=True, expand_y=True)
    window["-PROJECT-"].update(str(BASE_CFG.get("project_name") or ""))
    window["-INPUT_PATH-"].update(default_input_path)
    window["-TEXT_PATH-"].update(str((base_paths_cfg.get("input_text") or "")))
    window["-OUT-"].update(default_out_dir)
    window["-MOVE_TO_DIR-"].update((BASE_CFG.get("output") or {}).get("move_to_dir", ""))
    window["-GPU-"].update(True)
    window["-CLEANUP-"].update(True)
    window["-TRANSLATION_MODEL_ID-"].update(current_translation_model_id)
    lang_summary_text, lang_summary_color = _language_pair_summary(
        current_src_lang,
        current_tgt_lang,
        translate_enabled=bool(current_steps.get("translate", False)),
    )
    window["-LANG_SUMMARY-"].update(lang_summary_text, text_color=lang_summary_color)
    model_summary_text, model_summary_color = _translation_summary(
        current_translation_model_id,
        current_src_lang,
        current_tgt_lang,
    )
    window["-MODEL_SUMMARY-"].update(model_summary_text, text_color=model_summary_color)
    window["-STATUS-"].update(format_done_status(0, 0))
    _emit_info(window, f"Build: {build_info}")
    running = False
    last_run_count = 0
    set_source_mode(window, default_is_dir)
    sync_update_existing_controls(window, {
        "-UPDATE_EXISTING_FILE-": bool(window["-UPDATE_EXISTING_FILE-"].get()),
        "-SRC_DIR-": default_is_dir,
        "-INPUT_PATH-": window["-INPUT_PATH-"].get(),
    })
    if voices:
        _emit_info(window, f"Voices loaded: {len(voices)}")
    else:
        msg = voices_load_error or "Failed to load the voice list"
        window["-VOICE_ERROR-"].update(msg)
        _emit_info(window, f"Preview error: {msg}")

    while True:
        event, values = window.read(timeout=150)

        if event == "__TIMEOUT__":
            pass

        if preview._player is not None and preview._player.poll() is not None:
            preview._player = None
            if not preview.is_active():
                window["-PREVIEW-"].update("▶ Preview")

        if audio_play._player is not None and audio_play._player.poll() is not None:
            audio_play._player = None
            if not audio_play.is_active():
                window["-PLAY_AUDIO-"].update("▶ Play")

        if event == "-FILE-":
            handle_file_event(window, values, BASE_TITLE, progress_state)

        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        if event in ("-SRC_FILE-", "-SRC_DIR-"):
            set_source_mode(window, bool(values.get("-SRC_DIR-")))
            sync_update_existing_controls(window, values)

        # Автозаполнение Project name по имени файла
        if event == "-INPUT_PATH-" and values.get("-SRC_FILE-", True):
            path_str = values["-INPUT_PATH-"].strip()
            if path_str:
                p = Path(path_str)
                window["-PROJECT-"].update(p.stem)
            sync_update_existing_controls(window, values)

        if event == "-INPUT_PATH-" and values.get("-SRC_DIR-", False):
            sync_update_existing_controls(window, values)

        if event == "-TEXT_PATH-":
            path_str = values["-TEXT_PATH-"].strip()
            if path_str:
                p = Path(path_str)
                window["-PROJECT-"].update(p.stem)
                if not values.get("-OUT-", "").strip():
                    window["-OUT-"].update(str(p.parent))

        if event == "-BROWSE_INPUT-":
            selected_path = browse_input_path(
                is_dir_mode=bool(values.get("-SRC_DIR-", False)),
                video_file_types=video_file_types,
            )
            if selected_path:
                window["-INPUT_PATH-"].update(selected_path)
                if values.get("-SRC_FILE-", True):
                    window["-PROJECT-"].update(Path(selected_path).stem)
                sync_update_existing_controls(window, {
                    **values,
                    "-INPUT_PATH-": selected_path,
                })

        if event == "-BROWSE_TEXT-":
            selected_path = browse_text_file()
            if selected_path:
                window["-TEXT_PATH-"].update(selected_path)
                window["-PROJECT-"].update(Path(selected_path).stem)
                if not values.get("-OUT-", "").strip():
                    window["-OUT-"].update(str(Path(selected_path).parent))

        if event == "-UPDATE_EXISTING_FILE-":
            sync_update_existing_controls(window, values)

        # Выбор голоса (если надо, можно куда-то логировать)
        if event == "-VOICE-":
            current_voice = values["-VOICE-"]
            _emit_info(window, f"Voice selected: {current_voice}")

        if event in ("-LANG_SRC-", "-LANG_DST-"):
            current_src_lang = normalize_language_code(values.get("-LANG_SRC-", current_src_lang), default="en")
            current_tgt_lang = normalize_language_code(values.get("-LANG_DST-", current_tgt_lang), default="ru")
            lang_summary_text, lang_summary_color = _language_pair_summary(
                current_src_lang,
                current_tgt_lang,
                translate_enabled=bool(current_steps.get("translate", False)),
            )
            window["-LANG_SUMMARY-"].update(lang_summary_text, text_color=lang_summary_color)
            if current_translation_model_id:
                summary_text, summary_color = _translation_summary(
                    current_translation_model_id,
                    current_src_lang,
                    current_tgt_lang,
                )
                window["-MODEL_SUMMARY-"].update(summary_text, text_color=summary_color)
            _emit_info(window, f"Language pair selected: {current_src_lang} -> {current_tgt_lang}")

        if event == "-PREVIEW-":
            if preview.is_active():
                preview.stop()
                window["-PREVIEW-"].update("▶ Preview")
                _emit_info(window, "Preview stopped")
                continue

            selected_display = values.get("-VOICE-", "")
            selected_voice_id = voice_id_by_display.get(selected_display, selected_display)
            if not selected_voice_id:
                msg = "No voice selected for preview"
                window["-VOICE_ERROR-"].update(msg)
                _emit_info(window, f"Preview error: {msg}")
                continue

            preview_text = ((cfg.tts.preview_text if cfg is not None else "This is a preview of the selected voice.") or "").strip()
            if not preview_text:
                msg = "preview_text is empty in the configuration"
                window["-VOICE_ERROR-"].update(msg)
                _emit_info(window, f"Preview error: {msg}")
                continue

            window["-VOICE_ERROR-"].update("")
            window["-PREVIEW-"].update("⏹ Stop")
            _emit_info(window, f"Preview started: voice={selected_voice_id}")
            preview.start_preview(
                model_name=(cfg.tts.model_name if cfg is not None else "tts_models/multilingual/multi-dataset/xtts_v2"),
                voice_id=selected_voice_id,
                preview_text=preview_text,
                use_gpu=bool(values.get("-GPU-", cfg.usegpu if cfg is not None else True)),
                lang=current_tgt_lang,
                window=window,
            )

        if event == "-PREVIEW_READY-":
            payload = values["-PREVIEW_READY-"] or {}
            if payload.get("ok"):
                try:
                    preview.play_file(payload["file"])
                except Exception as ex:
                    msg = f"Preview playback error: {ex}"
                    window["-VOICE_ERROR-"].update(msg)
                    _emit_info(window, f"Preview error: {msg}")
                    window["-PREVIEW-"].update("▶ Preview")
            else:
                msg = payload.get("error", "Unknown preview error")
                window["-VOICE_ERROR-"].update(msg)
                _emit_info(window, f"Preview error: {msg}")
                window["-PREVIEW-"].update("▶ Preview")


        if event == "-PLAY_AUDIO-":
            if audio_play.is_active():
                audio_play.stop()
                window["-PLAY_AUDIO-"].update("▶ Play")
                _emit_info(window, "Audio playback stopped")
                continue

            selected_display = values.get("-VOICE-", "")
            selected_voice_id = voice_id_by_display.get(selected_display, selected_display)
            if not selected_voice_id:
                msg = "No voice selected for audio playback."
                window["-AUDIO_ERROR-"].update(msg)
                _emit_info(window, f"Audio play error: {msg}")
                continue

            ok, err = validate_text_file_path(values.get("-TEXT_PATH-", ""))
            if not ok:
                window["-AUDIO_ERROR-"].update(err)
                _emit_info(window, f"Audio play error: {err}")
                continue

            window["-AUDIO_ERROR-"].update("")
            window["-PLAY_AUDIO-"].update("■ Stop")
            _emit_info(window, f"Audio playback synthesis started: {Path(values.get('-TEXT_PATH-', '')).name}")
            audio_play.start(
                voice_id=selected_voice_id,
                text_file=values.get("-TEXT_PATH-", "").strip(),
                use_gpu=bool(values.get("-GPU-", cfg.usegpu if cfg is not None else True)),
                window=window,
            )

        if event == "-AUDIO_PLAY_READY-":
            payload = values["-AUDIO_PLAY_READY-"] or {}
            if payload.get("ok"):
                try:
                    audio_play.play_file(payload["file"])
                except Exception as ex:
                    msg = f"Audio playback error: {ex}"
                    window["-AUDIO_ERROR-"].update(msg)
                    _emit_info(window, f"Audio play error: {msg}")
                    window["-PLAY_AUDIO-"].update("▶ Play")
            else:
                msg = payload.get("error", "Unknown audio playback error")
                window["-AUDIO_ERROR-"].update(msg)
                _emit_info(window, f"Audio play error: {msg}")
                window["-PLAY_AUDIO-"].update("▶ Play")

        if event == "-MOVE_TO_DIR-":
            persist_move_to_dir(values.get("-MOVE_TO_DIR-", "").strip())

        if event == "-START-":
            if running:
                sg.popup("A process is already running. Please wait for it to finish.", title="Info")
                continue
            if bool(current_steps.get("translate", False)):
                ready_model_id = ensure_translation_model_ready_for_start(
                    window,
                    current_translation_model_id or values.get("-TRANSLATION_MODEL_ID-", ""),
                    src_lang=current_src_lang,
                    tgt_lang=current_tgt_lang,
                )
                if not ready_model_id:
                    continue
                current_translation_model_id = ready_model_id
                try:
                    synced, sync_message = sync_translation_start_values(
                        values,
                        ready_model_id,
                        src_lang=current_src_lang,
                        tgt_lang=current_tgt_lang,
                    )
                except Exception as exc:
                    synced = False
                    sync_message = f"Translation model cannot be checked: {exc}"
                if not synced:
                    sg.popup_error(sync_message, title="Translation model required", keep_on_top=True)
                    _emit_info(window, sync_message)
                    continue
                window["-TRANSLATION_MODEL_ID-"].update(values["-TRANSLATION_MODEL_ID-"])
                _emit_info(window, f"Translation model ready for start: {sync_message}")
            run_count = handle_start_event(values, current_steps, video_exts, window, voice_id_by_display, progress_state)
            if run_count:
                last_run_count = run_count
                running = True

        if event == "-LOG-":
            handle_log_event(window, values)

        if event == "-DONE-":
            running = False
            handle_done_event(window, values, BASE_TITLE, last_run_count, progress_state)

        if event == "-AUDIO_DONE-":
            running = False
            payload = values["-AUDIO_DONE-"] or {}
            if payload.get("ok"):
                out_file = payload.get("file", "")
                saved_file = payload.get("saved_file", "")
                window["-STATUS-"].update("Status: ok")
                _emit_info(window, f"Audio synthesis completed: {out_file}")
                if saved_file:
                    _emit_info(window, f"Saved WAV file: {saved_file}")
            else:
                msg = payload.get("error", "Unknown audio synthesis error")
                window["-STATUS-"].update("Status: error")
                window["-AUDIO_ERROR-"].update(msg)
                _emit_info(window, f"Audio synthesis error: {msg}")

        if event == "-MODELS-":
            selected_model_id = show_models_modal(
                window,
                current_translation_model_id,
                src_lang=current_src_lang,
                tgt_lang=current_tgt_lang,
            )
            if selected_model_id:
                current_translation_model_id = selected_model_id
                window["-TRANSLATION_MODEL_ID-"].update(current_translation_model_id)
                persist_languages(current_src_lang, current_tgt_lang)
                persist_translation_model(
                    current_translation_model_id,
                    src_lang=current_src_lang,
                    tgt_lang=current_tgt_lang,
                )
                summary_text, summary_color = _translation_summary(
                    current_translation_model_id,
                    current_src_lang,
                    current_tgt_lang,
                )
                window["-MODEL_SUMMARY-"].update(summary_text, text_color=summary_color)
                spec = get_model_spec(current_translation_model_id)
                _emit_info(window, f"Translation model selected: {spec.label} [{spec.id}]")

        if event == "-STEPS-":
            current_steps = show_steps_modal(window, current_steps)
            window["-STEPS_SUMMARY-"].update(steps_summary(current_steps))
            lang_summary_text, lang_summary_color = _language_pair_summary(
                current_src_lang,
                current_tgt_lang,
                translate_enabled=bool(current_steps.get("translate", False)),
            )
            window["-LANG_SUMMARY-"].update(lang_summary_text, text_color=lang_summary_color)

    preview.stop()
    audio_play.stop()
    window.close()


if __name__ == "__main__":
    main()
