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

from dubpipeline.config import save_pipeline_yaml, get_voice, normalize_audio_update_mode, load_pipeline_config_ex, pipeline_path
from dubpipeline.models.catalog import (
    build_model_choices,
    get_model_spec,
    get_model_status,
    legacy_translate_backend_for_model,
)
from dubpipeline.steps.step_tts import list_voices, synthesize_preview_text

from dubpipeline.utils.build_info import get_build_info
from dubpipeline.utils.logging import info, error
from dubpipeline.input_discovery import enumerate_input_files, source_mode_disabled_map
from dubpipeline.input_mode import resolve_saved_input_state, validate_input_path


def _preview_worker_target(q: Queue, *, model_name: str, voice_id: str, preview_text: str, out_file: str, use_gpu: bool) -> None:
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
        )
        q.put({"ok": True, "file": str(out_path)})
    except Exception as ex:
        q.put({"ok": False, "error": str(ex)})

STEP_LABELS = {
    "extract_audio": "Извлечение аудио",
    "asr_whisperx": "ASR (WhisperX)",
    "translate": "Перевод",
    "tts": "TTS",
    "align": "Синхронизация",
    "merge": "Сборка видео",
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

UPDATE_EXISTING_WARNING = (
    "Исходный файл будет заменён при успехе. "
    "При ошибке исходник останется без изменений."
)

AUDIO_MODE_TOOLTIP = (
    "Add: сохраняет исходные дорожки и добавляет дубляж в конец.\n"
    "Overwrite: сохраняет только новую дорожку дубляжа.\n"
    "Overwrite+Reorder: ставит дубляж первой дорожкой, затем оставшиеся."
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

    def start_preview(self, *, model_name: str, voice_id: str, preview_text: str, use_gpu: bool, window) -> None:
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


def run_pipeline(args_list, window):
    """
    args_list, например: ["run", "D:/Projects/DubPipeline/out/myproj.pipeline.yaml"]
    Работает в отдельном потоке и шлёт события -LOG- и -DONE- в окно.
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
    """Запускает несколько пайплайнов подряд в одном потоке.

    run_items: list[tuple[label:str, args_list:list[str]]]
    Шлёт события -LOG- и единый -DONE- в конце.
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
                _emit_info(window, f"Остановка: пайплайн завершился с кодом {exit_code}")
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
        return "Шаги: ничего не выбрано"
    return f"Шаги: {', '.join(enabled)}"


def show_steps_modal(parent, current_steps: dict) -> dict:
    steps = normalize_steps(current_steps)
    layout = [[sg.Text("Выберите шаги генерации:")]]
    for key in STEP_LABELS:
        layout.append([sg.Checkbox(STEP_LABELS[key], key=f"-STEP-{key}-", default=steps[key])])
    layout.append([sg.Button("Сохранить", key="-SAVE-"), sg.Button("Отмена", key="-CANCEL-")])

    window = sg.Window(
        "Шаги генерации",
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


def _translation_summary(model_id: str) -> tuple[str, str]:
    try:
        spec = get_model_spec(model_id)
        status = get_model_status(model_id)
    except Exception:
        return "Machine Translation: unknown model", "firebrick"

    if status.enabled:
        return f"Machine Translation: {spec.label}", "darkgreen"
    if status.reason == "not supported yet":
        return f"Machine Translation: {spec.label} (not supported yet)", "darkorange"
    return f"Machine Translation: {spec.label} (not installed)", "firebrick"


def persist_translation_model(model_id: str) -> None:
    spec = get_model_spec(model_id)
    BASE_CFG.setdefault("translation", {})
    BASE_CFG["translation"]["model_id"] = spec.id
    BASE_CFG["translation"]["backend"] = spec.backend
    BASE_CFG["translation"]["model_ref"] = spec.model_ref

    # Keep legacy keys for compatibility with older scripts/tools.
    BASE_CFG.setdefault("translate", {})
    BASE_CFG["translate"]["backend"] = legacy_translate_backend_for_model(spec)
    if BASE_CFG["translate"]["backend"] == "hf":
        BASE_CFG["translate"]["hf_model"] = spec.model_ref

    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(BASE_CFG, f, allow_unicode=True, sort_keys=False)


def show_models_modal(parent, current_model_id: str) -> str | None:
    choices = build_model_choices()
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
            sg.Text(
                "",
                key="-MODELS_STATUS-",
                text_color="black",
                size=(72, 3),
                expand_x=True,
            )
        ],
        [sg.Button("Apply", key="-MODELS_APPLY-"), sg.Button("Cancel", key="-MODELS_CANCEL-")],
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

    def _update_state(display_value: str) -> None:
        nonlocal selected_model_id
        choice = by_display.get(display_value)
        if choice is None:
            window["-MODELS_STATUS-"].update(
                "Unknown selection. Please choose a model.",
                text_color="firebrick",
            )
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
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        if not choice.enabled:
            reason = choice.reason or "not installed"
            if reason == "not supported yet":
                msg = "This model is planned but not supported yet in this build."
            else:
                msg = "This model is not installed locally. Choose another model."
            window["-MODELS_STATUS-"].update(msg, text_color="firebrick")
            window["-MODELS_APPLY-"].update(disabled=True)
            selected_model_id = None
            return

        window["-MODELS_STATUS-"].update("Model is installed and ready.", text_color="darkgreen")
        window["-MODELS_APPLY-"].update(disabled=False)
        selected_model_id = choice.model_id

    _update_state(selected_display)

    result: str | None = None
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-MODELS_CANCEL-"):
            result = None
            break
        if event == "-MODELS_MT-":
            _update_state(values.get("-MODELS_MT-", ""))
        if event == "-MODELS_APPLY-":
            if selected_model_id:
                result = selected_model_id
                break

    window.close()
    return result


def persist_move_to_dir(path: str) -> None:
    BASE_CFG.setdefault("output", {})
    BASE_CFG["output"]["move_to_dir"] = path
    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(BASE_CFG, f, allow_unicode=True, sort_keys=False)


def set_source_mode(window, is_dir: bool) -> None:
    for key, disabled in source_mode_disabled_map(is_dir=is_dir).items():
        window[key].update(disabled=disabled)


def browse_input_path(*, is_dir_mode: bool, video_file_types):
    if is_dir_mode:
        return sg.popup_get_folder("Выберите папку с видео")
    return sg.popup_get_file("Выберите видеофайл", file_types=video_file_types)




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


def handle_file_event(window, values, base_title: str) -> None:
    data = values["-FILE-"] or {}
    idx = data.get("idx", 0)
    total = data.get("total", 0)
    name = data.get("name", "")
    if total and idx:
        window["-STATUS-"].update(f"Status: running ({idx}/{total}) - {name}")
        window["-PROJECT-"].update(name)
        window["-INPUT_PATH-"].update(name)
    else:
        window["-STATUS-"].update(f"Status: running - {name}")
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
        sg.popup_error("В выбранной папке нет видео-файлов (*.mp4, *.mkv, *.mov, *.avi).")
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


def handle_start_event(values, current_steps, video_exts, window):
    if bool(values.get("-SRC_DIR-")):
        run_items, run_count = _prepare_folder_run(values, current_steps, video_exts, window)
        if not run_items:
            return 0
        window["-LOGBOX-"].update("")
        window["-STATUS-"].update(f"Status: running ({run_count} files)")
        threading.Thread(target=run_pipeline_sequence, args=(run_items, window), daemon=True).start()
        return run_count

    run_items, run_count = _prepare_single_file_run(values, current_steps, window)
    if not run_items:
        return 0
    window["-LOGBOX-"].update("")
    window["-STATUS-"].update("Status: running")
    threading.Thread(target=run_pipeline_sequence, args=(run_items, window), daemon=True).start()
    return run_count


def handle_log_event(window, values) -> None:
    raw = values["-LOG-"]
    if raw is not None:
        raw = raw.replace("\r", "\n")
        for line in raw.splitlines():
            if line:
                print_parsed_log(window, line)


def handle_done_event(window, values, base_title: str, last_run_count: int) -> None:
    exit_code = values["-DONE-"]
    try:
        window.TKroot.title(base_title)
    except Exception:
        pass

    status = "ok" if exit_code == 0 else f"error (code {exit_code})"
    if last_run_count > 1 and exit_code == 0:
        window["-STATUS-"].update(f"Status: {status} ({last_run_count} files)")
    else:
        window["-STATUS-"].update(f"Status: {status}")


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
        for choice in build_model_choices():
            if choice.model_id and choice.enabled:
                current_translation_model_id = choice.model_id
                break

    current_steps = normalize_steps(BASE_CFG.get("steps"))
    base_output_cfg = BASE_CFG.get("output") or {}
    base_paths_cfg = BASE_CFG.get("paths") or {}
    default_input_mode, default_input_path = resolve_saved_input_state(BASE_CFG)
    default_is_dir = default_input_mode == "dir"
    default_update_existing = bool(base_output_cfg.get("update_existing_file", False))
    default_mode_display = _mode_to_display(base_output_cfg.get("audio_update_mode", BASE_CFG.get("mode", "add")))
    default_out_dir = str(base_paths_cfg.get("out_dir") or "")

    video_exts = {".mp4", ".mkv", ".mov", ".avi"}
    video_file_types = (("Видео файлы", "*.mp4;*.mkv;*.mov;*.avi"),)

    layout = [
        [sg.Text("Источник:"),
         sg.Radio("Один файл", "SRCMODE", key="-SRC_FILE-", default=not default_is_dir, enable_events=True),
         sg.Radio("Папка", "SRCMODE", key="-SRC_DIR-", default=default_is_dir, enable_events=True)],

        [sg.Text("Project name:"),
         sg.Input(key="-PROJECT-", expand_x=True)],

        [sg.Text("Input path:"),
         sg.Input(key="-INPUT_PATH-", expand_x=True, enable_events=True),
         sg.Button("Browse...", key="-BROWSE_INPUT-")],

        [sg.Checkbox(
            "Рекурсивно (включая подпапки)",
            key="-RECURSIVE-",
            default=False,
            disabled=True,
        )],

        [sg.Text("Голос TTS:"),
         sg.Combo(
             values=voices,
             key="-VOICE-",
             readonly=True,
             enable_events=True,
             default_value=(current_voice if current_voice in voices else (voices[0] if voices else "")),
             disabled=(len(voices) == 0),
             size=(40, 1),
         ),
         sg.Button("▶ Preview", key="-PREVIEW-", disabled=(len(voices) == 0)),
         sg.Text("", key="-VOICE_ERROR-", text_color="red", size=(40, 1))],

        [sg.Text("Выходная папка:"),
         sg.Input(key="-OUT-", expand_x=True),
         sg.FolderBrowse("...", key="-BROWSE_OUT-")],

        [sg.Checkbox(
            "Обновлять существующий видеофайл",
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

        [sg.Text("Переместить в папку:"),
         sg.Input(key="-MOVE_TO_DIR-", expand_x=True, enable_events=True),
         sg.FolderBrowse("...", key="-BROWSE_MOVE_DIR-", target="-MOVE_TO_DIR-")],

        [sg.Checkbox("Использовать GPU?", key="-GPU-")],
        [sg.Checkbox("Удалять субтитры?", key="-SRT-")],
        [sg.Checkbox("Перегенерировать все шаги (игнорировать кэш)", key="-REBUILD-")],
        [sg.Checkbox("Убирать мусор (удалять временные файлы после успеха)", key="-CLEANUP-")],
        [sg.Button("Шаги генерации...", key="-STEPS-"),
         sg.Text(steps_summary(current_steps), key="-STEPS_SUMMARY-", expand_x=True)],
        [sg.Button("Models...", key="-MODELS-"),
         sg.Text("", key="-MODEL_SUMMARY-", expand_x=True)],
        [sg.Text("Режим добавления аудио:"),
         sg.Combo(
             values=audio_mode_labels,
             key="-MODES-",
             readonly=True,
             enable_events=True,
             default_value=default_mode_display,
             tooltip=AUDIO_MODE_TOOLTIP,
             size=(40, 1),
         )],

        [sg.Text("Доп. аргументы CLI (опционально):")],
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

        [sg.Button("Старт", key="-START-"),
         sg.Button("Выход", key="-EXIT-"),
         sg.Text("Status: idle", key="-STATUS-")],
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

    # на всякий случай ещё раз говорим, что лог должен тянуться
    window["-LOGBOX-"].expand(expand_x=True, expand_y=True)
    window["-PROJECT-"].update(str(BASE_CFG.get("project_name") or ""))
    window["-INPUT_PATH-"].update(default_input_path)
    window["-OUT-"].update(default_out_dir)
    window["-MOVE_TO_DIR-"].update((BASE_CFG.get("output") or {}).get("move_to_dir", ""))
    window["-GPU-"].update(True)
    window["-CLEANUP-"].update(True)
    window["-TRANSLATION_MODEL_ID-"].update(current_translation_model_id)
    model_summary_text, model_summary_color = _translation_summary(current_translation_model_id)
    window["-MODEL_SUMMARY-"].update(model_summary_text, text_color=model_summary_color)
    window["-STATUS-"].update(f"Status: idle | build: {build_info}")
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
        msg = voices_load_error or "Не удалось загрузить список голосов"
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

        if event == "-FILE-":
            handle_file_event(window, values, BASE_TITLE)

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

        if event == "-UPDATE_EXISTING_FILE-":
            sync_update_existing_controls(window, values)

        # Выбор голоса (если надо, можно куда-то логировать)
        if event == "-VOICE-":
            current_voice = values["-VOICE-"]
            _emit_info(window, f"Voice selected: {current_voice}")

        if event == "-PREVIEW-":
            if preview.is_active():
                preview.stop()
                window["-PREVIEW-"].update("▶ Preview")
                _emit_info(window, "Preview stopped")
                continue

            selected_display = values.get("-VOICE-", "")
            selected_voice_id = voice_id_by_display.get(selected_display, selected_display)
            if not selected_voice_id:
                msg = "Не выбран голос для preview"
                window["-VOICE_ERROR-"].update(msg)
                _emit_info(window, f"Preview error: {msg}")
                continue

            preview_text = ((cfg.tts.preview_text if cfg is not None else "Это тестовое воспроизведение выбранного голоса.") or "").strip()
            if not preview_text:
                msg = "preview_text пустой в конфигурации"
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
                window=window,
            )

        if event == "-PREVIEW_READY-":
            payload = values["-PREVIEW_READY-"] or {}
            if payload.get("ok"):
                try:
                    preview.play_file(payload["file"])
                except Exception as ex:
                    msg = f"Ошибка воспроизведения preview: {ex}"
                    window["-VOICE_ERROR-"].update(msg)
                    _emit_info(window, f"Preview error: {msg}")
                    window["-PREVIEW-"].update("▶ Preview")
            else:
                msg = payload.get("error", "Unknown preview error")
                window["-VOICE_ERROR-"].update(msg)
                _emit_info(window, f"Preview error: {msg}")
                window["-PREVIEW-"].update("▶ Preview")


        if event == "-MOVE_TO_DIR-":
            persist_move_to_dir(values.get("-MOVE_TO_DIR-", "").strip())

        if event == "-START-":
            if running:
                sg.popup("Процесс уже запущен, дождитесь окончания.", title="Инфо")
                continue
            run_count = handle_start_event(values, current_steps, video_exts, window)
            if run_count:
                last_run_count = run_count
                running = True

        if event == "-LOG-":
            handle_log_event(window, values)

        if event == "-DONE-":
            running = False
            handle_done_event(window, values, BASE_TITLE, last_run_count)

        if event == "-MODELS-":
            selected_model_id = show_models_modal(window, current_translation_model_id)
            if selected_model_id:
                current_translation_model_id = selected_model_id
                window["-TRANSLATION_MODEL_ID-"].update(current_translation_model_id)
                persist_translation_model(current_translation_model_id)
                summary_text, summary_color = _translation_summary(current_translation_model_id)
                window["-MODEL_SUMMARY-"].update(summary_text, text_color=summary_color)
                spec = get_model_spec(current_translation_model_id)
                _emit_info(window, f"Translation model selected: {spec.label} [{spec.id}]")

        if event == "-STEPS-":
            current_steps = show_steps_modal(window, current_steps)
            window["-STEPS_SUMMARY-"].update(steps_summary(current_steps))

    preview.stop()
    window.close()


if __name__ == "__main__":
    main()
