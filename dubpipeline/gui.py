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
from dubpipeline.steps.step_tts import list_voices, synthesize_preview_text

from dubpipeline.utils.logging import info, error
from dubpipeline.input_discovery import enumerate_input_files, source_mode_disabled_map


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


def persist_move_to_dir(path: str) -> None:
    BASE_CFG.setdefault("output", {})
    BASE_CFG["output"]["move_to_dir"] = path
    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(BASE_CFG, f, allow_unicode=True, sort_keys=False)


def set_source_mode(window, is_dir: bool) -> None:
    for key, disabled in source_mode_disabled_map(is_dir=is_dir).items():
        window[key].update(disabled=disabled)




def sync_update_existing_controls(window, values) -> None:
    update_existing = bool(values.get("-UPDATE_EXISTING_FILE-", False))
    window["-OUT-"].update(disabled=update_existing)
    window["-BROWSE_OUT-"].update(disabled=update_existing)
    window["-UPDATE_WARNING-"].update(visible=update_existing)

    if not update_existing:
        return

    is_dir_mode = bool(values.get("-SRC_DIR-", False))
    if is_dir_mode:
        candidate = values.get("-IN_DIR-", "").strip()
    else:
        in_path = values.get("-IN-", "").strip()
        candidate = str(Path(in_path).expanduser().parent) if in_path else ""

    if candidate:
        window["-OUT-"].update(candidate)


def handle_file_event(window, values, base_title: str) -> None:
    data = values["-FILE-"] or {}
    idx = data.get("idx", 0)
    total = data.get("total", 0)
    name = data.get("name", "")
    if total and idx:
        window["-STATUS-"].update(f"Статус: running ({idx}/{total}) — {name}")
        window["-PROJECT-"].update(name)
        window["-IN-"].update(name)
    else:
        window["-STATUS-"].update(f"Статус: running — {name}")
    try:
        if total and idx:
            window.TKroot.title(f"{base_title} — {idx}/{total}: {name}")
        else:
            window.TKroot.title(f"{base_title} — {name}")
    except Exception:
        pass

def _prepare_folder_run(values, current_steps, video_exts, window):
    in_dir = values.get("-IN_DIR-", "").strip()
    if not in_dir or not os.path.isdir(in_dir):
        sg.popup_error("Укажите корректную папку с видео.")
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
    input_path = values["-IN-"].strip()
    if not input_path or not os.path.isfile(input_path):
        sg.popup_error("Укажите корректный путь к входному видео.")
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
        window["-STATUS-"].update(f"Статус: running ({run_count} files)")
        threading.Thread(target=run_pipeline_sequence, args=(run_items, window), daemon=True).start()
        return run_count

    run_items, run_count = _prepare_single_file_run(values, current_steps, window)
    if not run_items:
        return 0
    window["-LOGBOX-"].update("")
    window["-STATUS-"].update("Статус: running")
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
        window["-STATUS-"].update(f"Статус: {status} ({last_run_count} files)")
    else:
        window["-STATUS-"].update(f"Статус: {status}")


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
    current_steps = normalize_steps(BASE_CFG.get("steps"))
    base_output_cfg = BASE_CFG.get("output") or {}
    default_update_existing = bool(base_output_cfg.get("update_existing_file", False))
    default_mode_display = _mode_to_display(base_output_cfg.get("audio_update_mode", BASE_CFG.get("mode", "add")))

    video_exts = {".mp4", ".mkv", ".mov", ".avi"}

    layout = [
        [sg.Text("Источник:"),
         sg.Radio("Один файл", "SRCMODE", key="-SRC_FILE-", default=True, enable_events=True),
         sg.Radio("Папка", "SRCMODE", key="-SRC_DIR-", enable_events=True)],

        [sg.Text("Project name:"),
         sg.Input(key="-PROJECT-", expand_x=True)],

        [sg.Text("Входное видео:"),
         sg.Input(key="-IN-", expand_x=True, enable_events=True),
         sg.FileBrowse("...", key="-BROWSE_FILE-", file_types=(("Видео файлы", "*.mp4;*.mkv;*.mov;*.avi"),))],

        [sg.Text("Папка с видео:"),
         sg.Input(key="-IN_DIR-", expand_x=True, enable_events=True, disabled=True),
         sg.FolderBrowse("...", key="-BROWSE_DIR-", target="-IN_DIR-", disabled=True)],

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

        [sg.Button("Старт", key="-START-"),
         sg.Button("Выход", key="-EXIT-"),
         sg.Text("Статус: idle", key="-STATUS-")],
    ]

    window = sg.Window(
        "DubPipeline GUI",
        layout,
        resizable=True,
        finalize=True,
    )

    BASE_TITLE = "DubPipeline GUI"
    preview = PreviewController()

    # на всякий случай ещё раз говорим, что лог должен тянуться
    window["-LOGBOX-"].expand(expand_x=True, expand_y=True)
    window["-PROJECT-"].update("2")
    window["-IN-"].update("J:/Projects/!!!AI/DubPipeline/tests/data/2.mp4")
    window["-OUT-"].update("J:/Projects/!!!AI/DubPipeline/tests/output")
    window["-MOVE_TO_DIR-"].update((BASE_CFG.get("output") or {}).get("move_to_dir", ""))
    window["-GPU-"].update(True)
    window["-CLEANUP-"].update(True)
    running = False
    last_run_count = 0
    set_source_mode(window, False)
    sync_update_existing_controls(window, {
        "-UPDATE_EXISTING_FILE-": bool(window["-UPDATE_EXISTING_FILE-"].get()),
        "-SRC_DIR-": False,
        "-IN-": window["-IN-"].get(),
        "-IN_DIR-": window["-IN_DIR-"].get(),
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
        if event == "-IN-" and values.get("-SRC_FILE-", True):
            path_str = values["-IN-"].strip()
            if path_str:
                p = Path(path_str)
                window["-PROJECT-"].update(p.stem)
            sync_update_existing_controls(window, values)

        if event == "-IN_DIR-":
            sync_update_existing_controls(window, values)

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

        if event == "-STEPS-":
            current_steps = show_steps_modal(window, current_steps)
            window["-STEPS_SUMMARY-"].update(steps_summary(current_steps))

    preview.stop()
    window.close()


if __name__ == "__main__":
    main()
