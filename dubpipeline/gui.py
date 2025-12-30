import threading
import subprocess
import sys
import os
from pathlib import Path

import FreeSimpleGUI as sg
import yaml
import re
import time
import shutil

from dubpipeline.config import save_pipeline_yaml, get_voice
from dubpipeline.steps.step_tts import getVoices

from dubpipeline.utils.logging import  info


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


def main():
    sg.theme("SystemDefault")
    mpeg_modes = ("Замена", "Добавление")
    voices = getVoices()
    current_voice = get_voice()

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

        [sg.Text("Голос TTS:"),
         sg.Combo(
             values=voices,
             key="-VOICE-",
             readonly=True,
             enable_events=True,
             default_value=current_voice if current_voice != "" else voices[0] ,
             size=(40, 1),
         )],

        [sg.Text("Выходная папка:"),
         sg.Input(key="-OUT-", expand_x=True),
         sg.FolderBrowse("...")],

        [sg.Checkbox("Использовать GPU?", key="-GPU-")],
        [sg.Checkbox("Удалять субтитры?", key="-SRT-")],
        [sg.Checkbox("Перегенерировать все шаги (игнорировать кэш)", key="-REBUILD-")],
        [sg.Text("Режим добавления аудио:"),
         sg.Combo(
             values=mpeg_modes,
             key="-MODES-",
             readonly=True,
             enable_events=True,
             default_value="Добавление",
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

    # на всякий случай ещё раз говорим, что лог должен тянуться
    window["-LOGBOX-"].expand(expand_x=True, expand_y=True)
    window["-PROJECT-"].update("2")
    window["-IN-"].update("J:/Projects/!!!AI/DubPipeline/tests/data/2.mp4")
    window["-OUT-"].update("J:/Projects/!!!AI/DubPipeline/tests/output")
    window["-GPU-"].update(True)
    running = False
    last_run_count = 0

    def set_source_mode(is_dir: bool) -> None:
        window["-IN-"].update(disabled=is_dir)
        window["-BROWSE_FILE-"].update(disabled=is_dir)
        window["-IN_DIR-"].update(disabled=not is_dir)
        window["-BROWSE_DIR-"].update(disabled=not is_dir)

    set_source_mode(False)

    while True:
        event, values = window.read()

        if event == "-FILE-":
            data = values["-FILE-"] or {}
            idx = data.get("idx", 0)
            total = data.get("total", 0)
            name = data.get("name", "")

            # Строка статуса (и для одиночного файла, и для папки)
            if total and idx:
                window["-STATUS-"].update(f"Статус: running ({idx}/{total}) — {name}")
            else:
                window["-STATUS-"].update(f"Статус: running — {name}")

            # Заголовок окна
            try:
                if total and idx:
                    window.TKroot.title(f"{BASE_TITLE} — {idx}/{total}: {name}")
                else:
                    window.TKroot.title(f"{BASE_TITLE} — {name}")
            except Exception:
                pass

        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        if event in ("-SRC_FILE-", "-SRC_DIR-"):
            set_source_mode(bool(values.get("-SRC_DIR-")))

        # Автозаполнение Project name по имени файла
        if event == "-IN-" and values.get("-SRC_FILE-", True):
            path_str = values["-IN-"].strip()
            if path_str:
                p = Path(path_str)
                window["-PROJECT-"].update(p.stem)

        # Выбор голоса (если надо, можно куда-то логировать)
        if event == "-VOICE-":
            current_voice = values["-VOICE-"]
            _emit_info(window, f"Выбран голос: {current_voice}")

        if event == "-START-":
            if running:
                sg.popup("Процесс уже запущен, дождитесь окончания.", title="Инфо")
                continue

            is_dir_mode = bool(values.get("-SRC_DIR-"))
            base_out = values["-OUT-"].strip()
            base_project = values["-PROJECT-"].strip()

            # --- Режим: папка ---
            if is_dir_mode:
                in_dir = values.get("-IN_DIR-", "").strip()
                if not in_dir or not os.path.isdir(in_dir):
                    sg.popup_error("Укажите корректную папку с видео.")
                    continue

                in_dir_p = Path(in_dir)
                files = sorted(
                    [p for p in in_dir_p.iterdir() if p.is_file() and p.suffix.lower() in video_exts],
                    key=lambda p: p.name.lower(),
                )
                if not files:
                    sg.popup_error("В выбранной папке нет видео-файлов (*.mp4, *.mkv, *.mov, *.avi).")
                    continue

                # Если выход не указан — берём саму папку
                if not base_out:
                    base_out = str(in_dir_p)
                    window["-OUT-"].update(base_out)

                run_items = []
                for p in files:
                    project_name = f"{base_project}_{p.stem}" if base_project else p.stem
                    out_dir = str(Path(base_out) / project_name)  # под-папка, чтобы не было коллизий

                    v = dict(values)
                    v["-IN-"] = str(p)
                    v["-PROJECT-"] = project_name
                    v["-OUT-"] = out_dir

                    pipeline_file = Path(out_dir) / f"{project_name}.pipeline.yaml"
                    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
                    if not pipeline_file.exists():
                        shutil.copy2(TEMPLATE_PATH, pipeline_file)

                    pipeline_path = save_pipeline_yaml(v, pipeline_file)
                    run_items.append((p.name, ["run", str(pipeline_path)]))

                last_run_count = len(run_items)
                window["-LOGBOX-"].update("")
                window["-STATUS-"].update(f"Статус: running ({last_run_count} files)")
                running = True
                threading.Thread(
                    target=run_pipeline_sequence, args=(run_items, window), daemon=True
                ).start()
                continue

            # --- Режим: один файл ---
            input_path = values["-IN-"].strip()
            if not input_path or not os.path.isfile(input_path):
                sg.popup_error("Укажите корректный путь к входному видео.")
                continue

            # Если выход не указан — берём папку входного файла
            if not base_out:
                base_out = str(Path(input_path).parent)
                window["-OUT-"].update(base_out)

            project_name = base_project or Path(input_path).stem
            values["-PROJECT-"] = project_name
            out_dir = str(Path(base_out) / project_name)
            values["-OUT-"] = out_dir

            pipeline_file = Path(out_dir) / f"{project_name}.pipeline.yaml"
            pipeline_file.parent.mkdir(parents=True, exist_ok=True)
            if not pipeline_file.exists():
                shutil.copy2(TEMPLATE_PATH, pipeline_file)

            pipeline_path = save_pipeline_yaml(values, pipeline_file)
            args = ["run", str(pipeline_path)]

            last_run_count = 1
            window["-LOGBOX-"].update("")
            window["-STATUS-"].update("Статус: running")
            running = True

            worker_thread = threading.Thread(
                target=run_pipeline_sequence, args=([(Path(input_path).name, args)], window), daemon=True
            )
            worker_thread.start()

        if event == "-LOG-":
            log_repr = repr(values["-LOG-"])
            info(f"CATCH -LOG- {log_repr}")
            raw = values["-LOG-"]
            if raw is not None:
                raw = raw.replace("\r", "\n")

                for line in raw.splitlines():
                    if not line:
                        continue
                    print_parsed_log(window, line)

        if event == "-DONE-":
            exit_code = values["-DONE-"]
            running = False

            try:
                window.TKroot.title(BASE_TITLE)
            except Exception:
                pass

            status = "ok" if exit_code == 0 else f"error (code {exit_code})"
            if last_run_count > 1 and exit_code == 0:
                window["-STATUS-"].update(f"Статус: {status} ({last_run_count} files)")
            else:
                window["-STATUS-"].update(f"Статус: {status}")

    window.close()


if __name__ == "__main__":
    main()
