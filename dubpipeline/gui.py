import threading
import subprocess
import sys
import os
from pathlib import Path

import FreeSimpleGUI as sg
import yaml
import re

from dubpipeline.config import save_pipeline_yaml
from dubpipeline.steps.step_tts import getVoices

from dubpipeline.utils.logging import step, info, warn, error, debug


USE_SUBPROCESS = True

TEMPLATE_PATH = Path(__file__).with_name("video.pipeline.yaml")
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    BASE_CFG = yaml.safe_load(f)

LOG_LINE_RE = re.compile(
    r"^\[(?P<level>\w+\s*)\]\s*"                # [LEVEL]
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
    # отладка
    # print("PRINT_PARSED_LOG -LOG-", repr(line))
    debug(f"[DEBUG] print_parsed_log -LOG- {repr(line)}")
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

def main():
    sg.theme("SystemDefault")

    voices = getVoices()

    layout = [
        [sg.Text("Project name:"),
         sg.Input(key="-PROJECT-", expand_x=True)],

        [sg.Text("Входное видео:"),
         sg.Input(key="-IN-", expand_x=True, enable_events=True),
         sg.FileBrowse("...", file_types=(("Видео файлы", "*.mp4;*.mkv;*.mov;*.avi"),))],

        [sg.Text("Голос TTS:"),
         sg.Combo(
             values=voices,
             key="-VOICE-",
             readonly=True,
             enable_events=True,
             default_value=voices[0] if voices else (None if voices else None),
             size=(40, 1),
         )],

        [sg.Text("Выходная папка:"),
         sg.Input(key="-OUT-", expand_x=True),
         sg.FolderBrowse("...")],

        [sg.Checkbox("Использовать GPU?", key="-GPU-")],
        [sg.Checkbox("Генерировать субтитры?", key="-SRT-")],
        [sg.Checkbox("Перегенерировать все шаги (игнорировать кэш)", key="-REBUILD-")],

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

    # на всякий случай ещё раз говорим, что лог должен тянуться
    window["-LOGBOX-"].expand(expand_x=True, expand_y=True)
    window["-PROJECT-"].update("2")
    window["-IN-"].update("J:/Projects/!!!AI/DubPipeline/tests/data/2.mp4")
    window["-OUT-"].update("J:/Projects/!!!AI/DubPipeline/tests/out")
    window["-GPU-"].update(True)
    running = False

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        # Автозаполнение Project name по имени файла
        if event == "-IN-":
            path_str = values["-IN-"].strip()
            if path_str:
                p = Path(path_str)
                window["-PROJECT-"].update(p.stem)

        # Выбор голоса (если надо, можно куда-то логировать)
        if event == "-VOICE-":
            current_voice = values["-VOICE-"]
            line = info(f"Выбран голос: {current_voice}")
            window.write_event_value("-LOG-", line)
            #window["-LOGBOX-"].print(f"[INFO] Выбран голос: {current_voice}")

        if event == "-START-":
            if running:
                sg.popup("Процесс уже запущен, дождитесь окончания.", title="Инфо")
                continue

            input_path = values["-IN-"].strip()
            output_dir = values["-OUT-"].strip()
            rebuild = values["-REBUILD-"]

            if not input_path or not os.path.isfile(input_path):
                sg.popup_error("Укажите корректный путь к входному видео.")
                continue

            # Если выход не указан — берём папку входного файла
            if not output_dir:
                output_dir = str(Path(input_path).parent)
                window["-OUT-"].update(output_dir)

            project_name = values["-PROJECT-"].strip() or Path(input_path).stem
            values["-PROJECT-"] = project_name

            # Здесь можно передать GPU/SRT/доп. аргументы в save_pipeline_yaml через values
            # (вы уже это сделали в своей реализации)
            pipeline_path = save_pipeline_yaml(values, TEMPLATE_PATH)

            args = ["run", pipeline_path]

            window["-LOGBOX-"].update("")  # очистить лог
            window["-STATUS-"].update("Статус: running")
            running = True

            worker_thread = threading.Thread(
                target=run_pipeline, args=(args, window), daemon=True
            )
            worker_thread.start()

        if event == "-LOG-":
            str=repr(values[ "-LOG-"])
            info(f"CATCH -LOG- {str}")
            raw = values["-LOG-"]
            raw = raw.replace("\r", "\n")

            for line in raw.splitlines():
                if not line:
                    continue
                print_parsed_log(window, line)

        if event == "-DONE-":
            exit_code = values["-DONE-"]
            running = False
            status = "ok" if exit_code == 0 else f"error (code {exit_code})"
            window["-STATUS-"].update(f"Статус: {status}")

    window.close()


if __name__ == "__main__":
    main()
