import threading
import subprocess
import sys
import os
import FreeSimpleGUI  as sg
from pathlib import Path
import  yaml

from dubpipeline.config import save_pipeline_yaml
from dubpipeline.steps.step_tts import getVoices

# Если хотите вызывать напрямую Python-модуль:
USE_SUBPROCESS = True  # можно потом переключить на False и вызывать main() напрямую
TEMPLATE_PATH = Path(__file__).with_name("video.pipeline.yaml")
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    BASE_CFG = yaml.safe_load(f)
print(TEMPLATE_PATH)

def run_pipeline(args_list, window):
    """
    Функция, которая запускает ваш пайплайн и пишет логи в окно.
    Работает в отдельном потоке, чтобы не вешать GUI.
    """
    try:
        if USE_SUBPROCESS:
            # Пример вызова: python -m dubpipeline.cli --input ... --output ...
            cmd = [sys.executable, "-m", "dubpipeline.cli"] + args_list
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace"
            )

            for line in process.stdout:
                window.write_event_value("-LOG-", line)

            process.wait()
            exit_code = process.returncode

        else:
            # Вариант: прямой импорт и вызов main()
            from dubpipeline.cli import main as dp_main
            # Здесь можно перехватить stdout, если нужно, но для простоты опустим
            exit_code = dp_main(args_list)

        window.write_event_value("-DONE-", exit_code)

    except Exception as e:
        window.write_event_value("-LOG-", f"[ERROR] {e}\n")
        window.write_event_value("-DONE-", 1)


def main():
    sg.theme("SystemDefault")
    voices = getVoices()
    current_voice = voices[0] if voices else None

    layout = [
        [sg.Text("Входное видео:"),
         sg.Input(key="-IN-", expand_x=True),
         sg.FileBrowse("...", file_types=(("Видео файлы", "*.mp4;*.mkv;*.mov;*.avi"),))],
        [sg.Text("Голос TTS:"),
        sg.Combo(
            values=voices,
            key="-VOICE-",
            size=(80, 20),
            readonly=True,  # чтобы нельзя было вручную печатать
            enable_events=True,  # чтобы получать событие при выборе
            default_value=voices[0] if voices else None
        )],
        [sg.Text("Выходная папка:"),
         sg.Input(key="-OUT-", expand_x=True),
         sg.FolderBrowse("...")],
        [sg.Checkbox("Использовать GPU?", key="-GPU-")],
        [sg.Checkbox("Генерировать  субтитры?", key="-SRT-")],
        [sg.Checkbox("Перегенерировать все шаги (игнорировать кэш)", key="-REBUILD-")],
        [sg.Text("Доп. аргументы CLI (опционально):")],
        [sg.Input(key="-EXTRA-", expand_x=True)],
        [sg.Multiline(key="-LOGBOX-", size=(80, 20), autoscroll=True, disabled=True, font=("Consolas", 9))],
        [sg.Button("Старт", key="-START-"),
         sg.Button("Выход", key="-EXIT-"),
         sg.Text("Статус: idle", key="-STATUS-")]
    ]

    window = sg.Window("DubPipeline GUI", layout, resizable=True)

    worker_thread = None
    running = False

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        if event == "-START-":
            if running:
                sg.popup("Процесс уже запущен, дождитесь окончания.", title="Инфо")
                continue

            input_path = values["-IN-"].strip()
            output_dir = values["-OUT-"].strip()
            extra = values["-EXTRA-"].strip()
            rebuild = values["-REBUILD-"]
            isGPU = values["-GPU-"]
            isSRT = values["-SRT-"]


            if not input_path or not os.path.isfile(input_path):
                sg.popup_error("Укажите корректный путь к входному видео.")
                continue

            if not output_dir:
                sg.popup_error("Укажите выходную папку.")
                continue

            # куда класть YAML – например, в выходную папку:
            #out_dir = Path(values["-OUT-"].strip() or ".")
            #pipeline_path = out_dir / f"{project_name}.pipeline.yaml"

            # 1) сохраняем YAML по данным из GUI
            values["-PROJECT-"] = Path(input_path).stem
            pipeline_path = save_pipeline_yaml(values, TEMPLATE_PATH)

            # 2) запускаем dubpipeline.cli, передавая путь к YAML
            cmd = [
                sys.executable,
                "-m",
                "dubpipeline.cli",
                "run",
                str(pipeline_path),
            ]

            # дальше как у вас уже сделано: через subprocess/поток, пишем логи и т.п.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )

            # os.makedirs(output_dir, exist_ok=True)
            #
            # # Формируем аргументы CLI
            # args = [
            #     "--input", input_path,
            #     "--output-dir", output_dir,
            #     "--lang", "en-ru",  # пока хардкод, но можно сделать выпадающий список
            # ]
            # if rebuild:
            #     args.append("--rebuild")
            #
            # if extra:
            #     # простое разбиение, можно улучшить парсером shlex.split
            #     import shlex
            #     args.extend(shlex.split(extra))
            #
            # window["-LOGBOX-"].update("")  # очистить лог
            # window["-STATUS-"].update("Статус: running")
            # running = True
            #
            # worker_thread = threading.Thread(
            #     target=run_pipeline, args=(args, window), daemon=True
            # )
            # worker_thread.start()
        elif event == "-VOICE-":
            current_voice = values["-VOICE-"]
            print("Выбран голос:", current_voice)

        elif event == "-LOG-":
            line = values["-LOG-"]
            # включаем запись в Multiline
            window["-LOGBOX-"].update(disabled=False)
            window["-LOGBOX-"].update(window["-LOGBOX-"].get() + line)
            window["-LOGBOX-"].update(disabled=True)

        elif event == "-DONE-":
            exit_code = values["-DONE-"]
            running = False
            status = "ok" if exit_code == 0 else f"error (code {exit_code})"
            window["-STATUS-"].update(f"Статус: {status}")

    window.close()


if __name__ == "__main__":
    main()
