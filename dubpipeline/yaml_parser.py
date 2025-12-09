import yaml
from pathlib import Path

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Шаг 1: соберём переменные, которые можно подставлять
    # можно просто взять корневой словарь:
    variables = dict(cfg)

    # Шаг 2: рекурсивно обойдём структуру и форматнём все строки
    def format_values(obj):
        if isinstance(obj, dict):
            return {k: format_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [format_values(v) for v in obj]
        elif isinstance(obj, str):
            try:
                return obj.format(**variables)
            except KeyError:
                # если вдруг в строке есть {что-то}, чего нет в variables
                return obj
        else:
            return obj

    return format_values(cfg)

config = load_config("config.yaml")

print(config["paths"]["input_video"])      # lecture_01.mp4
print(config["paths"]["segments_json"])    # out/segments/lecture_01.segments.json
