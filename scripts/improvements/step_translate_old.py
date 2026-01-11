import json
from argostranslate import package, translate
from dubpipeline.utils.logging import info, step, warn, error, debug

from dubpipeline.config import PipelineConfig

# =========================
# Translation step (EN -> RU) Version OLD
# =========================
#

DEBUG:bool = False

def run(cfg:PipelineConfig):
    # Создание json файла с английским и русским текстом и временными метками.

    ensure_argos_model_installed()
    translate_segments(cfg.paths.segments_file, cfg.paths.segments_ru_file)

def ensure_argos_model_installed():
    """Устанавливает пакет EN→RU, если его ещё нет."""
    packages = package.get_installed_packages()
    for p in packages:
        if p.from_code == "en" and p.to_code == "ru":
            return

    step("Скачивание модели EN→RU (Argos)…\n")
    available = package.get_available_packages()
    for p in available:
        if p.from_code == "en" and p.to_code == "ru":
            download_path = p.download()
            package.install_from_path(download_path)
            info("[OK] Модель EN→RU установлена.\n")
            return

    raise RuntimeError("Не удалось скачать модель Argos EN→RU.")

def translate_segments(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    translated = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "")
        text_ru = translate.translate(text, "en", "ru")

        seg_out = {
            "id": idx,          # ← вот здесь присваиваем id
            **seg,
            "text_ru": text_ru  # добавляем перевод
        }

        translated.append(seg_out)

        info(f"[{idx}] {text} -> {text_ru}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    if DEBUG:
        info(f"\n[OK] Переведено {len(translated)} сегментов.\n")
        info(f"[SAVE] {output_file}\n")
