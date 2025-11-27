import json
from argostranslate import package, translate

def ensure_argos_model_installed():
    """Устанавливает пакет EN→RU, если его ещё нет."""
    packages = package.get_installed_packages()
    for p in packages:
        if p.from_code == "en" and p.to_code == "ru":
            return

    print("[STEP] Скачивание модели EN→RU (Argos)…")
    available = package.get_available_packages()
    for p in available:
        if p.from_code == "en" and p.to_code == "ru":
            download_path = p.download()
            package.install_from_path(download_path)
            print("[OK] Модель EN→RU установлена.")
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

        print(f"[{idx}] {text} -> {text_ru}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Переведено {len(translated)} сегментов.")
    print(f"[SAVE] {output_file}")

if __name__ == "__main__":
    ensure_argos_model_installed()
    translate_segments("d:\AI\DubPipeline\out\lecture_sample.segments.json", "d:\AI\DubPipeline\out\lecture_sample.segments_translated.json")
