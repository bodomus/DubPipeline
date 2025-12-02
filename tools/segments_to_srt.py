import json
import pathlib


INPUT_SEGMENTS = "out/video_sample.segments.json"   # поправьте путь, если нужно
OUTPUT_SRT = "out/video_sample.from_segments.en.srt"


def format_timestamp(seconds: float) -> str:
    """
    Преобразует секунды в формат SRT: HH:MM:SS,mmm
    """
    ms = int(round(seconds * 1000))
    h = ms // (3600 * 1000)
    ms %= 3600 * 1000
    m = ms // (60 * 1000)
    ms %= 60 * 1000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments, with_speaker=False):
    """
    segments: список словарей {speaker, start, end, text}
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"]

        if with_speaker and seg.get("speaker"):
            text = f"{seg['speaker']}: {text}"

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # пустая строка между субтитрами

    return "\n".join(lines)


def main():
    input_path = pathlib.Path(INPUT_SEGMENTS)
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл сегментов: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    srt_text = segments_to_srt(segments, with_speaker=True)

    output_path = pathlib.Path(OUTPUT_SRT)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    print(f"[DONE] SRT сохранён в: {output_path}")


if __name__ == "__main__":
    main()
