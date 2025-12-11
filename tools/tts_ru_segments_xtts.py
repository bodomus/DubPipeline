import json
from pathlib import Path

import torch
from TTS.api import TTS


def main():
    # --- НАСТРОЙКИ ---
    segments_path = Path("out/21305.segments_translated.json")
    out_dir = Path("out/tts_ru_segments")

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    language = "ru"

    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bold green][INFO] Using device: {device}[/bold green]")

    print(f"[bold green][INFO] Loading TTS model: {model_name}[/bold green]")
    tts = TTS(model_name).to(device)

    # --- Дебаг: какие вообще есть спикеры и языки ---
    speakers = getattr(tts, "speakers", None)
    languages = getattr(tts, "languages", None)
    print("[bold green][INFO] Available speakers:[/bold green]", speakers)
    print("[bold green][INFO] Available languages:[/bold green]", languages)

    if not speakers:
        raise RuntimeError(
            "[bold red][ERROR]XTTS не возвращает список speakers. "
            "Нужно либо обновить coqui-tts, либо использовать speaker_wav.[/bold red]"
        )

    default_speaker = speakers[0]
    print(f"[bold green][INFO] Using default speaker: {default_speaker!r}[/bold green]")

    # --- ЗАГРУЗКА СЕГМЕНТОВ ---
    print(f"[bold green][INFO] Loading segments from {segments_path}[/bold green]")
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    segments = sorted(segments, key=lambda s: s["start"])

    # --- ГЕНЕРАЦИЯ ---
    for seg in segments:
        seg_id = seg["id"]
        text_ru = (seg.get("text_ru") or "").strip()

        if not text_ru:
            print(f"[bold blue][WARN] Segment {seg_id} has empty 'text_ru', skipping[/bold blue]")
            continue

        out_wav = out_dir / f"seg_{seg_id:04d}.wav"
        if out_wav.exists():
            print(f"[SKIP] {out_wav} already exists")
            continue

        print(f"[bold yellow][TTS] id={seg_id}  {seg['start']:.2f}s–{seg['end']:.2f}s[/bold yellow]")
        print(f"      RU: {text_ru}")

        # КЛЮЧЕВАЯ ЧАСТЬ: задаём и language, и speaker
        tts.tts_to_file(
            text=text_ru,
            file_path=str(out_wav),
            language=language,
            speaker=default_speaker,
        )

    print("[bold green][DONE] Russian TTS segments generated in:[bold yellow][/bold green]", out_dir)


if __name__ == "__main__":
    main()
