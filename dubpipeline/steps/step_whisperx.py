from datetime import timedelta

from ..config import PipelineConfig

import os
import json
import pathlib

import torch
import whisperx
from rich import print
from dubpipeline.utils.logging import info, step, warn, error, debug

# === НАСТРОЙКИ =================================================================

# Путь к аудио (поставьте сюда тот же файл, что вы используете для SRT-теста)
#AUDIO_PATH = Path()

# Модель Whisper / Faster-Whisper через whisperx
MODEL_NAME = "large-v2"                  # как в вашем тесте
BATCH_SIZE = 1

# Порог склейки слов в один сегмент (секунды)
MAX_GAP_BETWEEN_WORDS = 0.8

# OUTPUT_SRT = "out/video_sample.from_segments.en.srt"

# === ВСПОМОГАТЕЛЬНАЯ ЛОГИКА ====================================================

def merge_words_to_segments(words, max_gap=0.8):
    """
    Простая логика:
    - пока спикер тот же и пауза между словами <= max_gap, накапливаем слова в сегмент
    - при смене спикера или большой паузе — закрываем сегмент и начинаем новый
    """
    segments = []
    if not words:
        return segments

    # на всякий случай сортируем по времени начала
    words = sorted(words, key=lambda w: w["start"])

    current = {
        "speaker": words[0]["speaker"],
        "start": words[0]["start"],
        "end": words[0]["end"],
        "text": words[0]["text"],
    }

    for w in words[1:]:
        same_speaker = (w["speaker"] == current["speaker"])
        small_gap = (w["start"] - current["end"] <= max_gap)

        if same_speaker and small_gap:
            current["text"] += " " + w["text"]
            current["end"] = w["end"]
        else:
            segments.append(current)
            current = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
            }

    segments.append(current)
    return segments


# === ОСНОВНОЙ PIPELINE =========================================================

def run(cfg:PipelineConfig):
    #Создание json файла с английским текстом и временными метками.
    # Папка для результатов
    OUT_DIR = cfg.paths.out_dir
    OUT_DIR.mkdir(exist_ok=True)

    audio_path = pathlib.Path(cfg.paths.audio_wav)
    if not audio_path.exists():
        raise FileNotFoundError(f"Не найден аудиофайл: {audio_path}")

    base_name = audio_path.stem

    device = "cuda" if torch.cuda.is_available() and cfg.usegpu else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    info(f"[bold green]Device: {device}, compute_type: {compute_type}[/bold green]\n")
    info(f"[bold green]Audio: {audio_path}[/bold green]\n")

    # 1) Загружаем аудио
    audio = whisperx.load_audio(str(audio_path), )

    # 2) ASR (Whisper / Faster-Whisper через whisperx)
    step("[bold magenta]Loading ASR model...[/bold magenta]\n")
    model = whisperx.load_model(MODEL_NAME, device, compute_type=compute_type)

    step("[bold magenta]Transcribing...[/bold magenta]\n")
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    # result["segments"] — фразовые сегменты (без выравнивания по словам)
    segments = result["segments"]
    info(f"[bold yellow]Segments: {len(segments)}[/bold yellow]\n")
    srt_file = cfg.paths.srt_file_en
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

    info(f"[bold green]SRT saved to:[/bold green] {srt_file}")

    # 3) Alignment (wav2vec2) для точных таймкодов
    step("[bold magenta]Loading alignment model...[/bold magenta]\n")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )

    step("[bold magenta]Aligning...[/bold magenta]\n")
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    # aligned_result["segments"][i]["words"] – слова с точными таймкодами

    step("[bold magenta][STEP] Running diarization (if available)...[/bold magenta]\n")
    with_speakers = run_diarization_safe(audio, aligned_result, device=device)

    # with_speakers["segments"] — список сегментов, внутри "words" со speaker/…

    # 6) Приводим слова к плоскому списку для удобства
    words = []
    for seg in with_speakers["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        for w in seg.get("words", []):
            # иногда встречаются слова без таймкодов — пропускаем
            if "start" not in w or "end" not in w:
                continue
            text = w.get("word") or w.get("text") or ""
            text = text.strip()
            if not text:
                continue

            words.append({
                "speaker": speaker,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "text": text,
            })

    # 7) Склеиваем слова в более крупные сегменты по спикеру + паузам
    step("[bold magenta][STEP] Merging words to segments...[/bold magenta]\n")
    segments = merge_words_to_segments(words, max_gap=MAX_GAP_BETWEEN_WORDS)

    # 8) Сохраняем результаты
    #words_path = OUT_DIR / f"{base_name}.words.json"
    segments_path = OUT_DIR / f"{base_name}.segments.json"
    cfg.paths.segments_file = segments_path
    cfg.paths.segments_ru_file = OUT_DIR / f"{base_name}.segments.ru.json"
    #print(f"[SAVE] Words → {words_path}")
    #with open(words_path, "w", encoding="utf-8") as f:
    #    json.dump(words, f, ensure_ascii=False, indent=2)

    info(f"[bold yellow][SAVE] Segments → {segments_path}[/bold yellow]")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    info("[bold green][DONE] Готово. Теперь у вас есть words.json и segments.json[/bold green]\n")

def run_diarization_safe(audio, aligned_result, device="cpu"):
    """
    Пытаемся запустить диаризацию, если в whisperx есть нужные API.
    Если не получается — аккуратно назначаем всем словам SPEAKER_00.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        # Новый API (GitHub-версия): DiarizationPipeline
        if hasattr(whisperx, "DiarizationPipeline"):
            step("Running diarization via DiarizationPipeline...\n")
            diarize_model = whisperx.DiarizationPipeline(
                device=device,
                use_auth_token=hf_token
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            return result

        # Возможный старый API (на всякий случай, если он есть в вашей версии)
        if hasattr(whisperx, "load_diarization_model"):
            step("[bold magenta][STEP] Running diarization via load_diarization_model...[/bold magenta]\n")
            diarize_model = whisperx.load_diarization_model(
                device=device,
                use_auth_token=hf_token
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            return result

        warn("[bold blue][WARN] В установленной версии whisperx нет DiarizationPipeline/load_diarization_model.[/bold blue]\n")
        warn("[bold blue][WARN] Продолжаем без диаризации, ставим SPEAKER_00 для всех слов.[/bold blue]\n")

    except Exception as e:
        error(f"[bold blue][WARN] Diarization failed: {e}[/bold blue]\n")
        error("[bold blue][WARN] Продолжаем без диаризации, ставим SPEAKER_00 для всех слов.[/bold blue]\n")

    # --- Fallback: один спикер для всего аудио ---
    for seg in aligned_result.get("segments", []):
        seg["speaker"] = "SPEAKER_00"
        for w in seg.get("words", []):
            w["speaker"] = "SPEAKER_00"

    return aligned_result

# 3. Сохраняем в SRT (без диаризации — пока)
def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    # Приводим к формату HH:MM:SS,mmm
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

