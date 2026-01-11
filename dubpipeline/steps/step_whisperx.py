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
MODEL_NAME = "large-v3"                  # как в вашем тесте
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
    #OUT_DIR = cfg.paths.out_dir
    #OUT_DIR.mkdir(exist_ok=True)

    audio_path = pathlib.Path(cfg.paths.audio_wav)
    if not audio_path.exists():
        raise FileNotFoundError(f"Не найден аудиофайл: {audio_path}")

    base_name = audio_path.stem

    device = "cuda" if torch.cuda.is_available() and cfg.usegpu else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    compute_type = "float16" if device == "cuda" else "int8"
    info(f"Device: {device}, compute_type: {compute_type}\n")
    info(f"Audio: {audio_path}\n")

    # 1) Загружаем аудио
    audio = whisperx.load_audio(str(audio_path), )

    # 2) ASR (Whisper / Faster-Whisper через whisperx)
    step("Loading ASR model...\n")
    model = whisperx.load_model(MODEL_NAME, device, compute_type=compute_type)

    step("Transcribing...\n")
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    # result["segments"] — фразовые сегменты (без выравнивания по словам)
    segments = result["segments"]
    info(f"Segments: {len(segments)}\n")
    srt_file = cfg.paths.srt_file_en
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

    info(f"SRT saved to: {srt_file}")

    # 3) Alignment (wav2vec2) для точных таймкодов
    step("Loading alignment model...\n")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )

    step("Aligning...\n")
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    # aligned_result["segments"][i]["words"] – слова с точными таймкодами

    step("[STEP] Running diarization (if available)...\n")
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
    step("[STEP] Merging words to segments...\n")
    segments = merge_words_to_segments(words, max_gap=MAX_GAP_BETWEEN_WORDS)

    # 8) Сохраняем результаты
    info(f"[SAVE] Segments → {cfg.paths.segments_file}")
    with open(pathlib.Path(cfg.paths.segments_file), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


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
            step("[STEP] Running diarization via load_diarization_model...\n")
            diarize_model = whisperx.load_diarization_model(
                device=device,
                use_auth_token=hf_token
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            return result

        warn("[WARN] В установленной версии whisperx нет DiarizationPipeline/load_diarization_model.\n")
        warn("[WARN] Продолжаем без диаризации, ставим SPEAKER_00 для всех слов.\n")

    except Exception as e:
        error(f"[WARN] Diarization failed: {e}\n")
        error("[WARN] Продолжаем без диаризации, ставим SPEAKER_00 для всех слов.\n")

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

