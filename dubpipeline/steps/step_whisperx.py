from ..config import PipelineConfig

import os
import json
import pathlib

import torch
import whisperx

# === НАСТРОЙКИ =================================================================

# Путь к аудио (поставьте сюда тот же файл, что вы используете для SRT-теста)
#AUDIO_PATH = Path()

# Модель Whisper / Faster-Whisper через whisperx
MODEL_NAME = "large-v2"                  # как в вашем тесте
BATCH_SIZE = 1

# Порог склейки слов в один сегмент (секунды)
MAX_GAP_BETWEEN_WORDS = 0.8

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
    compute_type = "float16" if device == "cuda" else "int32"
    print(f"[INFO] Device: {device}, compute_type: {compute_type}")
    print(f"[INFO] Audio: {audio_path}")

    # 1) Загружаем аудио
    audio = whisperx.load_audio(str(audio_path))

    # 2) ASR (Whisper / Faster-Whisper через whisperx)
    print("[STEP] Loading ASR model...")
    model = whisperx.load_model(MODEL_NAME, device, compute_type=compute_type)

    print("[STEP] Transcribing...")
    result = model.transcribe(audio, batch_size=BATCH_SIZE)
    # result["segments"] — фразовые сегменты (без выравнивания по словам)

    # 3) Alignment (wav2vec2) для точных таймкодов
    print("[STEP] Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )

    print("[STEP] Aligning...")
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    # aligned_result["segments"][i]["words"] – слова с точными таймкодами

    print("[STEP] Running diarization (if available)...")
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
    print("[STEP] Merging words to segments...")
    segments = merge_words_to_segments(words, max_gap=MAX_GAP_BETWEEN_WORDS)

    # 8) Сохраняем результаты
    #words_path = OUT_DIR / f"{base_name}.words.json"
    segments_path = OUT_DIR / f"{base_name}.segments.json"
    cfg.paths.segments_file = segments_path
    cfg.paths.segments_ru_file = OUT_DIR / f"{base_name}.segments.ru.json"
    #print(f"[SAVE] Words → {words_path}")
    #with open(words_path, "w", encoding="utf-8") as f:
    #    json.dump(words, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Segments → {segments_path}")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print("[DONE] Готово. Теперь у вас есть words.json и segments.json")

def run_diarization_safe(audio, aligned_result, device="cpu"):
    """
    Пытаемся запустить диаризацию, если в whisperx есть нужные API.
    Если не получается — аккуратно назначаем всем словам SPEAKER_00.
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        # Новый API (GitHub-версия): DiarizationPipeline
        if hasattr(whisperx, "DiarizationPipeline"):
            print("[STEP] Running diarization via DiarizationPipeline...")
            diarize_model = whisperx.DiarizationPipeline(
                device=device,
                use_auth_token=hf_token
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            return result

        # Возможный старый API (на всякий случай, если он есть в вашей версии)
        if hasattr(whisperx, "load_diarization_model"):
            print("[STEP] Running diarization via load_diarization_model...")
            diarize_model = whisperx.load_diarization_model(
                device=device,
                use_auth_token=hf_token
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            return result

        print("[WARN] В установленной версии whisperx нет DiarizationPipeline/load_diarization_model.")
        print("[WARN] Продолжаем без диаризации, ставим SPEAKER_00 для всех слов.")

    except Exception as e:
        print(f"[WARN] Diarization failed: {e}")
        print("[WARN] Продолжаем без диаризации, ставим SPEAKER_00 для всех слов.")

    # --- Fallback: один спикер для всего аудио ---
    for seg in aligned_result.get("segments", []):
        seg["speaker"] = "SPEAKER_00"
        for w in seg.get("words", []):
            w["speaker"] = "SPEAKER_00"

    return aligned_result
