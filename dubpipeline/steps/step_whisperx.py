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

def merge_words_to_segments(words, max_gap=0.8, max_seg_dur=20.0, max_seg_chars=350):
    """
    Простая логика:
    - пока спикер тот же и пауза между словами <= max_gap, накапливаем слова в сегмент
    - при смене спикера или большой паузе — закрываем сегмент и начинаем новый

    Защита от "гигантских" сегментов (важно для качества перевода/TTS):
    - режем сегмент, если он разросся по длительности > max_seg_dur
    - режем сегмент, если текст превысил max_seg_chars (0/None => без лимита)
    """
    segments = []
    if not words:
        return segments

    # нормализуем лимиты
    try:
        max_gap = float(max_gap)
    except Exception:
        max_gap = 0.8
    try:
        max_seg_dur = float(max_seg_dur)
    except Exception:
        max_seg_dur = 20.0
    try:
        max_seg_chars = int(max_seg_chars) if max_seg_chars is not None else 0
    except Exception:
        max_seg_chars = 350

    # 0/отрицательное => без лимита по символам
    if max_seg_chars <= 0:
        max_seg_chars = 10**9

    # на всякий случай сортируем по времени начала
    words = sorted(words, key=lambda w: w["start"])

    def new_current(w: dict) -> dict:
        return {
            "speaker": w.get("speaker", "UNKNOWN"),
            "start": float(w["start"]),
            "end": float(w["end"]),
            "text": (w.get("text") or "").strip(),
        }

    current = new_current(words[0])

    for w0 in words[1:]:
        w = {
            "speaker": w0.get("speaker", "UNKNOWN"),
            "start": float(w0["start"]),
            "end": float(w0["end"]),
            "text": (w0.get("text") or "").strip(),
        }

        same_speaker = (w["speaker"] == current["speaker"])
        small_gap = (w["start"] - current["end"] <= max_gap)

        if same_speaker and small_gap:
            prospective_end = w["end"]
            prospective_dur = max(0.0, prospective_end - current["start"])
            prospective_text = (current["text"] + " " + w["text"]).strip()

            # если после добавления мы всё ещё в лимитах — продолжаем копить
            if (prospective_dur <= max_seg_dur) and (len(prospective_text) <= max_seg_chars):
                current["text"] = prospective_text
                current["end"] = prospective_end
                continue

        # иначе закрываем текущий сегмент и начинаем новый
        segments.append(current)
        current = new_current(w)

    segments.append(current)
    return segments


def post_merge_segments_for_tts(
    segments: list[dict],
    *,
    min_seg_dur: float = 1.0,
    min_seg_chars: int = 25,
    max_merge_gap: float = 0.35,
    max_seg_dur: float = 12.0,
    allow_cross_speaker: bool = True,
) -> list[dict]:
    """
    Пост-склейка сегментов именно для ускорения TTS:
    - уменьшает количество очень коротких сегментов (у них высокий overhead на TTS)
    - склеивает только соседние сегменты при маленьком промежутке (gap)
    - по умолчанию допускает склейку через смену спикера (пока TTS одним голосом)
      -> можно отключить allow_cross_speaker=False, если позже начнёте делать разные голоса.
    """
    if not segments:
        return []

    # На всякий случай сортируем
    segments = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    def is_short(seg: dict) -> bool:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = max(0.0, end - start)
        text = (seg.get("text") or "").strip()
        return (dur < float(min_seg_dur)) or (len(text) < int(min_seg_chars))

    merged: list[dict] = []
    cur = dict(segments[0])

    for nxt0 in segments[1:]:
        nxt = dict(nxt0)

        cur_start = float(cur.get("start", 0.0))
        cur_end = float(cur.get("end", cur_start))
        nxt_start = float(nxt.get("start", cur_end))
        nxt_end = float(nxt.get("end", nxt_start))

        gap = nxt_start - cur_end
        same_speaker = (cur.get("speaker") == nxt.get("speaker"))
        can_speaker = same_speaker or bool(allow_cross_speaker)

        # Решаем, стоит ли склеивать: маленький gap + хотя бы один из соседних сегментов короткий
        should_merge = (gap <= float(max_merge_gap)) and can_speaker and (is_short(cur) or is_short(nxt))

        # Не даём сегментам разрастаться слишком сильно (чтобы не ломать ритм и не копить ошибки)
        merged_dur = max(0.0, nxt_end - cur_start)
        if should_merge and (merged_dur <= float(max_seg_dur)):
            # merge nxt into cur
            cur["end"] = nxt_end
            # speaker: оставляем speaker первого сегмента (если allow_cross_speaker=True)
            cur_text = (cur.get("text") or "").strip()
            nxt_text = (nxt.get("text") or "").strip()
            if cur_text and nxt_text:
                cur["text"] = cur_text + " " + nxt_text
            else:
                cur["text"] = cur_text or nxt_text
            continue

        merged.append(cur)
        cur = nxt

    merged.append(cur)
    return merged


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
    wm_max_seg_dur = float(os.getenv("DUBPIPELINE_WORD_MERGE_MAX_SEG_DUR", "20.0"))
    wm_max_seg_chars = int(os.getenv("DUBPIPELINE_WORD_MERGE_MAX_SEG_CHARS", "350"))
    segments = merge_words_to_segments(
        words,
        max_gap=MAX_GAP_BETWEEN_WORDS,
        max_seg_dur=wm_max_seg_dur,
        max_seg_chars=wm_max_seg_chars,
    )
    info(f"[INFO] Word-merge limits: max_seg_dur={wm_max_seg_dur:.2f}s, max_seg_chars={wm_max_seg_chars}\\n")
    # 7.1) Пост-склейка для ускорения TTS (опционально, по env)
    raw_n = len(segments)
    try:
        min_seg_dur = float(os.getenv("DUBPIPELINE_MIN_SEG_DUR", "1.0"))
        min_seg_chars = int(os.getenv("DUBPIPELINE_MIN_SEG_CHARS", "25"))
        max_merge_gap = float(os.getenv("DUBPIPELINE_MERGE_MAX_GAP", "0.35"))
        max_seg_dur = float(os.getenv("DUBPIPELINE_MAX_SEG_DUR", "12.0"))
        allow_cross = os.getenv("DUBPIPELINE_MERGE_ALLOW_CROSS_SPEAKER", "1").strip().lower() not in {"0", "false", "no", "off"}
        segments = post_merge_segments_for_tts(
            segments,
            min_seg_dur=min_seg_dur,
            min_seg_chars=min_seg_chars,
            max_merge_gap=max_merge_gap,
            max_seg_dur=wm_max_seg_dur,
            allow_cross_speaker=allow_cross,
        )
    except Exception as ex:
        warn(f"[WARN] Post-merge for TTS skipped due to error: {ex}\n")

    info(f"[INFO] Segments raw={raw_n}, post={len(segments)}\n")

    # 8) Сохраняем результаты
    info(f"[SAVE] Segments → {cfg.paths.segments_file}")
    with open(pathlib.Path(cfg.paths.segments_file), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # Optional: release VRAM after WhisperX so TTS can run faster/more stable.
    if str(device).startswith("cuda"):
        rel = os.getenv("DUBPIPELINE_WHISPERX_RELEASE_VRAM", "1").strip().lower()
        if rel not in {"0", "false", "no", "off"}:
            try:
                import gc
                # try to drop model refs
                try:
                    del model
                except Exception:
                    pass
                try:
                    del model_a
                except Exception:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
                info("[VRAM] Released CUDA cache after whisperx.\n")
            except Exception as ex:
                warn(f"[VRAM] Failed to release CUDA cache after whisperx: {ex}\n")


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

