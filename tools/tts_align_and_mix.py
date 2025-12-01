import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def build_atempo_filter(total_factor: float) -> str:
    """
    Разбивает общий коэффициент total_factor на цепочку atempo=...
    с учётом ограничений ffmpeg (0.5–2.0 на один фильтр).

    Итоговый эффект = произведение всех atempo.
    """
    factors = []

    factor = float(total_factor)

    # Очень маленькая/большая скорость - подстрахуемся
    if factor <= 0:
        raise ValueError(f"Invalid atempo factor <= 0: {factor}")

    # Замедление сильнее, чем 0.5: цепочка atempo=0.5
    while factor < 0.5:
        factors.append(0.5)
        factor /= 0.5

    # Ускорение сильнее, чем 2.0: цепочка atempo=2.0
    while factor > 2.0:
        factors.append(2.0)
        factor /= 2.0

    factors.append(factor)

    # Собираем строку "atempo=...,atempo=..."
    parts = [f"atempo={f:.6f}" for f in factors]
    return ",".join(parts)


def align_segments(
    segments,
    tts_dir: Path,
    aligned_dir: Path,
    ffmpeg_path: str = "ffmpeg",
    eps: float = 0.01,
):
    """
    Выравнивает длительность каждого TTS-сегмента под (end - start)
    с помощью ffmpeg atempo. Результат пишет в aligned_dir.
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        seg_id = seg["id"]
        start = float(seg["start"])
        end = float(seg["end"])
        target_dur = end - start

        if target_dur <= 0:
            print(f"[WARN] Segment {seg_id} has non-positive duration, skipping")
            continue

        in_wav = tts_dir / f"seg_{seg_id:04d}.wav"
        out_wav = aligned_dir / f"seg_{seg_id:04d}.wav"

        if not in_wav.exists():
            print(f"[WARN] TTS file not found for segment {seg_id}: {in_wav}")
            continue

        # Читаем, чтобы узнать текущую длительность
        data, sr = sf.read(in_wav)
        cur_dur = len(data) / sr

        diff = abs(cur_dur - target_dur)
        print(
            f"[ALIGN] id={seg_id} target={target_dur:.3f}s "
            f"cur={cur_dur:.3f}s diff={diff:.3f}s"
        )

        # Если отличие маленькое (< eps), просто копируем файл
        if diff < eps:
            shutil.copy2(in_wav, out_wav)
            continue

        factor = cur_dur / target_dur
        filter_str = build_atempo_filter(factor)
        print(f"        atempo factor={factor:.4f} -> filter: {filter_str}")

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(in_wav),
            "-vn",
            "-acodec",
            "pcm_s16le",  # оставим PCM
            "-filter:a",
            filter_str,
            str(out_wav),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] ffmpeg failed for segment {seg_id}")
            print(result.stderr)
            # на всякий случай не падаем, а продолжаем
        else:
            print(f"[OK] Aligned segment written: {out_wav}")


def mix_aligned_segments_to_timeline(
    segments,
    aligned_dir: Path,
    out_wav: Path,
    eps: float = 0.001,
):
    """
    Склеивает выровненные сегменты в одну дорожку по таймлайну start/end.
    Паузы между сегментами остаются тишиной.
    """
    # Определим максимальный end, чтобы посчитать длину таймлайна
    max_end = max(float(seg["end"]) for seg in segments)
    print(f"[MIX] Max end time: {max_end:.3f}s")

    # Найдём первый существующий сегмент, чтобы узнать sr и кол-во каналов
    sr = None
    channels = None

    for seg in segments:
        seg_id = seg["id"]
        path = aligned_dir / f"seg_{seg_id:04d}.wav"
        if path.exists():
            data, sr_tmp = sf.read(path)
            sr = sr_tmp
            # soundfile возвращает (N,) для mono и (N, C) для multi
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]
            break

    if sr is None:
        raise RuntimeError("No aligned WAVs found to determine sample rate")

    print(f"[MIX] Using sample rate={sr}, channels={channels}")

    total_samples = int(math.ceil((max_end + 0.1) * sr))
    if channels == 1:
        timeline = np.zeros((total_samples,), dtype=np.float32)
    else:
        timeline = np.zeros((total_samples, channels), dtype=np.float32)

    # Размещаем сегменты
    for seg in segments:
        seg_id = seg["id"]
        start = float(seg["start"])
        end = float(seg["end"])
        target_dur = end - start

        path = aligned_dir / f"seg_{seg_id:04d}.wav"
        if not path.exists():
            print(f"[WARN] Aligned segment file missing, skipping: {path}")
            continue

        data, sr_tmp = sf.read(path)
        if sr_tmp != sr:
            raise RuntimeError(
                f"Sample rate mismatch for segment {seg_id}: {sr_tmp} vs {sr}"
            )

        if data.ndim == 1 and channels > 1:
            data = np.tile(data[:, None], (1, channels))
        elif data.ndim > 1 and channels == 1:
            data = data.mean(axis=1)

        cur_dur = len(data) / sr
        diff = abs(cur_dur - target_dur)
        if diff > 2 * eps:
            print(
                f"[WARN] After alignment, segment {seg_id} duration differs from "
                f"target by {diff:.3f}s (cur={cur_dur:.3f}, target={target_dur:.3f})"
            )

        start_idx = int(round(start * sr))
        end_idx = start_idx + len(data)

        if end_idx > len(timeline):
            end_idx = len(timeline)
            data = data[: end_idx - start_idx]

        print(
            f"[MIX] Place seg {seg_id} at {start_idx}..{end_idx} "
            f"({start:.3f}s–{start + len(data) / sr:.3f}s)"
        )

        timeline[start_idx:end_idx] += data

    # Записываем результат
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, timeline, sr)
    print(f"[DONE] Mixed timeline written to: {out_wav}")


def main():
    segments_path = Path("out/21305.segments_translated.json")
    tts_dir = Path("out/tts_ru_segments")
    aligned_dir = Path("out/tts_ru_segments_aligned")
    out_mix_path = Path("out/21305.ru_full.wav")

    print(f"[INFO] Loading segments from {segments_path}")
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    # сортируем по start на всякий случай
    segments = sorted(segments, key=lambda s: s["start"])

    print("[STEP] Aligning segment durations...")
    align_segments(segments, tts_dir, aligned_dir)

    print("[STEP] Mixing aligned segments into single WAV...")
    mix_aligned_segments_to_timeline(segments, aligned_dir, out_mix_path)


if __name__ == "__main__":
    main()
