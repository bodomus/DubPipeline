import json
import math
import shutil
import subprocess
import os
from time import perf_counter
from pathlib import Path

import numpy as np
import soundfile as sf

from dubpipeline.config import PipelineConfig
from dubpipeline.utils.logging import info, step, warn, error
from dubpipeline.utils.timing import timed

DEBUG:bool = False

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


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        warn(f"[ALIGN] Bad env {name}={v!r}, using default={default}")
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        warn(f"[ALIGN] Bad env {name}={v!r}, using default={default}")
        return int(default)


def _pad_or_trim_to_samples(data: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Возвращает сигнал ровно target_samples по длине:
    - если короче: дополняет тишиной в конце
    - если длиннее: обрезает хвост
    Поддерживает mono (N,) и multi (N,C).
    """
    if target_samples <= 0:
        return data[:0]

    if data.ndim == 1:
        n = data.shape[0]
        if n == target_samples:
            return data
        if n > target_samples:
            return data[:target_samples]
        out = np.zeros((target_samples,), dtype=np.float32)
        out[:n] = data.astype(np.float32, copy=False)
        return out

    # (N, C)
    n, c = data.shape[0], data.shape[1]
    if n == target_samples:
        return data
    if n > target_samples:
        return data[:target_samples, :]
    out = np.zeros((target_samples, c), dtype=np.float32)
    out[:n, :] = data.astype(np.float32, copy=False)
    return out


def _write_pcm16(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr, subtype="PCM_16")


def align_segments(
    segments,
    tts_dir: Path,
    aligned_dir: Path,
    ffmpeg_path: str = "ffmpeg",
    eps: float = 0.01,
):
    """
    Выравнивает длительность каждого TTS-сегмента под (end - start).

    ВАЖНО (качество голоса):
    - Если TTS короче окна сегмента -> НЕ замедляем (не тянем atempo < 1.0),
      а дополняем тишиной до target (padding). Это убирает "гнусавость/пьяницу".
    - Если TTS длиннее окна сегмента -> ускоряем (atempo > 1.0), но с ограничением
      max_speedup (по умолчанию 1.25). Если не влезает даже после ограничения —
      обрежем хвост до target (лучше чем наезжать на следующий сегмент).

    Управление через env:
      DUBPIPELINE_ALIGN_DISABLE_SLOWDOWN=1   (по умолчанию 1)
      DUBPIPELINE_ALIGN_MAX_SPEEDUP=1.25     (по умолчанию 1.25)
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)

    disable_slowdown = bool(_env_int("DUBPIPELINE_ALIGN_DISABLE_SLOWDOWN", 1))
    max_speedup = _env_float("DUBPIPELINE_ALIGN_MAX_SPEEDUP", 1.25)

    metrics = []
    t_align0 = perf_counter()

    for seg in segments:
        seg_id = seg["id"]
        seg_t0 = perf_counter()
        start = float(seg["start"])
        end = float(seg["end"])
        target_dur = end - start

        if target_dur <= 0:
            warn(f"Segment {seg_id} has non-positive duration, skipping")
            metrics.append({"seg_id": int(seg_id), "status": "skip_bad_duration"})
            continue

        in_wav = tts_dir / f"seg_{seg_id:04d}.wav"
        out_wav = aligned_dir / f"seg_{seg_id:04d}.wav"

        if not in_wav.exists():
            warn(f"TTS file not found for segment {seg_id}: {in_wav}")
            metrics.append({"seg_id": int(seg_id), "status": "skip_missing_tts", "in_wav": str(in_wav)})
            continue

        # Читаем, чтобы узнать текущую длительность и sample rate
        t_read0 = perf_counter()
        data, sr = sf.read(in_wav)
        t_read1 = perf_counter()

        cur_dur = len(data) / sr
        diff = cur_dur - target_dur  # + => длиннее, - => короче
        abs_diff = abs(diff)

        if DEBUG:
            info(f"[ALIGN] id={seg_id} target={target_dur:.3f}s cur={cur_dur:.3f}s diff={diff:+.3f}s\n")

        # Если отличие маленькое (< eps), просто копируем файл
        if abs_diff < eps:
            shutil.copy2(in_wav, out_wav)
            seg_t1 = perf_counter()
            metrics.append({
                "seg_id": int(seg_id),
                "status": "copy",
                "target_dur": float(target_dur),
                "cur_dur": float(cur_dur),
                "diff": float(diff),
                "read_sec": round(t_read1 - t_read0, 4),
                "total_sec": round(seg_t1 - seg_t0, 4),
            })
            continue

        target_samples = int(round(target_dur * sr))

        # СЛИШКОМ КОРОТКО: padding вместо замедления (сохраняем тембр)
        if cur_dur < target_dur and disable_slowdown:
            t_fix0 = perf_counter()
            fixed = _pad_or_trim_to_samples(np.asarray(data, dtype=np.float32), target_samples)
            _write_pcm16(out_wav, fixed, sr)
            t_fix1 = perf_counter()

            seg_t1 = perf_counter()
            metrics.append({
                "seg_id": int(seg_id),
                "status": "pad_silence",
                "target_dur": float(target_dur),
                "cur_dur": float(cur_dur),
                "diff": float(diff),
                "read_sec": round(t_read1 - t_read0, 4),
                "fix_sec": round(t_fix1 - t_fix0, 4),
                "total_sec": round(seg_t1 - seg_t0, 4),
            })

            if DEBUG:
                info(f"[ALIGN] id={seg_id} short by {-diff:.3f}s -> pad silence\n")
            continue

        # СЛИШКОМ ДЛИННО: ускоряем atempo (но ограничиваем), затем trim/pad до target
        required_factor = cur_dur / target_dur  # >1 => нужно ускорять
        used_factor = required_factor

        capped = False
        if required_factor > max_speedup:
            used_factor = max_speedup
            capped = True

        filter_str = build_atempo_filter(used_factor)

        tmp_wav = aligned_dir / f"seg_{seg_id:04d}.tmp.wav"
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except Exception:
                pass

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(in_wav),
            "-vn",
            "-acodec", "pcm_s16le",
            "-filter:a", filter_str,
            "-ar", str(sr),
            str(tmp_wav),
        ]

        t_ff0 = perf_counter()
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        t_ff1 = perf_counter()

        ffmpeg_ok = (result.returncode == 0) and tmp_wav.exists()
        if not ffmpeg_ok:
            warn(f"[ALIGN] ffmpeg failed for segment {seg_id}, fallback to hard trim/pad. rc={result.returncode}")
            if DEBUG:
                warn(result.stderr[-2000:])
            tmp_data = np.asarray(data, dtype=np.float32)
            tmp_sr = sr
        else:
            t_r20 = perf_counter()
            tmp_data, tmp_sr = sf.read(tmp_wav)
            t_r21 = perf_counter()
            if tmp_sr != sr:
                warn(f"[ALIGN] Sample rate changed for seg {seg_id}: {tmp_sr} vs {sr}. Using {tmp_sr}.")
                sr = tmp_sr
                target_samples = int(round(target_dur * sr))
            # доп. метрика чтения tmp
            read2_sec = round(t_r21 - t_r20, 4)
            # чистим tmp
            try:
                tmp_wav.unlink()
            except Exception:
                pass

        # добиваем точную длину, чтобы не было наложений в таймлайне
        t_fix0 = perf_counter()
        fixed = _pad_or_trim_to_samples(np.asarray(tmp_data, dtype=np.float32), target_samples)
        _write_pcm16(out_wav, fixed, sr)
        t_fix1 = perf_counter()

        seg_t1 = perf_counter()
        status = "speedup"
        if capped:
            status = "speedup_capped_trim"

        metrics_row = {
            "seg_id": int(seg_id),
            "status": status,
            "target_dur": float(target_dur),
            "cur_dur": float(cur_dur),
            "diff": float(diff),
            "required_factor": float(required_factor),
            "used_factor": float(used_factor),
            "read_sec": round(t_read1 - t_read0, 4),
            "ffmpeg_sec": round(t_ff1 - t_ff0, 4),
            "fix_sec": round(t_fix1 - t_fix0, 4),
            "total_sec": round(seg_t1 - seg_t0, 4),
        }
        # если читали tmp успешно
        if ffmpeg_ok:
            metrics_row["read2_sec"] = read2_sec

        metrics.append(metrics_row)

        if DEBUG:
            info(f"[ALIGN] id={seg_id} long by {diff:.3f}s -> atempo {used_factor:.4f} (need {required_factor:.4f})\n")

    t_align1 = perf_counter()
    try:
        metrics_path = aligned_dir / "align_metrics.json"
        payload = {
            "count": len(metrics),
            "total_sec": round(t_align1 - t_align0, 4),
            "settings": {
                "disable_slowdown": bool(disable_slowdown),
                "max_speedup": float(max_speedup),
                "eps": float(eps),
            },
            "segments": metrics,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        info(f"[METRICS] Align saved: {metrics_path}\n")
    except Exception as ex:
        warn(f"[METRICS] Failed to write align_metrics.json: {ex}\n")

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
    if not segments:
        warn("[MIX] No segments to mix. Keeping existing audio as-is.")
        return

    # Определим максимальный end, чтобы посчитать длину таймлайна
    t_mix0 = perf_counter()
    mix_metrics = []
    max_end = max(float(seg["end"]) for seg in segments)
    info(f"[MIX] Max end time: {max_end:.3f}s")

    # Найдём первый существующий сегмент, чтобы узнать sr и кол-во каналов
    sr = None
    channels = None

    for seg in segments:
        seg_id = seg["id"]
        path = aligned_dir / f"seg_{seg_id:04d}.wav"
        if path.exists():
            t_read0 = perf_counter()
            data, sr_tmp = sf.read(path)
            t_read1 = perf_counter()
            sr = sr_tmp
            # soundfile возвращает (N,) для mono и (N, C) для multi
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]
            break

    if sr is None:
        warn("[MIX] No aligned WAVs found to determine sample rate. Keeping existing audio as-is.")
        return

    if DEBUG:
        warn(f"[MIX] Using sample rate={sr}, channels={channels}\n")

    total_samples = int(math.ceil((max_end + 0.1) * sr))
    if channels == 1:
        timeline = np.zeros((total_samples,), dtype=np.float32)
    else:
        timeline = np.zeros((total_samples, channels), dtype=np.float32)

    # Размещаем сегменты
    for seg in segments:
        seg_id = seg["id"]
        seg_t0 = perf_counter()
        start = float(seg["start"])
        end = float(seg["end"])
        target_dur = end - start

        path = aligned_dir / f"seg_{seg_id:04d}.wav"
        if not path.exists():
            warn(f"Aligned segment file missing, skipping: {path}")
            continue

        t_read0 = perf_counter()
        data, sr_tmp = sf.read(path)
        t_read1 = perf_counter()
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
            warn(
                f"After alignment, segment {seg_id} duration differs from "
                f"target by {diff:.3f}s (cur={cur_dur:.3f}, target={target_dur:.3f})"
            )

        start_idx = int(round(start * sr))
        end_idx = start_idx + len(data)

        if end_idx > len(timeline):
            end_idx = len(timeline)
            data = data[: end_idx - start_idx]

        info(
            f"[MIX] Place seg {seg_id} at {start_idx}..{end_idx} "
            f"({start:.3f}s–{start + len(data) / sr:.3f}s)"
        )

        timeline[start_idx:end_idx] += data
        seg_t1 = perf_counter()
        mix_metrics.append({
            "seg_id": int(seg_id),
            "read_sec": round(t_read1 - t_read0, 4),
            "place_sec": round(seg_t1 - seg_t0, 4),
            "samples": int(len(data)),
        })

    # Записываем результат
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, timeline, sr)
    info(f"Mixed timeline written to: {out_wav}")
    t_mix1 = perf_counter()
    try:
        metrics_path = out_wav.parent / "mix_metrics.json"
        payload = {
            "count": len(mix_metrics),
            "total_sec": round(t_mix1 - t_mix0, 4),
            "segments": mix_metrics,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        info(f"[METRICS] Mix saved: {metrics_path}\n")
    except Exception as ex:
        warn(f"[METRICS] Failed to write mix_metrics.json: {ex}\n")

@timed("step_align", log=info)
def run(cfg:PipelineConfig):

    segments_path = Path(cfg.paths.segments_ru_file)
    tts_dir = Path(cfg.paths.segments_path)
    aligned_dir = Path(cfg.paths.segments_align_path)
    out_mix_path = cfg.paths.audio_wav

    info(f"Loading segments from {segments_path}")
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    # сортируем по start на всякий случай
    segments = sorted(segments, key=lambda s: s["start"])

    if not segments:
        warn("[ALIGN] No translated segments found. Keeping extracted audio without remix.")
        return

    step("Aligning segment durations...")
    align_segments(segments, tts_dir, aligned_dir)

    any_aligned = any((aligned_dir / f"seg_{int(seg['id']):04d}.wav").exists() for seg in segments if "id" in seg)
    if not any_aligned:
        warn("[ALIGN] No aligned segment files were produced. Keeping extracted audio without remix.")
        return

    step("Mixing aligned segments into single WAV...")
    mix_aligned_segments_to_timeline(segments, aligned_dir, out_mix_path)
