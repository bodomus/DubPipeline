from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List

import numpy as np
import soundfile as sf


PathLike = Union[str, Path]


def concat_wavs(
    input_wavs: Iterable[PathLike],
    output_wav: PathLike,
    *,
    gap_ms: int = 0,
    target_sr: Optional[int] = None,
    target_channels: Optional[int] = None,
    subtype: str = "PCM_16",
    block_frames: int = 65536,
) -> Path:
    """
    Склеивает несколько WAV в один (streaming), без ffmpeg.

    Требования по умолчанию:
      - sample rate и число каналов у всех входных файлов должны совпадать.
    Можно принудить:
      - target_sr: ожидаемый SR (иначе берётся из первого файла)
      - target_channels: 1 или 2 (иначе берётся из первого файла)

    gap_ms: добавляет тишину между файлами.

    Возвращает Path к output_wav.
    """
    in_list: List[Path] = [Path(p) for p in input_wavs]
    if not in_list:
        raise ValueError("concat_wavs: input_wavs is empty")

    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Берём эталонные параметры из первого файла
    with sf.SoundFile(str(in_list[0]), mode="r") as f0:
        ref_sr = f0.samplerate
        ref_ch = f0.channels

    sr = int(target_sr or ref_sr)
    ch = int(target_channels or ref_ch)
    if ch not in (1, 2):
        raise ValueError(f"target_channels must be 1 or 2, got {ch}")

    gap_frames = int(sr * (gap_ms / 1000.0))
    gap_block = None
    if gap_frames > 0:
        # блок тишины (float32)
        gap_block = np.zeros((min(gap_frames, block_frames), ch), dtype=np.float32)

    def _convert_channels(x: np.ndarray, in_ch: int, out_ch: int) -> np.ndarray:
        """x shape: (frames, in_ch)"""
        if in_ch == out_ch:
            return x
        if in_ch == 1 and out_ch == 2:
            return np.repeat(x, 2, axis=1)
        if in_ch == 2 and out_ch == 1:
            return x.mean(axis=1, keepdims=True)
        raise ValueError(f"Unsupported channel conversion: {in_ch} -> {out_ch}")

    # Пишем выходной файл (float32 внутри, subtype задаёт формат на диске)
    with sf.SoundFile(str(out_path), mode="w", samplerate=sr, channels=ch, subtype=subtype) as fout:
        for idx, in_wav in enumerate(in_list):
            with sf.SoundFile(str(in_wav), mode="r") as fin:
                if fin.samplerate != sr:
                    raise ValueError(
                        f"Sample rate mismatch in {in_wav}: {fin.samplerate} != {sr}. "
                        f"(Сделайте единый SR на этапе TTS или добавьте ресэмплинг.)"
                    )

                in_ch = fin.channels
                # Читаем блоками
                while True:
                    data = fin.read(block_frames, dtype="float32", always_2d=True)
                    if data.size == 0:
                        break
                    data = _convert_channels(data, in_ch, ch)
                    fout.write(data)

            # Тишина между кусками (кроме последнего)
            if gap_frames > 0 and idx != len(in_list) - 1:
                remaining = gap_frames
                while remaining > 0:
                    n = min(remaining, block_frames)
                    if n == gap_block.shape[0]:
                        fout.write(gap_block)
                    else:
                        fout.write(np.zeros((n, ch), dtype=np.float32))
                    remaining -= n

    return out_path
