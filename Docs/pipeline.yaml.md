# pipeline.yaml: HQ Ducking merge

## Финальный аудио-мерж без stem separation

Режим `audio_merge.mode: hq_ducking` строит финальный звук так:

1. Берется оригинальная аудио-дорожка из видео.
2. Добавляется TTS дорожка.
3. Оригинал приглушается sidechain-компрессором во время TTS.
4. После TTS оригинал возвращается.
5. Опционально применяется loudness normalisation (`loudnorm`).

Это режим без `background.wav`, без KaraFan/Demucs/MDX и без вырезания оригинального голоса.

## Конфиг

```yaml
audio_merge:
  mode: hq_ducking

  original_track: auto   # auto | index:N | lang:xx

  tts_gain_db: 0.0
  original_gain_db: 0.0

  ducking:
    enabled: true
    amount_db: 10.0
    threshold_db: -30.0
    attack_ms: 10
    release_ms: 250
    ratio: 6.0
    knee_db: 6.0

  loudness:
    enabled: true
    target_i: -16.0
    true_peak: -1.5
```

## Параметры

- `original_track`: выбор исходной аудиодорожки (`auto`, `index:N`, `lang:xx`).
- `tts_gain_db`, `original_gain_db`: pre-gain для входных дорожек.
- `ducking.amount_db`: усиление sidechain-сигнала для глубины ducking.
- `ducking.threshold_db`, `ratio`, `attack_ms`, `release_ms`, `knee_db`: параметры компрессора.
- `loudness.enabled`: включает/выключает `loudnorm` на финальном миксе.

## CLI overrides

```bash
python -m dubpipeline.cli run video.pipeline.yaml ^
  --merge-mode hq_ducking ^
  --tts-gain-db 0.0 ^
  --original-gain-db 0.0 ^
  --ducking-amount-db 12 ^
  --ducking-threshold-db -30 ^
  --ducking-attack-ms 10 ^
  --ducking-release-ms 250
```

Отключение loudnorm:

```bash
python -m dubpipeline.cli run video.pipeline.yaml --merge-mode hq_ducking --no-loudnorm
```
