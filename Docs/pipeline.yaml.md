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

## Languages

Current public multilingual flow is built around a selected `source -> target` pair.

Supported GUI/CLI language codes:

- `de`
- `fr`
- `es`
- `ru`

CLI overrides:

```bash
python -m dubpipeline.cli run video.pipeline.yaml --lang-src fr --lang-dst de
```

Rules:

- if `steps.translate=true`, `source` and `target` must be different
- model availability is resolved for the selected pair
- GUI and CLI validate the public selection against `de/fr/es/ru`
- legacy YAML values such as `en` are still tolerated for backward compatibility

## Target-Aware Artifacts

Important output naming now follows `languages.tgt`:

```yaml
languages:
  src: fr
  tgt: de

paths:
  segments_tgt_json: "{out_dir}/{project_name}.segments.{target_lang}.json"
  tts_segments_dir: "{out_dir}/segments/tts_{target_lang}_segments"
  tts_segments_aligned_dir: "{out_dir}/segments/tts_{target_lang}_segments_aligned"
  final_video: "{out_dir}/{project_name}.{target_lang}.muxed.mp4"
```

The final mux metadata also follows the selected target language. By default:

- `de -> deu / German (DubPipeline)`
- `fr -> fra / French (DubPipeline)`
- `es -> spa / Spanish (DubPipeline)`
- `ru -> rus / Russian (DubPipeline)`
