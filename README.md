# DubPipeline
translate videos
Steps:
ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le output.wav
.\tools\test_whisperx_to_srt.py
.\tools\whisperx_dump_segments.py
.\tools\translate_segments.py
.\tools\tts_ru_segments_xtts.py
.\tools\tts_align_and_mix.py
.\tools\mux_ru_audio.py

Получение аудио файла из видео файла через ffmpeg

ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le output.wav
Папка .tests для тестов

# _Установка_
```bash
  python -m venv .venv
```
```bash
  .venv\Scripts\activate
```
```bash
  pip install -r requirements.txt
```

# **_Запуск 1 шага_**

получить первый .srt из реального аудио.
Положить фвйл .tests\data\lecture_sample.wav
Запустить
```bash
python .\tools\test_whisperx_to_srt.py
```
Ожидание
tests\output\lecture_sample.en.srt


# Запуск 2 шага
#### Диаризация + формирование сегментов для перевода.
```bash
python .\tools\whisperx_dump_segments.py
```


# **Запуск 3 шага**
pip install -r requirements_translate.txt

```bash
python .\tools\translate_segments.py
```

# **Запуск 4 шага**
## Шаг 4 — Генерация русской озвучки (TTS).
```bash
Prerequisites
pip install --upgrade coqui-tts soundfile
```
Run translate segments
```bash
python .\tools\tts_ru_segments_xtts.py
```
Run Скрипт: выровнять TTS-сегменты и склеить в один WAV
```bash
pip install soundfile numpy
```
Запуск 
```bash
python .\tools\tts_align_and_mix.py
```

# 5. Слияние звуковой дорожки в видео файл 

Запуск 
```bash
python .\tools\mux_ru_audio.py
```

1) Заменить оригинальную аудиодорожку:
```bash

python .\tools\mux_ru_audio.py ^
  --video in\lecture_sample.mp4 ^
  --audio out\lecture_sample.ru_full.wav ^
  --out out\lecture_sample.ru.replace.mp4 ^
  --mode replace
```
2) Добавить русскую дорожку к оригиналу (рекомендую в MKV):
(venv) λ python .\tools\mux_ru_audio.py ^
  --video in\lecture_sample.mp4 ^
  --audio out\lecture_sample.ru_full.wav ^
  --out out\lecture_sample.ru.with_both.mkv ^
  --mode add

3) Добавить русскую дорожку и сделать её первой:
(venv) λ python .\tools\mux_ru_audio.py ^
  --video in\lecture_sample.mp4 ^
  --audio out\lecture_sample.ru_full.wav ^
  --out out\lecture_sample.ru.rus_first.mkv ^
  --mode rus_first


# Опции

##### Включить явно (не обязательно, он и так включён):
$env:DUBPIPELINE_TRANSLATE_SENT_FALLBACK="1"

##### Отключить
$env:DUBPIPELINE_TRANSLATE_SENT_FALLBACK="0"

##### включает попытку синтезировать сегмент TTS одним вызовом
DUBPIPELINE_TTS_TRY_SINGLE_CALL

##### задаёт лимит символов для этой попытки; если не проходит — текст режется на чанки.
DUBPIPELINE_TTS_TRY_SINGLE_CALL_MAX_CHARS  

##### максимальный размер одного русского чанка для TTS (влияет на количество part-файлов и стабильность/качество)
DUBPIPELINE_TTS_MAX_RU_CHARS 

##### включить/выключить склейку хвостов (дефолт 1)
DUBPIPELINE_WHISPERX_MERGE_DANGLING=1/0

##### максимальный разрыв между сегментами (сек)
DUBPIPELINE_WHISPERX_DANGLING_MAX_GAP=0.60

##### максимум слов в следующем коротком сегменте
DUBPIPELINE_WHISPERX_DANGLING_MAX_NEXT_WORDS=6 

## CLI ключи (`python -m dubpipeline.cli run <pipeline.yaml> ...`)

- `--in-file <path>` — входной видеофайл (существующий файл), переопределяет `paths.input_video`.
- `--in-dir <path>` — входная директория (существующая папка), переопределяет `paths.input_video`.
- `--in-file` и `--in-dir` взаимно исключающие: нельзя указывать вместе.
- `--recursive` — рекурсивный обход входной директории (если вход — файл, ключ игнорируется с предупреждением).
- `--glob "*.mp4"` — glob-фильтр входных файлов для директории (если вход — файл, ключ игнорируется с предупреждением).
- `--out <dir>` — переопределяет `paths.out_dir` (рабочая/temp директория).
- `--lang-src <code>` — переопределяет `languages.src`.
  WhisperX uses this value for ASR and alignment. Use `--lang-src en` (or another concrete source language) to avoid wrong auto-detection; `--lang-src auto` keeps WhisperX auto-detection and is intended for ASR-only runs without the Translate step.
- `--lang-dst <code>` — переопределяет `languages.tgt`.
- `--steps ...` — управление шагами:
  - patch-форма: `--steps +asr,-tts,+merge` (включить/выключить поверх YAML);
  - list-форма: `--steps asr,translate,tts,merge` (полная замена списка включённых шагов).
- `--usegpu` / `--cpu` — выбор устройства (взаимоисключающие).
- `--rebuild` — принудительно пересоздать артефакты шагов.
- `--delete-temp` / `--keep-temp` — политика очистки temp/work файлов (взаимоисключающие).
- `--plan` — dry-run режим: печатает effective config + входные файлы + шаги и завершает работу без запуска шагов и без записи файлов.

При запуске `--plan`/`run` в summary выводится источник входа (`CLI` или `YAML/ENV/default`) и итоговый `input_video`.

Допустимые canonical id шагов для `--steps`:
- `extract_audio` → `01_extract_audio`
- `asr` → `02_asr_whisperx`
- `translate` → `03_translate`
- `tts` → `04_tts+align`
- `merge` → `05_merge`

## CLI `speak` (text -> WAV)

```bash
python -m dubpipeline.cli speak --text "Привет! Это тест озвучки." --out-audio out.wav --voice <voice_id>
```

```bash
python -m dubpipeline.cli speak --text-file book.txt --out-audio book.wav
```

```bash
python -m dubpipeline.cli speak --text-file chapter.txt --out-audio chapter.wav --speaker-wav sample.wav
```

Рекомендации:
- Для больших книг лучше запускать по главам.
- Управляйте длиной сегментов через `tts.text_max_chars` (по умолчанию 400).
- Пауза между сегментами задаётся `tts.gap_ms`.
- `--plan` строит только план сегментов и артефактов без загрузки модели.
- На CPU качество/скорость хуже, для больших текстов предпочтительнее GPU.

## GUI: единый входной путь

- В GUI используется один общий инпут `Input path` и переключатель режима `Один файл` / `Папка`.
- Режим `Один файл` трактует `Input path` как путь к видеофайлу.
- Режим `Папка` трактует `Input path` как путь к директории с видео.
- Кнопка `Browse...` открывает файловый или папочный диалог в зависимости от выбранного режима.

## Multilingual Translation

- Public language selection in the current GUI/CLI flow is supported for `de`, `fr`, `es`, `ru`.
- GUI now has explicit `source -> target` language selectors.
- CLI overrides are available through `--lang-src <code>` and `--lang-dst <code>`.
- If the `Translate` step is enabled, `source` and `target` must be different.
- Translation model availability is resolved per language pair, so a model may be available for one pair and unavailable for another.
- Legacy YAML with `en` can still be loaded for backward compatibility, but new GUI/CLI selection is intentionally limited to `de/fr/es/ru`.

## Translation Providers

Translation runtime code is routed through provider classes in `dubpipeline.translation.providers`.
The default provider is `auto`, which preserves the existing model-based behavior:
HF sequence-to-sequence models use the HF provider, and Argos models use the Argos provider.

YAML can select a provider through:

```yaml
translation:
  provider: auto
```

The CLI exposes the same setting:

```bash
python -m dubpipeline.cli run video.pipeline.yaml --translation-provider argos
```

Supported public values are `auto`, `argos`, and `qwen`. Future providers should add a provider
class, register it in the provider factory, add tests that do not download models, and only then
expose the value in config/CLI.

### Local Qwen Provider

The Qwen provider uses a locally cached Hugging Face snapshot and does not download model files
during translation. Install/cache the model first through the Models flow.

```yaml
translation:
  provider: qwen
  model_id: qwen3_8b
  model: Qwen/Qwen3-8B
  device: cuda
  dtype: auto
  max_new_tokens: 512
```

CLI smoke example:

```powershell
python -m dubpipeline.cli run video.pipeline.yaml --translation-provider qwen --set translation.device=cuda --set translation.model=Qwen/Qwen3-8B
```

Qwen loads the tokenizer/model once per process cache key and reuses it across segment translation
calls. Use `translation.device=cpu` only deliberately; `translation.device=cuda` fails clearly when
CUDA is unavailable.

## Target-Aware Outputs

- Translated segments are written to `*.segments.{target_lang}.json`.
- TTS segment folders are written to `segments/tts_{target_lang}_segments` and `segments/tts_{target_lang}_segments_aligned`.
- Final muxed video is written to `*.{target_lang}.muxed.mp4`.
- Final audio track metadata is derived from the selected target language:
  - `de -> deu / German (DubPipeline)`
  - `fr -> fra / French (DubPipeline)`
  - `es -> spa / Spanish (DubPipeline)`
  - `ru -> rus / Russian (DubPipeline)`

## Task Handoff

After each YouTrack task, save handoff context in a comment using this template:

- What was done
- Files changed
- Decisions taken
- What remains
- Risks / limitations
- Next step
