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
