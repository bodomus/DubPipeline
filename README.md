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

Получение аудио файла из видео

ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le output.wav



# **_Запуск_**
получить первый .srt из реального аудио.
Положить фвйл D:\AI\DubPipeline\tests\data\lecture_sample.wav
Запустить
```bash
python .\tools\test_whisperx_to_srt.py
```
Ожидание
tests\output\lecture_sample.en.srt

# ** Запуск 3 шага**
```bash
python .\tools\whisperx_dump_segments.py
```


# **Запуск 4 шага**
pip install -r requirements_translate.txt

```bash
python .\tools\translate_segments.py
```

# **Запуск 5 шага**
## Шаг 5 — Генерация русской озвучки (TTS).
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


3. Слияние звуковой дорожки в видео файл 
mux_ru_audio.py

1) Заменить оригинальную аудиодорожку:
```bash

python .\tools\mux_ru_audio.py ^
  --video in\lecture_sample.mp4 ^
  --audio out\lecture_sample.ru_full.wav ^
  --out out\lecture_sample.ru.replace.mp4 ^
  --mode replace
```
2) Добавить русскую дорожку к оригиналу (рекомендую в MKV):