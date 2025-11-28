# DubPipeline
translate videos

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

# **_Запуск_**

```bash
python .\tools\test_whisperx_to_srt.py
```


# **_Запуск_4 шага**
pip install -r requirements_translate.txt

```bash
python python .\tools\translate_segments.py
```

# **_Запуск_5 шага**
Шаг 5 — Генерация русской озвучки (TTS).
```bash
pip install --upgrade coqui-tts soundfile
```

```bash
python .\tools\tts_ru_segments_xtts.py
```


2. Скрипт: выровнять TTS-сегменты и склеить в один WAV
```bash
pip install soundfile numpy
```

Запуск 
.\tools\tts_align_and_mix.py