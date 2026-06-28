# WhisperX auto language detection may misdetect

DUB: Respect --lang-src in WhisperX ASR and alignment

Problem:
WhisperX auto language detection may misdetect English speech as Welsh (`cy`).
When this happens, ASR output becomes incorrect and alignment crashes because no default align model exists for `cy`.

Required behavior:
- If --lang-src is set to en/ru/de/etc, pass it into model.transcribe(language=...).
- Use the same configured source language for whisperx.load_align_model().
- If --lang-src=auto, keep current autodetect behavior.
- If detected language has no alignment model, fail with a clear error message suggesting --lang-src en.
- Log detected language, configured source language, and alignment language.
