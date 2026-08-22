# Implementation Report

## Ticket
DUB-83 — Fix unsafe source-text fallback after technical-token restore failure.

## Workflow
- Level: 2.
- Branch: `codex/dub-83-technical-token-fallback`.
- Graphify: queried translation/TTS technical-token flow.
- CRG: `code-review-graph update --brief` and `detect-changes --brief` completed with UTF-8 output.

## Root Cause
`dubpipeline.steps.step_translate.translate_segments` caught `TechnicalTokenError`
from `TechnicalTokenProtector.restore`, logged a warning, and cached the original
source text as the translated value. For RU targets this then populated both
`text_tgt` and `text_ru`, allowing English source text to reach TTS as a successful
translation.

## Changes
- Added `dubpipeline/text/translation_validation.py` with conservative translation
  validation:
  - rejects meaningful source-equals-target translations when source/target differ;
  - rejects long RU target text that is clearly still English;
  - allows short/code-like technical segments.
- Updated `dubpipeline/steps/step_translate.py`:
  - removed source-text fallback after restore failure;
  - retries damaged protected translations once with `sent_fallback=False`;
  - raises `TranslationModelError` after exhausted recovery or validation failure;
  - validates cached and newly translated values before writing `text_tgt` / `text_ru`;
  - logs segment id, source/target language, placeholder, attempt, reason, and action.
- Updated `dubpipeline/steps/step_tts_core.py`:
  - rejects unsafe source-text fallback before loading TTS/synthesizing audio.
- Updated `tests/test_text_technical_tokens.py` with DUB-83 regressions.

## Retry / Recovery
Attempt 1 translates protected text normally. If placeholder restore fails, the
segment is queued for a second translation attempt. Attempt 2 translates the same
protected text with sentence fallback disabled, then restore and validation run again.
If either still fails, the translation step fails explicitly and no successful output
JSON is written for that segment.

## Validation
- `text_tgt/text_ru == source` fails for meaningful EN -> RU sentences.
- A RU target with long, mostly Latin text and no Cyrillic fails as likely untranslated.
- Correct Russian text containing `Unreal Engine 5.7`, `Nanite`, `Lumen`, and `PCG`
  passes.
- Short technical segments such as `OK`, `UE`, and `PCG` pass.

## Tests Run
- Passed: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp C:\Users\bodom\.codex\visualizations\2026\08\15\01a00402-eca6-76c2-a078-b63ee676a223\pytest-dub83 tests/test_text_technical_tokens.py`
- Passed: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp C:\Users\bodom\.codex\visualizations\2026\08\15\01a00402-eca6-76c2-a078-b63ee676a223\pytest-dub83-related2 tests/test_translation_models.py tests/test_tts_provider_import.py -k "not coqui_provider_without_package_has_clear_error"`
- Passed: `.venv\Scripts\python.exe -m py_compile dubpipeline/steps/step_translate.py dubpipeline/steps/step_tts_core.py dubpipeline/text/translation_validation.py`
- Passed: `.venv\Scripts\python.exe -m dubpipeline.cli --help`
- Passed: `.venv\Scripts\python.exe -m dubpipeline.cli run dubpipeline/video.pipeline.yaml --in-file tests/data/dub_1.mp4 --plan`
- Passed: `.venv\Scripts\python.exe -m dubpipeline.cli speak --text "Тест" --out-audio <visualization-root>\dub83-plan.wav --plan`

## Not Run / Limitations
- Full real Qwen/GUI problem-case reproduction was not run because it requires the
  specific local video/model runtime.
- One environment-specific test was deselected:
  `test_coqui_provider_without_package_has_clear_error`. In this environment Coqui/TTS
  is importable, so the test no longer exercises the missing-package branch and fails
  earlier on missing segment fixture setup.
- Mux and `AudioTrackUpdateMode` were not changed; the defect occurs before mux.

## Diff Stat
```text
dubpipeline/steps/step_translate.py | 121 ++++++++++++++++++++++-
dubpipeline/steps/step_tts_core.py  |  21 +++-
tests/test_text_technical_tokens.py | 191 +++++++++++++++++++++++++++++++++++-
```

Additional new files:
- `dubpipeline/text/translation_validation.py`
- `Tickets/DUB-83.md`
- `Tickets/DUB-83-investigation.md`
- `Tickets/DUB-83-implementation-plan.md`
- `review-DUB-83.md`