# DUB-83 Investigation

## Workflow

- Level: 2.
- Branch: `codex/dub-83-technical-token-fallback`.
- Graphify: queried technical-token/translation/TTS flow.
- CRG: `code-review-graph update --brief` and `detect-changes --brief` completed with UTF-8 output.

## Root Cause

`dubpipeline.steps.step_translate.translate_segments` protected technical values,
translated protected text, then called `TechnicalTokenProtector.restore`. When restore
raised `TechnicalTokenError`, the code logged `using source text fallback` and wrote the
original source text into the translation cache. Later it filled `text_tgt` and `text_ru`
from that cache, so TTS consumed English text as if it were a valid RU translation.

## Data Flow

1. `segments.json` source `text`.
2. `TechnicalTokenProtector.protect` replaces numeric/code-like values with `DUBTECHTOKENxxxx`.
3. `TranslatorService.translate_texts` dispatches to HF/Qwen/Argos provider.
4. `TechnicalTokenProtector.restore` validates exactly one placeholder occurrence.
5. `translate_segments` writes `text_tgt`, and `text_ru` for RU target.
6. `step_tts_core._segment_text` prefers `text_tgt`, then `text_ru`, then `text`.

## Expected Fix

Replace source fallback with retry of damaged protected segments. If retry still cannot
restore placeholders or validation says the target is unsafe, raise `TranslationModelError`
before writing a successful translated segment. Add a TTS-side guard for malformed or
stale translated JSON.