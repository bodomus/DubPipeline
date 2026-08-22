# DUB-83 Implementation Plan

## Scope

- Translation step: remove unsafe source fallback, add retry, add validation.
- TTS step: reject unsafe source-text fallback before synthesis.
- Tests: update regression tests for placeholder preservation, retry success/failure,
  source-equals-target validation, technical-term false positive protection, and short
  segment allowance.

## Non-Scope

- No mux changes.
- No `AudioTrackUpdateMode` changes.
- No ASR/TTS model-quality changes.
- No provider/model dependency upgrades.

## Validation

- Focused: `tests/test_text_technical_tokens.py`.
- Broader: translation/TTS related tests.
- CLI smoke and plan mode when possible.
- CRG update after implementation.