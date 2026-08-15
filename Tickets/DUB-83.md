# DUB-83 — Fix unsafe source-text fallback after technical-token restore failure

Source: https://bodomus.youtrack.cloud/issue/DUB-83

## Summary

DubPipeline can currently treat an EN source segment as a successful RU translation
when technical-token restoration fails after translation. The unsafe chain is:

source segment -> technical-token protection -> translation provider -> restore failure
-> source text fallback -> `text_tgt` / `text_ru` -> TTS.

## Goal

Prevent technical-token restore failures from silently writing source English text as
successful target-language translation. Add controlled retry/recovery, conservative
translation validation, diagnostics, and regression coverage.

## Acceptance Notes

- Restore failure must not fall back to source text as success.
- Retry/recovery must run when placeholders are damaged.
- Exhausted recovery must raise an explicit translation failure.
- EN source text must not reach RU TTS as a successful translation.
- Legitimate technical terms such as Unreal Engine, Nanite, Lumen, PCG, and short
  technical segments must not be rejected by a coarse Latin-text rule.
