# DUB-78 Implementation Plan

## Plan

1. Add a small provider abstraction under `dubpipeline/translation/`.
2. Move runtime translation implementations behind provider classes:
   - `ArgosTranslationProvider`;
   - HF sequence-to-sequence provider for current `nllb` and `opus_mt` backends, to preserve current behavior.
3. Keep `TranslatorService` as the public orchestration facade used by CLI/steps/tests.
4. Add `translation.provider` to `TranslationConfig`.
   - Empty/default provider resolves from current model backend to preserve behavior.
   - Explicit `argos` selects the Argos model/provider.
   - Unsupported values fail with a clear `translation provider 'xyz' is not supported` error.
5. Add CLI `--translation-provider {auto,argos}` and map it to `translation.provider`.
6. Preserve legacy `translate.backend` / `translate.hf_model` synchronization.
7. Add tests for default provider, explicit Argos provider, invalid provider, factory/registry, existing translation behavior, and CLI/config parsing.
8. Update docs with a short note on provider architecture and how to add a provider.
9. Run focused tests first, then broader available tests.
10. Update CRG and inspect post-change impact.
11. Create `review-DUB-78.md` with the implementation report.

## Deliberate Non-Changes

- No Qwen implementation.
- No model downloads in tests.
- No ASR/TTS/media flow changes.
- No GUI redesign.

