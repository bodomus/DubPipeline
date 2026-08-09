# Implementation Plan

## DUB-79

1. Extend the translation provider registry with `qwen`.
2. Add `QwenTranslationProvider` using Transformers `AutoTokenizer` and `AutoModelForCausalLM`.
3. Implement local-only model loading, deterministic generation settings, output cleanup, and release behavior.
4. Extend `TranslationProviderContext` with Qwen-specific model/device/dtype/max-token/prompt settings.
5. Add `qwen3_8b` to the model catalog as a supported HF snapshot model.
6. Extend `TranslationConfig`, env mapping, YAML default, CLI choices, and runtime resolution.
7. Add tests with fake tokenizer/model/torch paths so no model downloads happen.
8. Update README and create `review-DUB-79.md`.
9. Run focused tests, CLI help/plan validation, update CRG, refresh Graphify because architecture changes.

