# Implementation Plan

## DUB-79

1. Extend the translation provider registry with `qwen`.
2. Add `QwenTranslationProvider` using Transformers `AutoTokenizer` and `AutoModelForCausalLM`.
3. Implement local-only model loading, Qwen3 non-thinking prompt formatting, output cleanup/validation, and release/offload behavior.
4. Extend `TranslationProviderContext` with Qwen-specific model/device/dtype/quantization/max-token/prompt settings.
5. Add `qwen3_8b` to the model catalog as a supported HF snapshot model.
6. Extend `TranslationConfig`, env mapping, YAML default, CLI choices, and runtime resolution.
7. Add tests with fake tokenizer/model/torch paths so no model downloads happen, including cache identity and lifecycle behavior.
8. Install/check the real local Qwen FP8 model through the existing installer and run CUDA smoke with VRAM numbers.
9. Update README and create `review-DUB-79.md`.
10. Run focused tests, CLI help/plan validation, update CRG, refresh Graphify because architecture changes.
