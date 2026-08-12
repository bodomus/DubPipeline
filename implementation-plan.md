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

## Final Translation Quality Pass

1. [completed] Add a fixed 25-line EN technical translation dataset and a local-only Qwen GPU comparison runner.
2. [completed] Strengthen the default prompt with general technical-context interpretation guidance without phrase-specific substitutions.
3. [completed] Compare current sampling, conservative sampling, and deterministic decoding on the same dataset.
4. [completed] Run five uncached repetitions for the three required difficult phrases and assess meaning, terminology, natural Russian, conciseness, and TTS suitability manually.
5. [completed] Select and encode one versioned production generation profile in a single structured definition.
6. [completed] Preserve prompt hash and generation profile in Qwen cache identity and add focused tests for prompt guidance, settings, thinking disablement, validation, and token protection.
7. [completed] Run the final production profile on the real RTX 4080, record full outputs, elapsed time, and peak VRAM.
8. [completed] Update CRG and Graphify, run focused and regression tests, self-review the branch diff, and update `review-DUB-79.md` with the final verdict.
