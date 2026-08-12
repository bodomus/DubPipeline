# Implementation Report

## Ticket

DUB-79 - Local Qwen Translation Provider.

Roadmap document number: DUB-76.

Final verdict: READY FOR REVIEW.

## Review Fixes Completed

- Switched the production Qwen default to the official local `Qwen/Qwen3-8B-FP8` snapshot for `qwen3_8b`.
- Added `translation.quantization` with `auto` default and explicit `fp8`, `none`, `awq`, `bnb_4bit`, `bnb_8bit` validation paths.
- Added explicit runtime dependencies required by Qwen3 FP8 Transformers loading: `transformers>=4.51.0`, `accelerate>=0.26.0`, and Triton with Windows/Linux markers.
- Added Qwen cache identity isolation by provider, model id/ref, language pair, quantization, dtype, prompt hash, generation profile, and context version.
- Added folder-run lifecycle support: single-file runs still release after translate; folder runs can keep the CPU-side Qwen cache between files and offload from CUDA before TTS.
- Added final shared Qwen cache cleanup after multi-file CLI runs.
- Added Qwen output validation for empty output, prompt echo, assistant labels, markdown fences, and `<think>` markers.
- Strengthened Qwen prompt rules for placeholders, technical tokens, numbers, units, paths, timestamps, versions, and uppercase abbreviations.
- Added logging for model load, cache hits, offload, and release without logging full prompts.
- Added tests for cache identity, quantization resolution, config parsing, output validation, provider lifecycle, and installer/model behavior.

## Runtime Decisions

- `translation.quantization=auto` resolves to FP8 on CUDA and to non-quantized model refs on CPU.
- `qwen3_8b` installs/checks `Qwen/Qwen3-8B-FP8` through the existing HF snapshot installer.
- Full-precision `Qwen/Qwen3-8B` remains possible through explicit `translation.quantization=none` plus `translation.model`, but it is not the safe default for RTX 4080 16 GB.
- AWQ/bitsandbytes modes are recognized but not defaulted. AutoAWQ can downgrade Transformers below the Qwen3 minimum, so it was not made mandatory.
- Triton cache is redirected to the DubPipeline models root when `TRITON_CACHE_DIR` is not already set, avoiding fragile writes to the user home `.triton` directory.

## Real GPU Smoke

Command shape: real `TranslatorService`, local-only Qwen provider, 5 EN->RU phrases, CUDA, `translation.quantization=fp8`, explicit `_load_qwen()` before generation to separate load/generation timing.

Environment:

- GPU: NVIDIA GeForce RTX 4080
- GPU memory: 16375.5 MiB total
- CUDA capability: 8.9
- Driver: 610.47
- Python: 3.12.10
- Torch: 2.6.0+cu124
- Torch CUDA: 12.4
- Transformers: 4.57.3
- Accelerate: 1.14.0
- Triton: 3.2.0 via `triton-windows 3.2.0.post21`
- Model: `Qwen/Qwen3-8B-FP8`
- Local path: `C:\Users\bodom\AppData\Local\DubPipeline\models\hf\Qwen\Qwen3-8B-FP8`

Measured result:

- Load time: 8.624 s
- Generation time for 5 phrases: 7.551 s
- VRAM before load: 15074.0 MiB free / 16375.5 MiB total
- VRAM after load: 6072.0 MiB free
- VRAM after generation: 5942.0 MiB free
- VRAM after release: 14996.0 MiB free
- Torch peak allocated: 9053.5 MiB
- Torch peak reserved: 9074.0 MiB

Smoke outputs:

- `Feed it into the node.` -> `Покорми его через узел.`
- `Set the value to 42.5 dB.` -> `Установите значение в 42,5 дБ.`
- `Use version 1.2.3 of the API.` -> `Используйте версию 1.2.3 API.`
- `The file is saved as C:/Temp/output.wav.` -> `Файл сохранен как C:/Temp/output.wav.`
- `Wait 00:01:23 before starting TTS.` -> `Подожди 00:01:23 перед началом TTS.`

Smoke verdict: runtime and VRAM are acceptable for RTX 4080 16 GB. Quality is good enough for provider smoke, but phrase-level style tuning remains a normal DUB-80/DUB-81 context/prompt task.

## Validation

- `python -m pytest tests\test_translation_models.py tests\test_model_installer.py` -> 43 passed, 2 warnings.
- `python -m dubpipeline.cli --help` -> passed.
- Real model install: `ModelInstaller().install("qwen3_8b", src_lang="en", tgt_lang="ru")` -> installed `Qwen/Qwen3-8B-FP8`.
- Real GPU smoke: passed with the VRAM numbers above.

## Notes and Risks

- Initial smoke attempts exposed missing runtime dependencies in the local venv: `accelerate`, Triton, and torch version mismatch. The venv was updated to the project-pinned torch 2.6.0+cu124 and the new requirements now document the dependency set.
- The provider still uses Transformers `torch_dtype`; Transformers logs that this is deprecated in favor of `dtype`. This is warning-only and should be cleaned in a future dependency-maintenance pass.
- Qwen output remains probabilistic because Qwen3 documentation recommends sampling for non-thinking mode. Cache identity includes the prompt/generation profile so prompt changes invalidate stale translations.
- No merge to `master` was done.
