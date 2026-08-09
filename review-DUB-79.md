# Implementation Report

## Ticket

DUB-79 - Local Qwen Translation Provider.

Roadmap document number: DUB-76.

## Workflow

- Level: 2.
- Graphify: refreshed before and after implementation.
- CRG: full build before implementation, update after implementation.
- Working tree before changes: branch created from `master`; only pre-existing untracked `tests/TXT/1.txt`.

## Scope

- CLI: added `qwen` to `--translation-provider` choices.
- GUI: no direct UI changes.
- Config: added Qwen provider-specific fields under `translation`.
- Pipeline steps: `step_translate` behavior unchanged, still uses `TranslatorService`.
- Models/runtime: added Qwen HF snapshot model and provider lifecycle.
- Media/artifacts: no changes to timestamps, segment ids, TTS, muxing, or output names.

## Investigation

- Current behavior: translation providers existed for Argos and internal HF seq2seq backends.
- Expected behavior: Qwen selectable as a local provider without changing default behavior.
- Root cause/gap: provider registry had no local LLM provider and catalog Qwen entries were planned/unsupported.
- Main symbols: `TranslationProviderContext`, `QwenTranslationProvider`, `TranslatorService._resolve_runtime`, `TranslationConfig`, `build_parser`, model catalog.
- Expected blast radius: translation runtime/config/CLI/tests/docs.
- Compatibility concerns: existing YAML may contain `model_id: opus_mt`; explicit `provider: qwen` now selects `qwen3_8b` to avoid provider/backend mismatch.

## Changes

- Added `QwenTranslationProvider` using Hugging Face Transformers `AutoTokenizer` and `AutoModelForCausalLM`.
- Added default Qwen EN->RU instruction prompt and defensive output cleanup.
- Added local-only model loading, deterministic generation, cache reuse, and resource release.
- Added `qwen3_8b` catalog model for `Qwen/Qwen3-8B` with `hf_snapshot` installer.
- Added `translation.model`, `translation.device`, `translation.dtype`, `translation.max_new_tokens`, and `translation.prompt`.
- Added CLI/config docs and sample YAML fields.
- Added unit tests for registration, config parsing, prompt construction, cleanup, device/dtype validation, lifecycle reuse, and CLI parsing.

## Backend Choice

Hugging Face Transformers was selected because the repository already uses torch/HF snapshot storage and model installer patterns. No llama.cpp/GGUF lifecycle exists in DubPipeline today, so adding that backend would increase scope and duplicate model-management infrastructure.

## Model Lifecycle

Qwen provider lifecycle:

```text
create provider
-> load tokenizer/model once for (model_ref, device, dtype)
-> translate N segments/calls
-> release cache entry and CUDA cache when requested by existing translate.release_vram flow
```

Translation loading uses `local_files_only=True`, so unit tests and normal translation do not perform hidden downloads.

## Configuration Example

```yaml
translation:
  provider: qwen
  model_id: qwen3_8b
  model: Qwen/Qwen3-8B
  device: cuda
  dtype: auto
  max_new_tokens: 512
```

CLI smoke:

```powershell
python -m dubpipeline.cli run video.pipeline.yaml --translation-provider qwen --set translation.device=cuda --set translation.model=Qwen/Qwen3-8B
```

## Graph and Source Validation

- Graphify identified translation providers, service, config/CLI, model catalog/installer, TTS, and WhisperX lifecycle code as the relevant working set.
- CRG post-change update succeeded with expected translation/config/CLI impact.
- Source validation confirmed `step_translate` still depends only on `TranslatorService`.
- No ASR/TTS/mux/timing code paths were changed.

## Validation

- Focused: `pytest tests/test_translation_models.py tests/test_cli.py -q` -> 63 passed, 3 warnings, 6 subtests passed.
- Affected: `pytest tests/test_translation_models.py tests/test_model_installer.py tests/test_cli.py -q` -> 73 passed, 3 warnings, 6 subtests passed.
- Wide `pytest tests -q` -> 139 passed, 2 existing unrelated failures, 3 warnings, 6 subtests passed.
- Root `pytest -q` -> existing collection error in `tools/test_whisperx_basic.py`.
- CLI help: `python -m dubpipeline.cli --help` passed.
- Run help: `python -m dubpipeline.cli run --help` passed and lists `auto, argos, qwen`.
- Plan mode: `python -m dubpipeline.cli run dubpipeline\video.pipeline.yaml --in-file tests\TXT\1.txt --translation-provider qwen --set translation.device=cuda --set translation.model=Qwen/Qwen3-8B --plan` passed without model loading.
- Compile smoke: `python -m compileall dubpipeline tests -q` passed.

## Known Limitations

- Manual real Qwen GPU translation was not run because no confirmed local `Qwen/Qwen3-8B` snapshot/VRAM run was available in this task.
- Full suite remains blocked by pre-existing unrelated failures:
  - `tests/test_audio_mix_step.py::AudioMixStepTests::test_external_voice_skips_extract_audio_step`
  - `tests/test_tts_provider_import.py::TtsProviderImportTests::test_coqui_provider_without_package_has_clear_error`
- Root pytest remains blocked by `tools/test_whisperx_basic.py` collection against a mocked `whisperx`.

## Prerequisites for DUB-80

- Use `TranslationProviderContext` as the extension point for context-aware request data.
- Keep segment count, ids, timestamps, speaker metadata, and order unchanged in `step_translate`.
- Add provider capability/context request tests with mocked Qwen provider; do not load real Qwen in unit tests.

