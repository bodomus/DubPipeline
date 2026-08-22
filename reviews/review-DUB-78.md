# Implementation Report

## Ticket

DUB-78 — Translation Provider Architecture.

Note: the ticket text refers to DUB-75, but the YouTrack issue number is DUB-78.

## Workflow

- Level: 2.
- Graphify:
  - `graphify update .`
  - Initial sandboxed attempt failed with `[WinError 5] Access is denied`.
  - Escalated preflight succeeded: `969 nodes`, `2324 edges`, `54 communities`.
  - Post-change refresh succeeded: `1012 nodes`, `2433 edges`, `59 communities`.
- CRG:
  - `code-review-graph build --repo .`
  - Preflight result: `74 files`, `770 nodes`, `7691 edges`.
  - Post-change: `$env:PYTHONIOENCODING='utf-8'; code-review-graph update --repo . --brief`
  - Post-change result: incremental update completed; overall risk score shown as `1.00`.
- Working tree before changes:
  - New branch from `master`: `codex/dub-78-translation-provider-architecture`.
  - Pre-existing untracked file: `tests/TXT/1.txt`.

## Scope

- CLI: added `--translation-provider`.
- GUI: no behavior changes.
- Config: added `translation.provider`.
- Pipeline steps: no step order changes.
- Models/runtime: translation runtime is now routed through providers.
- Media/artifacts: no output path or mux changes.

## Investigation

- Current behavior: `TranslatorService` owned model resolution and runtime translation implementations directly.
- Expected behavior: translation should use a provider abstraction while preserving current defaults.
- Root cause/gap: Argos/HF translation paths were hard-wired behind backend conditionals inside `TranslatorService`.
- Main symbols:
  - `TranslatorService`
  - `TranslationConfig`
  - `build_parser`
  - `_build_cli_set`
  - `translate_segments`
- Expected blast radius:
  - translation service/provider modules;
  - config load/save;
  - CLI parse/set generation;
  - translation and CLI tests.
- Compatibility concerns:
  - preserve model-based default behavior;
  - keep `translate.*` legacy fields synchronized;
  - keep `TranslatorService._HF_CACHE` and `_load_hf()` compatibility hooks for existing tests/runtime expectations.

## Changes

- Added `dubpipeline.translation.providers`.
- Added provider abstraction and implementations:
  - `ArgosTranslationProvider`;
  - `HfSeq2SeqTranslationProvider`.
- Kept `TranslatorService` as the public facade.
- Added provider registry/factory:
  - `create_translation_provider`;
  - `resolve_translation_provider_id`;
  - `provider_id_for_backend`.
- Added config field:
  - `translation.provider`.
- Added CLI option:
  - `--translation-provider {auto,argos}`.
- Added docs for translation providers.
- Updated tests for provider factory, explicit Argos selection, invalid provider, config parsing, CLI parsing, and existing translation behavior.

## Graph and Source Validation

- Graphify identified `TranslatorService`, `step_translate.translate_segments`, `TranslationConfig`, `TranslateConfig`, `build_parser`, `_build_cli_set`, and translation tests as the relevant working set.
- CRG confirmed the changed surface spans config, CLI, translation service, and tests.
- Source validation confirmed:
  - pipeline still calls `step_translate.run`;
  - `step_translate.run` still uses `TranslatorService.from_config`;
  - `translate_segments` still receives an object with `translate_texts`;
  - legacy `translate.backend` / `translate.hf_model` synchronization remains in config.
- Discrepancies:
  - CRG `detect-changes` initially failed printing Unicode under cp1251; rerun with `PYTHONIOENCODING=utf-8` succeeded.

## Post-Change Impact

- CRG updated: yes.
- Graphify refreshed: yes, because a new provider module changed structural relationships.
- Blast radius:
  - expected config/CLI/translation/tests/docs impact only.
- Unexpected dependants:
  - existing tests directly call `TranslatorService._load_hf()`, so a compatibility proxy was preserved.
- Artifact compatibility:
  - default translated output path remains unchanged.
  - no media artifacts or mux paths changed.

## Validation

- Focused tests:
  - `$env:TEMP='J:\Projects\!!!AI\DubPipeline\.temp'; $env:TMP='J:\Projects\!!!AI\DubPipeline\.temp'; .\.venv\Scripts\python.exe -m pytest tests/test_translation_models.py tests/test_cli.py -q`
  - Result: `52 passed`, `3 warnings`, `6 subtests passed`.
- Affected tests:
  - `$env:TEMP='J:\Projects\!!!AI\DubPipeline\.temp'; $env:TMP='J:\Projects\!!!AI\DubPipeline\.temp'; .\.venv\Scripts\python.exe -m pytest tests/test_translation_models.py tests/test_model_installer.py tests/test_cli.py -q`
  - Result: `61 passed`, `3 warnings`, `6 subtests passed`.
- Full `tests` package:
  - `$env:TEMP='J:\Projects\!!!AI\DubPipeline\.temp'; $env:TMP='J:\Projects\!!!AI\DubPipeline\.temp'; .\.venv\Scripts\python.exe -m pytest tests -q`
  - Result: `127 passed`, `2 failed`, `3 warnings`, `6 subtests passed`.
  - Existing unrelated failures:
    - `tests/test_audio_mix_step.py::AudioMixStepTests::test_external_voice_skips_extract_audio_step`
    - `tests/test_tts_provider_import.py::TtsProviderImportTests::test_coqui_provider_without_package_has_clear_error`
- Full root pytest:
  - `$env:TEMP='J:\Projects\!!!AI\DubPipeline\.temp'; $env:TMP='J:\Projects\!!!AI\DubPipeline\.temp'; .\.venv\Scripts\python.exe -m pytest -q`
  - Result: collection error in `tools/test_whisperx_basic.py`, because local `whisperx` is a `types.SimpleNamespace` without `load_model`.
- CLI help:
  - `.\.venv\Scripts\python.exe -m dubpipeline.cli --help`
  - Passed.
  - `.\.venv\Scripts\python.exe -m dubpipeline.cli run --help`
  - Passed and shows `--translation-provider`.
- Plan mode:
  - `.\.venv\Scripts\python.exe -m dubpipeline.cli run dubpipeline\video.pipeline.yaml --plan`
  - Failed because the YAML input video path is missing locally.
  - `.\.venv\Scripts\python.exe -m dubpipeline.cli run dubpipeline\video.pipeline.yaml --in-file tests\TXT\1.txt --plan`
  - Passed. No models loaded.
- CPU run: not run; out of scope and would require real media/model setup.
- GPU run: not run; out of scope and would require real media/model setup.
- GUI: not run; no GUI behavior changed.
- FFmpeg/FFprobe: not run; no media command changed.
- Multi-file: not run; provider architecture does not change input enumeration.

## Benchmark

- Required: no.
- Method: not applicable.
- Results: not applicable.

## Documentation and Handoff

- Updated:
  - `README.md`;
  - `dubpipeline/video.pipeline.yaml`;
  - `Tickets/DUB-78-translation-provider-architecture.md`;
  - `investigation.md`;
  - `implementation-plan.md`;
  - `review-DUB-78.md`.
- Handoff:
  - DUB-78 introduces provider architecture but does not add Qwen.
  - Next ticket can add a future provider by implementing a provider class and registering it in `dubpipeline.translation.providers`.

## Remaining Risks

- Explicit `translation.provider: argos` intentionally selects the Argos model for the current language pair.
- Public CLI exposes only `auto` and `argos`; HF remains an internal provider selected by current model backends.
- Full project test suite is not green due unrelated existing tests/tool collection issues documented above.

