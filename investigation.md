# DUB-78 Investigation

## Workflow

- Level: 2, because this changes translation architecture, configuration, CLI, tests, and documentation.
- Branch: `codex/dub-78-translation-provider-architecture`.
- YouTrack: `DUB-78`, source text says DUB-75 due to external numbering drift.

## Graphify

- Command: `graphify update .`
- First sandboxed attempt failed with `[WinError 5] Access is denied`.
- Escalated retry succeeded: `969 nodes`, `2324 edges`, `54 communities`.
- Focused query: `graphify query "current translation entry points, provider/model selection, config and CLI dependencies in DubPipeline" --budget 5000`.
- Relevant candidates: `TranslatorService`, `step_translate.translate_segments`, `PipelineConfig`, `TranslationConfig`, `TranslateConfig`, `build_parser`, `_build_cli_set`, `TranslationStepIntegrationTests`, `CliTests`.

## CRG

- Command: `code-review-graph build --repo .`
- Result: `74 files`, `770 nodes`, `7691 edges`.
- Command: `$env:PYTHONIOENCODING='utf-8'; code-review-graph detect-changes --brief`
- Result: completed; risk panel currently reflects existing working-tree changes as well as this branch context.

## Current Behavior

- `dubpipeline.steps.step_translate.run` constructs `TranslatorService.from_config(cfg)`.
- `step_translate.translate_segments` calls `translator.translate_texts(...)` and writes `text_tgt` plus `text_ru` for Russian targets.
- `TranslatorService` currently owns model resolution, availability checks, HF model cache/release, Argos package lookup, and the actual Argos/HF translation implementations.
- `PipelineConfig.translation.model_id` drives model selection. `TranslationConfig.backend` and `model_ref` are derived from model catalog resolution.
- Legacy `translate.backend` and `translate.hf_model` are still synchronized for backward compatibility.
- CLI has language/device/step options but no translation provider option.

## Expected Behavior

- Translation runtime should go through a provider abstraction.
- Argos should be a provider implementation.
- Existing default model/provider behavior should remain unchanged.
- Unknown provider values should fail clearly.
- Tests must avoid external downloads.

## Root Cause / Gap

The translation runtime implementation is coupled directly to `TranslatorService` through `_translate_argos`, `_translate_hf`, `_load_hf`, `_hf_generate`, and backend conditionals in `translate_texts`. This makes adding Qwen or other future providers require editing the service internals rather than adding provider implementations.

## Impact Radius

- Direct: `dubpipeline/translation/service.py`, new provider module(s), `dubpipeline/config.py`, `dubpipeline/cli.py`.
- Tests: `tests/test_translation_models.py`, `tests/test_cli.py`.
- Docs: `README.md` and/or focused docs for adding a provider.
- Adjacent but not intended: ASR, TTS, diarization, segment timing, muxing, GUI redesign.

## Compatibility Concerns

- Do not change target-aware output paths.
- Do not change default selected model behavior.
- Do not download models during tests.
- Preserve `translate.*` legacy fields and environment variables.
- Avoid changing GUI model picker behavior unless needed for config persistence.

