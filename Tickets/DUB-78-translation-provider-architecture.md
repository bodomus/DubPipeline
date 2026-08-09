# DUB-78 — Translation Provider Architecture

YouTrack: https://bodomus.youtrack.cloud/issue/DUB-78

Note: the attached/source ticket text refers to this as DUB-75, but the YouTrack issue is DUB-78. Subsequent ticket numbers are shifted.

## Goal

Refactor the current DubPipeline translation stage so translation engines are no longer hard-wired to a single implementation.

The current Argos Translate path must remain operational as a compatibility/fallback provider.

This ticket is architectural only. Do not add Qwen inference yet.

## Requirements

- Introduce a translation provider abstraction.
- Support source language, target language, input text, translated text, provider identification, provider-specific initialization, and provider-specific shutdown/cleanup when needed.
- Do not force all future providers into Argos-specific assumptions.
- Add a translation provider setting using the existing configuration style.
- Preserve configuration precedence.
- Default behavior must remain equivalent to current behavior.
- Add CLI provider selection without breaking existing translation options.
- Unknown provider should produce a clear error: `translation provider 'xyz' is not supported`.
- Provider initialization failure should include provider name and high-level failure reason.

## Tests

Add/update tests for:

- default provider selection;
- explicit Argos selection;
- invalid provider;
- provider factory/registry;
- existing translation behavior;
- configuration parsing;
- CLI parsing if applicable.

Tests must not download external models.

## Non-goals

Do not integrate Qwen, NLLB, ASR, TTS, diarization, segment timing changes, GUI redesign, Argos removal, or translation quality optimization.

## Acceptance Criteria

- Translation code uses a provider abstraction.
- Argos is implemented through that abstraction.
- Existing commands still work.
- Default translation output path remains unchanged.
- Tests pass.
- No external model is downloaded during tests.
- Graphify/CRG research findings are reflected in the implementation.
- Documentation briefly explains how another provider can be added.

