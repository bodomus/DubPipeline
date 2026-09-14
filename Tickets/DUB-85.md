# DUB-85: Native Headless Source Separation Provider

Source: https://bodomus.youtrack.cloud/issue/DUB-85

## Summary

Add a native headless source-separation backend to DubPipeline using the Python API
of `audio-separator`.

The provider must run without Ultimate Vocal Remover GUI, support BS Roformer /
InstVoc, use CUDA when configured and available, produce `vocals.wav` and
`background.wav`, reuse the model across multiple input files in one process, and
integrate with the DUB-84 source-separation cache/fallback/stem contracts.

## Scope

- Investigate whether the local UVR checkpoint `BS Roformer - InstVoc 2.ckpt`
  maps directly to `audio-separator`, or identify the supported equivalent model.
- Inspect dependency compatibility before changing dependencies.
- Add `provider: audio_separator` while preserving `provider: bs_roformer`.
- Initialize `audio_separator.separator.Separator` lazily.
- Load the model once for a batch/process-scoped provider and reuse it.
- Add explicit close/unload lifecycle behavior.
- Respect DubPipeline GPU/CPU configuration with clear failure behavior.
- Map native separator outputs to existing DUB-84 paths:
  `separation/<project_name>/vocals.wav` and
  `separation/<project_name>/background.wav`.
- Keep DUB-84 cache/fallback behavior and prevent stale stem reuse.
- Add tests that mock/fake `audio-separator` and do not download models or require
  a GPU.
- Validate manually with `tests/Test_for_SPLIT/en_710_760.wav` if environment setup
  succeeds.

## Non-Goals

- Residual English suppression.
- Vocal activity detection.
- Post-ducking controlled by the vocal stem.
- Final RU/background mix tuning.
- GUI redesign.
- Demucs, MDX, or broad model-selection frameworks.
- Translation/TTS changes.

## Acceptance Notes

The full ticket text is available in YouTrack and this local file is the concise
working copy required by the project ticket workflow.
