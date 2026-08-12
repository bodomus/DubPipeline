# DUB-79 - Local Qwen Translation Provider

YouTrack: https://bodomus.youtrack.cloud/issue/DUB-79

Document-number mapping: this YouTrack ticket corresponds to roadmap document DUB-76.

## Goal

Add a local Qwen-based EN->RU translation provider to DubPipeline through the provider abstraction introduced by DUB-78.

Initial target model:

```text
Qwen/Qwen3-8B
```

The exact model must be configurable. Tests must not download external models.

## Non-goals

- Do not add context-aware translation yet.
- Do not remove Argos.
- Do not change ASR, TTS, WhisperX, XTTS, diarization, segment timing, or muxing.
- Do not add cloud translation dependencies.

