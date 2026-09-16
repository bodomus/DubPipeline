# DUB-86 — Residual English Suppression and RU Background Mixing

## Summary

Improve final dubbed audio quality by suppressing residual English speech that remains in the separated background stem before mixing Russian TTS.

DUB-84 introduced the source-separation architecture. DUB-85 introduced the production native headless BS-RoFormer provider and canonical outputs:

```text
separation/<project_name>/vocals.wav
separation/<project_name>/background.wav
```

DUB-86 must use these artifacts to create a softly attenuated `background.cleaned.wav`, controlled by vocal activity in `vocals.wav`, before mixing Russian TTS. The legacy ducking path must remain available.

## Goal and pipeline

```text
vocals.wav -> activity/envelope control
background.wav -> controlled residual suppression -> background.cleaned.wav
background.cleaned.wav + Russian TTS -> loudness normalization -> final mix -> mux
```

Residual suppression must run after source separation and before final merge as a distinct, independently testable concern. It must use the separated vocal stem rather than ASR timestamps or RU TTS activity.

## Behavior

- Attenuate `background.wav` only while source English vocals are active.
- Preserve ambience, music, effects, and room sound between vocal-active regions.
- Keep suppression soft and configurable; avoid gating, pumping, abrupt gain jumps, and unnecessary silence.
- Target practical attenuation around 6–15 dB rather than muting the background.
- Mix RU TTS only after residual suppression.
- Never reintroduce the full original English track in separated-background mode.

## Configuration

Add a dedicated repository-consistent configuration section with conservative defaults, conceptually:

```yaml
source_separation:
  mode: separated_background

residual_suppression:
  enabled: true
  threshold_db: -35.0
  ratio: 6.0
  attack_ms: 10
  release_ms: 250
  max_reduction_db: 12.0
  control_gain_db: 0.0
  floor_db: -60.0
```

Exact names must follow repository conventions. Existing configurations must continue working, with suppression disabled by default. The implementation should prefer FFmpeg (`sidechaincompress`, `agate`, `volume`, envelope shaping, or another deterministic filter chain) unless Python DSP is clearly safer and simpler. Named config values must be used instead of an opaque hard-coded filter string.

## Modes

### Legacy

```text
source_separation.mode: legacy_ducking
original EN + RU-TTS-controlled ducking + RU TTS
```

No behavior change.

### Separated background, suppression disabled

```text
source_separation.mode: separated_background
residual_suppression.enabled: false
background.wav + RU TTS
```

### Separated background, suppression enabled

```text
source_separation.mode: separated_background
residual_suppression.enabled: true
vocals.wav controls attenuation of background.wav
background.cleaned.wav + RU TTS -> final mix
```

## Artifact and fallback rules

- Produce canonical `separation/<project_name>/background.cleaned.wav` or an equivalent convention-consistent artifact.
- Merge uses the cleaned artifact only when suppression succeeds and is enabled.
- Disabled suppression continues to use `background.wav` unchanged.
- On failure, never reuse stale cleaned output; remove or invalidate it.
- With `source_separation.fallback_mode: legacy_ducking`, follow the existing DUB-84/DUB-85 fallback path.
- With `fallback_mode: none`, raise an explicit error. No silent success.

## Cache

Residual suppression must have a logically independent cache identity including:

- background identity;
- vocals identity;
- suppression configuration;
- implementation/schema version.

Changes to threshold, ratio, attack, release, or maximum reduction must invalidate only the residual-suppression cache. They must not rerun BS-RoFormer.

## Observability

Log concise information about enablement, control/background/output paths, tuning values, cache hit/miss, and processing time. Do not print verbose FFmpeg output in normal mode unless processing fails.

## Tests

Tests must use generated WAV fixtures or fake canonical stems and must not require the real GPU separator. Cover:

- disabled mode returns/uses `background.wav` unchanged;
- enabled mode produces `background.cleaned.wav`;
- identical inputs/config produce a cache hit;
- changed suppression configuration produces a cache miss;
- stale cleaned output with invalid metadata is not reused;
- fallback and strict failure behavior;
- merge routing for legacy, separated without suppression, and separated with suppression.

Existing DUB-84/DUB-85 tests must remain green.

## Real validation

Use `tests/Test_for_SPLIT/en_710_760.wav`. Run DUB-85 source separation first, then produce `background.cleaned.wav`. If practical, create local, uncommitted comparison artifacts:

```text
A_background_original.wav
B_background_cleaned.wav
C_mix_without_residual_suppression.wav
D_mix_with_residual_suppression.wav
```

Record duration, sample rate, channels, mean/RMS and peak where practical, comparing vocal-active and vocal-inactive windows. Manual listening is required; RMS alone is insufficient.

## Quality acceptance

1. Residual English speech is audibly reduced.
2. Background ambience/music remains natural.
3. No obvious pumping or abrupt gain jumps.
4. Non-vocal portions remain largely unchanged.
5. RU TTS remains clearly dominant.
6. No full original EN track leaks into separated mode.

## GUI and non-goals

No GUI redesign is required. Preserve shared-config compatibility. Do not add new separation models, Demucs/MDX integration, automatic model selection, ASR/translation/TTS/diarization changes, ML residual-noise removal, or mastering redesign.

## Suggested implementation boundary

Prefer `dubpipeline/residual_suppression.py` and optionally `dubpipeline/steps/step_residual_suppression.py`, exposing a small independently testable contract such as `run_residual_suppression(cfg) -> Path | None`. Do not bury the concern inside mux logic.

## Acceptance criteria

- [ ] Residual suppression is a distinct configurable pipeline concern.
- [ ] `vocals.wav` is the control signal.
- [ ] `background.wav` is attenuated.
- [ ] Canonical cleaned-background artifact is produced.
- [ ] Suppression runs only in separated-background mode.
- [ ] Legacy ducking remains unchanged.
- [ ] Suppression can be disabled explicitly.
- [ ] Attack/release and attenuation are configurable.
- [ ] Background is preserved during vocal-inactive windows.
- [ ] Stale cleaned artifacts cannot be reused.
- [ ] Suppression config changes invalidate only its own cache.
- [ ] BS-RoFormer is not rerun for suppression-only changes.
- [ ] Merge routes to cleaned/background/original sources correctly by mode.
- [ ] Fallback behavior is explicit.
- [ ] Focused and DUB-84/DUB-85 tests pass.
- [ ] Real validation and listening comparison use `en_710_760.wav`.
- [ ] Review documents architecture, strategy, defaults, cache, fallback, validation, tuning, paths, limitations, and verdict.

## Required deliverables

```text
Tickets/DUB-86.md
Tickets/DUB-86-investigation.md
Tickets/DUB-86-implementation-plan.md
reviews/review-DUB-86.md
```

`READY FOR REVIEW` is permitted only after real listening validation.
