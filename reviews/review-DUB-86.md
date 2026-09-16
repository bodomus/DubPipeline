# Review DUB-86 — Residual English Suppression and RU Background Mixing

## Verdict

**AWAITING MANUAL LISTENING — not READY FOR REVIEW yet.**

The implementation, automated validation, real-file processing, cache checks, and
objective audio measurements are complete. The ticket explicitly requires listening
validation before `READY FOR REVIEW`; that human check remains open.

## Branch and baseline

- Branch: `codex/DUB-86-residual-english-suppression`
- Base: `master` at `07e1d652587bee6231e21dd7a0c2da7ade8074ea`
- Scope: residual source-vocal suppression between DUB-85 source separation and the
  existing HQ RU-TTS merge.
- Dependencies/models: unchanged.

## What was implemented

- Added `ResidualSuppressionConfig`, disabled by default, with named threshold, ratio,
  attack, release, maximum reduction, control gain, knee, and cache settings.
- Added canonical paths:
  - `separation/<project_name>/background.cleaned.wav`
  - `separation/<project_name>/residual_suppression.json`
- Added the distinct pipeline step `01c_residual_suppression` immediately after
  `01b_source_separation`.
- Added a dedicated FFmpeg implementation in `dubpipeline/residual_suppression.py`.
  `vocals.wav` is the sidechain control; `background.wav` is the processed signal.
- Added a dry-floor calculation through `sidechaincompress`'s dry/wet `mix`, limiting
  worst-case attenuation to `max_reduction_db` while preserving inactive regions.
- Padded the control/output streams and bounded output with the exact background
  duration. Output validation checks sample rate, channel count, and frame count.
- Changed HQ merge routing so it selects:
  - original input audio for legacy ducking/fallback;
  - raw `background.wav` when residual suppression is disabled;
  - valid `background.cleaned.wav` when suppression is enabled.
- Preserved the existing HQ mix filter graph and RU-TTS ducking behavior.
- Added README and sample-pipeline documentation.

## Effective defaults and tuning

```yaml
steps:
  residual_suppression: true

residual_suppression:
  enabled: false
  threshold_db: -35.0
  ratio: 6.0
  attack_ms: 10
  release_ms: 250
  max_reduction_db: 12.0
  control_gain_db: 0.0
  knee: 2.828427
  cache_enabled: true
```

The step flag permits orchestration, but effective processing requires both
`source_separation.mode: separated_background` and
`residual_suppression.enabled: true`. Existing pipeline files therefore retain their
previous behavior.

## Cache and artifact safety

Residual suppression has an independent metadata file and cache identity containing:

- SHA-256, size, mtime, and resolved path for both stems;
- every residual-suppression config value;
- implementation and metadata schema versions;
- FFmpeg binary and output identity.

Changing residual tuning invalidates only `background.cleaned.wav`; it does not
invalidate or rerun BS-RoFormer. Processing writes a temporary WAV, validates it, then
promotes it atomically with `os.replace`. Stale output and metadata are removed before
a rebuild and after failure.

## Fallback behavior

- `fallback_mode: legacy_ducking`: missing DUB-85 stems or a suppression failure returns
  to the existing original-audio legacy mix. Stale cleaned artifacts are removed.
- `fallback_mode: none`: processing or missing-artifact failure is explicit.
- Disabled suppression always routes the valid raw separated background unchanged.

A regression test specifically verifies that a missing DUB-85 background reaches the
legacy fallback before any WAV-duration inspection.

## Automated validation

### Focused suite

```text
76 passed, 4 subtests passed
```

Covered residual config/path/precedence, generated-WAV FFmpeg behavior, cache hit and
tuning miss, stale metadata, strict and legacy fallback, source-separation behavior,
HQ merge, merge routing, and CLI behavior.

### Project suite

```text
180 passed, 11 subtests passed
```

This is `pytest tests -q --ignore=tests/test_tts_provider_import.py`. The unfiltered run
reported `184 passed, 11 subtests passed, 1 failed`. The single failure is environment
dependent and unrelated to DUB-86:
`test_coqui_provider_without_package_has_clear_error` assumes Coqui TTS is absent, but
the current `.venv` has it installed, so the test proceeds to the next missing-segments
check.

Additional checks passed:

- `python -m compileall -q dubpipeline`
- `python -m dubpipeline.cli --help`
- real pipeline `--plan`, which reports `01c_residual_suppression: enabled`
- `git diff --check` (only expected LF-to-CRLF working-copy notices)

## Real-file validation

Input and existing DUB-85 stems:

```text
tests/Test_for_SPLIT/en_710_760.wav
tests/output/separation/en_710_760_native/vocals.wav
tests/output/separation/en_710_760_native/background.wav
```

Produced:

```text
tests/output/separation/en_710_760_native/background.cleaned.wav
tests/output/separation/en_710_760_native/residual_suppression.json
tests/output/dub86_validation_20260916/comparisons/A_background_original.wav
tests/output/dub86_validation_20260916/comparisons/B_background_cleaned.wav
```

Settings were the documented defaults. Processing existing stems took approximately
0.13 seconds; a repeat was a residual-cache hit. Both A and B are 50.000 seconds,
44.1 kHz, stereo, and the cleaned output has the exact source frame count.

Objective 0.5-second window analysis:

| Measurement | Result |
|---|---:|
| Windows | 100 |
| Vocal-active windows (`vocals RMS > -35 dB`) | 22 |
| Vocal-inactive windows (`vocals RMS < -60 dB`) | 67 |
| Active attenuation, median | -8.03 dB |
| Active attenuation, range | -10.55 to -0.17 dB |
| Inactive delta, median | 0.00 dB |
| Inactive delta, range | -1.63 to 0.00 dB |
| Original global RMS | -53.24 dB |
| Cleaned global RMS | -53.49 dB |
| Original / cleaned peak | -26.51 / -26.51 dB |

The inactive negative tail is consistent with the configured 250 ms release. The
measured active attenuation is within the intended soft-suppression range and never
exceeds the 12 dB configured floor.

Comparison C/D mixes were not created because no matching real RU-TTS artifact was
available. The ticket marks those two artifacts as optional (“if practical”). Merge
routing is covered by automated tests.

## Repository-intelligence postflight

- CRG was updated after implementation. Its diff-based report found 14 changed symbols,
  risk score 0.40, and no affected flow anomaly. New untracked files are not included in
  CRG's Git diff count, but they were parsed by the graph update and checked directly.
- Graphify was structurally refreshed after adding the module/step: 1232 nodes and 3645
  edges. Focused traversal connects the new resolver to `step_merge_py`, CLI reachability,
  config, source-separation fallback, and residual tests.
- Dynamic FFmpeg behavior, YAML/environment/CLI precedence, and artifact routing were
  verified through direct source inspection and executable tests rather than inferred
  from graph edges.

## Remaining manual acceptance

Listen to A/B with level-matched playback and confirm:

1. residual English speech is audibly lower in B;
2. ambience/music remains natural;
3. no obvious pumping or abrupt transitions occur;
4. non-vocal regions remain effectively unchanged.

After that confirmation, the ticket can be marked `READY FOR REVIEW`. RU-TTS dominance
in a C/D comparison remains a useful optional follow-up if a matching TTS artifact is
provided.
