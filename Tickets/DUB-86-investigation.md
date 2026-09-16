# DUB-86 Investigation

## Workflow and baseline

- Workflow level: Level 2 (audio/FFmpeg, configuration, cache, artifact routing, and fallback behavior).
- Branch: `codex/DUB-86-residual-english-suppression`.
- Base: `master` at `07e1d652587bee6231e21dd7a0c2da7ade8074ea`.
- Working tree was clean before the ticket file was added.
- Python: 3.12.10.
- FFmpeg/FFprobe: available, 2025-01-08 build.
- Active torch environment: 2.5.1+cu121, CUDA 12.1, RTX 4080; this differs from repository pins but DUB-86 does not change torch/CUDA dependencies.

## Repository intelligence

- Graphify was refreshed with `graphify update . --no-cluster`: 1184 nodes, 3439 edges.
- Focused queries covered source-separation configuration/runtime, canonical stems, HQ merge routing, cache, fallback, and related tests.
- CRG was refreshed with `code-review-graph update --base master --brief` and inspected with `PYTHONUTF8=1 code-review-graph detect-changes --base master --brief`.
- CRG reported only the ticket document before implementation (no changed symbols, risk 0). Its first report rendering failed under cp1251; the UTF-8 rerun succeeded.
- Graph candidates were verified directly in `config.py`, `cli.py`, `source_separation.py`, `step_source_separation.py`, `step_merge_py.py`, `step_merge_hq.py`, and focused tests.

## Current behavior

- DUB-85 writes canonical `vocals.wav`, `background.wav`, and `metadata.json` below `separation/<project_name>/`.
- `run_pipeline()` executes source separation as `01b_source_separation` before ASR.
- `step_merge_py.run()` calls `resolve_background_audio_for_merge()` immediately before HQ mix rendering.
- When valid separated stems exist, HQ mix uses `background.wav` as FFmpeg input 2. Otherwise configured legacy fallback returns to the original-video audio path.
- Existing HQ ducking is driven by RU TTS and must remain unchanged in legacy mode.
- Source-separation cache already validates source/model/config identity and prevents stale stems from entering merge.

## Expected behavior

- Add an independently testable residual-suppression concern after source separation.
- Use `vocals.wav` as sidechain control and attenuate `background.wav` into `background.cleaned.wav`.
- Route the cleaned artifact to merge only when enabled and valid.
- Preserve raw separated background when suppression is disabled.
- Preserve legacy original-audio ducking on configured fallback.
- Keep residual cache independent from source-separation cache so tuning changes never rerun BS-RoFormer.

## Chosen DSP boundary

Use a dedicated FFmpeg-based module and step. The filter graph will:

1. apply configurable gain to the vocal control stem;
2. run `sidechaincompress` on the background with that control signal;
3. use the filter's dry/wet `mix` to retain a fixed dry floor.

For a configured maximum reduction `R`, the dry-floor amplitude is `10^(-R/20)`. The output is:

```text
dry_floor * background + (1 - dry_floor) * compressed_background
```

When the sidechain is inactive, compressed background equals dry background and the weights sum to 1, preserving the signal. Under strong activity, the dry floor prevents attenuation beyond the configured maximum. This avoids loading long media into Python RAM and gives deterministic streaming behavior.

## Configuration and compatibility

- Add `ResidualSuppressionConfig` with conservative defaults and `enabled: false`.
- Add a `residual_suppression` step flag defaulting to true; effective enablement also requires separated-background mode and config `enabled: true`.
- Preserve precedence: defaults -> YAML -> environment -> CLI `--set`.
- Add derived cleaned-WAV and residual-metadata paths.
- Existing YAML remains compatible because all new fields have defaults.
- GUI redesign is out of scope; template/default merging preserves the new section.

## Cache and failure behavior

- Cache identity includes schema version, both stem identities, all DSP settings, FFmpeg implementation identity, and output path.
- Cache metadata is separate from DUB-85 `metadata.json`.
- Invalid metadata or missing/empty output is a cache miss.
- Before a fresh run and after any failure, stale cleaned output and residual metadata are removed.
- FFmpeg writes to a temporary WAV and promotes it with `os.replace` only after validation.
- With `fallback_mode: legacy_ducking`, suppression failure returns no separated background to merge, selecting the existing legacy path.
- With `fallback_mode: none`, failure is explicit.

## Impact and risks

- CLI: new `01c_residual_suppression` step and public step id.
- Config: new dataclass, defaults, environment group/mappings, and derived paths.
- Pipeline: one new step between source separation and ASR.
- Merge: resolver chooses cleaned/raw/legacy background; HQ filter graph itself remains unchanged.
- Models/GPU: no model or dependency change.
- Media: sample format is re-encoded to PCM WAV; duration/channel behavior requires FFmpeg integration tests and real validation.
- GUI: no control redesign.
- Benchmark: no performance claim; benchmark protocol is not required.

## Validation required

- Config/path/precedence tests.
- Filter graph and FFmpeg command tests.
- Generated-WAV integration test measuring active vs inactive windows and maximum reduction.
- Cache hit/miss/stale-output tests.
- Strict and fallback failure tests.
- Merge routing tests for all three modes.
- Existing source-separation and HQ merge tests.
- Full pytest, CLI help, and plan-mode checks.
- Real DUB-85 stems from `tests/Test_for_SPLIT/en_710_760.wav`, objective measurements, and manual listening comparison when runtime assets are available.
