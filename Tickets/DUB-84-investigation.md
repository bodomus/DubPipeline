# DUB-84 Investigation

## Repository Intelligence

- Branch: `codex/dub-84-source-separation-background-provider`.
- Base commit: `302914ab4465583c6ecaf618e4139c5af312f403`.
- Graphify query used: `DubPipeline audio extraction mix mux config cache steps source separation background provider`.
- CRG status: 868 nodes, 8615 edges, 78 files; last updated on this branch at base commit.
- CRG CLI has no `query` subcommand in this checkout; scoped relationship analysis was completed via `code-review-graph status`, Graphify candidates, `rg`, and direct source inspection.

## Current Flow

- CLI dispatch lives in `dubpipeline/cli.py`.
- Steps currently run as `extract_audio -> asr_whisperx -> translate -> tts+align -> merge`.
- `step_extract_audio` writes `cfg.paths.audio_wav`.
- HQ merge is selected by `audio_merge.mode: hq_ducking`.
- `step_merge_py.run` renders a temporary HQ mix with `render_hq_mix_audio`, then muxes it through `mux_smart`.
- `step_merge_hq.build_filtergraph` currently mixes the selected original video audio stream with generated TTS and applies existing ducking/loudness.

## Relevant Configuration

- `StepsConfig` currently contains `extract_audio`, `asr_whisperx`, `translate`, `tts`, `align`, `merge`.
- `PathsTemplatesConfig` currently derives audio, segments, TTS folders, and final video.
- `AudioMergeConfig` contains current ducking and loudness parameters.
- Environment override routing supports short groups such as `AMR` for `audio_merge`.
- CLI `--plan` loads config with `create_dirs=False` and must stay side-effect-free.

## Risk Assessment

- Media behavior risk is localized to HQ mixing when the new mode is explicitly enabled.
- Default behavior must remain byte-for-byte path-compatible for legacy configs.
- Source separation may be GPU/model-heavy, so implementation must avoid import-time model initialization and support fake tests.
- Cache must not be file-exists-only because stale stems would silently contaminate later mixes.
- GUI impact should be minimal through load/save config compatibility; richer GUI controls can be deferred.

## Decision

Implement a new `source_separation` config section and a `source_separation` pipeline step. The initial BS Roformer integration will be command-template based so users can wire their locally installed separator without adding or downloading a new dependency in this ticket.
