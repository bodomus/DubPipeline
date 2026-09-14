# DUB-85 Investigation

## Repository Intelligence

- Workflow level: Level 2, because this changes model/runtime/media behavior.
- Branch: `codex/DUB-85-native-headless-source-separation-provider`.
- Base commit: `f056c0c6c961845284dc98b902978ed47362e364`.
- Graphify command used:
  `graphify query "DubPipeline source separation provider bs_roformer command native headless audio stems merge" --budget 3000`
- Focused Graphify lifecycle command used:
  `graphify query "run_pipeline _build_cfg_for_input source_separation provider lifecycle batch TranslatorService release_shared_cache" --context call --budget 4000`
- CRG command used:
  `code-review-graph update --brief`
- CRG status after update: 914 nodes, 9038 edges, 81 files, branch
  `codex/DUB-85-native-headless-source-separation-provider`, commit `f056c0c6c961`.
- CRG limitation: `--brief` and `detect-changes --brief` updated/analyzed data, but
  printing failed under cp1251 with `UnicodeEncodeError`. `code-review-graph status`
  succeeds.
- Working tree before implementation included unrelated/unowned `tests/TXT/1.txt` and
  permission warnings for generated temp directories under `.codex_tmp/` and
  `tests/.tmp_runtime/`.

## Ticket Source

- YouTrack: `https://bodomus.youtrack.cloud/issue/DUB-85`
- Local working copy: `Tickets/DUB-85.md`
- YouTrack exposes only read tools in this session, so fields/comments cannot be
  updated from Codex unless write tools become available.

## Current Behavior

- DUB-84 added `dubpipeline/source_separation.py` with:
  - `AudioBackgroundProvider`;
  - `OriginalAudioBackgroundProvider`;
  - command-template `BsRoformerProvider`;
  - DUB-84 metadata cache;
  - fallback to legacy original-audio ducking when configured;
  - merge integration through `resolve_background_audio_for_merge`.
- `step_source_separation.run()` calls `run_source_separation(cfg)`.
- `run_source_separation()` currently creates a provider per call.
- CLI batch processing discovers files in `main()`, then calls
  `_build_cfg_for_input()` and `run_pipeline()` once per file.
- Existing batch lifecycle releases the shared translation cache only after all files.
  Source separation has no equivalent run-scoped provider lifecycle yet.

## Expected Behavior

- Add `provider: audio_separator` backed by `audio_separator.separator.Separator`.
- Keep `provider: bs_roformer` command-template behavior available.
- Load native separator lazily on first actual separation.
- Reuse the loaded native model across multiple input files in one CLI process.
- Close/unload the native provider once the batch completes.
- Keep cache checks ahead of native model initialization.
- Map native stems into `vocals.wav` and `background.wav` without exposing native
  output filenames downstream.
- Respect `cfg.usegpu` and CUDA availability explicitly.

## Dependency and Environment Findings

- Current Python: 3.12.10.
- Current runtime `torch`: `2.5.1+cu121`; CUDA is available as CUDA 12.1.
- GPU: NVIDIA GeForce RTX 4080.
- Repository policy documents pins `torch==2.6.0+cu124`,
  `torchvision==0.21.0+cu124`, `torchaudio==2.6.0+cu124`, and
  `pyannote.audio==3.4.0`. The active shell environment does not match those pins.
- `audio-separator`, `audio_separator`, `onnxruntime`, and `onnxruntime-gpu` are not
  installed in the active environment.
- Shell access to PyPI JSON failed with network refusal, so dependency compatibility
  is documented with degraded external-package validation. The implementation must not
  upgrade torch/CUDA packages.

## Model Compatibility Findings

- Ticket local UVR checkpoint:
  `C:\Users\bodom\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\BS Roformer - InstVoc 2.ckpt`
- Local checkpoint exists:
  - size: 639331213 bytes;
  - MD5: `261E8988646EC49498E46FDAF97F0703`.
- Local UVR config directory contains:
  `model_bs_roformer_ep_368_sdr_12.9628.yaml`.
- That config identifies a mel-band RoFormer-style architecture with:
  - sample rate 44100;
  - stereo input;
  - `dim: 512`;
  - `depth: 12`;
  - target instrument `Vocals`;
  - instruments `Vocals` and `Instrumental`;
  - inference batch size 1.
- This supports the architecture/stem expectation for BS RoFormer InstVoc, but the
  friendly UVR checkpoint name is not the same as the managed audio-separator model
  filename. The provider must not silently substitute a managed model when an explicit
  local model path was configured.

## Relevant Symbols

- `dubpipeline.config.SourceSeparationConfig`
- `dubpipeline.source_separation.SourceSeparationRequest`
- `dubpipeline.source_separation.create_provider`
- `dubpipeline.source_separation.run_source_separation`
- `dubpipeline.source_separation.read_cached_result`
- `dubpipeline.source_separation.model_identity`
- `dubpipeline.cli.run_pipeline`
- `dubpipeline.cli.main`
- `dubpipeline.steps.step_source_separation.run`
- `tests/test_source_separation.py`

## Risk Assessment

- CLI: batch lifecycle changes are needed to pass one provider through multiple files.
- GUI: no GUI redesign; `run_pipeline()` must preserve a default call path for GUI and
  tests.
- Config: add minimal fields without changing default behavior.
- Model/runtime: native import must be lazy, and no model load may occur in plan mode or
  cache-hit paths.
- Media/artifacts: output contract remains DUB-84 `vocals.wav`, `background.wav`, and
  `metadata.json`.
- Cache: provider/model/device changes must invalidate cache.
- Fallback: no stale stems may survive failed fresh runs.

## Validation Required

- Focused unit tests for source separation provider creation, lazy load, batch reuse,
  stem mapping, failures, fallback, and cache-before-load behavior.
- Existing merge HQ tests.
- CLI `--help`.
- CLI `run --plan` to verify no heavy imports/model loading.
- Real `audio-separator` run on `tests/Test_for_SPLIT/en_710_760.wav` only if dependency
  installation/setup succeeds.
