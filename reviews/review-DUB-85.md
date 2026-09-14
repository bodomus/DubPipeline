# DUB-85 Review

Issue: https://bodomus.youtrack.cloud/issue/DUB-85/Native-Headless-Source-Separation-Provider

Branch: `codex/DUB-85-native-headless-source-separation-provider`

Base commit: `f056c0c6c961845284dc98b902978ed47362e364`

## Summary

Implemented a native headless source separation provider using the `audio-separator`
Python API while preserving the existing DUB-84 command-template provider and legacy
ducking behavior.

The new provider is selected with:

```yaml
source_separation:
  mode: separated_background
  provider: audio_separator
  model: model_bs_roformer_ep_368_sdr_12.9628.ckpt
  device: auto
```

## Done

- Added native `audio_separator` provider with lazy model initialization.
- Reused one native provider instance across `--in-dir` runs so the model can stay
  loaded for multiple files.
- Preserved cache-before-model-load behavior: cache hits do not import or initialize
  `audio-separator`.
- Added explicit `cuda`, `cpu`, and `auto` device handling.
- Added deterministic output normalization to existing DUB-84 paths:
  `vocals.wav`, `background.wav`, and `metadata.json`.
- Added metadata schema version 2 with native provider parameters.
- Added explicit error handling for missing native dependency, model load failures,
  separation failures, missing vocals stem, and missing background/instrumental stem.
- Preserved `fallback_mode: legacy_ducking` for dependency/model-load/runtime failures.
- Added config fields: `model`, `model_file_dir`, `device`, `output_format`, and
  `sample_rate`.
- Added `audio-separator[gpu]` to `requirements.txt`.
- Documented native provider usage in `README.md`.
- Saved ticket/preflight artifacts:
  - `Tickets/DUB-85.md`
  - `Tickets/DUB-85-investigation.md`
  - `Tickets/DUB-85-implementation-plan.md`

## Files Changed

- `dubpipeline/source_separation.py`
- `dubpipeline/config.py`
- `dubpipeline/cli.py`
- `dubpipeline/steps/step_source_separation.py`
- `tests/test_source_separation.py`
- `requirements.txt`
- `README.md`
- `Tickets/DUB-85.md`
- `Tickets/DUB-85-investigation.md`
- `Tickets/DUB-85-implementation-plan.md`

## Validation

- `python -m unittest discover -s tests -p "test_source_separation.py"`:
  22 tests passed.
- `python -m unittest discover -s tests -p "test_step_merge_hq.py"`:
  6 tests passed.
- `python -m unittest discover -s tests -p "test_cli.py"`:
  35 tests passed.
- `python -m dubpipeline.cli --help`: passed.
- `python -m dubpipeline.cli speak --text "Тест" --out-audio tmp_unittest_speak.wav --plan`:
  passed.
- `python -m dubpipeline.cli run dubpipeline\video.pipeline.yaml --in-file tests\Test_for_SPLIT\en_710_760.wav --set source_separation.mode=separated_background --set source_separation.provider=audio_separator --set source_separation.device=cuda --plan`:
  passed; plan mode did not load the native model.
- Real native separation on `tests\Test_for_SPLIT\en_710_760.wav`:
  passed, generated valid 44.1 kHz stereo WAV stems, both 50.0 seconds.
- Native cache-hit rerun:
  passed, returned from metadata/stems without model initialization.
- Post-change CRG update:
  passed with `PYTHONIOENCODING=utf-8`.
- Post-change Graphify refresh:
  passed.

`python -m pytest ...` was attempted but pytest is not installed in the active
environment, so focused tests were run through `unittest`.

## Runtime Notes

- Active environment has `torch 2.5.1+cu121`; the project instructions mention
  `torch 2.6.0+cu124`.
- Installing `audio-separator[gpu]` installed `audio-separator 0.47.0` and
  `onnxruntime-gpu 1.29.0`.
- Pip reported dependency conflicts:
  - `py-key-value-aio 0.4.5` requires `beartype>=0.20.0`, while
    `audio-separator` installed `beartype 0.18.5`.
  - `thinc 8.3.2` requires `numpy<2.1.0,>=2.0.0`, while the environment now has
    `numpy 2.1.3`.
- Real CUDA separation emitted an ONNX Runtime warning because installed
  `onnxruntime-gpu 1.29.0` expects CUDA 13 DLLs while PyTorch uses CUDA 12.1.
  RoFormer still loaded and the real single-file separation completed.
- Additional two-input same-process native runtime validation was attempted after the
  successful single-file run. It loaded the cached model once, but then stopped
  producing progress during the first separation and was interrupted after several
  minutes. Unit coverage verifies provider reuse, but the local GPU/ONNX runtime stack
  should be normalized before treating batch runtime behavior as production-validated.
- The real separated stems were structurally valid. `ffmpeg astats` showed the vocals
  stem was nearly silent on this sample; no previous UVR reference output was found in
  the repository for qualitative comparison.
- `git status` emits permission warnings for pre-existing `.codex_tmp` and
  `tests/.tmp_runtime` directories. A few `tmp_unittest/tmp*` temp directories created
  during validation also remain inaccessible to the current process and could not be
  removed.

## Remaining Risks

- Native provider behavior depends on the installed `audio-separator`/ONNX/Torch CUDA
  compatibility matrix.
- Batch same-process real GPU runtime needs another validation pass after the CUDA/ONNX
  environment mismatch is resolved.
- Manual auditory comparison against Ultimate Vocal Remover output was not completed
  because no matching baseline output was available in the repository.
