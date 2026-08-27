# Review DUB-84

## Done

- Created branch `codex/dub-84-source-separation-background-provider` from master.
- Added `source_separation` config with default `legacy_ducking` mode, explicit `separated_background` mode, configurable provider/model path/command, cache flag, and fallback mode.
- Added derived separation paths under `paths.templates`:
  - `separation/{project_name}/vocals.wav`
  - `separation/{project_name}/background.wav`
  - `separation/{project_name}/metadata.json`
- Added `dubpipeline.source_separation` provider/cache boundary:
  - `AudioBackgroundProvider`
  - `OriginalAudioBackgroundProvider`
  - `BsRoformerProvider`
  - source identity metadata using path, size, mtime, and SHA-256
  - cache validation that requires matching metadata and non-empty stems
- Added `step_source_separation` after `extract_audio`; it runs only when `source_separation.mode: separated_background`.
- Integrated separated background into HQ mix rendering as an optional third ffmpeg input, preserving existing ducking/loudness behavior.
- Added explicit failure handling: fallback to legacy original audio only when `source_separation.fallback_mode: legacy_ducking`; otherwise separation failures raise.
- Preserved external voice track behavior by skipping extract-audio and using the copied voice input as the source audio for downstream steps.
- Added tests for config parsing, per-input paths, command provider execution, cache hit/miss, fallback behavior, merge background selection, and ffmpeg command inputs.

## Corrective Pass

- Fixed stale-stem fallback bug.
- Root cause: `run_source_separation()` could fall back to legacy after a cache miss and failed fresh separator run, while `resolve_background_audio_for_merge()` later trusted `background.wav` by file existence alone. That allowed an old stale stem to bypass legacy fallback.
- Final fallback behavior: merge uses separated background only when cache metadata is valid for the current source audio, provider, command, and model file. If separation fails and `fallback_mode: legacy_ducking` is configured, stale stems are removed and merge uses the original audio path.
- Fresh separation attempts now remove stale `vocals.wav` and `background.wav` before invoking the external provider.
- Cache model identity now includes normalized model path, file size, `mtime_ns`, and SHA-256. Replacing a checkpoint at the same path invalidates the cache.
- Missing BS Roformer model path/file now raises a clear `SourceSeparationError` before the external command is invoked.
- Added regression coverage for stale stems + invalid metadata/cache miss + separator failure + legacy fallback.
- Removed the unrelated `video.pipeline.yaml` output path content change; `git diff --exit-code -- dubpipeline/video.pipeline.yaml` reports no content diff. The file may still appear modified in `git status` due line-ending normalization metadata, and `git restore` was blocked by the safety policy because this file had been pre-existing user work.

## Validation

- `python -m py_compile` on changed Python files: passed.
- `python -m py_compile dubpipeline/source_separation.py tests/test_source_separation.py`: passed.
- `python -m pytest tests/test_source_separation.py -q`: 11 passed, 1 warning.
- `python -m pytest tests/test_source_separation.py tests/test_cli.py tests/test_step_merge_py.py tests/test_audio_mix_step.py tests/test_step_merge_hq.py -q`: 56 passed, 3 warnings, 2 subtests passed.
- `python -m dubpipeline.cli --help`: passed.
- `python -m dubpipeline.cli run dubpipeline/video.pipeline.yaml --in-file tests/TXT/Emar_Krasnyiy_Kedr_3_Stepnyie_razboyniki.txt --plan`: passed; source separation disabled by default.
- `python -m dubpipeline.cli run dubpipeline/video.pipeline.yaml --in-file tests/TXT/Emar_Krasnyiy_Kedr_3_Stepnyie_razboyniki.txt --set source_separation.mode=separated_background --plan`: passed; source separation enabled in plan summary.
- `code-review-graph update --brief`: completed with UTF-8 output.
- `graphify update .`: completed outside sandbox after sandbox access-denied on temporary directories.

## Real BS Roformer Validation

- UVR installation found at `C:\Users\bodom\AppData\Local\Programs\Ultimate Vocal Remover`.
- BS Roformer InstVoc 2 checkpoint found at `C:\Users\bodom\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\BS Roformer - InstVoc 2.ckpt`.
- A headless real separation run was not completed. `UVR.exe --help` starts the GUI/PyInstaller application rather than a documented CLI path and failed during startup with a numba/librosa cache error: `RuntimeError: cannot cache function '__shear_dense': no locator available for file 'librosa\\util\\utils.py'`. The process did not exit cleanly and was stopped.
- No installed command-line separator was found via `Get-Command` for `audio-separator`, `uvr`, or vocal-remover style commands.
- Package check showed no installed `audio-separator`, `uvr5`, `demucs`, or `onnxruntime-gpu` in the DubPipeline virtual environment.
- Because no safe headless UVR command/API was available, the real run, cache-hit rerun, and audio-quality comparison remain manual validation items. The code-level command-template adapter and cache/fallback contracts were validated with fakes.

## Not Run

- Manual GUI validation was not run; GUI load/save continues through the shared config path, but no dedicated GUI controls were added in this ticket.

## Risks And Notes

- The BS Roformer provider is command-template based. Users must configure `source_separation.command` and `source_separation.model_path` for their local separator installation.
- Existing `hq_ducking` behavior remains the default. Separated background affects mix only when `source_separation.mode` is explicitly set to `separated_background`.
- Pre-existing dirty/user files were preserved: `tests/TXT/1.txt`; `dubpipeline/video.pipeline.yaml` has no content diff after the corrective pass.

## Verdict

READY FOR REVIEW for code correctness. Real BS Roformer execution is documented as blocked by the installed UVR application's lack of a safe headless CLI path in this environment.
