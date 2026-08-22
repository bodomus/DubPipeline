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

## Validation

- `python -m py_compile` on changed Python files: passed.
- `python -m pytest tests/test_source_separation.py tests/test_cli.py tests/test_step_merge_py.py tests/test_audio_mix_step.py tests/test_step_merge_hq.py -q`: 53 passed, 3 warnings, 2 subtests passed.
- `python -m dubpipeline.cli --help`: passed.
- `python -m dubpipeline.cli run dubpipeline/video.pipeline.yaml --in-file tests/TXT/Emar_Krasnyiy_Kedr_3_Stepnyie_razboyniki.txt --plan`: passed; source separation disabled by default.
- `python -m dubpipeline.cli run dubpipeline/video.pipeline.yaml --in-file tests/TXT/Emar_Krasnyiy_Kedr_3_Stepnyie_razboyniki.txt --set source_separation.mode=separated_background --plan`: passed; source separation enabled in plan summary.
- `code-review-graph update --brief`: completed with UTF-8 output.
- `graphify update .`: completed outside sandbox after sandbox access-denied on temporary directories.

## Not Run

- Real BS Roformer / InstVoc 2 separation was not run: no local model command/path was provided in the ticket context and this implementation intentionally does not auto-download models or dependencies.
- Manual GUI validation was not run; GUI load/save continues through the shared config path, but no dedicated GUI controls were added in this ticket.

## Risks And Notes

- The BS Roformer provider is command-template based. Users must configure `source_separation.command` and `source_separation.model_path` for their local separator installation.
- Existing `hq_ducking` behavior remains the default. Separated background affects mix only when `source_separation.mode` is explicitly set to `separated_background`.
- Pre-existing dirty files were left untouched: `dubpipeline/video.pipeline.yaml` and `tests/TXT/1.txt`.
