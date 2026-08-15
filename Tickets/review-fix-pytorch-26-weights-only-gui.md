# Implementation Report

## Ticket
Local prompt: Fix PyTorch 2.6 `weights_only` checkpoint loading error in DubPipeline GUI.

## Workflow
- Level: 2.
- Graphify: used `graphify query` and `graphify path`; confirmed direct `gui.py -> cli.py`, `gui.py -> config.py`, and `gui.py -> step_tts.py` import relationships.
- CRG: `code-review-graph update --brief` and `detect-changes --brief` completed after setting `PYTHONIOENCODING=utf-8`.
- Working tree before changes: pre-existing `dubpipeline/video.pipeline.yaml` modification and untracked `tests/TXT/1.txt`.

## Scope
- GUI: early startup bootstrap added.
- CLI: unchanged; GUI-launched CLI subprocesses inherit the configured environment.
- Config: unchanged.
- Pipeline steps: unchanged.
- Models/runtime: PyTorch checkpoint compatibility environment only.
- Media/artifacts: unchanged.

## Investigation
- Current behavior: GUI imports `dubpipeline.cli`, `dubpipeline.config`, and `step_tts` at module load; `config.py` imports `torch` at top level.
- Expected behavior: set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before any PyTorch-adjacent imports during GUI startup.
- Root cause/gap: no early GUI runtime bootstrap existed before import-time PyTorch reachability.
- Main symbols: `dubpipeline.runtime.configure_pytorch_checkpoint_loading`, `dubpipeline.gui`.
- Expected blast radius: GUI import/startup and child processes spawned from GUI.
- Compatibility concerns: uses `setdefault`, preserving explicit user values such as `0`.

## Changes
- Added `dubpipeline/runtime.py`.
- Updated `dubpipeline/gui.py` to configure PyTorch checkpoint compatibility before CLI/config/TTS imports.
- Added a one-time startup info log with the effective env value.
- Added `tests/test_runtime.py` for default enablement, explicit value preservation, and GUI import-order guard.

## Graph and Source Validation
- Source confirmed `dubpipeline/config.py` imports `torch` at line 11.
- Source confirmed GUI CLI subprocesses use inherited environment through `subprocess.Popen(...)`.
- Source confirmed Windows spawn workers re-import the GUI module, so the same top-level bootstrap applies.
- CRG reported no affected flows and the expected broad changed-file set from existing working tree state.

## Validation
- Passed: `.venv\Scripts\python.exe -m pytest tests/test_runtime.py`.
- Passed: GUI import smoke with env absent; output env value was `1`.
- Passed: GUI import smoke with explicit env `0`; output env value stayed `0`.
- Passed: `.venv\Scripts\python.exe -m dubpipeline.cli --help`.
- Passed: `.venv\Scripts\python.exe -m dubpipeline.cli run dubpipeline/video.pipeline.yaml --in-file tests/data/dub_1.mp4 --plan`.
- Blocked: `tests/test_gui.py` has unrelated temporary-directory ACL failures when writing under `A:\TEMP` and also under `tests/.tmp_runtime`.
- Blocked: plan run without `--in-file` against `dubpipeline/video.pipeline.yaml` fails before pipeline planning because the YAML's referenced input video is missing.
- Manual GUI checkpoint reproduction was not run in this environment.

## Remaining Risks
- Full acceptance still needs normal GUI launch and the previously failing trusted checkpoint operation.
- Legacy pickle checkpoint loading should remain limited to trusted/local checkpoints.
