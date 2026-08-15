# Investigation: Fix PyTorch 2.6 weights_only GUI Startup

## Ticket
Local prompt: Fix PyTorch 2.6 `weights_only` checkpoint loading error in DubPipeline GUI.

## Workflow
- Level: 2, because this touches GUI startup and ML runtime import order.
- Graphify: `graphify query` and `graphify path` used for GUI/config/CLI/step relationships.
- CRG: `code-review-graph status`, `detect-changes --brief`, and `update --brief` confirmed availability; console rendering hit cp1251 Unicode errors after producing summaries.

## Current Behavior
- `dubpipeline/gui.py` imports `dubpipeline.cli`, `dubpipeline.config`, and `dubpipeline.steps.step_tts` at module import time.
- `dubpipeline/config.py` imports `torch` at top level.
- `dubpipeline.steps.step_tts` and `step_tts_core` import `torch`/Coqui-related code at top level or near startup.
- When the GUI starts without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, PyTorch 2.6 may attempt `weights_only=True` checkpoint loading and fail on trusted legacy checkpoint config objects such as `omegaconf.listconfig.ListConfig`.

## Expected Behavior
- GUI startup configures `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before any import path can load PyTorch-related modules.
- Existing explicit environment values are preserved.
- CLI subprocesses launched by the GUI inherit the configured environment.
- Multiprocessing preview/audio workers re-import the GUI module under Windows spawn and run the same early bootstrap.

## Source Validation
- `dubpipeline/gui.py` line 15 imports `dubpipeline.cli`; line 16 imports `dubpipeline.config`.
- `dubpipeline/config.py` line 11 imports `torch`.
- `dubpipeline/gui.py` lines 256 and 339 start multiprocessing workers for preview/audio synthesis.
- `dubpipeline/gui.py` lines 393 and 439 launch `python -u -m dubpipeline.cli` subprocesses with inherited environment.
- Graphify confirmed direct import paths: `gui.py -> config.py`, `gui.py -> step_tts.py`, and `gui.py -> cli.py`.

## Root Cause
The compatibility variable was not set before GUI import-time dependencies reached PyTorch.

## Smallest Correct Change
- Add a small runtime bootstrap module with no ML imports.
- Call it at the top of `dubpipeline/gui.py` immediately after importing `os`, before importing `dubpipeline.cli`, `dubpipeline.config`, `step_tts`, or any other ML-adjacent modules.
- Add focused tests for `setdefault` behavior.

## Risks
- This enables pickle-based checkpoint loading, which is acceptable only for trusted local checkpoints.
- Full checkpoint reproduction requires a GUI/model runtime run with the affected trusted checkpoint.
