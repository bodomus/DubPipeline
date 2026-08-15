# Implementation Plan: Fix PyTorch 2.6 weights_only GUI Startup

## Scope
- GUI bootstrap: yes.
- CLI: only indirectly through GUI subprocess environment inheritance.
- Config: no behavior changes.
- Models/runtime: environment compatibility only; no dependency/model changes.
- Media/artifacts: no changes.

## Steps
1. Add `dubpipeline/runtime.py` with `configure_pytorch_checkpoint_loading()`.
2. Use `os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")`.
3. Import and call the bootstrap at the earliest top-level point in `dubpipeline/gui.py`.
4. Emit one startup info log with the effective variable value.
5. Add unit tests proving default enablement and explicit value preservation.
6. Run focused tests, import smoke checks, CLI help, and CRG post-change update.

## Validation Targets
- `python -m pytest tests/test_runtime.py`
- `python -m pytest tests/test_gui.py`
- `python -m dubpipeline.cli --help`
- GUI import/startup smoke where practical.

## Manual Verification Note
Full acceptance still requires launching the GUI through the normal user path and running the previously failing checkpoint operation on a machine with the trusted model checkpoint available.
