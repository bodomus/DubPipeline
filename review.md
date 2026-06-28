Review the latest diff.

Scope reviewed:
- WhisperX source language handling in `dubpipeline/steps/step_whisperx.py`
- CLI/config validation for `--lang-src`
- README drift for the CLI option
- Focused tests for CLI and WhisperX language helpers

Blockers:
- None found.

Warnings:
- `--lang-src auto` is allowed only for ASR-only flows; runs with the Translate step enabled now fail early with a clear message because translation still needs a concrete source language.
- Full media/WhisperX execution was not run because it would require model/audio runtime work.

Safe-to-merge verdict:
- Safe to merge based on focused unit coverage and code review.
