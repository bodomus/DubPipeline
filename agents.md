# AGENTS.md

## Purpose
This file defines how Codex should work in the DubPipeline repository.

DubPipeline is a local media dubbing pipeline. Make focused, minimal, verifiable changes without breaking the existing workflow.

---

## Working style
- Read the relevant code path before editing.
- For non-trivial tasks, make a short plan first.
- Keep changes minimal, local, and reversible.
- Do not refactor unrelated code.
- Preserve existing behavior unless the task explicitly requires changes.
- State assumptions clearly in the final response.

---

## Project scope
Typical pipeline:
1. extract audio
2. ASR
3. optional diarization / alignment
4. translation
5. TTS
6. timing / alignment of generated speech
7. concatenation
8. merge / mux into final output

Do not change pipeline order unless explicitly required.

---

## Important areas
- `dubpipeline/cli.py` — CLI entry
- `dubpipeline/gui.py` — GUI entry
- `dubpipeline/steps/` — pipeline steps
- `dubpipeline/utils/` — helpers
- `tests/` — automated tests
- `README.md` / `docs/` — documentation
- config files (`.yaml`, `.yml`, `.json`) — settings

If the actual structure differs, follow the repo instead of inventing a parallel one.

---

## Technical assumptions
- FFmpeg is a core dependency.
- YAML config is part of the workflow.
- Local models may be large and may be absent on the machine.
- Windows is a primary development environment.
- Existing CLI flags, config keys, output naming, and output layout are important.

---

## Commands Codex should prefer
### Install
Use the repository’s existing setup instructions first.

Typical examples:
- `python -m venv .venv`
- `.venv\\Scripts\\activate`
- `pip install -r requirements.txt`

### Run CLI
- `python -m dubpipeline.cli --help`
- `python -m dubpipeline.cli run <config-or-args>`

### Run GUI
- `python -m dubpipeline.gui`

### Tests
Prefer targeted tests first:
- `pytest -q`
- `pytest tests/<relevant_test_file>.py -q`

### Validation before finishing
Run the smallest sufficient validation set:
1. syntax / lint if configured
2. targeted tests for touched area
3. focused CLI or functional check if appropriate

Do not run expensive end-to-end media jobs unless needed.

---

## Editing rules
- Prefer editing existing files over creating new ones.
- Do not create duplicate pipeline paths or alternate implementations unless requested.
- Do not rename public CLI flags, config keys, output files, or pipeline steps unless explicitly required.
- Keep logging style consistent with the existing codebase.
- Preserve backward compatibility where practical.

---

## Coding rules
- Follow existing naming and module boundaries.
- Prefer simple, explicit code over clever abstractions.
- Avoid adding new dependencies unless clearly necessary.
- Add comments only where logic is non-obvious.
- Be careful with subprocess handling, encodings, and Windows path quoting.

---

## FFmpeg rules
- Treat FFmpeg command lines as business logic.
- Do not casually rewrite working filter graphs.
- If changing FFmpeg arguments:
  - preserve stream mapping intentionally
  - preserve codec/container compatibility intentionally
  - preserve sample rate and channel assumptions intentionally
  - verify output remains playable
- Prefer minimal argument changes and explain why each changed flag matters.

---

## Model rules
- Do not assume a model is installed unless code confirms it.
- Do not hardcode machine-specific model paths.
- Keep “installed”, “available”, and “placeholder / unsupported” states clearly separated.
- Large model downloads must check free disk space first.
- Prefer AppData or the existing configured storage location for model files.

---

## Current product constraints
- Qwen/Mistral entries may exist as placeholders; do not enable them unless the task explicitly implements real support.
- Preserve clear UX around installed vs not installed models.
- Do not activate unfinished features accidentally.

---

## Config rules
- Preserve existing config schema unless the task explicitly changes it.
- If adding config keys:
  - choose clear names
  - document defaults
  - keep backward compatibility where possible
- Update example configs when config behavior changes.

---

## GUI rules
- Keep GUI changes minimal and task-focused.
- Do not redesign the whole interface unless explicitly asked.
- Preserve existing control flow and UX expectations.
- If adding a control, wire it through actual pipeline behavior.
- For long-running operations, use visible status/progress in the existing style.

---

## CLI rules
- Maintain compatibility with existing commands and flags unless explicitly asked otherwise.
- If adding CLI options:
  - document them
  - validate inputs clearly
  - connect them to actual pipeline behavior
- Avoid silent behavior changes.

---

## Logging and diagnostics
- Keep logs practical and useful.
- Prefer logs that help diagnose:
  - missing files
  - invalid config
  - missing models
  - FFmpeg failures
  - pipeline step timings
- Do not add unnecessary log noise.

---

## Audio/output constraints
- Final outputs must remain playable in common desktop players.
- Be careful with sample rate, channel count, codec, and container compatibility.
- When changing merge logic, preserve original audio track behavior unless the task says otherwise.
- For TTS / alignment tasks, protect intelligibility first, then timing precision.
- For mux / merge tasks, protect playability first.

---

## Tests policy
- Add or update tests for behavior changes when feasible.
- Prefer focused tests around:
  - config parsing
  - CLI argument behavior
  - utility functions
  - file naming / output planning
  - command construction
- If end-to-end validation is too heavy, add the best lightweight coverage possible.
- If tests cannot be run, say so explicitly.

---

## Documentation policy
Update docs when changing:
- installation/setup
- required tools
- model handling
- config schema
- CLI flags
- GUI behavior
- pipeline step behavior
- output naming or output structure

At minimum check:
- `README.md`
- example configs
- `docs/`
- changelog / release notes if present

---

## Safety and constraints
- Never hardcode secrets, tokens, or local absolute paths.
- Never invent successful command results if commands were not run.
- Never delete user media, models, or outputs unless explicitly requested.
- Avoid destructive cleanup of caches or model folders unless explicitly requested.
- If a task is risky, state the risk clearly.

---

## Definition of done
A task is done only when:
- the requested change is implemented
- touched files are internally consistent
- minimal necessary validation was performed
- relevant docs/config examples were updated if needed
- risks, limitations, or manual follow-up were stated clearly

---

## Final response format
When finishing a task, respond with:
1. Summary
2. Files changed
3. Validation performed
4. Risks / limitations
5. Manual steps, if any

Be specific. Do not claim tests passed unless they were actually run.

---

## Subdirectory overrides
More specific `AGENTS.md` files in subfolders may define stricter local rules.
When working in a subdirectory, prefer the nearest applicable instructions.