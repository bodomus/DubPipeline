# AGENTS.md

## Mandatory pre-ticket workflow

Before starting any non-trivial ticket, feature, bugfix, refactor, investigation,
implementation-planning task, performance change, model/runtime change, GUI change,
CLI change, or code review:

1. Resolve the repository root:

   ```powershell
   git rev-parse --show-toplevel
   ```

2. Read the repository-root file `.codex/PRE_TICKET_WORKFLOW.md`.
3. Use `$graphify-repository-analysis`.
4. Use `$code-review-graph-analysis`.
5. Execute all applicable preflight phases.
6. Do not begin implementation until repository-intelligence preflight is complete.
7. After implementation, update CRG, inspect impact radius, run validation, and
   refresh Graphify when structural relationships changed.

For spelling, formatting, comment-only, or metadata-only changes, the full graph
preflight may be skipped when graph context cannot affect correctness.

## Project scope

This repository contains DubPipeline, a local video/audio dubbing and text-to-speech
pipeline.

Primary stack and runtime characteristics:

- Python;
- Windows-first local execution;
- CLI entry point: `python -m dubpipeline.cli`;
- FreeSimpleGUI desktop GUI;
- FFmpeg and FFplay subprocesses;
- WhisperX and faster-whisper for ASR/alignment;
- pyannote.audio for diarization/VAD;
- translation backends including Argos and Hugging Face models;
- XTTS v2 / Coqui TTS for speech synthesis;
- PyTorch with CUDA 12.4;
- GPU/CPU execution modes;
- YAML configuration with precedence:
  defaults -> pipeline YAML -> environment -> CLI;
- model installation, caching, VRAM lifecycle, intermediate artifacts, and media muxing.

Analysis must account for:

- long-running and resource-heavy model initialization;
- GPU VRAM ownership and release;
- CPU fallback;
- deterministic dry-run/plan behavior;
- Windows multiprocessing `spawn`;
- subprocess lifecycle and cancellation;
- thread-to-GUI event delivery;
- binary/media tool availability;
- generated media and temporary artifacts;
- partial pipeline execution and resume/rebuild semantics;
- language-pair validation;
- target-aware output paths and track metadata;
- external subtitle behavior;
- audio timing, alignment, loudness, ducking, and mux safety;
- backward-compatible YAML, environment variables, and CLI aliases.

## Repository layout

- `dubpipeline/` — production package.
- `dubpipeline/cli.py` — CLI parser and execution entry point.
- `dubpipeline/gui.py` — FreeSimpleGUI application and process/thread coordination.
- `dubpipeline/config.py` — typed configuration and precedence rules.
- `dubpipeline/steps/` — pipeline step implementations.
- `dubpipeline/models/` — model catalog, status, and installation behavior.
- `dubpipeline/translation/` — translation services/backends.
- `dubpipeline/utils/` — logging, timing, audio concatenation, output movement, and utilities.
- `tools/` — legacy or diagnostic scripts; verify whether a script is still part of the supported flow.
- `tests/` and/or `.tests/` — automated tests, fixtures, and sample inputs; verify actual current layout.
- `output/`, `out/`, generated segment directories, and media files — generated runtime artifacts.
- `.codex/` — Codex workflows.
- `.agents/skills/` — repository-local Codex skills.
- `.code-review-graph/` — generated CRG state.
- `graphify-out/` — generated Graphify state.

Do not assume README examples represent the only current execution path. Verify current
package entry points and tests.

## Generated and non-source directories

Do not treat these as production source:

- `.git/`
- `.idea/`
- `.vs/`
- `.vscode/`
- `.venv*/`, `venv*/`, `env*/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.tox/`, `.nox/`
- build, dist, egg, wheel, and packaging output;
- `output/`, `out/`, temp/work directories;
- generated WAV, MP4, SRT, JSON, model, cache, and benchmark output;
- `.code-review-graph/`
- `graphify-out/`
- downloaded third-party model repositories such as `stabilityai/`.

Do not index model weights, virtual environments, generated media, test output, or
large caches in Graphify or CRG.

Do not exclude project-owned YAML, Python, tests, or small textual fixtures merely
because they live near generated output.

## Repository intelligence routing

- Use Graphify for pipeline architecture, step ownership, configuration flow,
  model/translation/TTS relationships, GUI-to-CLI orchestration, and cross-module
  candidate discovery.
- Use CRG for exact functions/classes, imports, callers, dependants, tests,
  subprocess/process/thread relationships, and change-impact analysis.
- Treat graph results as candidate evidence.
- Use direct source inspection, `rg`, tests, dry-run output, logs, and actual runtime
  behavior as authoritative.
- Verify dynamic imports, environment-driven selection, subprocess entry points,
  multiprocessing targets, callback wiring, and configuration-derived paths directly
  in source.
- When Graphify, CRG, README, and source disagree, current source plus executable tests
  win.

## Configuration contract

`dubpipeline/config.py` is the primary configuration authority.

Preserve and verify the precedence contract:

```text
code defaults
  -> pipeline YAML
  -> environment
  -> CLI overrides
```

Configuration changes must inspect:

- typed dataclasses;
- normalization and validation;
- YAML load/save;
- environment variable mapping;
- legacy environment compatibility;
- CLI `--set` and explicit flags;
- GUI load/save behavior;
- derived paths;
- dry-run/plan output;
- backward compatibility.

Do not introduce a second independent configuration source.

Do not silently change default model, language, GPU mode, output path, cleanup policy,
or merge mode.

## CLI safety

The CLI is a public contract.

For `run` and `speak` changes:

- preserve parser validation and mutual exclusions;
- validate paths before expensive model loading;
- preserve `--plan` as side-effect-free;
- preserve `--in-file` / `--in-dir` semantics;
- preserve `--steps` patch/list behavior;
- preserve language-pair validation;
- preserve target-aware outputs;
- avoid creating directories or writing files in plan mode;
- preserve documented exit behavior;
- keep errors actionable;
- update README and tests with public option changes.

## GUI safety

The GUI coordinates threads, subprocesses, and Windows multiprocessing.

For GUI changes:

- keep GUI updates on the GUI event path;
- avoid blocking the main event loop;
- ensure worker process/subprocess cleanup;
- preserve `spawn`-safe top-level multiprocessing targets;
- do not pass unpicklable closures to child processes;
- preserve cancellation and window-close cleanup;
- verify preview/player process termination;
- verify redirected stdout/stderr handling;
- keep CLI behavior and GUI-generated YAML aligned;
- validate file/folder mode and input-path semantics;
- verify model installation/status feedback.

Manual GUI validation is required for behavior that cannot be covered by unit tests.

## Model and GPU safety

Model/runtime changes are high-risk.

Inspect:

- model catalog and resolver;
- installer/status logic;
- device and compute type;
- torch/CUDA compatibility;
- model initialization ownership;
- reuse versus repeated loading;
- VRAM release;
- batch size;
- precision;
- CPU fallback;
- cache locations;
- failure recovery;
- multi-file execution behavior.

Do not upgrade torch, torchaudio, torchvision, WhisperX, pyannote, Coqui TTS, CUDA,
or model identifiers casually.

The repository pins:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
pyannote.audio==3.4.0
```

Dependency changes require a compatibility investigation and focused runtime validation.

Do not claim performance improvement without comparable measurements.

## Pipeline and artifact safety

For step changes, inspect:

- input contract;
- output artifact contract;
- skip/rebuild behavior;
- artifact existence checks;
- target-language naming;
- partial failure behavior;
- cleanup/keep-temp behavior;
- batch/multi-file behavior;
- progress reporting;
- logging/timing;
- downstream consumers.

Avoid overwriting original media until final output is validated.

For update-existing-file behavior, use an atomic or rollback-safe strategy.

## Audio and media safety

For FFmpeg, alignment, TTS, and mux changes, verify:

- sample rate and channel assumptions;
- segment start/end/duration;
- silence and gap insertion;
- tempo/stretch behavior;
- clipping;
- loudness normalization;
- ducking threshold/attack/release;
- stream mapping;
- language tags and titles;
- container compatibility;
- original track preservation;
- temporary output and final replace sequence;
- FFmpeg command quoting on Windows.

Do not infer media correctness from a successful process exit alone. Inspect command,
metadata, durations, and representative output.

## Testing and benchmark discipline

Run narrow tests first.

Preferred general validation:

```powershell
python -m pytest
python -m dubpipeline.cli --help
python -m dubpipeline.cli run <pipeline.yaml> --plan
python -m dubpipeline.cli speak --text "Тест" --out-audio <path> --plan
```

Use the repository's actual test paths and supported command syntax.

For performance work, use the established policy when applicable:

```text
1 warmup run + 5 measured runs
report median
record commit, branch, device, compute type, batch size, model, and input
```

Do not compare runs with different models, inputs, hardware modes, or cache state without
calling out the difference.

GPU-heavy integration tests may be unavailable in CI. State what ran and what remains
manual.

## Dependency and environment safety

- Preserve `requirements.txt` compatibility unless the ticket explicitly changes it.
- Do not commit Hugging Face tokens, pyannote tokens, credentials, private paths, or
  environment files.
- Do not commit model weights or downloaded packages.
- Verify FFmpeg/FFplay availability before media tests.
- Verify CUDA and torch versions before GPU claims.
- Keep environment-specific instructions out of production defaults.

## Change safety

- Do not modify production code during investigation-only tasks.
- Do not reset, clean, stash, revert, or overwrite unrelated user changes.
- Keep scope aligned with the ticket.
- Avoid unrelated refactoring.
- Preserve backward compatibility unless removal is explicit.
- Distinguish direct impact, adjacent impact, generated-artifact impact, model/runtime
  impact, GUI-only impact, test-only impact, and graph-proximity noise.

## Documentation and task handoff

Update README/config examples when public CLI, GUI, language, model, output, or setup
behavior changes.

After each YouTrack task, preserve handoff context:

- what was done;
- files changed;
- decisions taken;
- what remains;
- risks/limitations;
- next step.

## Definition of done

A non-trivial ticket is not complete until:

- the applicable pre-ticket workflow was executed;
- graph findings were validated against source;
- CLI/GUI/config/model/media risks were assessed;
- the smallest coherent implementation was completed;
- tests were added or updated;
- CRG was updated after changes;
- post-change impact was inspected;
- required tests, plan runs, and manual checks were executed or reported unavailable;
- benchmark claims include reproducible evidence;
- documentation/handoff obligations were evaluated;
- remaining risks were documented;
- an implementation report was produced.
