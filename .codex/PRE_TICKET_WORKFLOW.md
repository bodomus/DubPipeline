# PRE_TICKET_WORKFLOW.md

> Mandatory repository-intelligence workflow for Codex before every non-trivial
> DubPipeline ticket, bugfix, feature, refactor, performance change, model/runtime
> change, GUI/CLI change, investigation, planning task, or code review.

## 0. Authority and purpose

This workflow defines how DubPipeline must be investigated, modified, and validated.

The repository uses:

1. **Graphify** for architectural and semantic orientation.
2. **code-review-graph (CRG)** for concrete structural relationships and impact analysis.

Neither graph is authoritative.

Current source, tests, plan output, logs, FFmpeg commands, generated artifact metadata,
and runtime behavior remain authoritative.

Explicit user instructions take precedence, then applicable `AGENTS.md`, then this workflow.

## 1. Workflow levels

### Level 0 — trivial

Examples:

- spelling;
- formatting;
- comment-only changes;
- metadata-only edits.

Required:

- read instructions;
- inspect Git state;
- validate the edited file.

No graph preflight required.

### Level 1 — local change

Examples:

- narrow parser validation fix;
- local path-handling correction;
- isolated utility or test fix;
- small GUI label/state correction;
- documentation-aligned config fix.

Required:

- repository baseline;
- CRG scoped analysis;
- direct source validation;
- focused tests;
- plan-mode validation when applicable;
- Graphify reuse/query when architecture context matters.

### Level 2 — structural, runtime, or performance change

Examples:

- new pipeline step;
- model lifecycle or RAM/VRAM reuse;
- multi-file batching;
- translation backend/model changes;
- WhisperX/pyannote/XTTS changes;
- GUI process/thread architecture;
- CLI/config precedence change;
- output naming or mux contract change;
- FFmpeg/audio alignment changes;
- dependency upgrade;
- broad refactor;
- performance optimization.

Required:

- full Graphify preflight;
- full CRG preflight;
- investigation;
- implementation plan;
- source/config/test/runtime validation;
- post-change CRG update;
- narrow plus broader tests;
- representative manual or integration run when required;
- benchmark protocol for performance claims;
- Graphify refresh when architecture changed.

When uncertain, choose Level 2.

## 2. Execution order

1. Read instructions and ticket.
2. Resolve repository root and record Git state.
3. Classify workflow level.
4. Identify affected execution surfaces:
   CLI, GUI, config, step, model, translation, TTS, media, tests, packaging.
5. Check Graphify when applicable.
6. Check CRG.
7. Gather scoped graph context.
8. Validate findings in source/config/tests.
9. Assess model, GPU, subprocess, artifact, and media risk.
10. Produce `investigation.md` and `implementation-plan.md` for Level 2.
11. Implement the smallest coherent change.
12. Update CRG.
13. Inspect blast radius.
14. Run static/unit/plan validation.
15. Run representative integration/manual validation when required and available.
16. Run benchmark protocol for performance work.
17. Refresh Graphify only for structural change.
18. Update docs and handoff.
19. Produce an implementation report.

Do not skip directly to implementation.

## 3. Repository baseline

Run:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
python --version
python -m pip --version
```

Read:

- root `AGENTS.md`;
- `.codex/PRE_TICKET_WORKFLOW.md`;
- ticket;
- `README.md`;
- `requirements.txt`;
- applicable YAML/config files;
- relevant tests and scripts.

Identify:

- active virtual environment;
- Python executable;
- CUDA availability when relevant;
- torch/torchaudio/torchvision versions;
- FFmpeg/FFplay availability;
- affected entry point;
- generated-output locations;
- pre-existing user changes.

Do not alter or delete unrelated user work.

## 4. Graphify preflight

Follow `$graphify-repository-analysis`.

For Level 2, identify:

- pipeline orchestration;
- step ownership and dependencies;
- CLI/config/GUI relationships;
- model catalog/installer/service relationships;
- translation and TTS boundaries;
- output and artifact flows;
- subprocess/process/thread boundaries;
- tests associated with the area.

Expected output may be under `graphify-out/`.

Exclude virtual environments, downloaded models, generated media, output directories,
caches, graph databases, build/dist output, and third-party source trees.

Use only confirmed installed Graphify commands. Do not invent slash commands or update flags.

Validate important findings in source.

## 5. CRG preflight

Follow `$code-review-graph-analysis`.

Use only confirmed CRG commands.

Expected local state may be under `.code-review-graph/`.

Collect:

- exact functions/classes/modules;
- imports and callers;
- subprocess and multiprocessing entry points;
- GUI callback and event paths;
- config consumers;
- step consumers/producers;
- tests;
- expected blast radius.

Do not treat index-file existence as freshness.

## 6. Mandatory investigation

Answer:

1. What is current behavior?
2. What is expected behavior?
3. What is the root cause or missing capability?
4. What is the smallest correct change?
5. Which CLI, GUI, config, steps, models, utilities, and tests are affected?
6. Does the change alter config precedence or backward compatibility?
7. Does it alter artifact names, locations, or resume behavior?
8. Does it alter model loading, reuse, VRAM/RAM, or cache behavior?
9. Does it alter multiprocessing, subprocess, threading, or cancellation?
10. Does it alter language validation or target-aware output?
11. Does it alter media timing, stream mapping, metadata, or loudness?
12. What tests and representative runs are required?
13. Is a benchmark required?
14. Which documentation/handoff files require updates?
15. Is there graph/source/README disagreement?

For Level 2, write:

```text
investigation.md
implementation-plan.md
```

## 7. Configuration change gate

For config/YAML/ENV/CLI changes:

1. inspect dataclasses and defaults;
2. inspect YAML load/save;
3. inspect ENV mapping and legacy variables;
4. inspect CLI explicit flags and `--set`;
5. inspect GUI read/write;
6. inspect derived paths;
7. inspect plan mode;
8. add precedence and backward-compatibility tests;
9. ensure plan mode has no writes/model loading.

Preserve:

```text
defaults -> YAML -> ENV -> CLI
```

## 8. Model/runtime change gate

For WhisperX, faster-whisper, pyannote, translation, XTTS, torch, CUDA, or model installer changes:

- inspect dependency pins;
- identify model ownership and initialization count;
- inspect per-file versus per-batch lifetime;
- inspect cache and download behavior;
- inspect GPU/CPU selection;
- inspect precision and batch size;
- inspect VRAM release;
- inspect failure cleanup;
- inspect multi-file behavior;
- run representative validation;
- benchmark comparable runs.

Do not upgrade coupled dependencies without a compatibility matrix and rollback plan.

## 9. CLI change gate

For CLI changes:

- inspect parser, config override generation, validation, execution dispatch, and tests;
- validate before expensive initialization;
- preserve mutual exclusions;
- preserve `--plan`;
- preserve `--steps` forms;
- preserve `run` and `speak`;
- preserve language validation;
- preserve file/folder behavior;
- verify errors and exit behavior;
- update README.

## 10. GUI change gate

For GUI changes:

- inspect main event loop;
- inspect background threads;
- inspect subprocess and multiprocessing controllers;
- inspect `window.write_event_value`;
- inspect stop/terminate/join/kill behavior;
- preserve top-level spawn targets;
- verify close/cancel cleanup;
- keep GUI and CLI/config contracts aligned;
- manually validate changed interactions.

## 11. Pipeline step and artifact gate

For any step:

- identify inputs;
- identify outputs;
- identify skip/rebuild checks;
- identify downstream consumers;
- verify target-language naming;
- verify partial failure and cleanup;
- verify batch/multi-file behavior;
- verify logging/timing/progress;
- add tests around contracts.

## 12. Audio/media gate

For FFmpeg, TTS alignment, mixing, or mux changes:

- inspect exact command construction;
- verify Windows path quoting;
- verify sample rate/channels;
- verify segment durations and gaps;
- verify tempo/stretch constraints;
- verify clipping and loudness;
- verify ducking;
- verify stream mapping;
- verify metadata and language tags;
- verify original file replacement safety;
- inspect representative output with FFprobe when available.

## 13. Implementation rules

- Preserve user changes.
- Keep scope focused.
- Avoid unrelated refactoring.
- Keep plan mode side-effect-free.
- Preserve backward compatibility unless removal is explicit.
- Keep heavy imports/model loading out of paths that should remain cheap.
- Clean up processes/threads/resources.
- Add or update tests.
- Re-query CRG for unexpected dependencies.
- Re-query Graphify for unexpected subsystem boundaries.
- Keep README/config examples synchronized.

## 14. Post-change CRG validation

After changes:

1. update CRG with confirmed command;
2. inspect changed symbols;
3. inspect import/caller/dependant impact;
4. inspect CLI/GUI reachability;
5. inspect config consumers;
6. inspect step/artifact consumers;
7. inspect tests;
8. investigate unexpected cross-module impact;
9. verify new code is reachable;
10. verify obsolete paths are not unintentionally active.

## 15. Validation order

Use the narrowest applicable validation first:

```powershell
python -m pytest <focused tests>
python -m pytest
python -m dubpipeline.cli --help
python -m dubpipeline.cli run <pipeline.yaml> --plan
python -m dubpipeline.cli speak --text "Тест" --out-audio <path> --plan
```

When relevant:

- import smoke test;
- config precedence test;
- FFmpeg/FFprobe command test;
- CPU representative run;
- GPU representative run;
- GUI manual test;
- multi-file folder run;
- output replacement safety test.

Do not claim execution that did not occur.

## 16. Benchmark protocol

For performance claims:

1. define fixed input;
2. record commit and branch;
3. record Python, torch, CUDA, GPU, model, precision, batch size, and config;
4. distinguish cold model load from warm processing;
5. run one warmup;
6. run five measured runs;
7. report individual times and median;
8. record peak RAM/VRAM when relevant;
9. compare equivalent artifact/cache state;
10. report quality or correctness regressions.

For multi-file model-reuse work, report separately:

- first-file cold cost;
- subsequent-file cost;
- total folder throughput;
- model initialization count;
- peak RAM/VRAM;
- cleanup behavior after batch completion.

## 17. Graphify post-change policy

Refresh Graphify only if architecture changed:

- new step/module;
- new model/service boundary;
- CLI/GUI orchestration redesign;
- major config flow change;
- batching/lifecycle redesign;
- broad refactor.

Update CRG after every non-trivial code change.

## 18. Documentation and handoff

Evaluate updates to:

- README setup and commands;
- CLI options;
- GUI behavior;
- supported languages;
- model/dependency notes;
- output naming;
- config examples;
- benchmark documentation.

Save task handoff:

- what was done;
- files changed;
- decisions;
- remaining work;
- risks/limitations;
- next step.

## 19. Failure handling

If Graphify or CRG fails:

- record exact confirmed command and concise error;
- do not fabricate findings;
- continue with `rg`, source, tests, plan output, and runtime checks when safe;
- report degraded analysis.

If GPU/model/media validation is unavailable:

- state what was unavailable;
- state alternative validation;
- state remaining risk.

## 20. Required implementation report

```markdown
# Implementation Report

## Ticket
<ticket id and summary>

## Workflow
- Level: 1 / 2
- Graphify:
- CRG:
- Working tree before changes:

## Scope
- CLI:
- GUI:
- Config:
- Pipeline steps:
- Models/runtime:
- Media/artifacts:

## Investigation
- Current behavior:
- Expected behavior:
- Root cause/gap:
- Main symbols:
- Expected blast radius:
- Compatibility concerns:

## Changes
- ...

## Graph and source validation
- Graphify findings:
- CRG findings:
- Source/config validations:
- Discrepancies:

## Post-change impact
- CRG updated:
- Blast radius:
- Unexpected dependants:
- Artifact compatibility:

## Validation
- Focused tests:
- Full tests:
- CLI help:
- Plan mode:
- CPU run:
- GPU run:
- GUI:
- FFmpeg/FFprobe:
- Multi-file:

## Benchmark
- Required:
- Method:
- Results:

## Documentation and handoff
- Updated:
- Handoff:

## Remaining risks
- ...
```

## 21. Non-negotiable rules

1. Do not start non-trivial implementation before applicable preflight.
2. Do not invent Graphify or CRG commands.
3. Do not trust graph output without source validation.
4. Do not destroy unrelated user changes.
5. Do not load heavy models during plan-mode validation.
6. Do not claim GPU/media correctness from unit tests alone.
7. Do not change coupled dependency versions casually.
8. Do not claim performance improvements without comparable measurements.
9. Preserve config precedence and backward compatibility unless explicitly changed.
10. Update CRG after implementation.
11. Source and executable evidence win over graph inference.
