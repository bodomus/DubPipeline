# DubPipeline Codex workflow package

Prepared for `bodomus/DubPipeline` from the current `master` repository state.

Copy into the repository root while preserving paths:

```text
DubPipeline/
├── AGENTS.md
├── .codex/
│   └── PRE_TICKET_WORKFLOW.md
├── .agents/
│   └── skills/
│       ├── graphify-repository-analysis/
│       │   └── SKILL.md
│       └── code-review-graph-analysis/
│           └── SKILL.md
└── .gitignore.recommended
```

Commit:

- `AGENTS.md`;
- `.codex/`;
- `.agents/skills/`.

Review `.gitignore.recommended` and merge only appropriate rules.

The package intentionally does not include Graphify/CRG wrapper scripts because the exact
installed CLI syntax and output configuration must first be confirmed locally. The workflow
forbids Codex from inventing commands.

Repository-specific facts incorporated:

- CLI entry point `python -m dubpipeline.cli`;
- `run` and `speak` commands;
- `--plan` dry-run behavior;
- FreeSimpleGUI GUI;
- Windows multiprocessing/subprocess concerns;
- config precedence: defaults -> YAML -> ENV -> CLI;
- supported translation targets `de`, `fr`, `es`, `ru`;
- target-aware output paths and track metadata;
- torch/torchvision/torchaudio 2.6/0.21/2.6 with CUDA 12.4;
- pyannote.audio 3.4.0 compatibility pin;
- WhisperX, faster-whisper, Argos/HF translation, XTTS;
- FFmpeg/FFplay media flow;
- model loading, RAM/VRAM, batching, cleanup, and benchmark requirements;
- YouTrack handoff template from README.
