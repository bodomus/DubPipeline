---
name: graphify-repository-analysis
description: Use Graphify to orient within DubPipeline architecture, discover relationships among CLI, GUI, configuration, steps, models, translation, TTS, media handling, and tests, and produce source-verified context before structural implementation or review.
---

# Graphify Repository Analysis

## Repository workflow precedence

When `AGENTS.md` or `.codex/PRE_TICKET_WORKFLOW.md` requires this skill, that workflow
takes precedence.

- Level 2: mandatory full preflight.
- Level 1: reuse or query when architecture context matters.
- Level 0: normally unnecessary.

## Purpose

Use Graphify for architectural orientation and cross-module candidate discovery.

Graphify is supporting evidence only. Validate important findings in current source,
configuration, tests, plan output, logs, and runtime behavior.

## Scope model

Orient around:

- `dubpipeline.cli`;
- `dubpipeline.gui`;
- `dubpipeline.config`;
- `dubpipeline.steps`;
- `dubpipeline.models`;
- `dubpipeline.translation`;
- `dubpipeline.utils`;
- input discovery and input-mode logic;
- external subtitles;
- tests and supported diagnostic scripts.

Use Graphify to find:

- pipeline orchestration;
- step boundaries;
- config consumers;
- GUI-to-CLI execution paths;
- model installer/catalog/service relationships;
- translation/TTS/mux boundaries;
- output/artifact flows;
- subprocess/process/thread boundaries;
- related tests.

## Exclusions

Exclude:

- virtual environments;
- downloaded models and model repositories;
- generated media and output directories;
- build/dist/package output;
- caches and test reports;
- `.code-review-graph`;
- `graphify-out`;
- large binary fixtures unless explicitly in scope.

Retain Python, YAML, tests, and small textual fixtures.

## Workflow

1. Resolve repository root.
2. Read `AGENTS.md`, workflow, ticket, README, requirements, and relevant config.
3. Confirm Graphify availability and exact installed commands.
4. Assess graph freshness and ticket coverage.
5. Reuse, build, or refresh only when justified.
6. Run focused queries using concrete symbols and pipeline terms.
7. Build a compact candidate working set.
8. Validate all implementation-relevant conclusions in source.
9. Record commands, findings, validation, and limitations.

Do not invent slash commands, output flags, backends, or update syntax.

A local installation may use Ollama, but confirm local configuration before execution.
Do not commit machine-specific model/backend settings.

## Query guidance

Useful concepts include:

- `build_parser`;
- `load_pipeline_config_ex`;
- `PipelineConfig`;
- CLI override generation;
- `run` and `speak`;
- GUI `run_pipeline`;
- preview/playback controllers;
- model catalog and installer;
- ASR/alignment;
- translation service;
- TTS synthesis;
- output mover;
- FFmpeg mux/mix;
- target-aware paths;
- multi-file input discovery;
- timing and run metadata.

Use graph paths only as navigation, never as proof of runtime order.

## Source-validation rules

Verify directly:

- parser dispatch;
- dynamic config selection;
- YAML/ENV/CLI precedence;
- subprocess commands;
- multiprocessing targets;
- GUI event callbacks;
- model ownership and loading;
- artifact paths;
- skip/rebuild behavior;
- language validation;
- cleanup;
- tests.

## Failure handling

Record exact confirmed command and concise error. Preserve existing graph data. Continue
with CRG, `rg`, source, tests, plan output, and runtime checks. Report partial/unavailable
Graphify analysis.

## Definition of done

- availability/freshness assessed;
- graph reused/refreshed only when justified;
- focused task-specific queries run;
- source/config findings verified;
- large generated/model content excluded;
- limitations reported.
