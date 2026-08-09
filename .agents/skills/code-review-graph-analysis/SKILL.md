---
name: code-review-graph-analysis
description: Use code-review-graph in DubPipeline for exact imports, functions, classes, callers, dependants, GUI/CLI entry paths, config consumers, step relationships, tests, and blast-radius analysis before and after implementation.
---

# Code Review Graph Analysis

## Repository workflow precedence

When `AGENTS.md` or `.codex/PRE_TICKET_WORKFLOW.md` requires this skill, that workflow
takes precedence.

- Level 1 and Level 2: mandatory scoped preflight and post-change update.
- Level 0: normally unnecessary.

## Purpose

Use CRG as a structural dependency and review aid.

It does not replace source inspection, tests, plan-mode validation, GPU/model runs,
GUI checks, or media inspection.

## Scope

Analyze:

- CLI parser and dispatch;
- config types/load/save/precedence;
- GUI controllers and callbacks;
- subprocess, multiprocessing, and threading paths;
- pipeline steps and artifact consumers;
- model catalog/installer/runtime;
- translation and TTS services;
- input discovery;
- output movement and muxing;
- tests.

## Exclusions

Exclude virtual environments, caches, downloaded models, generated media, output,
build/dist/package artifacts, graph databases, and large binary fixtures.

## Workflow

1. Resolve repository root.
2. Read instructions and ticket.
3. Inspect diff first for reviews.
4. Discover exact CRG command/configuration.
5. Verify freshness with successful update/query.
6. Collect task-specific symbols and dependants.
7. Validate important relationships in source.
8. After implementation, update CRG and inspect blast radius.
9. Record coverage and limitations.

Do not invent CLI syntax.

Possible command families are examples only:

```powershell
code-review-graph --help
code-review-graph build
code-review-graph update --brief
code-review-graph detect-changes --brief
```

Confirm locally before execution.

## Required analysis

Inspect as applicable:

- imports;
- functions/classes;
- parser construction and dispatch;
- config consumers;
- GUI event handlers;
- background threads;
- child-process targets;
- subprocess command builders;
- model initialization;
- step inputs/outputs;
- artifact naming;
- cleanup;
- tests.

Answer:

1. What symbols change?
2. Who imports/calls them?
3. What depends on them?
4. Which entry points make them reachable?
5. Which tests assert behavior?
6. Which config and artifacts are adjacent?
7. What is the expected blast radius?
8. Are new paths disconnected?
9. Are obsolete paths still active?
10. Does impact cross CLI, GUI, model, or media boundaries unexpectedly?

## Dynamic-runtime caveat

CRG may not fully represent:

- subprocess execution;
- multiprocessing `spawn`;
- GUI callback strings;
- environment-based configuration;
- YAML-derived paths;
- dynamic model selection;
- external FFmpeg behavior;
- imported native/CUDA code.

Verify these directly.

## Post-change review

After implementation:

- update CRG;
- inspect changed symbols and dependants;
- inspect CLI/GUI reachability;
- inspect config consumers;
- inspect step/artifact consumers;
- inspect tests;
- investigate unexpected cross-module impact.

## Failure handling

Record confirmed command/error. Preserve existing data. Continue with Graphify, `rg`,
source, tests, plan output, and representative runtime checks. Report partial/unavailable CRG.

## Definition of done

- availability/freshness assessed;
- confirmed invocation used or absence reported;
- scoped dependency analysis completed;
- important relationships source-verified;
- tests and dynamic-runtime caveats examined;
- CRG updated after implementation;
- blast radius and limitations documented.
