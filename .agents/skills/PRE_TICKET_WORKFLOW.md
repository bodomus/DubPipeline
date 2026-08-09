# PRE_TICKET_WORKFLOW.md

> Mandatory workflow for Codex before every non-trivial ticket, bugfix, refactor, feature, or review task.

## 0. Purpose

This file defines the mandatory repository-intelligence workflow that must run **before work on every ticket**.

The repository uses two complementary code intelligence systems:

1. **Graphify**
   - architecture and subsystem exploration;
   - concept relationships;
   - cross-file connections;
   - architecture-oriented `query`, `explain`, and `path` navigation.

2. **code-review-graph (CRG)**
   - structural code graph;
   - callers, callees, inheritance, imports, and dependency analysis;
   - impact radius / blast radius;
   - review context;
   - change-aware and incremental analysis.

Neither graph is the final source of truth.

**Source code, tests, build output, runtime behavior, and project documentation remain authoritative.**

Graph-derived conclusions that affect implementation decisions must be validated against source code.

---

# 1. Mandatory execution order

For every ticket, execute the following phases in order:

1. Read repository instructions.
2. Verify Graphify availability and graph health.
3. Create or refresh the Graphify graph when required.
4. Verify code-review-graph availability and graph health.
5. Create or update the CRG graph when required.
6. Collect ticket-specific context from both graphs.
7. Validate important findings against source code.
8. Inspect relevant tests and build configuration.
9. Produce an investigation summary and implementation plan when the ticket is non-trivial.
10. Implement the ticket.
11. Update CRG after implementation.
12. Inspect blast radius / review context.
13. Run targeted tests, then broader validation when required.
14. Refresh Graphify only when architecture or subsystem structure changed.
15. Produce a concise implementation report.

Do not skip directly to implementation.

---

# 2. Phase A — repository instructions

Before any repository-wide search or implementation:

1. Read the root `AGENTS.md`.
2. Read any nested `AGENTS.md` files that apply to the target files.
3. Read ticket-specific instructions supplied by the user.
4. Identify:
   - repository root;
   - current branch;
   - current commit;
   - dirty working-tree state;
   - relevant solution/project/module boundaries;
   - build and test commands already documented by the repository.

Record the initial state.

Suggested baseline commands:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

Do not overwrite, revert, clean, stash, reset, or otherwise destroy pre-existing user changes.

---

# 3. Phase B — Graphify preflight

## 3.1 Verify Graphify CLI / skill availability

Check whether Graphify is available through the project-installed Codex integration or CLI.

Examples:

```powershell
graphify --help
```

and/or use the installed Graphify Codex skill.

If Graphify itself is unavailable:

1. Do not silently install or upgrade global software unless the ticket explicitly permits environment modification.
2. Report the missing prerequisite clearly.
3. Continue with CRG and direct source inspection if possible.
4. Mark Graphify analysis as unavailable in the final report.

A missing tool must not cause fabricated graph findings.

---

## 3.2 Check Graphify graph existence

Expected Graphify artifacts normally include:

```text
graphify-out/
├── graph.json
├── GRAPH_REPORT.md
└── graph.html
```

At minimum, verify that the persisted graph exists and is readable.

If the graph does not exist or is unusable:

1. Build it for the repository using the installed Graphify skill/workflow.
2. Respect repository-specific exclusions.
3. Do not include generated/vendor/build/cache directories unless the repository instructions explicitly require them.
4. Verify that `graphify-out/graph.json` can be queried successfully.

Preferred Graphify build workflow:

```text
/graphify .
```

or the equivalent installed Graphify project command.

After creation, run at least one scoped query relevant to the repository or ticket.

---

## 3.3 Check Graphify freshness

Determine whether the graph is suitable for the current ticket.

Use the current repository state, Graphify metadata, recent commits, and affected paths.

Refresh Graphify when one or more of the following is true:

- no graph exists;
- the graph is unreadable or query commands fail;
- the graph predates major architectural changes relevant to the ticket;
- a new subsystem/module/plugin/application was added;
- important files relevant to the ticket are absent from the graph;
- a previous refactor materially changed subsystem boundaries;
- repository instructions explicitly require a refresh.

For changed files or incremental refresh, prefer the Graphify update workflow rather than a blind full rebuild when supported by the installed integration.

Example conceptual workflow:

```text
/graphify . --update
```

Use project-specific backend/environment configuration already documented by the repository.

Do not invent backend credentials or switch inference providers without explicit permission.

---

## 3.4 Query Graphify for ticket context

Before broad source reading, use ticket-specific Graphify queries.

Generate questions from the ticket itself.

Minimum query set:

1. **Subsystem ownership**
   - Which subsystem/module/community owns the requested behavior?

2. **Main concepts**
   - Which classes, functions, services, modules, widgets, controllers, workers, or commands are central to the ticket?

3. **Cross-file relationships**
   - Which important relationships cross module or file boundaries?

4. **Architecture explanation**
   - Explain the main symbol or concept named in the ticket.

Examples:

```powershell
graphify query "Which subsystem owns <ticket behavior> and what are its main components?"
graphify explain "<MainSymbol>"
graphify query "What connects <EntryPoint> to <TargetBehavior>?"
```

Use `graphify path` only as a navigation aid.

**Never treat a shortest path as proof of runtime execution flow.**

Graph paths may reflect structural or semantic graph connectivity rather than actual runtime call order.

---

# 4. Phase C — code-review-graph preflight

## 4.1 Verify CRG availability

Check:

```powershell
code-review-graph --help
```

If CRG itself is unavailable:

1. Do not silently install or upgrade global software unless environment modification is explicitly permitted.
2. Report the missing prerequisite.
3. Continue with Graphify and direct source inspection.
4. Mark CRG analysis as unavailable in the final report.

---

## 4.2 Verify or create the CRG graph

The preferred behavior is:

1. Try to update/query the existing graph.
2. If the graph is missing, uninitialized, corrupted, or incompatible, build it.
3. Verify that analysis commands succeed.

Initial build:

```powershell
code-review-graph build
```

For an existing graph that may be stale:

```powershell
code-review-graph update --brief
```

When hooks or daemon/watch mode are known to have already kept the graph fresh, a read-only change check may be used:

```powershell
code-review-graph detect-changes --brief
```

When freshness is uncertain, prefer:

```powershell
code-review-graph update --brief
```

Do not assume the graph is fresh merely because a database file exists.

A successful file-existence check is weaker than a successful graph update/query.

---

## 4.3 Collect CRG context for the ticket

Use CRG CLI and/or MCP tools available in the Codex session.

Collect, as applicable:

- architecture overview;
- symbol search;
- callers;
- callees;
- imports;
- inheritance relationships;
- dependants;
- related tests;
- change review context;
- impact radius / blast radius;
- execution-flow context when supported.

For a bugfix or feature, answer:

1. What symbols are likely to change?
2. Who calls them?
3. What depends on them?
4. Which tests are connected to the affected area?
5. What files form the minimal review context?
6. What is the likely blast radius?

Do not dump the entire graph into context.

Prefer scoped graph queries.

---

# 5. Phase D — merge both intelligence sources

Before implementation, combine the results.

Create a working understanding with four layers:

## Layer 1 — Ticket intent

What behavior is requested?

What is the acceptance condition?

What must not change?

## Layer 2 — Graphify architecture view

Identify:

- owning subsystem;
- important architectural concepts;
- cross-file/module relationships;
- likely entry points;
- likely data/control boundaries.

## Layer 3 — CRG structural view

Identify:

- concrete symbols;
- callers and callees;
- dependants;
- inheritance relationships;
- imports;
- tests;
- likely impact radius.

## Layer 4 — Source validation

Open the minimum necessary source files and verify:

- Graphify conclusions;
- CRG conclusions;
- actual control flow;
- actual data flow;
- error handling;
- configuration;
- test behavior.

When graph output and source code disagree, source code wins.

Record the discrepancy if it is relevant.

---

# 6. Mandatory ticket investigation

For non-trivial tickets, do not implement until these questions are answered:

1. What is the current behavior?
2. What is the expected behavior?
3. What is the root cause, missing capability, or architectural gap?
4. What is the smallest correct change?
5. Which symbols are directly affected?
6. Which callers/dependants may be indirectly affected?
7. Which tests already cover the area?
8. Which tests are missing?
9. What build/test commands validate the change?
10. Is there any graph/source disagreement that requires caution?

For substantial tasks, produce:

```text
investigation.md
implementation-plan.md
```

unless the ticket or repository workflow defines different artifact names.

---

# 7. Implementation rules

During implementation:

1. Follow `AGENTS.md`.
2. Keep scope aligned with the ticket.
3. Prefer the smallest coherent change.
4. Do not perform unrelated refactors.
5. Preserve existing public behavior unless change is explicitly required.
6. Add or update tests for changed behavior.
7. Re-query CRG when implementation reveals unexpected dependencies.
8. Re-query Graphify when implementation reveals an unexpected subsystem boundary.
9. Validate risky assumptions directly against source.
10. Preserve user changes already present in the working tree.

---

# 8. Post-implementation CRG validation

After code changes:

```powershell
code-review-graph update --brief
```

Then inspect change impact using available CLI and/or MCP analysis tools.

Required post-change questions:

1. What changed?
2. What is the blast radius?
3. Which callers or dependants are now affected?
4. Which tests should run?
5. Are there suspicious untested paths?
6. Did the change reach outside the intended subsystem?

If the blast radius is larger than expected:

- stop expanding implementation blindly;
- inspect the additional affected files;
- determine whether the implementation should be narrowed;
- add missing validation when necessary.

---

# 9. Testing order

Run tests from narrowest to broadest.

Preferred order:

1. directly affected unit tests;
2. subsystem/module tests;
3. integration tests relevant to the ticket;
4. project/solution test suite when justified;
5. build/package/editor validation required by repository instructions.

Do not claim tests passed unless they were actually executed successfully.

If a test cannot be run, state:

- which test was not run;
- why;
- what evidence is available instead.

---

# 10. Post-implementation Graphify refresh policy

Do **not** blindly rebuild Graphify after every small ticket.

Refresh/update Graphify after implementation when the ticket changed:

- architecture;
- subsystem boundaries;
- major cross-module relationships;
- important entry points;
- public workflows;
- large refactors;
- module/plugin/application structure.

For a small local bugfix with no architectural effect, Graphify refresh is normally unnecessary.

CRG should be updated more aggressively because it is used for change impact and review context.

---

# 11. Failure handling

## Graphify failure

If Graphify build/update/query fails:

1. capture the exact failure;
2. do not fabricate results;
3. continue with CRG and source inspection when possible;
4. mention degraded analysis in the report.

## CRG failure

If CRG build/update/query fails:

1. capture the exact failure;
2. do not fabricate callers, dependants, or blast radius;
3. continue with Graphify and source inspection when possible;
4. mention degraded impact analysis in the report.

## Both tools fail

If both intelligence tools are unavailable:

1. continue using repository instructions and direct source analysis;
2. explicitly state that graph-assisted analysis was unavailable;
3. avoid claiming graph-derived confidence.

Tool failure should degrade the workflow, not cause hallucinated evidence.

---

# 12. Required implementation report

At the end of the ticket, provide a concise report with this structure:

```markdown
# Implementation Report

## Ticket
<ticket id and summary>

## Preflight
- Graphify: available / unavailable
- Graphify graph: existing / created / updated / unchanged
- CRG: available / unavailable
- CRG graph: existing / created / updated
- Working tree before changes: clean / dirty

## Investigation
- Current behavior:
- Root cause or implementation gap:
- Main symbols:
- Owning subsystem:
- Expected blast radius:

## Changes
- ...

## Graph validation
- Graphify findings used:
- CRG findings used:
- Source validations performed:
- Graph/source discrepancies:

## Post-change impact
- CRG updated: yes / no
- Blast radius:
- Unexpected dependants:
- Related tests:

## Validation
- Build:
- Targeted tests:
- Broader tests:
- Manual validation:

## Remaining risks
- ...
```

---

# 13. Non-negotiable rules

1. **Do not start implementation before preflight is complete.**
2. **Do not assume graph databases exist. Check them.**
3. **If a graph is missing, create it.**
4. **If a graph is stale for the task, update it.**
5. **Do not trust graph output blindly. Validate important conclusions against source code.**
6. **Do not use Graphify shortest paths as proof of runtime flow.**
7. **Use Graphify primarily for architecture and relationships.**
8. **Use CRG primarily for structural dependencies, review context, and impact analysis.**
9. **Update CRG after implementation before final review.**
10. **Do not fabricate tool results when a tool fails.**
11. **Do not destroy or overwrite unrelated user changes.**
12. **Do not declare the ticket complete without validation evidence.**

---

# 14. Ticket execution handoff

After completing all mandatory preflight phases above, continue with the ticket supplied in the current task.

The ticket instructions define **what to implement**.

This workflow defines **how the repository must be investigated, analyzed, implemented, and validated**.

When ticket instructions conflict with this workflow:

1. follow explicit user instructions first;
2. follow applicable `AGENTS.md` instructions next;
3. use this workflow for all remaining decisions.

Begin every ticket by executing this workflow.
