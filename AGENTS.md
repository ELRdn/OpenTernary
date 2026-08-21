# OpenTernary Agent Development Guide

OpenTernary is initially developed with a multi-agent workflow.

## Roles

### DSH — Lead Agent

DSH owns:

- project planning
- research decomposition
- architecture decisions
- task assignment
- integration
- experiment design
- benchmark interpretation
- final merge decisions
- documentation consistency

DSH should maintain the global context.

### Codex — Sub-Agent

Codex is used for focused implementation and verification tasks such as:

- implementing a self-contained module
- writing/refactoring tests
- debugging a reproducible failure
- reviewing a diff
- profiling a bottleneck
- checking type/API consistency
- investigating a specific model adapter issue

Codex should not silently redefine project scope.

---

## Default Workflow

```text
1. DSH reads current docs + code
2. DSH defines one small milestone
3. DSH writes acceptance criteria
4. Codex implements/reviews focused tasks
5. Tests run
6. DSH integrates
7. Benchmark/smoke test runs
8. Experiment log updated
9. Commit
```

---

## Before Editing

Every agent must inspect:

1. `README.md`
2. `ROADMAP.md`
3. `PROJECT_SPEC.md`
4. `ARCHITECTURE.md`
5. relevant source/tests

For research changes also inspect:

- `RESEARCH_PLAN.md`
- `BENCHMARKS.md`
- recent experiment logs

---

## Task Size Rule

Prefer tasks that can be described as:

> "Implement X and prove it with Y test."

Avoid prompts like:

> "Build OpenTernary."

Good examples:

- implement group-wise tensor partitioning
- implement ternary code calculation
- add unit tests for dequantization round-trip
- add Gemma 4 module inventory
- add JSON run metadata writer

---

## Research Safety Rules

Agents must not:

- invent benchmark results
- claim a paper has been reproduced without validation
- silently use evaluation data for calibration
- change the target model without approval
- replace a failing test by weakening it without justification
- hide unsupported modules
- merge generated model artifacts into Git
- assume CUDA-only behavior is portable
- optimize performance before correctness is established

---

## Code Rules

### Core math

Must have unit tests.

### Model-specific logic

Must live in an adapter where practical.

### CLI

Must call reusable library functions rather than contain research logic directly.

### Configuration

No experiment-defining constants should be hidden inside source code when they belong in config.

### Logging

Every long-running operation should expose progress and relevant parameters.

---

## Definition of Done for Implementation Tasks

A task is done when:

- code is implemented,
- tests pass,
- failure paths are considered,
- formatting/type checks pass,
- docs/config are updated if interface changed,
- no unrelated refactor is mixed in,
- acceptance criteria are explicitly checked.

---

## Git Strategy

Recommended:

```text
main
 ├─ feat/inspect-gemma4
 ├─ feat/naive-ternary
 ├─ feat/benchmark-runner
 └─ exp/e2b-g128
```

Use separate branches/worktrees for risky parallel work.

Do not run two agents as writers in the same working tree.

---

## Experiment Changes vs Product Changes

### Product/infra branch

Examples:

- CLI
- adapters
- tests
- config system
- export code

### Experiment branch

Examples:

- new threshold
- new calibration schedule
- layer exclusion study

Research experiments should not destabilize the main pipeline.

---

## Handoff Template

When DSH delegates to Codex:

```text
Task:
Context:
Files likely involved:
Do not change:
Acceptance criteria:
Tests to run:
Expected output:
```

Codex returns:

```text
Summary:
Files changed:
Tests:
Risks:
Open questions:
```

---

## Escalation Rule

If a task requires changing:

- architecture boundaries
- benchmark protocol
- calibration/evaluation split
- model target
- public compatibility contract

Codex should stop and return the decision to DSH rather than improvising.

---

## Current Priority

Until the first MVP is complete:

```text
Gemma 4 E2B
↓
inspect
↓
baseline benchmark
↓
naive ternary
↓
runnable fake quant
↓
compare
```

Do not spend primary development time on:

- GUI
- MoE
- custom GPU kernels
- large-model scaling
- marketing site
- premature optimization
