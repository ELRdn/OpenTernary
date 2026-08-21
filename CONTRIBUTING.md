# Contributing to OpenTernary

OpenTernary is currently an experimental research project.

Contributions are welcome once the repository is publicly opened, but reproducibility takes priority over feature count.

---

## Before Contributing

Read:

- `README.md`
- `PROJECT_SPEC.md`
- `ARCHITECTURE.md`
- `ROADMAP.md`
- `RESEARCH_PLAN.md`
- `BENCHMARKS.md`

---

## Good First Contributions

Examples:

- unit tests
- tensor utility improvements
- benchmark adapters
- documentation
- environment diagnostics
- model adapter validation
- reproducible bug reports

---

## Pull Request Requirements

A PR should include:

1. what problem it solves,
2. why the chosen approach is appropriate,
3. tests,
4. benchmark impact if relevant,
5. compatibility impact,
6. docs/config updates when interfaces change.

Avoid unrelated refactors.

---

## Research Contributions

If proposing a new quantization/calibration method, include:

- source/reference
- exact implemented idea
- deviations from source
- baseline
- configuration
- evaluation data
- results
- limitations

A new method should not be advertised as superior based on a single cherry-picked prompt.

---

## Bug Reports

Include:

```text
OpenTernary commit/version:
Model:
Model revision:
Command/config:
Hardware:
OS:
Python:
Framework version:
Full error:
Minimal reproduction:
```

---

## Generated Artifacts

Do not commit:

- downloaded model weights
- large checkpoints
- calibration caches
- benchmark caches
- packed model binaries

unless the repository explicitly defines a small fixture.

---

## License Note

The repository license must be selected before accepting external code contributions.
