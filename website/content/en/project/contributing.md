---
description: "How to set up a development environment, run the checks CI runs, work on the docs, and contribute code or research to OpenTernary."
---

# Contributing

OpenTernary is a research project, and **reproducibility comes before feature count**. Contributions are accepted
under the project's licence, Apache-2.0 (Section 5). The full guidelines are in
[CONTRIBUTING.md](https://github.com/ELRdn/OpenTernary/blob/main/CONTRIBUTING.md).

## Development setup

```bash
git clone https://github.com/ELRdn/OpenTernary.git
cd OpenTernary
uv sync                                   # core + dev tools from uv.lock
uv sync --extra ml --extra calibration    # when working on conversion, evaluation or calibration
```

The repository pins its uv version (`[tool.uv] required-version` in `pyproject.toml`), so use that version, and
`.python-version` pins Python 3.12. Never edit `uv.lock` by hand: use `uv add` / `uv lock`.

## The checks CI runs

Run these before opening a pull request:

```bash
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
uv run --frozen mypy src/openternary
uv run --frozen pytest
```

CI also runs:

- a **core-only** job: the contracts that must work without PyTorch;
- `python scripts/validate_cli_offline.py`: the full suite with empty Hugging Face caches, no GPU and offline
  settings, writing logs to `.tmp/`;
- `scripts/validate_cli_real.py --require-wheel`: an installed wheel against real libraries and seeded synthetic
  models in fresh processes, on Linux and Windows;
- `scripts/validate_cli_plugins.py`: real entry-point plugins, including conflicts and API mismatches;
- `pip install -e .` with no uv, to check that the pip path works.

## Working on the documentation

This site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from `website/` and
`mkdocs.yml`:

```bash
uv sync --group docs
uv run --frozen mkdocs serve          # http://127.0.0.1:8000, reloads on save
uv run --frozen mkdocs build --strict # what CI publishes: fails on any warning or broken link
```

- English pages are in `website/content/en/` and Japanese pages in `website/content/ja/`, with the **same file
  names**. Change both in the same pull request.
- `tests/test_docs.py` fails when a CLI command, option or configuration key is missing from the reference pages
  in either language, or when the two languages have different pages.
- TUI screenshots in `website/content/assets/images/` are SVG files exported by Textual from the real
  interface. Regenerate them rather than editing them.
- Pushing to `main` publishes the site to GitHub Pages (`.github/workflows/docs.yml`).

## Pull requests

A pull request should explain what problem it solves and why the approach fits, and include tests, the impact on
benchmarks (if any), the compatibility impact, and documentation and config updates when interfaces change. Avoid
unrelated refactors.

**Research contributions** (a new quantization or calibration method) must include the source or reference, the
exact idea implemented, any deviations from the source, the baseline, configuration, evaluation data, results and
limitations. A method is never presented as better on the strength of one cherry-picked prompt, and a paper or
framework is not called reproduced unless the implementation has been checked against its source.

**Never commit** model weights, large checkpoints, calibration or benchmark caches, or packed model binaries,
unless the repository explicitly defines a small fixture.

## Reporting bugs

Open an [issue](https://github.com/ELRdn/OpenTernary/issues) with:

```text
OpenTernary version (openternary --version):
Model and revision:
Command and config:
Hardware, OS, Python:
Output of `openternary doctor --json`:
Full error panel:
Minimal reproduction:
```

## Releasing

Maintainers publish releases through PyPI Trusted Publishing. See
[RELEASING.md](https://github.com/ELRdn/OpenTernary/blob/main/RELEASING.md).
