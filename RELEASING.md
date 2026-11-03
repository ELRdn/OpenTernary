# Releasing OpenTernary

Releases are published to PyPI by `.github/workflows/release.yml` with Trusted Publishing
(OIDC). No API token is stored anywhere.

## One-time setup (done 2026-11-03)

- PyPI pending publisher: project `openternary`, owner `ELRdn`, repository `OpenTernary`,
  workflow `release.yml`, environment `pypi`.
- Optional TestPyPI publisher with the same values and environment `testpypi`.
- GitHub repository environments `pypi` (and `testpypi`), ideally with yourself as a
  required reviewer so every publish waits for your approval.

## Each release

1. Make sure `main` is green in CI.
2. Set the version in **both** `pyproject.toml` and `src/openternary/__init__.py`
   (PEP 440, e.g. `0.1.0a1`, `0.1.0`), then refresh the lock with the pinned uv:
   `uv lock`. `tests/test_update.py` checks that the two agree.
3. Commit, merge to `main`, then tag that commit and push the tag:

   ```bash
   git tag v0.1.0a2
   git push origin v0.1.0a2
   ```

4. Approve the `pypi` environment in the Actions run. The workflow checks that the tag,
   `pyproject.toml` and `__init__.py` agree, runs the core tests, builds, runs
   `twine check --strict`, publishes, and creates a GitHub release (marked pre-release for
   `aN` / `bN` / `rcN`).

To rehearse without publishing to PyPI, run the workflow manually (Actions > Release >
Run workflow): it builds and uploads to TestPyPI only.

A published version can never be re-uploaded. If something is wrong, yank it on PyPI and
release the next version number.
