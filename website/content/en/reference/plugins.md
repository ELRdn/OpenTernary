---
description: "Extend OpenTernary with backends, passes, exporters and evaluators through versioned Python entry points, with a working pass plugin example."
---

# Plugins

OpenTernary can be extended with four kinds of plugins. Each is an installed Python package that registers an
object under a versioned **entry point group**.

| Kind | Entry point group | Selected by | Reserved names |
|---|---|---|---|
| backend | `openternary.backends.v1` | `--backend NAME` / `quantization.backend` | `ternary`, `torchao` |
| pass | `openternary.passes.v1` | `quantization.passes: [{name: NAME}]` | `clip`, `noop` |
| exporter | `openternary.exporters.v1` | `export --format NAME` | `safetensors`, `ternary-packed`, `torchao`, `gguf` |
| evaluator | `openternary.evaluators.v1` | `evaluate --evaluator NAME` | `diffusion` |

## Rules

- **Listing never runs plugin code.** `openternary plugins` and `backends list` read entry-point metadata only.
- **A plugin is imported only when selected by name**, and then it must have `api_version = 1`. Otherwise:
  `plugin API version mismatch` (exit code `3`).
- **Duplicates and reserved names are conflicts.** Two packages registering the same name, or a plugin using a
  built-in name, show up as `conflict` and cannot be selected.
- **Plugins are not sandboxed.** They run in the same process with your user's permissions. Install only plugins
  you trust.
- **Results stay honest.** Backend output is wrapped in the same manifest as built-in output, and exporter output
  is written and published atomically by OpenTernary.

```console
$ openternary plugins
Kind     Name     Version   status
────────────────────────────────────────
passes   shrink   0.1.0     · discovered
```

## Interfaces (API version 1)

| Kind | The registered object must provide |
|---|---|
| backend | a **class** with `api_version = 1` and `spec: BackendSpec` (whose `name` matches the entry point), `validate(config)`, and `convert(source, destination, config)` returning a `ConvertReport`-compatible result. It is instantiated with no arguments. |
| pass | an **object** with `api_version = 1`, `plan(options, backend) -> dict` returning `{"stage": "pre_quantization", "runtime_state": None}`, and `apply(tensor, options) -> tensor`, which must keep shape, dtype and device. Passes currently apply to the ternary backend only. |
| exporter | an **object** with `api_version = 1`, `validate(source_format)` (raise to refuse), and `export(source, staging_directory)`, which writes into the staging folder. Manifest and atomic publication are handled for you. |
| evaluator | an **object** with `api_version = 1`, `validate(raw_profile) -> profile`, and `evaluate(config, model, profile, run) -> dict` (the report, saved as `evaluation.json`). Evaluators version their own quality contract. |

`BackendSpec` lives in `openternary.backends`:

```python
from openternary.backends import BackendSpec

spec = BackendSpec(
    name="mybackend",
    distribution="my-openternary-backend",  # the installed distribution, for version reporting
    schemes=("myscheme",),
    weight_dtypes=("ternary",),
    formats=("safetensors",),
    implementation="my_backend:MyBackend",
)
```

## Example: a pass plugin

This plugin multiplies every weight by a constant before quantization. It is only an illustration: AbsMean rescales
anyway, so it does not change the codes. It is a complete, working package.

```text
openternary-shrink/
├── pyproject.toml
└── src/openternary_shrink/__init__.py
```

```toml title="pyproject.toml"
[project]
name = "openternary-shrink"
version = "0.1.0"
requires-python = ">=3.11"

[project.entry-points."openternary.passes.v1"]
shrink = "openternary_shrink:SHRINK"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```python title="src/openternary_shrink/__init__.py"
"""An example OpenTernary pass: multiply every weight by a constant before quantizing."""

import math


class Shrink:
    api_version = 1  # the plugin API this object implements

    def plan(self, options, backend):
        factor = options.get("factor")
        if set(options) != {"factor"} or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("shrink requires only a finite factor > 0")
        # Tensor passes must be stateless transformations applied before quantization.
        return {"stage": "pre_quantization", "runtime_state": None}

    def apply(self, tensor, options):
        return tensor * options["factor"]  # same shape, dtype and device


SHRINK = Shrink()
```

Install it into the same environment as OpenTernary and use it:

=== "uv tool install"

    ```bash
    uv tool install openternary --with-editable ./openternary-shrink --python 3.12 --reinstall
    openternary setup        # reinstalling rebuilt the environment, so put PyTorch back
    ```

=== "Project venv / source checkout"

    ```bash
    uv pip install -e ./openternary-shrink
    ```

```yaml title="shrink.yaml"
quantization:
  passes:
    - name: shrink
      options: {factor: 0.5}
```

```bash
openternary plan tiny-llama --config shrink.yaml       # "Passes  shrink"
openternary quantize tiny-llama --config shrink.yaml --device cpu --output runs/shrink
```

## Compatibility

Version 1 never guesses: unknown formats are not converted, and old manifests are not silently upgraded. An
incompatible change to a plugin interface will use a new entry point group (`….v2`) and a new `api_version`, so
existing plugins keep working or fail clearly.
