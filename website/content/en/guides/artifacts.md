---
description: "Validate artifacts against their manifests, list the artifacts under a folder, and remove them safely without breaking references."
---

# Manage artifacts

An **artifact** is a model folder written by OpenTernary with an `openternary-manifest.json`: a quantized
snapshot (`runs/<run>/artifacts/snapshot`), a calibrated snapshot, or any `export` output.

## Validate

```bash
openternary artifacts validate runs/tiny-packed                  # --level files (default)
openternary artifacts validate runs/tiny-packed --level tensors
```

| Level | Checks | Reads |
|---|---|---|
| `files` | the manifest's schema, and the SHA-256 of every file listed in it | every byte once |
| `tensors` | the above, plus: opens every tensor (safetensors, ternary-packed, TorchAO) and checks shape, dtype, finite values and the list of targets | every tensor |

The artifact itself is never modified. The result reports `artifact_validation: file_integrity` or
`tensor_integrity`. Neither means the model loads in a framework (`model_reload`) or is any good
(`quality_acceptance`). Those are separate checks. See [four questions](../concepts/runs-and-evidence.md#four-questions-four-answers).

## List

```bash
openternary cache list runs
openternary cache list runs --json
```

This lists every artifact under the folder with its format, source, size and validation state:

```text
{'artifact': '…/runs/tiny-packed',   'format': 'ternary-packed', 'source': '…/runs/tiny-ternary/artifacts/snapshot', 'bytes': 225364, …}
{'artifact': '…/runs/tiny-restored', 'format': 'safetensors',    'source': '…/runs/tiny-packed',                    'bytes': 1308018, …}
{'artifact': '…/runs/tiny-ternary/artifacts/snapshot', 'format': 'safetensors', 'source': '…/tiny-llama',          'bytes': 1307928, …}
```

The `source` column shows the chain: original model → quantized snapshot → packed → restored.

## Remove

Removal is deliberately two-step and refuses anything that could break something else:

```bash
openternary cache remove runs/tiny-restored --root runs            # check only: status "planned"
openternary cache remove runs/tiny-restored --root runs --execute  # actually delete
```

Without `--execute`, nothing is deleted. You see what would be removed, its size, and where references were
checked:

```text
target                      /home/user/openternary-demo/runs/tiny-restored
root                        /home/user/openternary-demo/runs
bytes                       1308018
references_checked_within   /home/user/openternary-demo/runs
external_references         unknown
status                      planned
```

It refuses (exit code `2`):

- an artifact that another artifact or a search journal **inside `--root`** still refers to:

    ```text
    ╭─ ✖ Invalid configuration, argument or input ─────────────────────────────────╮
    │ artifact is referenced by: ['/home/user/openternary-demo/runs/tiny-restored'] │
    ╰──────────────────────────────────────────────────────────────────── exit 2 ─╯
    ```

- the root itself, or anything outside it;
- a folder that is not an artifact (no manifest). To delete a whole **run directory**, use the TUI's
  [`/delete`](../getting-started/tui.md#runs), which asks you to type the run's name, or delete the folder yourself.

References from outside `--root` cannot be seen and are reported as `external_references: unknown`. OpenTernary
never deletes shared Hugging Face cache entries or the original models.

In the TUI, `/cache:remove` with `--execute` asks you to type the target's name before it starts.

## Cache locations

`openternary cache-info` shows where uv and Hugging Face keep their caches:

```text
Cache info
UV_CACHE_DIR: (default: %LOCALAPPDATA%/uv/cache)
HF_HOME: (default: ~/.cache/huggingface)
```

Downloaded models live in the Hugging Face cache, managed with `hf cache` commands, not by OpenTernary.
