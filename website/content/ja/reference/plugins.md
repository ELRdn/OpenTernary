---
description: "バージョン付きのPythonのentry pointで、backend・pass・exporter・evaluatorを追加する方法と、動作するpassのpluginの例。"
---

# Plugin

OpenTernaryには4種類のpluginを追加できます。どれも、バージョン付きの**entry pointのグループ**にオブジェクトを登録
した、導入済みのPythonパッケージです。

| 種類 | entry pointのグループ | 選び方 | 予約されている名前 |
|---|---|---|---|
| backend | `openternary.backends.v1` | `--backend NAME` / `quantization.backend` | `ternary`・`torchao` |
| pass | `openternary.passes.v1` | `quantization.passes: [{name: NAME}]` | `clip`・`noop` |
| exporter | `openternary.exporters.v1` | `export --format NAME` | `safetensors`・`ternary-packed`・`torchao`・`gguf` |
| evaluator | `openternary.evaluators.v1` | `evaluate --evaluator NAME` | `diffusion` |

## 規則 {#rules}

- **一覧ではpluginのコードを実行しません。** `openternary plugins` と `backends list` は、entry pointの情報だけを
  読みます。
- **pluginは名前で選んだときだけimportし**、そのとき `api_version = 1` を持っていなければなりません。持っていなければ
  `plugin API version mismatch`（終了コード `3`）です。
- **重複と予約名は衝突として扱います。** 2つのパッケージが同じ名前を登録した場合や、組み込みの名前を使った場合は
  `conflict` と表示し、選べません。
- **pluginはサンドボックスで動きません。** 同じプロセスで、利用者の権限で動きます。信頼できるpluginだけを入れて
  ください。
- **結果は正直なまま。** backendの出力は組み込みと同じmanifestで包み、exporterの出力はOpenTernaryが原子的に書いて
  公開します。

```console
$ openternary plugins
Kind     Name     Version   status
────────────────────────────────────────
passes   shrink   0.1.0     · discovered
```

## インターフェース（APIバージョン1） {#interfaces-api-version-1}

| 種類 | 登録するオブジェクトが持つべきもの |
|---|---|
| backend | `api_version = 1` と `spec: BackendSpec`（`name` がentry pointと一致）、`validate(config)`、`ConvertReport` 互換の結果を返す `convert(source, destination, config)` を持つ**クラス**。引数なしでインスタンス化される |
| pass | `api_version = 1`、`{"stage": "pre_quantization", "runtime_state": None}` を返す `plan(options, backend) -> dict`、形・dtype・デバイスを保つ `apply(tensor, options) -> tensor` を持つ**オブジェクト**。passは今のところternary backendにだけ適用される |
| exporter | `api_version = 1`、`validate(source_format)`（拒否するなら例外）、ステージングのフォルダに書く `export(source, staging_directory)` を持つ**オブジェクト**。manifestと原子的な公開はOpenTernaryが行う |
| evaluator | `api_version = 1`、`validate(raw_profile) -> profile`、`evaluate(config, model, profile, run) -> dict`（`evaluation.json` として保存されるレポート）を持つ**オブジェクト**。品質の約束事のバージョンは各evaluatorが管理する |

`BackendSpec` は `openternary.backends` にあります:

```python
from openternary.backends import BackendSpec

spec = BackendSpec(
    name="mybackend",
    distribution="my-openternary-backend",  # バージョン表示に使う、導入済みのdistribution名
    schemes=("myscheme",),
    weight_dtypes=("ternary",),
    formats=("safetensors",),
    implementation="my_backend:MyBackend",
)
```

## 例: passのplugin {#example-a-pass-plugin}

量子化の前に、すべての重みに定数を掛けるpluginです。あくまで例で、AbsMeanはどのみちscaleを計算し直すので、符号は
変わりません。そのまま動く完全なパッケージです。

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

OpenTernaryと同じ環境に入れて使います:

=== "uv tool install"

    ```bash
    uv tool install openternary --with-editable ./openternary-shrink --python 3.12 --reinstall
    openternary setup        # 入れ直しで環境が作り直されたので、PyTorchを戻す
    ```

=== "プロジェクトのvenv・ソースから"

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
openternary plan tiny-llama --config shrink.yaml       # 「pass  shrink」と表示される
openternary quantize tiny-llama --config shrink.yaml --device cpu --output runs/shrink
```

## 互換性 {#compatibility}

バージョン1は推測をしません。不明な形式を変換したり、古いmanifestを黙って新しくしたりはしません。pluginの
インターフェースに互換性のない変更をするときは、新しいentry pointのグループ（`….v2`）と新しい `api_version` を
使うので、既存のpluginはそのまま動くか、はっきり失敗します。
