---
description: "openternary.apiの安定したバージョン1のPython API: 設定、計画、変換、書き出し、検査、評価、探索。"
---

# Python API

`openternary.api` を使うと、CLIと同じ機能を、子プロセスなしでPythonのコードから使えます。安定した**バージョン1**の
インターフェースです。importしてもPyTorchは読み込まず、重い依存はそれを必要とする関数の中でだけ読み込みます。

## 例 {#example}

[はじめての実行](../getting-started/first-run.md)の小さなモデルで動きます:

```python
from pathlib import Path

from openternary.api import (
    build_plan,
    convert_artifact,
    doctor,
    export_artifact,
    load_config,
    validate_artifact,
)

config = load_config(cli_overrides={
    "model.id": "tiny-llama",
    "quantization.scale_granularity": "per_group",
    "quantization.group_size": 64,
    "device": "cpu",
})

plan = build_plan(config)
print("targets:", sum(t["quantizable"] for t in plan["targets"]))

report = convert_artifact(Path("tiny-llama"), Path("api-out/ternary"), config)
print("content fingerprint:", report.content_fingerprint)

export_artifact(Path("api-out/ternary"), Path("api-out/packed"), "ternary-packed")
print("validation:", validate_artifact(Path("api-out/packed"))["artifact_validation"])

print("torch:", doctor()["dependencies"].get("torch"))
```

```text
targets: 14
content fingerprint: sha256:2a25a7bffeeb0d0103ab21c3d42336bd72df75359e6bccd564c33fb16475f54f
validation: file_integrity
torch: 2.15.0
```

このfingerprintは、同じモデルと設定でCLIが表示したものと同じです。CLIと違い、これらの関数は実行フォルダを作りません。
結果をどこに置くかは呼び出す側が決めます。

## 関数 {#functions}

### 設定と計画 {#configuration-and-planning}

`load_config(config_path=None, cli_overrides=None) -> AppConfig`
:   YAMLのファイル（または既定値）を読み、ドット区切りのキーで指定した上書きを適用します。例:
    `{"model.id": "./m", "quantization.group_size": 64}`。検証はコマンドラインと同じく厳格です。

`AppConfig`
:   設定全体のPydanticのモデルです（[設定](configuration.md)を参照）。`config.model_dump()` でふつうのdictになります。

`build_plan(config, *, operation="quantize") -> dict`
:   `plan` が表示する内容: 解決した設定、アダプタ、backend、`quantizable` 付きのテンソルの一覧 `targets`。ヘッダだけを
    読みます。

`plan(config_path=None, overrides=None) -> dict`
:   `load_config` のあとに `build_plan` を呼ぶ近道です。

`doctor() -> dict`
:   `openternary doctor --json` の `result` と同じ内容: Python、プラットフォーム、依存、backend。

### 変換と成果物 {#conversion-and-artifacts}

`convert_artifact(source, destination, config) -> ConvertReport`
:   `source` のsnapshotを量子化し、`destination` にmanifest付きの新しい成果物を作ります。結果には
    `content_fingerprint`・`file_hashes`・`shard_count`・`tensor_payload_bytes`・`shard_file_bytes`・
    `snapshot_total_bytes`・`quantization_json` があります。

`export_artifact(source, destination, format="safetensors") -> dict`
:   `safetensors`・`ternary-packed`・`torchao`・pluginの形式で書き出します。GGUFはconverterが必要なので、専用の
    関数 `openternary.export.gguf.export_gguf` を使います。

`validate_artifact(root) -> dict`
:   成果物のmanifestとファイルのハッシュを確認します。テンソル単位の確認には
    `openternary.services.tensor_validation.validate_tensors` を使います。

### 評価 {#evaluation}

`benchmark(config, snapshot_path, suite) -> dict`
:   snapshotでsmokeベンチマーク（`suite="smoke"`）を実行し、`benchmark.json` の内容を返します。

`quality(config, dataset_path, *, snapshot_path=None, model=None, processor=None, split, max_length, stride) -> dict`
:   品質を評価し、`quality.json` の内容を返します。`snapshot_path` を渡すか、読み込み済みの `model` と `processor` を
    一緒に渡します。

`compare_many(baseline, candidates) -> dict`
:   1つの基準の実行フォルダと、候補の実行フォルダのリストを比べます。

diffusionの評価は `openternary.services.diffusion.evaluate_diffusion` にあります。

### 探索 {#search}

`run_search(config, profile, candidates, output, budget, *, target_vram, objective="speed", resume=False, executor=None) -> dict`
:   `optimize` の中身です。`profile` は `QualityProfile`、`candidates` は量子化の設定のリスト、`budget` は
    `SearchBudget`、`target_vram` はバイト数です。

`QualityProfile(dataset, baseline, name="balanced-llm-v1", split="validation", max_length=512, stride=256)`
:   検証のprofile。`split` は `validation` だけです。

`SearchBudget(max_candidates=4, wall_seconds=3600, gpu_seconds=3600, artifact_bytes=20 GiB)`
:   探索の上限。

## 安定性 {#stability}

`openternary.api` から公開している名前は、バージョン1の約束に従い、既存の動作を互換性のない形で変えません。それ以外
（`openternary.services`・`openternary.quant` など）は内部のもので、リリースの間で変わることがあります。エラーは
例外として送出され、CLIはそれを[終了コード](output.md#exit-codes)に対応付けます。
