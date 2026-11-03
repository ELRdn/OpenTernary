---
description: "モデルを手元に用意し、inspectで量子化できるテンソルを一覧し、planで量子化を事前に確認する。どれも重みを読み込みません。"
---

# 調査と計画

モデルを量子化する前に、OpenTernaryがそのモデルに何をするかを確認しましょう。このガイドの2つのコマンドは
ファイルのヘッダだけを読むので、PyTorchもGPUもダウンロードも要りません。

## モデルを用意する {#getting-a-model}

OpenTernaryは**モデルを自分でダウンロードしません。** 渡せるのは次のどちらかです:

- `config.json` と `*.safetensors` を含む**フォルダ**（Hugging Faceのsnapshot、またはOpenTernaryの以前の出力）
- **Hugging FaceのモデルID**（ローカルのHugging Faceキャッシュの中だけを探します）

モデルをキャッシュに入れるには、`huggingface_hub` に含まれるHugging Face公式のCLIを使います:

```bash
uvx --from huggingface_hub hf auth login     # 初回のみ。Gemmaのようなゲート付きモデルに必要
uvx --from huggingface_hub hf download google/gemma-4-E2B-it-qat-q4_0-unquantized \
    --revision 6befbaca7398925921802abd1f277b495b78b738
```

Gemmaのモデルはゲート付きなので、先にHugging Faceのモデルのページで利用規約に同意してください。キャッシュに
モデルがなければ、OpenTernaryは実行すべき `hf download` のコマンドを示して止まります。

!!! info "revisionを固定する"

    Hugging Faceのモデルは、ダウンロードしたあとで更新されることがあります。上の `--revision` や、設定ファイルの
    `model.revision` のように、revision（commitのハッシュ）を必ず記録して渡してください。runの `model.json` に
    記録されるので、比較のときに同じ重みを使ったかを確認できます。

## `inspect`: モデルの中身 {#inspect-what-is-in-the-model}

```bash
openternary inspect google/gemma-4-E2B-it-qat-q4_0-unquantized
openternary inspect ./my-model --output runs/inspect-my-model
```

`inspect` はsafetensorsのヘッダと `config.json` を読み、次のことを行います:

- 構造に合う**アダプタ**（Gemma 4・汎用Transformers・Diffusers）を選ぶ
- すべてのテンソルを、形・dtype・役割、**量子化の対象**かどうかとともに一覧する
- パラメータを数え、BF16とFP32での大きさを見積もる
- モデルのテンソル一覧のfingerprintを計算する

新しい実行フォルダに `inspection.json` を書き、要約を表示します（例は[はじめての実行](../getting-started/first-run.md#2-inspect-and-plan)
を参照）。テンソルの完全な一覧は `inspection.json` にあり、`mixed_precision` に書く正確な名前はここで調べます。

`tensors` 配列の各要素には、`name`・`shape`・`dtype`・`param_count`・`role`・`quantizable` があり、元の精度で
残すテンソルには `exclude_reason` もあります:

```json
{"name": "lm_head.weight", "shape": [99, 128], "dtype": "F32", "param_count": 12672,
 "role": "lm_head", "quantizable": false,
 "exclude_reason": "high precision policy: lm_head (tie_word_embeddings)"}
```

```bash
# 量子化の対象のテンソル名
jq -r '.tensors[] | select(.quantizable) | .name' runs/<run>/inspection.json
```

`--load-weights` を付けると、重みの実際の値も読んで分布の統計を加えます。PyTorchが必要で、モデル全体を
テンソル1つずつ読みます。

## `plan`: quantizeが何をするか {#plan-what-quantize-would-do}

```bash
openternary plan google/gemma-4-E2B-it-qat-q4_0-unquantized --backend ternary --scheme absmean
openternary plan ./my-model --config my-quant.yaml --json
```

`plan` は設定をすべて解決し（CLIのフラグ ＞ YAML ＞ 既定値）、アダプタとbackendを選び、対象のテンソルを一覧
します。**実行フォルダは作らず**、何も書きません。オフラインでは分からないこと（ハードウェアで動くか、成果物を
読み込み直せるか、モデルの品質）は「不明」と報告します。

`plan` が受け取るのは、backendのオプション（`--backend`・`--scheme`・`--weight-dtype`・`--component`）と
`--config` です。グループ単位の設定は、設定ファイルに書くか、`quantize --dry-run` で確認します:

```bash
openternary quantize ./my-model --scale-granularity per_group --group-size 128 --dry-run
```

## 長い実行の前に設定ファイルを確認する {#checking-a-config-file-before-a-long-run}

`--config` を受け取るコマンドは、何かをする前に設定を完全に検証します。未知のキー、有限でない数値、未対応の
codebook、矛盾する校正の設定は、どれもエラー（終了コード `2`）になり、黙って無視されることはありません。
いちばん手早い確認方法はdry-runです:

```bash
openternary calibrate --config my-calibration.yaml --dry-run
```

解決した設定をYAMLで表示し、何も作りません。[TUI](../getting-started/tui.md#starting-a-job-form-plan-start)の
計画画面でも、ジョブを始める前に同じ出力が表示されます。

## 次へ {#next}

モデルを[量子化](quantize.md)します。
