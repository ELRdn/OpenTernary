---
description: "小さなモデルを使い、CPUだけ・オフラインで、OpenTernaryの全工程を1分ほどで動かします。"
---

# はじめての実行

このチュートリアルでは、自分で作る**小さなランダム初期化のモデル**を使って、主なコマンドをすべて動かします。
ダウンロードもGPUも不要で、ノートPCでも1分ほどで終わります。

この章でやること:

1. 2層のLlama型モデル（320,896パラメータ）を作る
2. 調査し、量子化の計画を確認する
3. 三値に量子化する
4. 重みを2bitに詰め、元に戻し、結果を検査する
5. 品質を測り、三値モデルと元のモデルを比較する
6. TUIで結果を見る

!!! note "なぜランダムなモデルを使うのか"

    実際のモデルは、数分から数時間と数GBのディスクが必要です。小さなモデルなら各ステップが一瞬で終わるので、
    各コマンドが何をして何を書き出すかをすぐ確認できます。出力は意味のない文字列ですが、それが
    [ステップ6](#6-measure-quality-and-compare)で役に立ちます。OpenTernaryがこれを品質の結果として認めない様子を
    確認できるからです。

以下の出力はすべて、LinuxのCPU版PyTorch 2.15.0とOpenTernary 0.1.0a2（`OPENTERNARY_LANG=ja`）で実際に実行した
ものです。同じバージョンなら、調査とcontentのfingerprint（`sha256:a1fd…`、`sha256:2a25…`）は一致します。
manifestのfingerprintにはパスや時刻が含まれるので一致しません。時間も環境によって変わります。

## 始める前に {#before-you-start}

OpenTernaryとPyTorchが入っている必要があります（[インストール](installation.md)）。GPUがなければ
`openternary setup --backend cpu` で十分です。

空のフォルダを作って、その中で作業します:

```bash
mkdir openternary-demo
cd openternary-demo
```

## 1. 小さなモデルを作る {#1-build-the-tiny-model}

次のスクリプトを `make_tiny_model.py` として保存します:

```python title="make_tiny_model.py"
--8<-- "make_tiny_model.py"
```

このスクリプトには `torch` と `transformers` が必要です。どちらも `openternary setup` が入れた環境にあるので、
その環境のPythonで実行します:

=== "Windows（PowerShell）"

    ```powershell
    & "$(uv tool dir)\openternary\Scripts\python.exe" make_tiny_model.py
    ```

=== "macOS / Linux"

    ```bash
    "$(uv tool dir)/openternary/bin/python" make_tiny_model.py
    ```

=== "プロジェクトのvenv・ソースから"

    ```bash
    uv run python make_tiny_model.py
    ```

```console
wrote tiny-llama
```

`tiny-llama` フォルダは、Hugging Faceからダウンロードしたモデルと同じ形をしています。`config.json`、
`model.safetensors`、トークナイザのファイルに加えて、このモデルが人工のものであることを示す
`openternary-fixture.json` が1つ入っています。

## 2. 調査と計画 {#2-inspect-and-plan}

`inspect` は、既定ではsafetensorsファイルの**ヘッダ**だけを読みます。すべてのテンソルを一覧し、どれを量子化
できるか判定します。重みは読み込みません:

```bash
openternary inspect tiny-llama
```

```text
                Inspection — tiny-llama
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Section               ┃ Value                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Revision              │ unknown                     │
│ Adapter               │ transformers                │
│ Architecture          │ ['LlamaForCausalLM']        │
│ Text layers           │ ?                           │
│ Hidden / Intermediate │ ? / ?                       │
│ Vocab                 │ ?                           │
│ Total tensors         │ 21                          │
│ Quantizable           │ 14 (0.919027)               │
│ Total params          │ 320,896                     │
│ Quantizable params    │ 294,912                     │
│ BF16 GB               │ 0.0012                      │
│ FP32 GB               │ 0.0012                      │
│ Fingerprint           │ sha256:a1fd8a0cf17a9fd22... │
└───────────────────────┴─────────────────────────────┘
Inspection completed: /home/user/openternary-demo/runs/2026-11-03_tiny-llama-naive-g128
inspection.json saved to /home/user/openternary-demo/runs/2026-11-03_tiny-llama-naive-g128/inspection.json
```

21個のテンソルのうち14個が量子化の対象です。2つのデコーダ層それぞれにある7つの線形層（`q_proj`・`k_proj`・
`v_proj`・`o_proj`・`gate_proj`・`up_proj`・`down_proj`）で、パラメータの91.9%を占めます。埋め込み・出力層・
正規化の重みは元の精度のまま残ります。`?` の行は、汎用のTransformers用アダプタが読まない項目で、Gemma 4用の
アダプタでは値が入ります。

`plan` はさらに一歩進めて、`quantize` が*何をするか*を、やはり何も読み込まずに示します:

```bash
openternary plan tiny-llama
```

```text
╭─ 計画（ダウンロード・読込・書込なし） ──────────╮
│ 項目               値                           │
│ ─────────────────────────────────────────────── │
│ モデル             tiny-llama                   │
│ backend / 方式     ternary / absmean → ternary  │
│ scale / group      per_tensor / g128            │
│ アダプタ           transformers                 │
│ 対象               14 / 21                      │
│ デバイス / dtype   cpu / source dtype preserved │
│ pass               —                            │
│ オフラインでは未確定                            │
│   ? hardware/runtime execution                  │
│   ? artifact reload                             │
│   ? model quality                               │
╰─────────────────────────────────────────────────╯
```

大事なのは最後の3行です。計画からは、結果が動くか・読み込み直せるか・品質がよいかは分かりません。分かるのは
測定したときだけです。

## 3. 三値に量子化する {#3-quantize-to-ternary}

既定ではテンソルごとに1つのscaleを使います。グループごとのscaleのほうが重みに忠実なので、64個ずつのグループに
します。まず `--dry-run` で確認します。何も作られません:

```bash
openternary quantize tiny-llama --scale-granularity per_group --group-size 64 --device cpu --dry-run
```

```text
Planned: scale_granularity=per_group group_size=64
grouping_scheme=last-dim-rowwise-v1 method=naive
No run directory was created (--dry-run).
```

実際に実行します:

```bash
openternary quantize tiny-llama --scale-granularity per_group --group-size 64 --device cpu --output runs/tiny-ternary
```

```text
                 Quantize — tiny-llama
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Section               ┃ Value                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Scale granularity     │ per_group                   │
│ Group size / scheme   │ 64 / last-dim-rowwise-v1    │
│ Quantizable tensors   │ 14                          │
│ Total groups          │ 4608                        │
│ Shards                │ 1                           │
│ Tensor payload        │ 1283584                     │
│ Shard file bytes      │ 1285736                     │
│ Packed estimate bytes │ 196096                      │
│ Content FP            │ sha256:2a25a7bffeeb0d010... │
└───────────────────────┴─────────────────────────────┘
Quantize completed: /home/user/openternary-demo/runs/tiny-ternary
ternary artifact: /home/user/openternary-demo/runs/tiny-ternary/artifacts/snapshot
```

量子化の対象の重みは、グループごとに \(\{-s, 0, +s\}\) の3値のどれかになりました（\(s\) はそのグループの
scale）。ただし、まだ**ふつうの浮動小数点数として保存されている**ので、snapshotは元のモデルと同じ大きさ
（1,285,736バイト）です。この*fake-quant*の形式は、どのTransformersのコードでも読み込めるので、そのまま評価
できます。`Packed estimate bytes` は、同じ重みを1つ2bitで保存し、scaleと量子化しないテンソルを加えた大きさです。
計算方法は[三値量子化](../concepts/ternary-quantization.md)で説明しています。

`runs/tiny-ternary` は**実行フォルダ（run）**です:

```text
runs/tiny-ternary/
├── config.yaml          # 実際に使った、すべて解決済みの設定
├── environment.json     # バージョン、gitのcommit、ハードウェア、シード
├── model.json           # モデルIDとrevision
├── metrics.json         # 状態と主な数値
├── quantization.json    # テンソルごとの詳細とfingerprint
├── logs/
└── artifacts/snapshot/  # 量子化したモデル（ふつうのHFモデルとして読める）
    ├── model-00001-of-00001.safetensors
    ├── openternary-manifest.json   # 形式、来歴、全ファイルのSHA-256
    └── config.json、トークナイザのファイルなど
```

各ファイルの説明は[実行フォルダのリファレンス](../reference/run-directory.md)にあります。

## 4. packed化・復元・検査 {#4-pack-restore-and-validate}

三値の重みを1つ2bitに詰めます:

```bash
openternary export runs/tiny-ternary --format ternary-packed --output runs/tiny-packed
```

```text
項目                  値
────────────────────────────────────────
artifact              /home/user/openternary-demo/runs/tiny-packed
manifest              schema_version=1, format=ternary-packed,
                      execution=completed, artifact_validation=file_integrity,
                      model_reload=not_run, quality_acceptance=not_run,
                      manifest_fingerprint=5e15f72a740a53eb777c57da17528fde3c52…
artifact_validation   file_integrity
```

テンソルのデータは1,285,736バイトから**198,600バイト**へ、約6.5分の1になります。残りの大部分は、元の精度の
ままの埋め込みと出力層です。packedの成果物はまだ直接実行できないので、safetensorsに戻します:

```bash
openternary export runs/tiny-packed --format safetensors --output runs/tiny-restored
```

そして念入りに検査します。`--level tensors` はすべてのテンソルを開き、形・dtype・有限値かどうかをmanifestと
照合します:

```bash
openternary artifacts validate runs/tiny-restored --level tensors
```

```text
項目                   値
────────────────────────────────────────
artifact               /home/user/openternary-demo/runs/tiny-restored
artifact_validation    tensor_integrity
manifest_fingerprint   62adeab647febbb4eef8c923bfdee60b41504c38ded7db75aed1aaa9…
tensor_count           21
model_reload           ○ not run
quality_acceptance     ○ not run
```

復元したテンソルは、fake-quantのsnapshotとバイト単位で一致します。三値の重みのpacked化は可逆です。結果が
**主張していないこと**にも注目してください。`model_reload` と `quality_acceptance` は `○ not run` です。
ファイルの整合性から分かるのはファイルが壊れていないことだけで、モデルが動くことではありません。

## 5. ベンチマークで振る舞いを比べる {#5-benchmark-and-compare-behaviour}

smokeベンチマークは、固定の5つのプロンプトをgreedyで生成し、生成されたトークンをすべて記録します:

```bash
openternary benchmark tiny-llama --device cpu --dtype fp32 --output runs/bench-original
openternary benchmark runs/tiny-ternary/artifacts/snapshot --device cpu --dtype fp32 --output runs/bench-ternary
openternary compare runs/bench-original runs/bench-ternary --output runs/compare-bench
```

```text
     Compare — behavioral delta (observational)
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric              ┃ Value                       ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Protocol match      │ True                        │
│ Result observation  │ different                   │
│ Result same?        │ False                       │
│ Content match       │ None                        │
│ Baseline result FP  │ sha256:6c4af247d0a4f7029... │
│ Quantized result FP │ sha256:c5b2dce616da98b50... │
└─────────────────────┴─────────────────────────────┘
      Per-prompt token diff
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Prompt                 ┃ Same? ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ smoke.code.python      │ False │
│ smoke.en.short         │ False │
│ smoke.ja.short         │ False │
│ smoke.ja.summary       │ False │
│ smoke.reasoning.simple │ False │
└────────────────────────┴───────┘
Result fingerprint equality is observational, NOT pass/fail.
```

`Protocol match: True` は、2つのrunがまったく同じプロンプトと生成設定を使ったことを意味します。つまり公平な
比較です。出力はすべてのプロンプトで異なりました。これは**観測**であって判定ではありません。違う文章が悪い
文章とは限らないからです。品質を判断するには、次のステップが必要です。

## 6. 品質を測って比較する {#6-measure-quality-and-compare}

`quality` は、**固定したデータセットのファイル**を使って、英語と日本語のパープレキシティを測り、指示ケースを
実行します。わざと小さくした次の例を `quality-mini.json` として保存してください。実際のデータセットにはもっと
多くのレコードが必要です。形式は[評価と比較](../guides/evaluate-and-compare.md#quality-datasets)で説明しています:

??? example "quality-mini.json"

    ```json
    --8<-- "quality-mini.json"
    ```

```bash
openternary quality tiny-llama --data quality-mini.json --device cpu --dtype fp32 --output runs/quality-original
openternary quality runs/tiny-ternary/artifacts/snapshot --data quality-mini.json --device cpu --dtype fp32 --output runs/quality-ternary
openternary compare runs/quality-original runs/quality-ternary --require-acceptance --output runs/compare-gate
```

主な数値（各runの `quality.json` より）:

| | 英語のパープレキシティ | 日本語のパープレキシティ | 指示スコア |
|---|---:|---:|---:|
| 元のモデル | 111.90 | 81.60 | 0 |
| 三値 | 105.36 | 78.13 | 0 |

三値モデルのほうがパープレキシティが*低く*なっています。実際の重みでこうなれば驚くべきことですが、ここでは
意味がありません。元のモデルがランダムなノイズで、それを量子化したものもノイズだからです。そのため `compare`
はこれを受け入れません:

```text
╭─ ✖ 受入条件を満たしていない ─────────────────────────────────────────────────────────╮
│ required quality acceptance failed or evidence is incomplete                         │
╰─────────────────────────────────────────────────────────────────────── 終了コード 6 ─╯
```

理由は `runs/compare-gate/acceptance.json` に書かれています:

```json
{
  "accepted": false,
  "status": "insufficient_evidence",
  "reason": "schema 3 with matched source/interface/runtime and non-synthetic model evidence required"
}
```

品質Gateが受け入れるのは、人工のものではないモデルを、同じ元モデル・トークナイザ・デバイス・dtype・
ソフトウェアで測った場合だけです。`--require-acceptance` を付けると、Gateを通らなければ終了コード**6**で
終わるので、スクリプトやCIはそこで止まります。なぜここまで厳しくしているかは
[実行結果と証拠](../concepts/runs-and-evidence.md)で説明しています。

## 7. TUIで見る {#7-see-it-in-the-tui}

同じフォルダで、引数なしの `openternary` を実行します。**実行結果**タブに、ここまでに作ったものが、各runの
ファイルと一緒に一覧されます:

<figure class="ot-screenshot" markdown>
![実行結果タブ: チュートリアルで作ったrunの一覧と、選択中のrunの詳細とファイル](../assets/images/tui-runs-ja.svg)
</figure>

`/quantize` と入力して ++enter++ を押すと、同じ量子化をフォームで指定し、計画を確認して、バックグラウンドの
ジョブとして実行できます。やり方は[全画面インターフェース](tui.md)で説明しています。

## 片付け {#clean-up}

作ったものはすべて最初に作ったフォルダの中（`tiny-llama/` と `runs/`）にあります。終わったらフォルダごと
削除してください。

## 次に読むもの {#what-next}

- 実際のモデルで同じことをする: [量子化](../guides/quantize.md)、[評価と比較](../guides/evaluate-and-compare.md)
- 数字の意味を理解する: [三値量子化](../concepts/ternary-quantization.md)
- 結果の改善を試す: [校正](../guides/calibrate.md)
