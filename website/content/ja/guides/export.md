---
description: "成果物を、safetensors・2bitのternary-packed・TorchAOの状態・指定したllama.cppのconverterを使ったGGUFとして書き出す。"
---

# 書き出し

`export` は、成果物を別の形式で書き出します。**元は決して変更せず**、既存の書き出し先を上書きすることもありません。

```bash
openternary export <runまたは成果物> --format <形式> --output <新しいフォルダ>
```

元には、実行フォルダ（その `artifacts/snapshot` か校正済みのsnapshotを使います）か、以前の書き出しのような成果物の
フォルダを指定できます。

## 形式 {#formats}

| `--format` | 書き出すもの | 制限 |
|---|---|---|
| `safetensors`（既定） | snapshotのそのままのコピー、または `ternary-packed` からの**復元** | fake-quantの重みは浮動小数点で保存される |
| `ternary-packed` | 2bitの符号、scale、形、詰め物、量子化しないテンソル | 符号とscaleから*正確に*復元できる入力のみ。直接実行できるものはない |
| `torchao` | TorchAO独自の状態を、読み込み直す入口と一緒にコピー | 読み込み直すには同じTorchAO（0.18.0）が必要。safetensorsへは変換できない |
| `gguf` | 指定したllama.cppのconverterで作る、浮動小数点のGGUF | Llama・Mistral・Qwen2のメタデータのみ。出力は `f16`/`bf16`/`f32`。下記を参照 |

## packed化と復元 {#packing-and-restoring}

```bash
openternary export runs/e2b-g128 --format ternary-packed --output runs/e2b-g128-packed
openternary export runs/e2b-g128-packed --format safetensors --output runs/e2b-g128-restored
openternary artifacts validate runs/e2b-g128-restored --level tensors
```

packedのフォルダには、`packed.json`（配置）、`tensors/`（テンソルごとに1ファイル。量子化したテンソルは2bitの符号と
scale、それ以外は元のデータ）、モデルの設定とトークナイザのファイル、manifestが入ります。符号の詰め方は
[2bitの形式](../concepts/ternary-quantization.md#the-2-bit-packing-format)で説明しています。

往復は**可逆**です。復元したテンソルは、packed化する前と値・dtype・形・バイト列が一致します。CPUのfixtureと
[はじめての実行](../getting-started/first-run.md#4-pack-restore-and-validate)で確認しています。実際のファイルの
大きさにはscale・詰め物・量子化しないテンソルが含まれ、理論上の大きさとは別に報告します。

符号とscaleから正確に復元できないsnapshotは、近似せずにpacked化を拒否します。たとえば、packed形式に保存しない
情報が必要な閾値で校正した、一部のsnapshotがこれにあたります。

## llama.cppを使ったGGUF {#gguf-through-llamacpp}

OpenTernaryはGGUFのconverterを同梱もダウンロードもしません。手元にあるllama.cppの `convert_hf_to_gguf.py` を
指定します:

```bash
openternary export runs/my-llama-restored --format gguf \
    --converter ~/src/llama.cpp/convert_hf_to_gguf.py \
    --converter-python ~/src/llama.cpp/.venv/bin/python \
    --output-dtype f16 \
    --output runs/my-llama-gguf
```

| オプション | 既定 | |
|---|---|---|
| `--converter PATH` | 必須 | converterのスクリプト。SHA-256をmanifestに記録する |
| `--converter-python PATH` | OpenTernaryを動かしているPython | converter自身の依存を入れた、隔離した環境 |
| `--output-dtype` | `f16` | `f16`・`bf16`・`f32`。三値のGGUFの型は作らない |
| `--timeout-seconds` | 3600 | この時間でconverterを止め、子プロセスごと片付ける |

- 元は、`llama`・`mistral`・`qwen2` の構造のメタデータを持つsafetensorsでなければなりません。**現在のGemmaの研究用
  snapshotは、この橋渡しの対象外です。**
- 通常の書き出しで確認するのは、GGUFのヘッダとファイルのハッシュまでです。実際に動かす確認は別で、リポジトリの
  [`scripts/validate_cli_gguf.py`](https://github.com/ELRdn/OpenTernary/blob/main/scripts/validate_cli_gguf.py)
  が、固定したllama.cppの本物のreaderとCPUでの生成で確かめます。
- これまでに検証したのは、人工のLlama（FP32）を固定したllama.cppで読み戻し、CPUで生成するところまでです。ほかの
  構造・精度、GPUでの実行は未検証です。

## 原子的な書き込みとmanifest {#atomic-writes-and-manifests}

書き出しはすべて、次の順に行います:

1. 書き出し先の隣の、一意な一時フォルダに書く
2. `openternary-manifest.json` を書く: schema、形式、全ファイルのSHA-256、元とそのインターフェース、設定、
   backend、アダプタ、component、pass、コードのハッシュ
3. 結果を検査する
4. そのあとで初めて `--output` の名前に変える

途中で何かが失敗すれば、書き出し先は現れません。manifestは変換の履歴を引き継ぐので、復元したsnapshotにも、
packedの成果物から来たこと、その成果物が量子化のrunから来たことが記録されています。

```bash
openternary artifacts validate runs/e2b-g128-packed                 # manifestと全ファイルのハッシュ
openternary artifacts validate runs/e2b-g128-packed --level tensors # 全テンソルも開いて確認
```

## 古いバージョンからの移行 {#migrating-from-older-versions}

| 以前 | 現在 |
|---|---|
| `export --format fake-quant` | `--format safetensors` |
| converterを指定しない `export --format gguf` | `--converter` の明示と、対応する構造が必要 |
