---
description: "smokeベンチマークと、英語・日本語の固定した品質評価を実行し、品質データセットを作り、品質Gateでrunを比較する。"
---

# 評価と比較

OpenTernaryは、モデルを2つの方法で測り、1つの方法で比べます:

| コマンド | 答える問い | 判定 |
|---|---|---|
| `benchmark` | 動くか。速さは。固定の5プロンプトに対して正確に何を生成するか | なし（振る舞いだけ） |
| `quality` | 英語と日本語の文章、完全一致の指示にどれだけ強いか | 単独ではなし |
| `compare` | 同じ条件で測った2つのrunはどう違うか | 品質のrunなら品質Gate |

まず**元のモデル**を、まったく同じ設定で測ってください。判定はすべて、その基準との相対で決まります。

## `benchmark`: smokeスイート {#benchmark-the-smoke-suite}

```bash
openternary benchmark google/gemma-4-E2B-it-qat-q4_0-unquantized --device cuda --dtype bf16 --output runs/e2b-bench
openternary benchmark runs/e2b-g128/artifacts/snapshot          --device cuda --dtype bf16 --output runs/e2b-g128-bench
```

suite `smoke v1` は、ウォームアップのプロンプトを1つ実行したあと、固定の5つのプロンプトを、モデルのchat template
を使ってgreedyで生成します（`do_sample: false`、`num_beams: 1`、新しいトークンは最大64）:

| ID | プロンプト |
|---|---|
| `smoke.en.short` | Write a one-sentence greeting in English. |
| `smoke.ja.short` | 日本語で一文の挨拶を書いてください。 |
| `smoke.ja.summary` | OpenTernaryについての文を、日本語の一文で要約させる |
| `smoke.reasoning.simple` | 一文で答える三段論法の問題（Bloops・Razzies・Loppies） |
| `smoke.code.python` | Write a Python function add(a, b) that returns a + b. Reply with code only. |

`benchmark.json` には次のものが記録されます:

- **protocolのfingerprint**: suite・プロンプト・chat templateの設定・生成設定のハッシュ。2つのrunを比べられるのは、
  これが一致するときだけです
- **結果のfingerprint**: 生成したトークンIDすべてのハッシュ。出力が同じならfingerprintも同じです
- 各プロンプトで生成したトークンと文章、レイテンシ、出力のトークン毎秒、モデルの読み込み時間
- 段階ごとの資源の測定値（メモリ、GPUならVRAM）。測れたときだけで、測れなければ `null`

オプション: `--dtype bf16|fp16|fp32`（既定 `bf16`）、`--device auto|cpu|cuda`、`--seed`、
`--thinking/--no-thinking`（Gemma 4のthinkingモード、既定はオフ）。生成設定は設定ファイルの `benchmark` セクション
で変えられますが、変えるとprotocolのfingerprintも変わります。

smokeスイートには、あえて品質のスコアがありません。モデルが動くことの確認と、出力が変わったか*どうか*を見るための
ものです。

## `quality`: パープレキシティと指示 {#quality-perplexity-and-instructions}

```bash
openternary quality google/gemma-4-E2B-it-qat-q4_0-unquantized --data quality.json --device cuda --dtype bf16 --output runs/e2b-q
openternary quality runs/e2b-g128/artifacts/snapshot          --data quality.json --device cuda --dtype bf16 --output runs/e2b-g128-q
```

`quality` は、固定したデータセットの1つのsplit（既定は `--split validation`、または `test`）を評価します:

**パープレキシティ**は、英語と日本語の文章で別々に、スライディングウィンドウ（`--max-length 512`・
`--stride 256`）で計算します。文書はそれぞれ独立に扱い、パディングや切り詰めはしません。低いほどよく、言語ごとと
全体で報告します。

**指示ケース**は、期待する答えが1つに決まっているプロンプトです。応答が `expected` と**完全に**一致したときだけ
正解になります。正規化もLLMによる採点もしないので、空白や句読点も違いになります。`instruction_score` は完全一致の
割合（%）です。

**崩壊の検出**では、空の応答、記号だけの応答、短い単位の繰り返しだけの応答（`ああああああああ` など）に印を
付けます。`collapse_count` は印の付いた応答の数で、1つでもあれば品質Gateに不合格になります。

`quality.json`（レポートschema 3）には、元モデルとトークナイザ・processorのファイルの同一性（内容のハッシュ）、
実際に使ったデバイスとdtype、ライブラリとOSのバージョン、資源の測定値も記録されます。`compare` が結果を受け入れる
には、これらがすべて必要です。

`--dry-run` は、モデルを読み込まずにデータセット（schema・来歴・split・重なり）を検査します。

### 品質データセット {#quality-datasets}

品質データセットは1つのJSONファイルで、来歴、3つの文章のsplit、2つの指示のsplitからなります:

```json
{
  "schema_version": 1,
  "dataset": {
    "id": "your-org/quality-v1",
    "revision": "v1",
    "license": "CC-BY-4.0",
    "source": [{"id": "文章の出どころ", "revision": "…"}]
  },
  "splits": {
    "calibration": [{"id": "c1", "text": "…", "language": "en"}],
    "validation":  [{"id": "v1", "text": "…", "language": "en"}, {"id": "v2", "text": "…", "language": "ja"}],
    "test":        [{"id": "t1", "text": "…", "language": "en"}, {"id": "t2", "text": "…", "language": "ja"}]
  },
  "instruction_cases": {
    "validation": [{"id": "vi1", "prompt": "…", "expected": "…"}],
    "test":       [{"id": "ti1", "prompt": "…", "expected": "…"}]
  }
}
```

規則（すべて検査されます）:

- `dataset.id`・`revision`・`license`・`source` は必須です。来歴も証拠の一部です。
- 3つの文章のsplitと2つの指示のsplitはすべて必要で、空ではいけません。各レコードには `id`・`text`・
  `language`、各ケースには `id`・`prompt`・`expected` が必要です。
- IDはsplit内で一意、指示のIDは2つのsplitを通して一意です。
- **重なりの禁止:** 同じ文章が2つのsplitに現れてはいけません。完全一致（Unicode NFKCと空白の正規化後）も、
  ほぼ重複（文字5-gramのMinHashで類似度0.9以上）も対象です。

研究用のデータセット `OpenTernary/quality-squad-jsquad-v1` は、WikiText（校正）、SQuAD（英語）、JGLUEの
JSQuAD（日本語）から決定的に作ります。言語・splitごとに文脈8件、指示ケース4件を、レコードIDのSHA-256の順で
選びます。作成する関数は `openternary.benchmark.quality_dataset.build_quality_dataset` です。作ったファイルには
元データのライセンス（CC-BY-SA・CC-BY）が適用されます。

!!! warning "testのsplitは1回だけ"

    候補の選択は `validation` で行います。`--split test` は、最終的に選んだ候補にだけ使ってください。多くの候補を
    testで評価していちばんよいものを残すと、testがvalidationと同じ役割になってしまいます。

## `compare` {#compare}

```bash
openternary compare runs/e2b-q runs/e2b-g128-q                         # 観測
openternary compare runs/e2b-q runs/e2b-g128-q --require-acceptance    # Gate: 受け入れられなければ終了コード6
openternary compare runs/e2b-q runs/e2b-g128-q --candidate runs/e2b-g64-q --candidate runs/e2b-calib-q
```

- 1つ目の引数が**基準**、2つ目が**候補**です。`--candidate`（複数指定可）で候補を追加でき、どれも同じ基準と
  比べます。
- 2つのrunは同じ種類（benchmarkどうし、またはqualityどうし）で、手順・データセット・モデルのrevision・dtype・
  デバイスが一致している必要があります。違えば、`compare` は違う項目を示して終了コード `2` で止まります。
- **benchmarkのrun**は観測の比較です。出力が全体で、またプロンプトごとに同じかを示します。
- **qualityのrun**は、証拠が揃っていれば[品質Gate](../concepts/runs-and-evidence.md#comparing-observation-vs-acceptance)で
  判定します。パープレキシティの上昇は各言語で2%まで、指示スコアの低下は2ポイントまで、崩壊はなし、重み付きの
  compositeは正でなければなりません。
- `--require-acceptance` を付けると、拒否や証拠不足のときに終了コード `6` で終わります。スクリプトやCI向けです。

`compare` は、`compare.json`（詳細）と `acceptance.json`（判定）を含むrunを書きます:

```json
{"accepted": false, "status": "insufficient_evidence",
 "reason": "schema 3 with matched source/interface/runtime and non-synthetic model evidence required"}
```

Gateを適用した場合は、指標ごとの変化、compositeのスコア、不合格になったGateの一覧
（例: `["general_ppl_regression", "collapse"]`）が判定に含まれます。

!!! tip "compareには `--output` を付ける"

    `--output` がないと、compareのrunは比べたrunではなく、設定の既定のモデルにちなんだ名前
    （`…_gemma-4-E2B-it-qat-q4_0-unquantized-naive-g128`）になり、紛らわしくなります。

## TUIで {#in-the-tui}

`/compare` で、実行結果から基準と候補を選べます。**実行結果**タブでrunを選んで ++b++ か ++c++ を押す
（または `/baseline`・`/candidate` と入力する）方法もあります。**実行結果**の表には、各runの品質の判定が表示されます。

## 古いバージョンからの移行 {#migrating-from-older-versions}

| 以前 | 現在 |
|---|---|
| `benchmark --suite baseline` | `--suite smoke`（唯一のsuite） |
| 比べるものがない `compare` が成功した | 入力エラー（終了コード `2`）。runに記録される |
| schemaのない品質レポートが合格になりえた | 観測のみ。`accepted: false` |
