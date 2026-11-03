---
description: "時間・GPU時間・VRAM・ディスクの予算内で量子化の設定の一覧を探索する。journalで再開でき、testのsplitでは選ばない。"
---

# 制約付き探索

`optimize` は、量子化の設定をいくつか試して測定し、品質Gateを通り、VRAMの目標に収まるものの中から最もよいものを
選びます。設定を勝手に作ることも、モデルやデータを変えることも、testのsplitを見ることもありません。

## 入力 {#inputs}

**探索空間**: 量子化の設定のJSONリスト。設定ファイルの `quantization` セクションと同じ形です:

```json title="space.json"
[
  {"backend": "ternary", "scheme": "absmean", "weight_dtype": "ternary", "scale_granularity": "per_tensor"},
  {"backend": "ternary", "scheme": "absmean", "weight_dtype": "ternary", "scale_granularity": "per_group", "group_size": 128},
  {"backend": "ternary", "scheme": "absmean", "weight_dtype": "ternary", "scale_granularity": "per_group", "group_size": 64}
]
```

**品質profile**: どの検証データで測り、どの基準と比べるか。パスはprofileのファイルからの相対で解決します:

```json title="profile.json"
{"schema_version": 1, "name": "balanced-llm-v1", "split": "validation",
 "dataset": "validation.jsonl", "baseline": "baseline-run", "max_length": 512, "stride": 256}
```

`split` に指定できるのは `validation` だけです。`baseline` は、同じ方法で測った元のモデルの `quality.json` を
持つ実行フォルダです。

## 探索を実行する {#running-a-search}

```bash
openternary optimize ./my-model \
    --search-space space.json --quality-profile profile.json \
    --target-vram 12GiB --objective speed \
    --max-candidates 4 --wall-seconds 3600 --gpu-seconds 3600 --max-artifact-bytes 20GiB \
    --output runs/search --dry-run
```

`--dry-run` を外すと実行します。中断した探索を続けるには、同じコマンドに `--resume` を付けます。

| オプション | 既定 | 意味 |
|---|---|---|
| `--target-vram` | 必須 | 推論時に測ったVRAMがこれ以下の候補だけが適格（`12GiB`・`8000MB` など） |
| `--objective` | `speed` | 適格な候補のうち、`speed` は最小のレイテンシ、`size` は最小の成果物を選ぶ |
| `--max-candidates` | 4 | 評価する候補の最大数（1〜256） |
| `--wall-seconds` | 3600 | 経過時間の予算の合計 |
| `--gpu-seconds` | 3600 | GPU時間の予算の合計（CPU以外のworkerは経過時間を保守的にGPU時間として数える） |
| `--max-artifact-bytes` | `20GiB` | 候補の成果物のディスク予算の合計 |
| `--config` | | 基になる設定（評価のデバイス、モデルのrevisionなど） |

## 仕組み {#how-it-works}

1. 各候補は**新しいworkerプロセス**で動きます。モデルを（CPUで）変換し、成果物を保存し、読み込み直し、validationの
    splitで品質を測り、レイテンシとメモリを記録します。
2. 結果は、profileの基準と比べて[品質Gate](../concepts/runs-and-evidence.md#comparing-observation-vs-acceptance)で
    確認します。受け入れられ、*かつ* `--target-vram` に収まる候補だけが**適格**です。
3. すべてのステップを**journal**に書きます。`--resume` では、元・設定・profile・基準・runtime・コードのハッシュが
    すべて一致する完了済みの候補だけを再利用します。出力や測定値の改ざんは検出して拒否します。
4. ディスクの使用量は100ミリ秒ごとに監視します。確認と確認の間に、一時的に上限を超えることがあります。
5. GPUでの評価が失敗したら失敗です。CPUの結果で置き換えることはありません。

## 結果 {#outcomes}

| 結果 | 終了コード | 意味 |
|---|---|---|
| 最良の候補 | `0` | 目的に照らして最もよい適格な候補。`search.json` にすべての候補がある |
| `no_feasible_candidate` | `6` | すべての候補がGateで拒否されたか、VRAMの目標を超えた |
| `budget_exhausted` | `7` | 探索が終わる前に予算が尽きた |

後の2つの場合、**最良の候補は公開しません**。「失敗の中でいちばんまし」なものを結果として示すことはありません。

探索は候補を別のプロセスで動かすので、親の進捗イベントには、候補ごとの開始と終了（`search.candidates`）だけが出て、
候補内部の変換のステップは出ません。

## 制限 {#limits}

探索の仕組み（選択・拒否・予算・再開・改ざんの検出）は、制御したfixtureと人工のモデルで検証しています。実際の
モデルでの有用性と、実デバイスでのVRAMのピークや速度の測定の正確さは、これから確かめる段階です。
