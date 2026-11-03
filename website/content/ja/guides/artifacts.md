---
description: "成果物をmanifestと照合して検査し、フォルダ内の成果物を一覧し、参照を壊さずに安全に削除する。"
---

# 成果物の管理

**成果物（artifact）**とは、OpenTernaryが `openternary-manifest.json` 付きで書いたモデルのフォルダのことです。
量子化したsnapshot（`runs/<run>/artifacts/snapshot`）、校正済みのsnapshot、`export` の出力がこれにあたります。

## 検査する {#validate}

```bash
openternary artifacts validate runs/tiny-packed                  # --level files（既定）
openternary artifacts validate runs/tiny-packed --level tensors
```

| レベル | 確認すること | 読むもの |
|---|---|---|
| `files` | manifestのschemaと、manifestにある全ファイルのSHA-256 | 全バイトを1回 |
| `tensors` | 上記に加えて、すべてのテンソル（safetensors・ternary-packed・TorchAO）を開き、形・dtype・有限値・対象の一覧を確認 | 全テンソル |

成果物そのものは変更しません。結果は `artifact_validation: file_integrity` または `tensor_integrity` です。どちらも、
フレームワークでモデルを読み込めること（`model_reload`）や、モデルがよいこと（`quality_acceptance`）は意味しません。
それらは別の確認です。[4つの問い](../concepts/runs-and-evidence.md#four-questions-four-answers)を参照してください。

## 一覧する {#list}

```bash
openternary cache list runs
openternary cache list runs --json
```

フォルダの下にあるすべての成果物を、形式・元・大きさ・検査の状態とともに一覧します:

```text
{'artifact': '…/runs/tiny-packed',   'format': 'ternary-packed', 'source': '…/runs/tiny-ternary/artifacts/snapshot', 'bytes': 225364, …}
{'artifact': '…/runs/tiny-restored', 'format': 'safetensors',    'source': '…/runs/tiny-packed',                    'bytes': 1308018, …}
{'artifact': '…/runs/tiny-ternary/artifacts/snapshot', 'format': 'safetensors', 'source': '…/tiny-llama',          'bytes': 1307928, …}
```

`source` を追うと、元のモデル → 量子化したsnapshot → packed → 復元、というつながりが分かります。

## 削除する {#remove}

削除はわざと2段階にしてあり、ほかのものを壊すおそれがあれば拒否します:

```bash
openternary cache remove runs/tiny-restored --root runs            # 確認だけ（status は "planned"）
openternary cache remove runs/tiny-restored --root runs --execute  # 実際に削除
```

`--execute` がなければ何も削除しません。削除されるもの、その大きさ、参照を確認した範囲が表示されます:

```text
項目                        値
──────────────────────────────────────────────────────────────────────────
target                      /home/user/openternary-demo/runs/tiny-restored
root                        /home/user/openternary-demo/runs
bytes                       1308018
references_checked_within   /home/user/openternary-demo/runs
external_references         unknown
status                      planned
```

次のものは拒否されます（終了コード `2`）:

- **`--root` の中**の別の成果物や探索のjournalが、まだ参照している成果物:

    ```text
    ╭─ ✖ 設定・引数・入力の誤り ───────────────────────────────────────────────────╮
    │ artifact is referenced by:                                                   │
    │ ['/home/user/openternary-demo/runs/tiny-restored']                           │
    ╰─────────────────────────────────────────────────────────────── 終了コード 2 ─╯
    ```

- root自体と、rootの外にあるもの
- 成果物ではない（manifestのない）フォルダ。**実行フォルダ**ごと削除するには、TUIの
  [`/delete`](../getting-started/tui.md#runs)（runの名前の入力で確認）を使うか、自分でフォルダを削除します

`--root` の外からの参照は確認できないので、`external_references: unknown` と表示します。OpenTernaryが、共有の
Hugging Faceキャッシュや元のモデルを削除することはありません。

TUIでは、`/cache:remove` に `--execute` を付けると、開始する前に対象の名前の入力を求めます。

## キャッシュの場所 {#cache-locations}

`openternary cache-info` は、uvとHugging Faceのキャッシュの場所を表示します:

```text
Cache info
UV_CACHE_DIR: (default: %LOCALAPPDATA%/uv/cache)
HF_HOME: (default: ~/.cache/huggingface)
```

ダウンロードしたモデルはHugging Faceのキャッシュにあり、管理するのはOpenTernaryではなく `hf cache` のコマンドです。
