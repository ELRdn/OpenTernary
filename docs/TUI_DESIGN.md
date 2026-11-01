# OpenTernary TUI / UX 設計

2026-11-01 / status: draft (P1完了、ブランド案レビュー待ち)

## 目的

既存CLIの機械契約（`--json`・終了コード・events JSONL）を1バイトも変えずに、人間向けの体験をRich/Textualで高品質化する。`AGENTS.md`の「GUIを主開発対象にしない」に対し、本件は**主開発ではなくUI/UX改善として例外扱い**（2026-11-01 オーナー承認）。

## 決定事項ログ

| # | 項目 | 決定 |
|---|---|---|
| D1 | 優先度 | UI/UX改善として例外扱い |
| D2 | 起動 | TTYで引数なし`openternary` → TUI。非TTYは従来どおりhelp。`openternary ui`も提供 |
| D3 | 大文字別名 | 追加しない（Windowsは大文字小文字を区別しない） |
| D4 | Textual | 必須依存。ただし遅延import（`--json`経路では読み込まない） |
| D5 | 配布 | パッケージレジストリから導入・更新する。レジストリは**未確定**（下記O1） |
| D6 | 版管理 | これからタグ運用を始める（SemVer、`vX.Y.Z`） |
| D7 | ML依存 | `update`はtorch等に触れない。再同期が必要なら案内のみ |
| D8 | 更新確認 | 既定で有効。ネットワークは「確認」にだけ使う。キャッシュ付き、失敗しても起動を妨げない |
| D9 | TUI操作範囲 | 全操作（実行起動・比較・削除含む）。実行前に計画確認、削除は文字入力で確認 |
| D10 | 言語 | 英語・日本語の両対応 |
| D11 | ロゴ | `OPEN-TERNARY`をHermes風の3D AA（ANSI Shadow系） |
| D12 | アイコン | 三つ巴をモチーフ。3つの巴＝三値 {-1, 0, +1} |

### 未決事項

- **O1 配布レジストリ**: npmかPyPIか（後述の比較を参照）。`update`の実装はこれに依存する。
- **O2 ブランド最終案**: 配色・ロゴ・アイコンの最終決定はオーナーが行う（本書の案は叩き台）。

## O1: npm と PyPI の比較

事実（2026-11-01確認）: `openternary`はnpm・PyPIとも未登録（registry 404）。`open-ternary`もnpm未登録。

| 観点 | PyPI（`uv tool install openternary`） | npm（`npm i -g openternary`） |
|---|---|---|
| 本体の格納 | wheelをそのまま配布 | npmにはPython wheelを載せられない。Nodeラッパーが初回起動時にuv/Pythonを用意し、wheelを**別の場所（PyPIかGitHub Releases）から**取得する二段構成になる |
| 利用者の前提 | Python（またはuv単体） | Node.js + Python/uv |
| 依存解決 | uvが`[ml]`などのextrasを扱う | ラッパーがuvを呼ぶので結局同じ処理 + Node層の保守 |
| 更新確認 | `https://pypi.org/pypi/openternary/json` | `https://registry.npmjs.org/openternary/latest` |
| ROCm torch保護 | `uv tool upgrade`は対象toolの環境だけを更新。ML環境は別に保つ設計が必要 | 同じ問題 + ラッパーが環境を作り直す危険 |

推奨: **PyPIを正（source of truth）とし、必要ならnpmは薄い入口として後から追加する。** npmだけでは完結せず、どちらにせよwheelの配布先が必要になる。

## アーキテクチャ

```
openternary <subcommand> [--json]   ← 既存。契約は不変（tests/test_cli_json_contract.py で固定）
openternary / openternary ui        ← Textual App（TTYのみ）
   └─ 実行は子プロセス:  python -m openternary <cmd> --json --events-jsonl <file>
        - 既存契約をそのまま消費。GPU OOMなどでTUIが巻き込まれない
        - Ctrl+C → 子プロセスへ伝播（exit 130 / run lock / last_failure.json は既存のまま）
openternary/ui/
   brand.py   パレット・3Dワードマーク・三つ巴マーク（実装済み・草案）
   theme.py   状態バッジ・意味的スタイル（P3）
   i18n.py    en/ja メッセージカタログ（P3）
   app/       Textual 画面群（P5）
```

### 状態表示の原則（正直さ）

`execution=completed`は品質合格を意味しない（`docs/CLI.md`）。UIもこれを守る。

| 状態 | 記号 | 色 |
|---|---|---|
| accepted（品質受入済み） | ✔ | sage |
| completed（処理完了・品質未評価） | ● | champagne（緑にしない） |
| running | ◐ | gold |
| planned | ○ | ash |
| failed / interrupted | ✖ | copper |

色だけに頼らず、必ず記号と文字列を併記する。

## ブランド案（叩き台）

- ワードマーク: ANSI Shadow書体。面（`█`）はchampagne→gold→copperの金属グラデーション、ストローク（`╔═╝`）は押し出しの影としてbronze→copper。幅98桁。100桁未満では1行版に切り替え。
- マーク: 手続き的に生成した三つ巴を点字ドット（U+2800–28FF）で描く。画像表示に対応していない端末でも描け、任意サイズに拡大縮小できる。巴の色は +1=gold / 0=champagne / -1=copper、外輪はbronze、ラボのアイコンにある軌道の点をemberで配置。
- パレット: ink `#0b0b0c` / champagne `#f0e4cc` / gold `#e2c28a` / bronze `#8a7350` / copper `#c65a3a` / ember `#ffd9a0` / ash `#6e6a63`
- プレビュー: `docs/design/home-120.svg`, `docs/design/home-80.svg`（`python scripts/render_brand_preview.py OUT.svg WIDTH`で再生成。表示データは見本）

## 作業計画と終了条件

| Phase | 内容 | 終了条件 |
|---|---|---|
| P1 ✅ | `--json`契約の固定（14プローブ）、ブランド部品 | golden一致・CI登録 |
| P2 | 長時間処理（calibrate/quantize/benchmark/export）の構造化イベント（追加のみ、schema_version 1のまま） | 実機runでイベント時刻とログが一致 |
| P3 | theme・i18n・helpのグループ化と内部用語除去・`doctor`等の人間向け表示・エラーパネル | golden差分ゼロ、スナップショットをオーナー承認 |
| P4 | タグ運用・リリースworkflow・`update`（確認→計画→実行→検証→ロールバック）・起動時の更新確認 | シナリオ表をすべて通過、torchの`pip freeze`が不変 |
| P5 | Textual App（Home / Runs / Job monitor / Compare / Doctor / Update） | Pilotテスト・snapshot・実機での監視 |
| P6 | `docs/CLI.md`更新、移行表、Windows CI | DoD確認 |

## 既知の事実（P1で判明）

- `backends list`/`info`、`cache list`の`--json`では`command`が`list`/`info`になる（グループ名を含まない）。現行の契約として固定し、変更しない。
- `NO_COLOR`では色を消し、太字などは残す（no-color.orgの仕様どおり）。パイプ出力では装飾なし。
- 起動時のimport時間: CLI全体で約0.26秒、`textual.app`の追加分は約0.2秒（`-X importtime`）。
