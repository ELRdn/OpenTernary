# OpenTernary TUI / UX 設計

2026-11-03 / status: draft (P1〜P3実装済み、P3の見た目はオーナー確認待ち。次はP4)

## 目的

既存CLIの機械契約（`--json`・終了コード・events JSONL）を1バイトも変えずに、人間向けの体験をRich/Textualで高品質化する。`AGENTS.md`の「GUIを主開発対象にしない」に対し、本件は**主開発ではなくUI/UX改善として例外扱い**（2026-11-01 オーナー承認）。

## 決定事項ログ

| # | 項目 | 決定 |
|---|---|---|
| D1 | 優先度 | UI/UX改善として例外扱い |
| D2 | 起動 | TTYで引数なし`openternary` → TUI。非TTYは従来どおりhelp。`openternary ui`も提供 |
| D3 | 大文字別名 | 追加しない（Windowsは大文字小文字を区別しない） |
| D4 | Textual | 必須依存。ただし遅延import（`--json`経路では読み込まない） |
| D5 | 配布 | **PyPIを正（source of truth）**とする（`uv tool install openternary`）。npmは後から薄い入口として追加（下記O1比較） |
| D6 | 版管理 | これからタグ運用を始める（SemVer、`vX.Y.Z`） |
| D7 | ML依存 | `update`はtorch等に触れない。再同期が必要なら案内のみ |
| D8 | 更新確認 | 既定で有効。ネットワークは「確認」にだけ使う。キャッシュ付き、失敗しても起動を妨げない |
| D9 | TUI操作範囲 | 全操作（実行起動・比較・削除含む）。実行前に計画確認、削除は文字入力で確認 |
| D10 | 言語 | 英語・日本語の両対応 |
| D11 | ロゴ | `OPEN-TERNARY`をHermes風の3D AA（ANSI Shadow系） |
| D12 | アイコン | 三つ巴をモチーフ。3つの巴＝三値 {-1, 0, +1}。**巴ごとに色を塗り分ける** |
| D13 | 回転方向 | 頭が進行方向（頭が先、尾が後ろに流れる） |
| D14 | アイコン配色 | 公式SVGの配色を使う: +1 = amber/orange、0 = mint/cyan、-1 = blue/violet、外周リングは連続した虹色（`docs/brand/`） |
| D15 | UIパレット | ラボのアイコン（金・銅）には合わせない。**Yoru（夜）に決定** |
| D16 | シンボル | **オーナー作成の公式SVG v1.0**（`docs/brand/openternary-v1.0.svg`）を採用。ストック画像を参考に作った組紐（A）・渦（B）の再構成は破棄 |
| D17 | アニメーション | **時計回り**（頭が先に進む向き。2026-11-03 オーナー決定）。`OPENTERNARY_SPIN=ccw`・`--ccw`・`mark(..., clockwise=False)`で反転。資産は反時計回りで格納し、時計回りは逆順再生。48コマ・12fps＝1周4秒。非TTYと`OPENTERNARY_REDUCED_MOTION`/`NO_MOTION`では静止 |

### 未決事項

- ~~O4 回転方向~~: 解消（D17、時計回り）。
- ~~O3 参考画像の出典~~: 解消。オーナー作成のオリジナルSVGに置き換え、ストック由来の再構成（`scripts/design/fit/`、比較画像5・6）はリポジトリから削除した。ただし、このブランチの過去のコミットには残っているので、mainへは squash merge で取り込む。

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

## ブランド（確定: 公式SVG v1.0・Yoru）

実装: `src/openternary/ui/brand.py`（パレット・ワードマーク・マーク読込）、`ui/splash.py`（回転デモ: `python -m openternary.ui.splash`）、`ui/assets/tomoe.otmk`（`scripts/design/bake_mark.py`が公式SVGをChromiumで正確にラスタライズして生成。16/24/30桁 × 48コマ、38KB、標準ライブラリだけで読込）。資産の拡張子は`.otmk`。`.bin`は`.gitignore`とsdistの除外対象（モデル重み用）になっているため使わない。

以下は選定時の比較記録。

画像: `docs/design/candidates/`（生成: `scripts/design/`、要 numpy・scipy・pillow）

### ワードマーク
ANSI Shadow系の3D文字。面（`█`）にアイコンと同じ知覚的虹色グラデーション、ストローク（`╔═╝`）は影としてborder色。100桁未満では1行版。

### 三つ巴（参考画像から4スタイル）

| 候補 | 特徴 | ターミナル実寸での判読性（観測） |
|---|---|---|
| A 組紐 | 3本の輪が上下に編まれる | 30桁で構造は読めるが煩雑。16桁では潰れる |
| B 渦 | 頭が中心、尾が渦状の帯 | 帯の隙間が点字ドット未満で消える。採用時は帯の幅・隙間を作り直す必要あり |
| C 大頭・細線 | 大きな頭、面を細線で分割（参考は二つ巴→三つに変更） | 細線は無調整だと消える。ヒンティング後は16桁でも判読可 |
| D 風車 | 中心の隙間が風車状に開く | ヒンティング後は16桁でも最も明瞭 |

**発見（観測）**: 点字ドット1個の間隔は30桁で約0.033（単位円基準）。デザイン上の細線（0.032）はこれ未満なので描画されない。表示サイズごとに隙間を最低1.6ドットへ広げる「光学サイズ調整（ヒンティング）」を行う。

### 虹色
OKLCHで明度・彩度を固定し、色相だけを回す（HSVの虹と違い、黄色だけ眩しい・青だけ暗いといった明るさのムラが出ない）。巴kは色相 `25° + 120k` から `+120°` を担当し、頭→尾の位置は形の内側を通る測地距離で測る。Google/Geminiのブランド色そのものは使わない（他社ブランドとの混同を避けるため）。ライトテーマでは明度を下げる（L 0.74 → 0.60）。

### UIパレット候補
虹色アイコンを主役にするため、UIの地は無彩色寄りにする。値はOKLCHで設計し、検証は機械的に行った。

| 案 | mode | bg / text / accent | コントラスト text / muted / 状態色の最小値（WCAG） | 成功↔失敗の色差 ΔE（通常 / 1型 / 2型 / 3型色覚, Machado 2009） |
|---|---|---|---|---|
| Sumi 墨 | dark | `#0e0f12` / `#e5e8ec` / `#97c9fd` | 15.59 / 6.14 / 6.69 | 32.2 / 27.7 / 18.6 / 38.1 |
| Yoru 夜 | dark | `#090c1c` / `#e9ebf3` / `#b89ffe` | 16.32 / 7.21 / 6.78 | 32.2 / 27.7 / 18.6 / 38.1 |
| Shiro 白 | light | `#f8f7f2` / `#1c1a17` / `#5550c0` | 16.19 / 5.6 / 5.15 | 27.1 / 10.4 / 16.8 / 31.0 |

- 基準: 本文4.5以上（AA）。全案・全色が満たす。
- 初版では成功=緑・失敗=赤で、2型色覚のΔEが3.3（ほぼ同じ色）だった。成功を青緑（hue 195）、失敗を朱（hue 32）にし、さらに明度差をつけて改善した。
- 状態は色だけで区別しない（✔ ● ◐ ✖ ▲ を必ず併記）。
- Textualのテーマ機能でダーク/ライトを切り替える。端末の背景色はOSC 11で問い合わせ、自動選択を試みる（未検証）。

## 作業計画と終了条件

| Phase | 内容 | 終了条件 |
|---|---|---|
| P1 ✅ | `--json`契約の固定（14プローブ）、ブランド部品 | golden一致・CI登録 |
| P2 ✅ | 長時間処理の構造化イベント（追加のみ、schema_version 1のまま）。一覧は`docs/CLI.md`の「JSON・終了コード」 | オーナー実機のsmoke runで欠落・順序乱れなし（下記）。ログとの時刻突合は`logs/`未入手のため未実施 |
| P3 ✅（オーナー確認待ち） | theme（記号のみ着色）・i18n（`OPENTERNARY_LANG`）・helpの3パネル化と内部用語除去・`doctor`/`backends`/`plan`/`plugins`の表表示・エラーの1回パネル化 | golden差分ゼロ（確認済み）、見た目をオーナー承認 |
| P4 | タグ運用・リリースworkflow・`update`（確認→計画→実行→検証→ロールバック）・起動時の更新確認 | シナリオ表をすべて通過、torchの`pip freeze`が不変 |
| P5 | Textual App（Home / Runs / Job monitor / Compare / Doctor / Update） | Pilotテスト・snapshot・実機での監視 |
| P6 | `docs/CLI.md`更新、移行表、Windows CI | DoD確認 |

## 実機計測（P2、2026-11-03）

オーナーのWindows PC（CPU実行）で`configs/gemma4-e2b-calibration-smoke.yaml`（205モジュール × 2ステップ、8サンプル × 64トークン）を実行した`--events-jsonl`の集計。

| 段階 | 所要 | 備考 |
|---|---|---|
| 開始〜データセット準備 | 11.7秒 | wikitext初回ダウンロードを含む。当時はイベントがなく無言 → `calibrate.dataset`を追加 |
| 教師モデル読込 | 8.6秒 | |
| 活性値収集 | 6.3秒 | train 7バッチ・heldout 1バッチ |
| 収集後の準備 | 15.2秒 | 無言 → `calibrate.prepare`を追加 |
| モジュール最適化 | 150.7秒 | 1モジュールあたり中央値0.45秒・最大1.33秒（後半層のMLPが重い） |
| 評価 | 26.8秒 | |
| 書き出し | 145.0秒 + hash 6.9秒 | `embed_tokens_per_layer`の処理とshard書込で最大31秒無言 → `materialize.flush`を追加 |
| 結果の作成 | 10.3秒 | 無言 → `calibrate.report`を追加 |
| 合計 | 381.8秒 | オーナー体感の約8分との差は、コマンド開始前（`uv run`・Python・torchの起動）と推測 |

- 1394行・416KB。sequenceと`elapsed_s`は単調増加、全行のキー構成が同一、`completed == total`の最終行は全段階で出ている。
- 書き出し中のRSSはコンソールログで約18.6GB。メモリの少ないPCでは書き出しが最も厳しい段階になりうる。
- 本番設定（50ステップ・32サンプル × 128トークン）の所要は、モジュール最適化が計算量にほぼ比例すると仮定して、このPCのCPUで**おおよそ5〜7時間（推測、未計測）**。

## 既知の事実（P1で判明）

- `backends list`/`info`、`cache list`の`--json`では`command`が`list`/`info`になる（グループ名を含まない）。現行の契約として固定し、変更しない。
- `NO_COLOR`では色を消し、太字などは残す（no-color.orgの仕様どおり）。パイプ出力では装飾なし。
- 起動時のimport時間: CLI全体で約0.26秒、`textual.app`の追加分は約0.2秒（`-X importtime`）。
