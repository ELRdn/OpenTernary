---
description: "OpenTernaryで検証済みのことと、そうでないこと: CLIと環境、研究の結果、リリースの状況。"
---

# 現在の状況

バージョン **0.1.0a2**（2026年11月）時点の内容です。このページでは、**検証済み**のこと（とその方法）、**実装だけ**の
こと、**未解決**のことを分けて書いています。迷ったときの根拠はリポジトリの証拠の記録です:
[CLIの実装・検証状況](https://github.com/ELRdn/OpenTernary/blob/main/docs/CLI_IMPLEMENTATION_STATUS.md)、
[検証の記録](https://github.com/ELRdn/OpenTernary/blob/main/docs/CLI_VALIDATION_EVIDENCE.md)、
[研究のロードマップ](https://github.com/ELRdn/OpenTernary/blob/main/ROADMAP.md)、
[実験ログ](https://github.com/ELRdn/OpenTernary/blob/main/EXPERIMENT_LOG.md)。

## ひとことで {#in-one-paragraph}

**道具は動きます。** 調査・量子化・校正・packed化・書き出し・評価・比較・探索は最後まで動き、Windows・WSL・AMDの
GPUで、本物のライブラリと人工のモデルを使って試験しています。PyPIで公開しており、TUIもあります。**研究の目標は
まだ達成していません。** 研究対象のGemma 4 E2Bを三値化したもので品質Gateを通ったものはなく、最新の実験は、今の
校正の方式に構造上の限界があることを示しています。

## CLIと環境 {#the-cli-and-its-environments}

人工のモデル（シード固定でランダムに初期化したLlamaとStable Diffusionのpipeline）、本物のライブラリ、導入した
wheelで検証しました:

| | Windows CPU | WSL Ubuntu 24.04 CPU | Windows、Radeon RX 9070 XT |
|---|---|---|---|
| 人工Llama: 三値の変換、pass、層ごとのpreserve、packedの往復 | ✔ 検証済み | ✔ 検証済み | ✔ FP32/BF16のforward、別プロセスでの再読込 |
| TorchAO 0.18.0のINT8: 変換・保存・再読込 | ✔ 検証済み | ✔ 検証済み | ✔ FP32/BF16のforward、別プロセスでの再読込 |
| Diffusers 0.40.0: UNetの差し替え、32×32、2 step | ✔ 検証済み | ✔ 検証済み | ✔ FP32/BF16の生成、有限値、保存 |
| 品質レポートschema 3、探索のworker、再開 | ✔ 人工fixture | ✔ 人工fixture | 計算と資源の計測まで。探索の品質はCPUで確認 |
| 固定版のGGUF converter | ✔ 隔離したconverterとreader、人工LlamaのFP32 | ✔ CPUのruntimeで最大8トークン生成 | ○ GPUでの実行は未検証 |
| wheelの導入、本物のentry pointのplugin | ✔ 検証済み | ✔ 検証済み | ✔ 複製したROCm環境にwheelを導入 |
| 学習済みのモデル・画像の品質 | ○ 未検証 | ○ 未検証 | ○ 未検証 |
| nativeのpacked三値カーネル | – 提供なし | – 提供なし | – 提供なし |

最終の検証（オフライン、Windows CPU）: **326件合格、2件スキップ、失敗0件**。スキップした2件はROCmが必要な試験で、
ROCm環境で別に合格しています。Ruff・format・mypyは全ファイルで通過しました。GPUのメモリの計測値は最大約84.6MiB
でした。人工のモデルでの結果を、Gemma・Mistral・Qwenなどの実際のモデルに広げて解釈はしません。

それ以降の変更は、すべてLinuxとWindowsのGitHub CIを通しています。TUIの試験、Windowsでの本物のCtrl+Breakによる
中止の試験、ジョブ管理の試験も含みます。

## 実機でのPyTorchの環境構築 {#pytorch-setup-on-real-hardware}

オーナーのPC（Windows 11、Radeon RX 9070 XT、gfx1201）で、`openternary setup` はGPUを検出し、AMDの
`torch 2.14.0+rocm10.1.0` をgfx1201用のカーネルだけで入れ、FP32・BF16のデバイスの試験に合格しました。PyPIの
リリース（`uv tool install openternary`）からも、同じ構成で動くことを確認しています。NVIDIAとAppleの経路は公式の
一覧に沿って実装していますが、プロジェクトとしては実機で動かしていません。

## 研究 {#research}

研究対象は **Gemma 4 E2B**（`google/gemma-4-E2B-it-qat-q4_0-unquantized`、revision `6befbaca…`）で、正準の205の
線形層をグループ単位（G128）で量子化し、BF16で評価します。

| Phase | 結果 |
|---|---|
| 0 基盤 | ✔ 完了 |
| 1 調査とBF16のsmokeの基準 | ✔ 完了: 実際のCPUでの2回の実行で、fingerprintが一致 |
| 2 三値の中核とpackedの集計 | ✔ 完了: 19のGate、実際のテンソルでの往復がビット単位で一致 |
| 3 モデル全体の三値化 | ✔ 完了: テンソル単位とグループ単位G128の、モデル全体のsnapshot（12 shard、10.2GB）が読み込めてsmokeを実行できる。出力は想定どおり基準と異なる |
| 4.1 校正: scaleの学習 | ✔ 完了: fixtureで学習とホールドアウトの再構成損失が下がる。実際のモデルで流れが動く |
| 4.2 閾値の学習 | 実装済み。AMD GPUのruntimeのGateに合格。変わる符号はごくわずか（scaleだけと比べて0.009%） |
| 4.3 soft-to-hard | 3つのスケジュールを205の全層で実行。hard化のあとはすべて崩壊（指示スコア0、8件中8件が崩壊）。**合格したスケジュールはない** |
| 5以降 日本語、特殊な部分、packedの書き出し、大きなモデル | 研究のPhaseとしては未着手 |

**品質Gateを通った量子化モデルはありません。** soft-to-hardの結果の分析では、最適化が扱うのはscaleだけで、符号の
割り当てとhard化の境界は固定されたままであることが分かりました。attentionの `q_proj` がホールドアウトの誤差の86%を
占め、layer 0のものだけで約47%です。再設計の判断を待っている状態です
（[判断の記録](https://github.com/ELRdn/OpenTernary/blob/main/docs/plans/p4-soft-to-hard-redesign-decision.md)）。
それまでは、OpenTernaryで作った三値モデルは、元のモデルよりはるかに性能が低いと考えてください。

## リリース {#releases}

| バージョン | 日付 | 内容 |
|---|---|---|
| 0.1.0a2 | 2026-11-03 | バックグラウンドのジョブを持つ全画面のTUI。選択式のコマンドプロンプト。[変更履歴](changelog.md)を参照 |
| 0.1.0a1 | 2026-11-03 | 最初のPyPIリリース |

どちらもプレリリースで、安定版はまだありません。

## まだ検証していないこと {#not-yet-verified}

- 実際の重みでの三値モデルの品質（研究の問い）
- native低bitでの実行（カーネルを提供していない）
- GPUでのGGUFと、Llama以外の構造のGGUF
- 実際のNVIDIAとAppleのハードウェアでの `setup`
- TUIから見守る長い校正と、実際のWindowsでのTUIからの中止（CIでのみ確認）
