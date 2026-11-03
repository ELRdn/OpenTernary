---
description: "OpenTernaryの今後: CLIの製品ロードマップ（CLI-0〜CLI-8）、研究のPhase、リリースの節目。"
---

# ロードマップ

OpenTernaryには、あえて分けた2つのロードマップがあります。**製品**が完成していて信頼でき、ドキュメントも揃って
いても、**研究**ではまだよいモデルができていない、ということがありえます（その逆も）。

## 製品: CLI {#product-the-cli}

[CLI_ROADMAP.md](https://github.com/ELRdn/OpenTernary/blob/main/CLI_ROADMAP.md)で管理しています。すべての段階の
最初の実装が揃い、人工のモデルで検証しています（[現在の状況](status.md)のとおり）。

| 段階 | 目標 | 状況 |
|---|---|---|
| CLI-0 | 成功・失敗と、実際に起きたことを常に一致させる | ✔ 実装済み |
| CLI-1 | backendと外部利用の約束事: JSON・終了コード・設定の移行 | ✔ 実装済み |
| CLI-2 | モデル固有の処理をアダプタに閉じ込める | ✔ 実装済み |
| CLI-3 | 実機で確認した2つ目のbackend（TorchAO） | ✔ 実装済み |
| CLI-4 | 順序付きのpass | ✔ 実装済み |
| CLI-5 | 保存・packedの表現・runtimeとの連携を分ける | ✔ 実装済み。nativeのpackedカーネルはない |
| CLI-6 | 同じ条件で測ったものだけを比べる | ✔ 実装済み |
| CLI-7 | 実測の制約と予算の中での探索 | ✔ 実装済み。実モデルでの有用性は未確認 |
| CLI-8 | 配布と拡張の互換性 | ✔ PyPI・plugin・TUI |

製品として次にやること:

- 同じ受入のスクリプトを、実際のモデルとハードウェアで実行する
- TUIでの長いジョブの監視と、Windowsでの中止を実機で確かめる
- 要望に応じて、ほかの導入・実行方法（パッケージマネージャーの入口など）を増やす

## 研究 {#research}

[ROADMAP.md](https://github.com/ELRdn/OpenTernary/blob/main/ROADMAP.md)で管理し、予算とGateは
[P0〜P7の研究計画](https://github.com/ELRdn/OpenTernary/blob/main/docs/plans/p0-p7-research-acceptance.md)にあります。

| Phase | 内容 | 状況 |
|---|---|---|
| 0〜3 | 基盤、調査、基準、三値の中核、モデル全体のfake量子化 | ✔ 完了 |
| 4 | 校正と再構成: scale（4.1）、閾値（4.2）、soft-to-hard（4.3）、複数層の範囲（4.4）、再現実験（4.5） | 4.1は完了。4.2と4.3は実行したが合格する候補はなし。再設計の判断待ち |
| 5 | 日本語の保持: 日本語のホールドアウトのsuite、日本語を多く含む校正 | 予定 |
| 6 | Gemma 4の特殊な部分: per-layer embedding、マルチモーダル | 予定 |
| 7 | 研究の成果としてのpacked三値の書き出し | 予定 |
| 8 | より大きなGemma 4 | 予定 |
| 9 | GUI | TUIはある。GUIはその後 |
| 10 | 対応モデルの拡大: Qwen・Llama・ほかの公開モデル | 予定 |

変わらない研究の原則: Gateは実験の前に決める。testのsplitは1回だけ使う。小さなfixtureやsmokeの実行を品質の主張に
しない。論文やフレームワーク（BitNet・Bonsai・CAT-Q・ScaleQなど）の再現は、元と照らし合わせて確認しない限り主張しない。

## リリースの節目 {#release-milestones}

| 節目 | 意味 | 状況 |
|---|---|---|
| v0.1「三値のHello World」 | 実際のモデルでinspect・benchmark・quantize・compareが動く | ✔ 研究では達成。0.1.0a1/a2として配布 |
| v0.2「校正」 | ホールドアウトの品質を改善する校正 | 未達成 |
| v0.3「日本語」 | 日本語の保持を測って報告する | 未達成 |
