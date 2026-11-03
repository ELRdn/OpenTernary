---
description: "OpenTernaryの層ごとの校正（recon-scale・recon-threshold・recon-soft-to-hard）の仕組み、保存されるもの、研究でこれまでに分かったこと。"
---

# 校正

素朴なAbsMeanの量子化では、scaleと閾値をすべて決まった式で選びます。**校正**では、元のモデルが実際に何を
計算しているかを見て、それらを選びます。目的は、量子化した各層が、実際の入力に対して、元の層にできるだけ近い
出力を出すようにすることです。

## 層ごとの再構成 {#layer-local-reconstruction}

OpenTernaryは**線形層を1つずつ**校正します（`window: per-layer`）:

```mermaid
flowchart LR
    T[元のモデル<br/>teacher] -- 校正用の文章で実行 --> A[(対象の各層の入力と出力<br/>ディスク上のキャッシュ)]
    A --> L{{各層で<br/>‖ y_teacher − F.linear(x, q·s) ‖ を最小化}}
    L --> S[層ごとの学習済みパラメータ]
    S --> M[Materialize:<br/>hardな三値のsnapshot]
```

1. **記録。** 元のモデルを校正用の文章で動かし、対象の各層の入力と出力をディスクに保存します。最適化の間、大きな
    モデルをメモリに置き続ける必要がなく、同じteacher・データ・対象ならキャッシュを再利用できます。
2. **最適化。** 層ごとに、学習するパラメータを `steps` 回（Adamまたは SGD、学習率 `lr`）更新し、teacherの出力と
    量子化した層の出力 \(F.\mathrm{linear}(x,\; q \cdot s)\) の再構成損失（`mse` または `l1`）を最小にします。
3. **ホールドアウト。** サンプルの一部（`held_out_ratio`、既定0.2）は最適化に使いません。その損失を前後で報告する
    ので、改善が汎化しているかが分かります。学習用とホールドアウトのサンプルが重ならないことは、ハッシュで確認します。
4. **Materialize。** 最終的なパラメータを**hard**な三値の符号で適用し、ふつうのfake-quantのsnapshot
    （`artifacts/calibrated_snapshot`）として書き出します。`quantize` の出力と同じように、ベンチマーク・評価・
    書き出しができます。

scaleは、出発点が素朴なAbsMeanとちょうど一致するようにパラメータ化しています。step 0は校正前のモデルを再現し、
変化はすべて最適化によるものです。

## 3つの方式 {#the-three-methods}

| `calibration.method` | 学習するもの | 補足 |
|---|---|---|
| `recon-scale` | テンソル・グループごとのscale | 基準となる方式。符号はAbsMeanで固定し、大きさだけを動かします。 |
| `recon-threshold` | scale**と**、0かどうかを分ける閾値 | `threshold_enabled: true` が必要。hardな判定を、clipped straight-through estimator（`threshold_ste_width`）で勾配を通して学習します。閾値はテンソル単位かグループ単位です。 |
| `recon-soft-to-hard` | 学習中はsoftな三値の割り当てを使うscale | 符号をsoftな期待値から始め、温度のスケジュール（`linear`・`cosine`・`exponential`、`temperature_start` から `temperature_end` へ）でhardに近づけます。最後の `hard_fraction` のstepはhardな符号を使います。保存するsnapshotは常にhardです。 |

設定の誤りは最初に拒否します。`threshold_enabled: true` のない `recon-threshold`（とその逆）、温度が上がって
しまうsoft-to-hardのスケジュール、未知のキー、有限でない数値などです。

## データ {#data}

| `calibration.dataset` | 出どころ | 用途 |
|---|---|---|
| `synthetic` | 生成したトークン列 | 仕組みの確認だけ。研究の証拠にはならない |
| `wiki-tiny` | Hugging FaceのWikiText（`Salesforce/wikitext`、`dataset_revision` で固定） | 本番の校正。`datasets` パッケージと、初回のダウンロードが必要 |

`seq_len` トークンの系列を `num_samples` 個使います。`wiki-tiny` を読み込めなければ、校正は**失敗**します。
`allow_dataset_fallback: true` を設定した場合だけ人工データに切り替え、そのときは実際に使ったデータセットを
runに記録します。

## 少ないメモリで動かす {#running-on-limited-memory}

校正は、大きなモデルを一般向けのGPUで扱えるように作られています:

- **低VRAM配置**（`runtime.low_vram: true`、既定）: 最適化中の層だけをGPUに置きます。
- **VRAMの予算:** GPUのメモリの `runtime.vram.max_fraction`（既定80%）までしか使わず、デスクトップ用に
  `reserve_mb` を空けておき、空きが `min_free_mb` を下回ると確保をやめます。
- **マイクロバッチの縮小:** メモリ不足のエラーが起きると、そのstepを巻き戻し、小さなマイクロバッチでやり直します
  （`runtime.microbatch.min_size` まで、`runtime.microbatch.auto: true`（既定）のとき）。
- **チェックポイント:** `checkpoint_interval` stepごとに、層ごとの状態をファイルのハッシュのmanifestとともに
  保存します。`--resume` で最新のチェックポイントから、`--resume-from` で指定したものから再開し、
  `--materialize-only` で学習をせずにチェックポイントからsnapshotを書き出します。

コマンドと、落ちたあとの復旧の方法は[校正のガイド](../guides/calibrate.md)にあります。

## 研究でこれまでに分かったこと {#what-the-research-has-found-so-far}

校正はOpenTernaryの研究の中心で、正直に言えば、**品質Gateを通る三値のGemma 4 E2Bはまだできていません。**
記録した実験から分かっていること:

- `recon-scale` は、小さな構成で再構成損失とホールドアウトの損失を下げます。記録・チェックポイント・再開・
  materialize・再読込の流れは、実際のモデルで最後まで動きます。
- `recon-threshold` で変わるhardな符号はごくわずかです。モデル全体の比較では、scaleだけの校正と比べて
  1,835,532,288個中166,481個（0.009%）でした。
- 3つのsoft-to-hardのスケジュールは、いずれも205の対象層すべてで完了しました。しかしhard化と再読込のあとは、
  符号の変化なし、指示スコア0、8件中8件が崩壊という結果で、どのスケジュールもGateを通りませんでした。
- 残る誤差の大部分はattentionの `q_proj` に集中しています。ホールドアウトの二乗誤差の86%を占め、layer 0の
  `q_proj` だけで約47%です。

現時点の分析では、最適化が扱うのはscaleだけで、符号の割り当てとhard化の境界は固定されたままです。これは
学習を長くすれば解決する問題ではなく、構造上の限界です。[研究の状況](../project/status.md#research)と
[判断の記録](https://github.com/ELRdn/OpenTernary/blob/main/docs/plans/p4-soft-to-hard-redesign-decision.md)を
参照してください。
