---
description: "三値量子化とは何か。OpenTernaryの符号とscaleの計算、テンソル単位とグループ単位、fake-quantとpackedの保存、ビットの数え方。"
---

# 三値量子化

## 考え方 {#the-idea}

言語モデルの重みは、ふつう16bitの浮動小数点数で保存されています。**三値**量子化では、各重みを3つの値の
どれかに置き換えます:

$$
w \;\longrightarrow\; \hat w \in \{-s,\; 0,\; +s\}
$$

\(s\) は、テンソル全体または小さな重みのグループで共有する**scale**です。重みごとに保存するのは
\(q \in \{-1, 0, +1\}\) という**符号**（*trit*）だけで、scaleはテンソルまたはグループごとに1回保存します。

これには次の魅力があります:

- **メモリ:** 1つのtritが持つ情報は \(\log_2 3 \approx 1.585\) bitで、BF16の16bitと比べて、原理的には重みを
  約10分の1にできます。
- **計算:** −1・0・+1を掛けるのに掛け算は要らず、足し算・引き算・スキップだけで済みます。専用のカーネルなら
  これを活かせます。

その代わりに精度が落ちます。16bitで学習したモデルは、重みが3値しかない状態で動くようには作られていません。
BitNet b1.58のように、最初から三値で*学習*したモデルはうまく動きます。一方、OpenTernaryが研究しているのは、
すでに学習したモデルを**学習後に**変換する方法で、これははるかに難しく、素朴に変換するとたいていモデルの能力は
失われます。OpenTernaryは、その損失を正直に測り、取り戻す方法を試すためにあります。

## 重みがtritになるまで（AbsMean） {#how-a-weight-becomes-a-trit-absmean}

OpenTernaryの基準の方式は**AbsMean**です。テンソル \(W\)（またはそのグループの1つ）について:

$$
s = \operatorname{mean}\lvert W \rvert, \qquad
q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{W}{s}\right), -1, +1\right), \qquad
\hat W = q \cdot s
$$

- 平均はFP32で計算します。
- `round` は**偶数丸め**（PyTorchの `torch.round`）です。実際には、\(\lvert w \rvert \le 0.5\,s\) の重みは0に、
  それ以外は \(\pm1\) になります。
- \(s = 0\)（重みがすべて0）のときは、符号もすべて0です。\(s\) に小さなepsilonを足すのではなく、明示的に
  分岐するので、scaleは式どおりの値になります。
- 空のテンソル、浮動小数点でないテンソル、NaNやInfを含むテンソルは、そのまま通さずエラーにします。

例として、1つのグループにある4つの重みで計算します。scaleは
\(s = (0.90 + 0.05 + 0.30 + 1.20) / 4 = 0.6125\) です:

| | \(w_1\) | \(w_2\) | \(w_3\) | \(w_4\) |
|---|---:|---:|---:|---:|
| 重み \(W\) | 0.90 | −0.05 | 0.30 | −1.20 |
| \(W / s\) | 1.47 | −0.08 | 0.49 | −1.96 |
| 符号 \(q\) | <span class="ot-trit plus">+1</span> | <span class="ot-trit zero">0</span> | <span class="ot-trit zero">0</span> | <span class="ot-trit minus">−1</span> |
| \(\hat W = q\,s\) | 0.6125 | 0 | 0 | −0.6125 |

小さな重み（−0.05と、\(0.5\,s\) をわずかに下回る0.30）は0になります。大きな重みは符号だけが残り、同じ大きさを
共有します。個々の重みの大きさの情報はすべて失われます。そこで、校正（[次の章](calibration.md)）でよりよい
scaleや閾値を探します。

## テンソル単位かグループ単位か {#per-tensor-or-per-group-scales}

| `--scale-granularity` | scaleを1つ持つ単位 | 保存の追加コスト（FP32のscale） | 忠実さ |
|---|---|---|---|
| `per_tensor`（既定） | テンソル全体 | ほぼなし | 低い: 数百万の重みで1つの大きさ |
| `per_group` | 行の中で連続する `--group-size` 個の重み | 重み1つあたり \(32 / G\) bit（\(G = 128\) なら0.25） | 高い: グループごとに大きさを持つ |

グループは **`last-dim-rowwise-v1`** の方式で作ります。テンソルを最後の次元に沿った行の並びとみなし、各行を
\(G\) 個ずつ連続したグループに分けます。行の長さが \(G\) の倍数でなければ、その行の最後のグループは短くなります。
\(G\) が行の長さ以上なら各行が1つのグループになるので、`per_group` が知らないうちに `per_tensor` になることは
ありません。

```text
行 0: [ g0: 128個 ][ g1: 128個 ] … [ g11: 128個 ]   ← 1536個、G = 128
行 1: [ g12 ]      [ g13 ]       … [ g23 ]
…
```

Gemma 4 E2Bをグループ単位G128で量子化すると、グループは約1,430万個になり、そのscaleはFP32で57MBです。同じ
モデルをテンソル単位にすると、scaleは205個です。

## 量子化される部分 {#what-gets-quantized}

既定では、**attentionとMLPの線形層**です:

- attention: `q_proj`・`k_proj`・`v_proj`・`o_proj`
- MLP: `gate_proj`・`up_proj`・`down_proj`

**埋め込み**・**出力層**・**正規化**・**バイアス**は元の精度のまま残ります。Gemmaのper-layer embeddingや
マルチモーダルの部分など、モデル固有の部分も同様です。どのテンソルが対象になるかはモデルの**アダプタ**が判断し、
分からないテンソルを推測で対象にすることはありません。次のことができます:

- グループごと外す: `quantization.target.attention: false` または `quantization.target.mlp: false`
- 個別のテンソルを元の精度で残す: `quantization.mixed_precision: {"model.layers.0.self_attn.q_proj.weight": preserve}`

量子化する前に正確な名前を一覧する方法は[調査と計画](../guides/inspect-and-plan.md)で説明しています。

## fake-quant・packed・native {#fake-quant-packed-and-native}

同じ三値の重みを、3つの方法で保存できます:

| 形式 | 保存するもの | 大きさ | 動かせる環境 | OpenTernaryでの状況 |
|---|---|---|---|---|
| **fake-quant** | \(\hat W = q\,s\) をふつうのBF16/FP32の数値で | 元と同じ | どのTransformersのコードでも | `quantize` と `calibrate` の出力 |
| **packed** | 2bitの符号＋scale＋量子化しないテンソル | 量子化した重み1つあたり約2bit＋追加分 | まだ何もない（先に展開する） | `export --format ternary-packed`、可逆な往復 |
| **nativeカーネル** | packedの符号を三値の行列積でそのまま実行 | 小さい | 専用のruntime | **提供していない** |

fake-quantはわざと効率を捨てた形式です。高速なカーネルを誰かが書く前に、三値の重みの*品質*を標準的な道具で
測れるようにするためです。packed化は、重みが本当に三値であることを示し、実際のファイルの大きさを明らかにします。
速度とメモリの利点を得るにはnativeのカーネルが必要で、OpenTernaryにはまだありません。

### 2bitのpacked形式 {#the-2-bit-packing-format}

各符号は2bitで、1バイトに4つ、上位ビットから詰めます:

```text
符号  →  ビット    byte = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3
 −1   →  00
  0   →  01        末尾の詰め物は 00 でなければならない
 +1   →  10        長さ = ceil(N / 4) バイト
       11 = 予約（展開時に拒否）
```

`unpack(pack(q)) == q` は完全に成り立ちます。OpenTernaryはこれを検査しており、safetensorsに復元した書き出しは、
元のfake-quantのsnapshotとバイト単位で一致します。符号とscaleから正確に復元できない入力（たとえば、追加の情報が
必要な閾値で校正したsnapshot）は、packed化を拒否します。

## ビットを正直に数える {#counting-bits-honestly}

「1.58bitモデル」という言葉は、しばしば大まかに使われます。OpenTernaryは次の量を分けて扱います:

| 量 | 量子化した重み1つあたり | 意味 |
|---|---|---|
| 理想のアルファベットのビット数 | \(\log_2 3 \approx 1.585\) | 3つの状態が等確率のときに区別するのに必要な情報量。符号の分布が偏っていれば理論上はもっと少ない |
| 論理的な2bit | 2 | 固定幅の2bitの符号が使う量 |
| packedのビット数 | \(8 \lceil N/4 \rceil / N\) | 2に、テンソルごとのバイトの詰め物の数bitを足したもの |
| scaleの追加分 | \(32 / G\)（グループ単位） | FP32のscale。別に数える |
| 対象外のテンソル | — | 埋め込み・出力層・正規化など。元の精度で、別に数える |

[はじめての実行](../getting-started/first-run.md)では、これらがすべて現れます。量子化した294,912個の重みは2bitで
73,728バイト、4,608個のFP32のscaleが18,432バイト、量子化しない25,984個のFP32の重みが103,936バイトで、合計
**196,096バイト**は、CLIの `Packed estimate bytes` とちょうど一致します。この小さなモデルでは三値で*ない*部分が
大半を占めます。大きなモデルではその割合は小さくなりますが、なくなりはしません。

## ほかのbackend {#other-backends}

`ternary` backendがOpenTernaryの中心です。確立された8bitの方式と比べるために、**TorchAO** backend
（`--backend torchao --scheme int8-weight-only --weight-dtype int8`、`torchao==0.18.0` が必要）も入っています。
サードパーティのbackendは[plugin](../reference/plugins.md)として追加できます。`openternary backends list` で、
何が入っていて、各backendで何が検証済みかを確認できます。

## pass {#passes}

量子化の前に、任意の**pass**で重みを順に変換できます:

- `clip`（`max_abs`）: 重みを \([-m, m]\) に収めます。値を変える処理で、品質を保つための方法では*ありません*。
- `noop`: 何もしません（passの仕組みの確認用）。

passの順序と重複は検査されます。回転のように実行時の変更が必要な変換は、未対応として拒否されます。

```yaml
quantization:
  passes:
    - name: clip
      options: {max_abs: 2.0}
```
