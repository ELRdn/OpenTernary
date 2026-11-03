---
description: "OpenTernary・三値モデル・ハードウェア・結果・プロジェクトについてのよくある質問。"
---

# よくある質問

## プロジェクトについて {#about-the-project}

??? question "OpenTernaryで、実際に使える小さくて速いモデルを作れますか？"

    まだできません。三値モデルは作れますが、このツールで作った三値モデルで品質Gateを通ったものはありません。素朴な
    変換ではモデルの能力の大部分が失われ、校正でもこれまでのところ取り戻せていません。また、高速な三値のカーネルが
    ないので、三値モデルの速度は元のモデルと同じです。今のところは**研究**のための道具で、三値化の代償を測り、それを
    減らす方法を試すためのものです。[現在の状況](status.md)を参照してください。

??? question "BitNetと何が違うのですか？"

    BitNet b1.58のモデルは、最初から三値の重みで*学習*します。OpenTernaryが研究しているのは、16bitで学習済みの
    モデルを、*学習後に*変換する方法です。はるかに難しい一方で、一から学習し直さずに、どの公開モデルにも使える
    可能性があります。OpenTernaryは、BitNetやほかの方式を再現したとは主張していません。

??? question "GGUFの量子化（Q4_K・IQ2など）やほかの低bitの道具とは何が違うのですか？"

    それらの形式は成熟していて速く実用的なので、今すぐ小さなモデルが欲しいならそちらを使ってください。OpenTernaryは、
    より極端な点（重み1つに3値）を狙い、測定・来歴・正直な評価を最優先にしています。GGUFへの書き出しは、対応する
    構造について浮動小数点でのみ行えます。

??? question "なぜGemma 4 E2Bなのですか？"

    一般向けのGPUで試行錯誤できる小ささで、新しく、指示調整版があるので品質の低下が見えやすいからです。研究では
    結果を比べられるよう、revisionを1つに固定しています。Gemmaは最初の対象であって唯一の対象ではありません。
    アダプタは汎用で、線形層を判別できるTransformersのモデルなら量子化できます。

??? question "なぜ日本語をこれほど重視するのですか？"

    英語の総合スコアだけでは、ほかの言語での大きな損失が隠れることがあり、低bitの量子化は、資源の少ない言語から
    先に損なうことが多いからです。日本語の品質は一級の指標として、英語とは別に測り、別にGateを設けています。

## 使い方 {#using-it}

??? question "GPUは必要ですか？"

    いいえ。調査・計画・量子化はCPUで動き、小さなモデルならチュートリアル・ベンチマーク・品質評価もCPUで動きます。
    実際のモデルの評価や校正を現実的な時間で行うには、GPUがあるとよいです。

??? question "NVIDIA・AMD・AppleのGPUに対応していますか？"

    `openternary setup` は、NVIDIA（CUDA）・AMD（ROCm、WindowsとLinux）・Apple silicon（MPS）向けのPyTorchを
    入れます。プロジェクトとして検証したのは、WindowsのRadeon RX 9070 XTでのAMDです。NVIDIAとAppleは公式の
    PyTorchのビルドを使いますが、プロジェクトとしてはまだ動かしていません。

??? question "どのモデルを量子化できますか？"

    safetensors形式のHugging Face Transformersのモデルで、アダプタが線形層を判別できるものです。Gemma 4には専用の
    アダプタがあり、汎用のアダプタはLlama型の構造を扱えます。Diffusersのpipelineのcomponentも量子化できます。
    どれについても、品質は検証していません。

??? question "量子化したsnapshotが元と同じ大きさなのはなぜですか？"

    `quantize` が書くのは*fake-quant*のsnapshotで、三値の値をふつうの浮動小数点数で保存しています。そうすれば、
    どのフレームワークでも読み込んで評価できるからです。`export --format ternary-packed` で1つ2bitで保存できます。
    [fake-quant・packed・native](../concepts/ternary-quantization.md#fake-quant-packed-and-native)を参照してください。

??? question "packedのモデルを動かせますか？"

    直接は動かせません。packed形式のruntimeはまだありません。動かすには `export --format safetensors` で
    safetensorsに戻してください。

??? question "OpenTernaryはどこかにデータを送りますか？"

    テレメトリはありません。自動で行う通信は、対話できる `doctor` やTUIでの、1日1回までのPyPIの公開JSON APIへの
    バージョン確認だけです（`OPENTERNARY_NO_UPDATE_CHECK=1` で止められます）。`setup` は実行したときに公開の
    パッケージの一覧を読み、`wiki-tiny` での `calibrate` はWikiTextをダウンロードします。モデルをダウンロードする
    ことはありません。

??? question "`compare` が比較を拒否するのはなぜですか？"

    2つのrunが同じ条件で測られていないか、証拠が揃っていない（人工のモデルなど）からです。条件の違う比較から出る
    数字には意味がありません。[実行結果と証拠](../concepts/runs-and-evidence.md#comparing-observation-vs-acceptance)を
    参照してください。

## 結果について {#results}

??? question "チュートリアルの三値モデルは、元よりパープレキシティが低くなりました。本当ですか？"

    いいえ。どちらのモデルもランダムなので、パープレキシティはノイズで、多少の差に意味はありません。だからこそ、
    品質Gateは人工のモデルを拒否します。[はじめての実行のステップ6](../getting-started/first-run.md#6-measure-quality-and-compare)を
    参照してください。

??? question "「1.58bit」とは何ですか？ なぜOpenTernaryは2と報告するのですか？"

    \(\log_2 3 \approx 1.585\) は、三値の1桁が持つ情報量です。固定幅の形式で保存すると、三値の重みは1つ2bitで、
    さらにscaleと量子化しないテンソルが加わります。OpenTernaryはそれぞれを分けて報告します。
    [ビットを正直に数える](../concepts/ternary-quantization.md#counting-bits-honestly)を参照してください。
