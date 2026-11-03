---
description: "OpenTernaryの背景にある考え方: 三値の重み、実行フォルダと証拠、校正、アーキテクチャ。"
---

# 考え方

この章では、OpenTernaryが*なぜ*そう動くのかを説明します。*どうやるか*はガイドの章にあります。

<div class="grid cards" markdown>

-   **[三値量子化](ternary-quantization.md)**

    ---

    重みを−1・0・+1に制限するとはどういうことか。scaleの仕組み、テンソル単位とグループ単位、fake-quantと
    packed、「1.58bit」がたいてい2bitを意味する理由。

-   **[実行結果と証拠](runs-and-evidence.md)**

    ---

    すべての結果は実行フォルダに残ります。「completed」が「よい」を意味しない理由、品質Gateの仕組み、状態の
    記号の意味。

-   **[校正](calibration.md)**

    ---

    層ごとの再構成で品質を取り戻す試み。scaleの学習、閾値の学習、soft-to-hardの割り当て。

-   **[アーキテクチャ](architecture.md)**

    ---

    CLI・サービス・アダプタ・backend・pass・exporter・evaluatorの関係と、pluginの差し込み口。

</div>
