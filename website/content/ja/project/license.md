---
description: "OpenTernaryのライセンス（Apache-2.0）、NOTICEの要件、名称とロゴの商標の方針、使うモデルとデータのライセンス。"
---

# ライセンスと商標

## コード: Apache License 2.0 {#code-apache-license-20}

OpenTernaryのソースコードは[Apache License 2.0](https://github.com/ELRdn/OpenTernary/blob/main/LICENSE)で提供して
います。Copyright 2026 The OpenTernary Authors.

要するに、次の条件のもとで、商用も含めてコードの利用・改変・再配布ができます:

- 再配布物や派生物（forkを含む）に、このプロジェクトが元であることを示す
  **[NOTICE](https://github.com/ELRdn/OpenTernary/blob/main/NOTICE)** のファイルを残す（第4条(d)）
- 変更したファイルに、その旨を示す（第4条(b)）
- ライセンスを同梱する

コントリビュートも同じライセンスで受け付けます（第5条）。この要約は法的な助言ではなく、適用されるのはライセンスの
本文です。

## 名称とロゴ: ライセンスの対象外 {#name-and-logo-not-licensed}

Apacheライセンスは、**OpenTernary** という名称、`OPEN-TERNARY` のワードマーク、三つ巴のシンボルへの権利を与えません
（第6条）。詳しくは[TRADEMARKS.md](https://github.com/ELRdn/OpenTernary/blob/main/TRADEMARKS.md)にあります:

| 許可なくできること | 書面の許可が必要なこと |
|---|---|
| マークを含む、無改変のリリースの再配布 | fork・製品・サービスに「OpenTernary」や紛らわしい名前を付ける |
| プロジェクトへの言及:「based on OpenTernary」「a fork of OpenTernary」「works with OpenTernary」 | fork・製品・サービスの顔としてロゴやワードマークを使う |
| そうした言及の横に、推薦を示唆しない形で無改変のロゴを置く | 改変版が公式である、推薦されていると示唆する |

公開するforkは、独自の名前とロゴを使ってください。`src/openternary/ui/brand.py` と素材を差し替え、NOTICEのファイルは
残します。

## 使うモデルとデータ {#models-and-data-you-use}

OpenTernaryには、モデルの重みもデータセットも含まれていません。量子化するモデルや評価に使うデータセットには、
それぞれのライセンスが適用され、量子化したモデルは元のモデルの派生物です。特に:

- **Gemma**のモデルには、Hugging Faceでダウンロード前に同意する、GoogleのGemmaの利用規約が適用されます。
- 研究用の品質データセットは、**WikiText**（CC BY-SA 3.0 / GFDL）、**SQuAD**（CC BY-SA 4.0）、**JGLUE/JSQuAD**
  （CC BY 4.0）から作ります。作ったファイルにはそれらの条件が適用されます。
- `wiki-tiny` での校正では、Hugging FaceからWikiTextをダウンロードします。

派生したものを公開する前に、すべてのモデルとデータセットのライセンスを確認してください。
