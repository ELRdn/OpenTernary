# OpenTernary Icon — SVG v1.0

OpenTernary の三値 LLM ツールキットを表す、円形のオリジナル三つ巴シンボル。

## 正式データ

- `openternary-v1.0.svg` — フルカラー版（背景透過、1024×1024 viewBox）
- `openternary-v1.0-mono.svg` — 単色版（`currentColor`、初期色 `#12182B`）
- `openternary-v1.0-16.png` — 16px フォールバック
- `openternary-v1.0-32.png` — 32px フォールバック
- `openternary-v1.0-512.png` — 512px 高解像度プレビュー・アセット
- `openternary-v1.0-preview.png` — 1024px 透過プレビュー
- `openternary-v1.0-dark-preview.png` — ダーク背景での確認用（背景不透明）
- `openternary-v1.0-mono-preview.png` — 単色表示確認用

## 設計コンセプト

- 円形の外周リングと 3 本の巴が基本構造。
- 巴の元形状は 1 つのカスタムベジェパスから構成し、同じ形を **0° / 120° / 240°** 回転配置。
- カラーは `+1 = amber/orange`、`0 = mint/cyan`、`−1 = blue/violet` に対応。
- 外周リングは連続した虹色グラデーション。
- PNG からの自動トレース・PNG の SVG 埋め込み・フォント依存・外部 URL 参照はなし。

## 利用方法

一般の Web / README / ドキュメントではカラー SVG を使用。白黒印刷・低色環境では単色 SVG を使用。

単色 SVG は CSS の `color` プロパティで色変更できる（通常の CSS で埋め込んだ SVG コンテンツをスタイリングする場合）。

## 再生成

標準 Python だけで SVG ファイルを再生成できる。

```bash
python3 generate_logo.py
```

PNG は Inkscape 等の SVG レンダラーから書き出す。

## 正式採用前に

これは OpenTernary 向けに新規に構成した図形だが、既存のロゴとの類似性・商標登録可能性は別途確認が必要。元の Adobe Stock 画像はトレースしていない。

16px では色と内部の余白がかなり小さくなるため、超小サイズ専用バリアントが必要かは実環境で判断する。
