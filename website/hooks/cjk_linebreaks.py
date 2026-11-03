"""Join soft line breaks between Japanese characters.

Markdown turns a line break inside a paragraph into a space. That is right for English, but between two
Japanese characters it shows up as a stray gap ("ジョブの リアルタイム"). The Japanese pages are wrapped at
about 120 columns for readable diffs, so this hook removes those breaks before rendering, except inside
fenced code blocks.
"""

from __future__ import annotations

import re
from typing import Any

# Hiragana, katakana, CJK ideographs, CJK and full-width punctuation.
_CJK = r"　-〿぀-ヿㇰ-ㇿ㐀-䶿一-鿿＀-￯"
_BREAK = re.compile(rf"(?<=[{_CJK}])\n[ \t]*(?=[{_CJK}])")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})")


def join_cjk_lines(markdown: str) -> str:
    out: list[str] = []
    chunk: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence is None and match:
            out.append(_BREAK.sub("", "".join(chunk)))
            chunk = []
            fence = match.group(2)[0] * 3
            out.append(line)
        elif fence is not None:
            out.append(line)
            if match and match.group(2).startswith(fence):
                fence = None
        else:
            chunk.append(line)
    out.append(_BREAK.sub("", "".join(chunk)))
    return "".join(out)


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    return join_cjk_lines(markdown)
