"""集中维护论文前后置章节标题的受控语义规则。"""

from __future__ import annotations

import re


_HEADING_NUMBER_PREFIX = re.compile(
    r"^(?:\d+(?:\.\d+)*(?:[.)]|\s+)|[IVXLC]+(?:[.)]|\s+))",
    flags=re.IGNORECASE,
)

_BIBLIOGRAPHY_HEADINGS = frozenset(
    {
        "references",
        "bibliography",
        "works cited",
        "literature cited",
    }
)

# 这里只接受具有稳定出版语义的完整标题，不在普通正文中做关键词包含匹配。
_BACK_MATTER_ROOT_HEADINGS = frozenset(
    {
        *_BIBLIOGRAPHY_HEADINGS,
        "credit authorship contribution statement",
        "authorship contribution statement",
        "declaration of competing interest",
        "declaration of competing interests",
        "conflict of interest",
        "conflicts of interest",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "data availability",
        "data availability statement",
    }
)


def normalize_semantic_heading(text: str) -> str:
    """去除标题编号并统一空白、大小写和末尾冒号，生成语义匹配键。"""
    normalized_whitespace = " ".join(text.split())
    without_number = _HEADING_NUMBER_PREFIX.sub("", normalized_whitespace)
    return " ".join(without_number.split()).casefold().rstrip(":")


def is_bibliography_heading(text: str) -> bool:
    """判断完整末级标题是否属于受控的参考文献标题集合。"""
    return normalize_semantic_heading(text) in _BIBLIOGRAPHY_HEADINGS


def is_back_matter_root_heading(text: str) -> bool:
    """判断标题是否应作为独立的论文后置根章节，而非继承 Conclusion。"""
    return normalize_semantic_heading(text) in _BACK_MATTER_ROOT_HEADINGS
