"""Conversation answer section parsing used by the Streamlit UI.

The backend persists the rendered answer as Markdown text. Some providers place a
``###`` heading and its body on the same physical line, so the UI must split on
known heading markers rather than assume headings occupy a whole line.
"""

from __future__ import annotations

import re


_SECTION_ALIASES = {
    "直接回答": "direct",
    "论文中的依据与推导": "explanation",
    "阅读理解": "interpretation",
    "如何理解": "interpretation",
    "证据覆盖说明": "coverage",
    "覆盖提示": "coverage",
    "direct answer": "direct",
    "evidence explanation": "explanation",
    "reading interpretation": "interpretation",
    "coverage note": "coverage",
}

# Match only the headings that belong to the answer contract.  This lets the
# body itself contain Markdown headings without accidentally splitting them.
_SECTION_MARKER = re.compile(
    r"###\s*(直接回答|论文中的依据与推导|阅读理解|如何理解|证据覆盖说明|覆盖提示|"
    r"Direct\s+Answer|Evidence\s+Explanation|Reading\s+Interpretation|Coverage\s+Note)"
    r"\s*",
    flags=re.IGNORECASE,
)


def split_answer_sections(answer: str) -> dict[str, str]:
    """Split a persisted assistant answer into stable UI sections.

    Both of the following are supported::

        ### 直接回答 正文…… ### 论文中的依据与推导 正文……

    and::

        ### 直接回答
        正文……
        ### 论文中的依据与推导
        正文……

    If no known marker exists, the whole answer is treated as ``direct`` so
    unresolved/fallback responses remain readable.
    """

    text = (answer or "").strip()
    result = {"direct": "", "explanation": "", "interpretation": "", "coverage": ""}
    if not text:
        return result

    matches = list(_SECTION_MARKER.finditer(text))
    if not matches:
        result["direct"] = _strip_leading_heading_noise(text)
        return result

    # Preserve any non-heading prefix instead of silently discarding it.
    prefix = text[: matches[0].start()].strip()
    if prefix:
        result["direct"] = _strip_leading_heading_noise(prefix)

    for index, match in enumerate(matches):
        heading = re.sub(r"\s+", " ", match.group(1).strip()).casefold()
        key = _SECTION_ALIASES.get(heading)
        if key is None:
            continue
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = _strip_leading_heading_noise(text[body_start:body_end].strip())
        if not body:
            continue
        result[key] = f"{result[key]}\n\n{body}".strip() if result[key] else body

    return result


def _strip_leading_heading_noise(text: str) -> str:
    """Remove leftover Markdown marker whitespace without rewriting content."""

    return re.sub(r"^#+\s*", "", text.strip())
