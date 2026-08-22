"""PaperBase 各页面共用的科学文本与公式渲染辅助函数。"""

from __future__ import annotations

import re

import streamlit as st


def render_scientific_text(text: str) -> None:
    """使用 Streamlit Markdown/KaTeX 渲染正文和常见 LaTeX 定界符。"""
    st.markdown(normalize_scientific_text(text))


def normalize_scientific_text(text: str) -> str:
    r"""把后端常见的 ``\(...\)``、``\[...\]`` 和复杂度写法标准化。"""
    normalized = text or ""
    normalized = re.sub(r"\\\((.*?)\\\)", r"$\1$", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", normalized, flags=re.DOTALL)
    return re.sub(
        r"(?<![$\\])\bO\([^\n)]{1,80}\)",
        lambda match: f"${match.group(0)}$",
        normalized,
    )

