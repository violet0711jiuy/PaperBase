"""Paper Overview 与 Explain Section 共用的来源折叠展示。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

if __package__ == "app.components":
    from app.components.evidence_preview import build_evidence_preview
    from app.components.scientific_text import render_scientific_text
else:
    from components.evidence_preview import build_evidence_preview
    from components.scientific_text import render_scientific_text


def render_source_expander(
    *,
    label: str,
    sources: Sequence[Any],
    state_prefix: str,
    source_key_prefix: str,
) -> None:
    """折叠展示 Section/Page/原文，并为每条原文保留独立展开状态。"""
    with st.expander(label, expanded=False):
        if not sources:
            st.caption("暂无可展示的来源原文。")
            return
        for index, source in enumerate(sources, start=1):
            _render_source_card(
                source,
                index=index,
                state_key=f"{state_prefix}::{index}",
                container_key=f"{source_key_prefix}-{index}",
            )


def _render_source_card(
    source: Any, *, index: int, state_key: str, container_key: str
) -> None:
    """按聊天 Evidence 卡片样式显示单条来源，缺失元数据会被优雅省略。"""
    with st.container(border=True, key=container_key):
        section = getattr(source, "section", None)
        page_start = getattr(source, "page_start", None)
        page_end = getattr(source, "page_end", None)
        st.markdown(
            f"<div class='pb-evidence-id'>E{index} · 论文证据</div>",
            unsafe_allow_html=True,
        )
        meta: list[str] = []
        if section:
            meta.append(f"章节：{section}")
        if page_start is not None:
            page = (
                f"p.{page_start}"
                if page_end in (None, page_start)
                else f"p.{page_start}–{page_end}"
            )
            meta.append(f"页码：{page}")
        if meta:
            st.markdown(
                f"<div class='pb-evidence-meta'>{' · '.join(meta)}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div class='pb-evidence-text-label'>证据原文</div>",
            unsafe_allow_html=True,
        )

        text = str(getattr(source, "text", "") or "")
        expanded = bool(st.session_state.get(state_key, False))
        preview, truncated = build_evidence_preview(text)
        render_scientific_text(text if expanded else preview)
        if truncated:
            button_label = "收起" if expanded else "展开全文"
            if st.button(button_label, key=f"{container_key}-toggle", type="secondary"):
                st.session_state[state_key] = not expanded
                st.rerun()
