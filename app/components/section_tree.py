"""Paper Workspace Explain Tab 左侧的真实 Section Tree 控件。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st


def render_section_tree(
    sections: Sequence[Any], selected_section_id: str | None
) -> str | None:
    """按 ParsedPaper 已保存的 level 顺序展示紧凑目录并返回点击项。"""
    st.markdown("<div class='pb-section-tree-title'>论文目录</div>", unsafe_allow_html=True)
    if not sections:
        st.caption("当前论文没有可用章节。")
        return None

    selected = None
    with st.container(key="paper-section-tree"):
        for section in sections:
            level = max(int(getattr(section, "level", 1)), 1)
            # 只用真实 section_level 做视觉缩进；父子关系本身仍来自 Service。
            indent_ratio = min(0.06 * (level - 1), 0.18)
            with st.container(key=f"paper-section-row-level-{level}-{section.section_id}"):
                indent_column, button_column = st.columns(
                    [indent_ratio, 1.0 - indent_ratio] if indent_ratio else [0.001, 0.999],
                    gap=None,
                )
                with indent_column:
                    st.empty()
                with button_column:
                    if st.button(
                        getattr(section, "title", ""),
                        key=f"paper-section-{section.section_id}",
                        type=(
                            "primary"
                            if section.section_id == selected_section_id
                            else "secondary"
                        ),
                        use_container_width=True,
                    ):
                        selected = section.section_id
    return selected
