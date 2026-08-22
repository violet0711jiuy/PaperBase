"""PaperBase 第一栏的全局导航。"""

from __future__ import annotations

import streamlit as st

if __package__ == "app.components":
    from app.state import KNOWLEDGE_BASE_PAGE, PAPER_WORKSPACE_PAGE
else:
    from state import KNOWLEDGE_BASE_PAGE, PAPER_WORKSPACE_PAGE


def render_navigation_pane() -> str:
    """绘制全局导航并返回当前页面名称。"""
    st.markdown(
        "<div class='pb-brand-row'><div class='pb-brand-logo'>"
        "<svg viewBox='0 0 32 32' aria-hidden='true'>"
        "<rect x='5' y='5' width='12' height='16' rx='1.8' fill='#e86d7a'/>"
        "<rect x='10' y='8' width='12' height='16' rx='1.8' fill='#5c8ed3'/>"
        "<rect x='15' y='11' width='12' height='16' rx='1.8' fill='#70b6aa'/>"
        "</svg></div><div><div class='pb-brand-name'>PaperBase</div>"
        "<div class='pb-brand-subtitle'>AI 论文知识库</div></div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<div class='pb-nav-section-title'>工作区</div>", unsafe_allow_html=True)

    current_page = st.session_state.get("current_page", KNOWLEDGE_BASE_PAGE)
    navigation_items = (
        (KNOWLEDGE_BASE_PAGE, "知识库"),
        (PAPER_WORKSPACE_PAGE, "论文工作区"),
    )
    for page, label in navigation_items:
        if st.button(
            label,
            key=f"global_nav_{page}",
            type="primary" if current_page == page else "secondary",
            use_container_width=True,
        ):
            st.session_state.current_page = page
            st.rerun()

    return current_page


def render_sidebar() -> str:
    """兼容旧调用方。"""
    return render_navigation_pane()
