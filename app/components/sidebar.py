"""PaperBase 第一栏的全局导航。

导航只输出一组普通 Streamlit 控件。品牌 HTML、分隔线和按钮按顺序渲染，
不再用一个跨越多个 widget 的 HTML ``div`` 假装包裹 Streamlit 组件。
"""

from __future__ import annotations

import streamlit as st

if __package__ == "app.components":
    from app.state import KNOWLEDGE_BASE_PAGE, PAPER_WORKSPACE_PAGE
else:
    from state import KNOWLEDGE_BASE_PAGE, PAPER_WORKSPACE_PAGE


def render_navigation_pane() -> str:
    """绘制全局导航并返回当前页面名称。"""
    # 品牌区域只包含静态 HTML，不试图包裹下面的 Streamlit 按钮。
    st.markdown(
        "<div class='pb-brand-row'><div class='pb-brand-logo'>"
        "<svg viewBox='0 0 48 48' aria-hidden='true'>"
        "<path d='M7 10.5C12 8 17 9 23 12v27c-6-3-11-4-16-1.5z' fill='#ffffff'/>"
        "<path d='M41 10.5C36 8 31 9 25 12v27c6-3 11-4 16-1.5z' fill='#d7e8f7'/>"
        "<path d='M24 12v27' stroke='#6d9cc4' stroke-width='2'/>"
        "</svg></div><div><div class='pb-brand-name'>PaperBase</div>"
        "<div class='pb-brand-subtitle'>AI 论文知识库</div></div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<div class='pb-nav-section'>工作区</div>", unsafe_allow_html=True)

    current_page = st.session_state.get("current_page", KNOWLEDGE_BASE_PAGE)
    navigation_items = (
        (KNOWLEDGE_BASE_PAGE, "📚  知识库"),
        (PAPER_WORKSPACE_PAGE, "📄  论文工作区"),
    )
    for page, label in navigation_items:
        # 当前页面使用 primary，其余页面使用 secondary；功能仍由原生按钮处理。
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
    """兼容旧调用方，但不再进入 Streamlit 默认 Sidebar。"""
    return render_navigation_pane()
