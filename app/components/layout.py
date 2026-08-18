"""PaperBase 统一 App Shell。

本模块只负责页面级骨架：三栏比例和一份很轻的全局主题。具体页面内容
由 ``pages/knowledge_base.py`` 和 ``pages/paper_workspace.py`` 自己渲染，
这样两个工作区在切换时始终使用同一个布局契约。
"""

from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def render_app_shell() -> tuple[DeltaGenerator, ...]:
    """创建统一的 Global Nav / Context Panel / Main Panel 三栏。

    ``st.columns`` 接收的是比例而不是像素，因此这里使用明确的比例列表。
    第一栏约占 14%，第二栏约占 19%，第三栏保留约 67% 空间给主内容。
    ``gap=None`` 让面板之间不出现 Streamlit 默认的大块留白。
    """
    _apply_shell_style()
    columns = st.columns([0.14, 0.19, 0.67], gap=None)
    # Streamlit container 的 key 会生成稳定的 ``st-key-*`` class；它只负责
    # 给三个 pane 提供背景和 padding，不参与 column 宽度计算。
    return tuple(
        column.container(key=key)
        for column, key in zip(
            columns,
            ("pb-global-nav", "pb-context-panel", "pb-main-panel"),
        )
    )


def _apply_shell_style() -> None:
    """应用不依赖 Streamlit 内部 DOM 的基础主题。

    这里只设置背景、字体、间距、按钮和卡片的基础观感；不改变 column 的
    flex 宽度，也不通过 marker、``:has`` 或 ``data-testid`` 重排页面结构。
    """
    st.markdown(
        """
        <style>
          .stApp { background: #f3f6fa; color: #243447; }
          .block-container { max-width: none; padding: 0; }
          .stButton button { border-radius: 10px; min-height: 2.65rem; }
          .stButton button[kind="primary"] { background: #4a6a88; border-color: #4a6a88; color: #fff; }
          .stButton button[kind="primary"]:hover { background: #3f5f7d; border-color: #3f5f7d; }
          .stButton button[kind="secondary"] { background: transparent; border-color: transparent; color: #526779; }
          .stButton button[kind="secondary"]:hover { background: #e5edf3; border-color: #dce4ed; color: #315e7d; }
          .st-key-pb-global-nav { box-sizing: border-box; background: #263f55; border-radius: 0; color: #fff; min-height: 100dvh; padding: 1.1rem 1rem; }
          .st-key-pb-global-nav .stButton button[kind="secondary"] { background: transparent; border-color: transparent; color: #e6eef5; justify-content: flex-start; }
          .st-key-pb-global-nav .stButton button[kind="secondary"]:hover { background: #344f69; border-color: #344f69; color: #fff; }
          .st-key-pb-global-nav .stButton button[kind="primary"] { background: #54779b; border-color: #6489ad; box-shadow: inset 4px 0 0 #52a9f4; justify-content: flex-start; }
          .st-key-pb-context-panel { box-sizing: border-box; background: #f3f6fa; border-left: 1px solid #dce4ed; border-right: 1px solid #dce4ed; min-height: 100dvh; padding: 1.1rem 1.15rem; }
          .st-key-pb-main-panel { box-sizing: border-box; background: #f7f9fc; min-height: 100dvh; padding: 1.1rem 1.4rem; }
          .stExpander { background: #f8fafc; border: 1px solid #dce4ed; border-radius: 10px; }
          .pb-brand-row { align-items: center; background: #263f55; border-radius: 12px; color: #fff; display: flex; gap: 0.65rem; padding: 0.75rem; }
          .pb-brand-logo { align-items: center; background: #f3f7fb; border-radius: 10px; display: flex; height: 2.65rem; justify-content: center; width: 2.65rem; }
          .pb-brand-logo svg { height: 2.15rem; width: 2.15rem; }
          .pb-brand-name { font-size: 1.35rem; font-weight: 750; line-height: 1.1; }
          .pb-brand-subtitle { color: #d3e0eb; font-size: 0.78rem; margin-top: 0.18rem; }
          .pb-nav-section { color: #b9c9d8; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; margin: 0.35rem 0 0.5rem; }
          .pb-chat-avatar { align-items: center; background: #dbe7f0; border: 1px solid #c5d7e4; border-radius: 50%; display: flex; font-size: 1.25rem; height: 2.35rem; justify-content: center; margin-top: 0.35rem; width: 2.35rem; }
          .pb-user-avatar { background: #e5eff6; }
          .pb-assistant-avatar { background: #eef4f8; }
          .pb-message-header { align-items: center; display: flex; font-size: 0.92rem; font-weight: 650; gap: 0.4rem; line-height: 1.3; margin-bottom: 0.5rem; }
          .pb-user-header { color: #4a6a88; justify-content: flex-end; }
          .pb-assistant-header { color: #243447; justify-content: flex-start; }
          .pb-message-time { color: #8a99a8; font-size: 0.74rem; font-weight: 450; }
          .pb-empty-panel { color: #718096; padding: 20vh 1rem; text-align: center; }
          .pb-empty-title { color: #34495b; font-size: 1rem; font-weight: 650; }
          .pb-empty-copy { font-size: 0.82rem; margin-top: 0.35rem; }
          .pb-paper-title { color: #243447; font-size: 1.55rem; font-weight: 750; line-height: 1.3; }
          .pb-paper-source { color: #8a99a8; font-size: 0.78rem; margin-top: 0.2rem; }
          .pb-answer-section { margin: 0.1rem 0 0.2rem; }
          .pb-answer-section-title { border-top: 1px solid #dce4ed; color: #315e7d; font-size: 1.02rem; font-weight: 700; line-height: 1.4; margin-top: 1rem; padding-top: 0.75rem; }
          .pb-answer-section:first-child .pb-answer-section-title { border-top: 0; margin-top: 0; padding-top: 0; }
          .pb-answer-section-muted .pb-answer-section-title { color: #526779; font-weight: 650; }
          .pb-evidence-id { color: #315e7d; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.18rem; }
          .pb-evidence-meta { color: #728092; font-size: 0.74rem; line-height: 1.5; margin-bottom: 0.48rem; }
          .pb-evidence-text-label { color: #5f7183; font-size: 0.76rem; font-weight: 650; margin-bottom: 0.18rem; }
          .pb-evidence-gap { height: 0.35rem; }
          div[data-testid="stForm"] { background: #fff; border: 1px solid #d7e1ea; border-radius: 14px; box-shadow: 0 2px 8px rgba(31, 45, 61, 0.045); margin-top: 0.7rem; padding: 0.42rem 0.5rem; }
          div[data-testid="stForm"] input { background: transparent; border: 0; box-shadow: none; color: #243447; font-size: 0.95rem; min-height: 2.7rem; padding: 0.62rem 0.72rem; }
          div[data-testid="stForm"] input:focus { border: 0; box-shadow: none; }
          div[data-testid="stFormSubmitButton"] button[kind="primary"] { background: #4a6a88; border-color: #4a6a88; color: #fff; border-radius: 10px; font-size: 1.35rem; min-height: 2.7rem; padding: 0; }
          div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: #3f5f7d; border-color: #3f5f7d; }
          .pb-pane-title { color: #243447; font-size: 1.35rem; font-weight: 700; line-height: 1.25; }
          .pb-pane-subtitle { color: #718096; font-size: 0.84rem; line-height: 1.45; }
          .pb-pane-kicker { color: #718096; font-size: 0.72rem; font-weight: 650; letter-spacing: 0.08em; }
          .pb-panel-divider { border-top: 1px solid #dce4ed; margin: 1rem 0; }
          .pb-muted { color: #718096; font-size: 0.82rem; }
          .pb-context-stat { color: #526779; font-size: 0.86rem; font-weight: 600; margin: 1rem 0 0.8rem; }
          .pb-context-section-title { color: #526779; font-size: 0.8rem; font-weight: 700; margin: 0.6rem 0 0.55rem; }
          .pb-paper-item { border-bottom: 1px solid #e3eaf1; padding: 0.58rem 0 0.65rem; }
          .pb-paper-item:last-child { border-bottom: 0; }
          .pb-paper-item-title { color: #34495b; font-size: 0.84rem; font-weight: 600; line-height: 1.45; }
          .pb-paper-item-meta { color: #8a99a8; font-size: 0.74rem; margin-top: 0.18rem; }
          .pb-chat-topline { align-items: flex-start; display: flex; justify-content: space-between; gap: 1rem; padding-top: 0.4rem; }
          .pb-chat-title { color: #1f2d3d; font-size: 1.3rem; font-weight: 750; line-height: 1.3; }
          .pb-chat-subtitle { color: #718096; font-size: 0.84rem; margin-top: 0.24rem; }
          .pb-chat-status { background: #eef4f8; border: 1px solid #d5e3ed; border-radius: 999px; color: #4a6a88; font-size: 0.74rem; padding: 0.38rem 0.7rem; white-space: nowrap; }
          .pb-chat-divider { border-top: 1px solid #dce4ed; margin: 0.9rem 0 0.7rem; }
          .pb-empty-chat { color: #718096; padding: 18vh 1rem 10vh; text-align: center; }
          .st-key-kb-message-scroll { box-sizing: border-box; height: calc(100dvh - 220px); max-height: calc(100dvh - 220px); min-height: 280px; overflow-y: auto; padding-right: 0.4rem; }
          .st-key-kb-message-scroll > div { padding-bottom: 0.4rem; }
          .st-key-kb-new-chat .stButton button[kind="primary"] { background: #4a6a88; border-color: #4a6a88; color: #fff; }
          .st-key-kb-new-chat .stButton button[kind="primary"]:hover { background: #3f5f7d; border-color: #3f5f7d; }
          .st-key-kb-conversation-list .stButton button { justify-content: flex-start; min-height: 3.35rem; padding: 0.55rem 0.7rem; text-align: left; white-space: pre-line; }
          .st-key-kb-conversation-list .stButton button[kind="primary"] { background: #dcebf4; border-color: #bfd7e6; box-shadow: inset 3px 0 0 #4a6a88; color: #315e7d; }
          .st-key-kb-conversation-list .stButton button[kind="secondary"] { background: transparent; border-color: transparent; color: #526779; }
          .st-key-kb-conversation-list .stButton button[kind="secondary"]:hover { background: #e8f0f6; border-color: #dce4ed; color: #315e7d; }
          [class*="st-key-kb-user-bubble-"] { background: #e5eff6; border: 1px solid #cbddea; border-radius: 15px; padding: 0.85rem 1rem; }
          [class*="st-key-kb-assistant-bubble-"] { background: #fff; border: 1px solid #dce4ed; border-left: 3px solid #4a6a88; border-radius: 15px; box-shadow: 0 1px 3px rgba(36, 52, 71, 0.03); padding: 0.95rem 1.05rem; }
          [class*="st-key-kb-user-bubble-"] [data-testid="stMarkdownContainer"], [class*="st-key-kb-assistant-bubble-"] [data-testid="stMarkdownContainer"] { color: #243447; font-size: 0.96rem; font-weight: 400; line-height: 1.75; }
          [class*="st-key-kb-user-bubble-"] [data-testid="stMarkdownContainer"] p, [class*="st-key-kb-assistant-bubble-"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.65rem; }
          [class*="st-key-kb-user-bubble-"] [data-testid="stMarkdownContainer"] p:last-child, [class*="st-key-kb-assistant-bubble-"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }
          [class*="st-key-kb-assistant-bubble-"] div[data-testid="stExpander"] { background: #f8fafc; border-color: #e2eaf1; border-radius: 10px; }
          [class*="st-key-kb-assistant-bubble-"] .pb-evidence-id { color: #4a6a88; font-size: 0.78rem; }
          [class*="st-key-kb-assistant-bubble-"] .pb-evidence-meta { color: #8795a3; font-size: 0.72rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
