"""PaperBase 统一三栏 App Shell。"""

from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

if __package__ == "app.components":
    from app.components.styles import apply_global_styles
else:
    from components.styles import apply_global_styles


def render_app_shell() -> tuple[DeltaGenerator, ...]:
    """创建 Global Nav / Context / Main 三栏工作台。

    第一栏与第二栏使用 ``gap=None`` 紧密相连；第三栏占最大空间。
    宽度比例按目标 UI 调整为约 15% / 22% / 63%。
    """
    apply_global_styles()
    columns = st.columns([0.15, 0.22, 0.63], gap=None)
    return tuple(
        column.container(key=key)
        for column, key in zip(
            columns,
            ("pb-global-nav", "pb-context-panel", "pb-main-panel"),
        )
    )
