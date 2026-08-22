"""Paper Overview 的轻量展示组件。

组件只负责把 Service 已经校验过的结构化字段排成卡片；论文概览的生成、
来源 chunk 读取和工件持久化仍由后端与 ``PaperBaseService`` 负责。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

import streamlit as st

from paperbase.overview.service import OverviewField, PaperOverview

if __package__ == "app.components":
    from app.components.source_evidence import render_source_expander
else:
    from components.source_evidence import render_source_expander


def render_paper_overview(
    overview: PaperOverview,
    *,
    workspace_id: str,
    source_loader: Callable[[str, Sequence[str]], Sequence[Any]],
) -> None:
    """渲染纵向摘要卡片，避免重复论文标题并保持阅读顺序清晰。"""
    # 论文标题已由 Paper Workspace 的 Header 展示；此处直接进入摘要内容。
    # 每个字段占一张完整宽度卡片，长段技术说明不会在两栏中被压缩得难以阅读。
    fields = (
        ("research_problem", "研究问题", overview.research_problem),
        ("main_method", "核心方法", overview.main_method),
        ("contributions", "主要贡献", overview.contributions),
        ("datasets", "数据集", overview.datasets),
        ("experimental_setup", "实验设置", overview.experimental_setup),
        ("main_results", "主要结果", overview.main_results),
        ("limitations", "局限性", overview.limitations),
    )
    for field_key, title, field in fields:
        _render_field_card(
            field_key,
            title,
            field,
            workspace_id=workspace_id,
            source_loader=source_loader,
        )


def _render_field_row(
    first: tuple[str, str, OverviewField],
    second: tuple[str, str, OverviewField],
    *,
    workspace_id: str,
    source_loader: Callable[[str, Sequence[str]], Sequence[Any]],
) -> None:
    """在宽屏下并排放置两个概览字段，窄屏时由 Streamlit 自适应堆叠。"""
    left, right = st.columns(2, gap="small")
    with left:
        _render_field_card(first[0], first[1], first[2], workspace_id=workspace_id, source_loader=source_loader)
    with right:
        _render_field_card(second[0], second[1], second[2], workspace_id=workspace_id, source_loader=source_loader)


def _render_field_card(
    field_key: str,
    title: str,
    field: OverviewField,
    *,
    workspace_id: str,
    source_loader: Callable[[str, Sequence[str]], Sequence[Any]],
) -> None:
    """渲染单个字段，正文保留 Streamlit Markdown/KaTeX 公式能力。"""
    with st.container(border=True, key=f"paper-overview-card-{field_key}"):
        source_sections = tuple(getattr(field, "source_sections", ()) or ())
        source_label = _short_source_label(source_sections[0]) if source_sections else ""
        source_chip = (
            f"<span class='pb-overview-source-chip'>{escape(source_label)}</span>"
            if source_label
            else ""
        )
        st.markdown(
            f"<div class='pb-overview-card-head'><div class='pb-overview-card-title'>{title}</div>"
            f"{source_chip}</div>",
            unsafe_allow_html=True,
        )
        # 不把正文拼进 HTML，交给 Streamlit Markdown 处理列表、粗体和 LaTeX。
        st.markdown(field.content)
        if field.source_chunk_ids:
            _render_sources(
                field_key,
                field,
                workspace_id=workspace_id,
                source_loader=source_loader,
            )


def _render_sources(
    field_key: str,
    field: OverviewField,
    *,
    workspace_id: str,
    source_loader: Callable[[str, Sequence[str]], Sequence[Any]],
) -> None:
    """在字段卡片内部折叠展示来源章节、页码和原文。"""
    source_count = len(field.source_chunk_ids)
    try:
        sources = tuple(source_loader(workspace_id, field.source_chunk_ids))
    except Exception:  # noqa: BLE001 - 单个来源失败不应阻塞整页概览。
        with st.expander(f"查看来源（{source_count}）", expanded=False):
            st.caption("来源暂时无法读取。")
        return
    render_source_expander(
        label=f"查看来源（{source_count}）",
        sources=sources,
        state_prefix=f"overview-source-expanded::{workspace_id}::{field_key}",
        source_key_prefix=f"paper-overview-source-{field_key}",
    )


def _short_source_label(section: str) -> str:
    """概览卡片只展示末级 section；完整路径仍保留在“查看来源”中。"""
    normalized = str(section or "").strip()
    if not normalized:
        return ""
    parts = [part.strip() for part in normalized.split(">") if part.strip()]
    return parts[-1] if parts else normalized
