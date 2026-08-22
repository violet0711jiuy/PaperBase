"""Explain Section 结果的用户可读展示组件。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re
from typing import Any

import streamlit as st

if __package__ == "app.components":
    from app.components.scientific_text import render_scientific_text
    from app.components.source_evidence import render_source_expander
else:
    from components.scientific_text import render_scientific_text
    from components.source_evidence import render_source_expander


_OVERVIEW_FIELDS = (
    "本章目标与范围",
    "整体流程",
    "子章节职责",
    "章节关系与作用",
)
_DETAIL_FIELDS = (
    "本节任务",
    "关键对象与输入输出",
    "处理过程与判断条件",
    "公式、符号或结果如何阅读",
    "设计作用与边界",
)


def render_section_explanation(
    explanation: Any,
    *,
    workspace_id: str,
    section: Any,
    direct_children: Sequence[Any],
    source_loader: Callable[[str, Sequence[str]], Sequence[Any]],
) -> None:
    """按 backend 返回的 mode 渲染父章节概览或叶章节详细解释。"""
    is_overview = explanation.mode == "section_overview"
    st.markdown(
        f"<div class='pb-explain-title'>{explanation.section_title}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='pb-explain-mode'>{'章节概览' if is_overview else '详细解释'}</div>",
        unsafe_allow_html=True,
    )
    if explanation.insufficient_evidence:
        st.info("该章节没有足够正文内容可供解释。")
    elif is_overview:
        _render_parent_explanation(explanation, direct_children)
    else:
        _render_leaf_explanation(explanation)

    source_ids = tuple(explanation.source_chunk_ids)
    if source_ids:
        try:
            sources = tuple(source_loader(workspace_id, source_ids))
        except Exception:  # noqa: BLE001 - 来源失败不阻塞已有解释正文。
            sources = ()
            st.caption("来源暂时无法读取。")
        render_source_expander(
            label=f"查看来源（{len(source_ids)}）",
            sources=sources,
            state_prefix=f"explain-source-expanded::{workspace_id}::{explanation.section_id}",
            source_key_prefix=f"explain-source-{explanation.section_id}",
        )


def _render_parent_explanation(explanation: Any, direct_children: Sequence[Any]) -> None:
    """父章节使用确定性的 direct children 结构辅助理解。"""
    _render_subheading("章节结构")
    if direct_children:
        for child in direct_children:
            st.markdown(f"- {child.title}")
    else:
        st.caption("当前章节没有直属子节。")
    _render_subheading("本章主要内容")
    _render_template_explanation(explanation.explanation, _OVERVIEW_FIELDS)
    if explanation.key_points:
        _render_subheading("整体逻辑")
        _render_key_points(explanation.key_points)


def _render_leaf_explanation(explanation: Any) -> None:
    """叶章节以不改变原意的前端分点方式展示正文和要点。"""
    _render_subheading("本节作用与核心过程")
    _render_template_explanation(explanation.explanation, _DETAIL_FIELDS)
    if explanation.key_points:
        _render_subheading("关键要点")
        _render_key_points(explanation.key_points)


def _render_template_explanation(text: str, fields: Sequence[str]) -> None:
    """按 Explain 的两套固定 JSON 文本模板渲染带标题的阅读要点。"""
    template_points = _extract_template_points(text, fields)
    if template_points:
        for index, (title, content) in enumerate(template_points, start=1):
            st.markdown(
                f"<div class='pb-explain-point-title'>{index}. {title}</div>",
                unsafe_allow_html=True,
            )
            # 内容仍由科学文本 renderer 处理，公式与普通 Markdown 不会退化成源码。
            render_scientific_text(content)
        return

    # 兼容旧 artifact：旧结果没有使用固定标签时仍能以编号方式阅读，
    # 不要求用户重新生成已经保存的解释。
    _render_explanation_points(text)


def _extract_template_points(
    text: str, fields: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """从固定的 ``字段名：内容`` 纯文本模板中提取字段与对应说明。"""
    normalized = (text or "").strip()
    if not normalized:
        return ()

    pattern = re.compile(
        "(?P<title>" + "|".join(re.escape(field) for field in fields) + r")\s*[：:]"
    )
    matches = tuple(pattern.finditer(normalized))
    if not matches:
        return ()

    contents_by_title: dict[str, str] = {}
    for index, match in enumerate(matches):
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        content = normalized[match.end() : content_end].strip()
        if content:
            contents_by_title[match.group("title")] = content
    return tuple(
        (field, contents_by_title[field])
        for field in fields
        if contents_by_title.get(field)
    )


def _render_explanation_points(text: str) -> None:
    """兼容旧自由文本：按段落或语义提示分点，不改变后端内容。"""
    points = _split_explanation_points(text)
    if not points:
        return

    # 统一交给 scientific renderer，确保每个条目中的 LaTeX 公式仍可正常显示。
    ordered_markdown = "\n\n".join(
        f"{index}. {point}" for index, point in enumerate(points, start=1)
    )
    render_scientific_text(ordered_markdown)


def _render_key_points(points: Sequence[str]) -> None:
    """一次性渲染要点列表，避免多个独立 Markdown 块显得零散。"""
    cleaned_points = [_strip_list_marker(point) for point in points if point.strip()]
    if not cleaned_points:
        return
    render_scientific_text("\n".join(f"- {point}" for point in cleaned_points))


def _split_explanation_points(text: str) -> tuple[str, ...]:
    """从 artifact 的自由文本中提取稳定的展示点。

    新产物通常保留段落或换行；旧产物可能已经被服务层压成单段，
    此时只在常见的中文语义标签前切分，避免按每一句机械拆散论述。
    """
    normalized = (text or "").strip()
    if not normalized:
        return ()

    # 优先使用现有换行：它最接近模型输出的原始段落边界。
    line_points = tuple(
        _strip_list_marker(line.strip())
        for line in normalized.splitlines()
        if line.strip()
    )
    if len(line_points) > 1:
        return line_points

    # 兼容已被压缩成一段的历史 artifact。只识别说明性标签，
    # 不拆普通句子，以保留公式、因果关系和完整上下文。
    semantic_boundary = re.compile(
        r"(?=(?:本[章节](?:任务|目标|作用|内容)?|核心(?:概念|方法|思路)|"
        r"处理(?:流程|过程)|关键(?:对象|步骤|公式|结果|发现)|"
        r"设计(?:作用与边界|作用|边界|动机)|输入(?:与输出)?|输出|实验(?:结果|设置)?|"
        r"局限(?:性)?|背景|总结)\s*[：:])"
    )
    semantic_points = tuple(
        _strip_list_marker(item.strip())
        for item in semantic_boundary.split(normalized)
        if item.strip()
    )
    return semantic_points or (normalized,)


def _strip_list_marker(text: str) -> str:
    """去掉已有项目符号，避免渲染后出现重复编号或重复圆点。"""
    return re.sub(r"^(?:[-*•]\s+|\d+[.、)]\s*)", "", text).strip()


def _render_subheading(title: str) -> None:
    st.markdown(f"<div class='pb-explain-subheading'>{title}</div>", unsafe_allow_html=True)
