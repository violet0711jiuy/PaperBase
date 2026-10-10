"""Knowledge Base 页面：三栏工作台、多会话聊天和用户可读 Evidence。"""

from __future__ import annotations

from datetime import datetime
from html import escape
import re

import streamlit as st

from paperbase.agentic.audit import AgenticAuditSnapshot

if __package__ == "app.pages":
    from app.components.answer_sections import split_answer_sections as _split_answer_sections
    from app.components.evidence_preview import build_evidence_preview
    from app.components.scientific_text import (
        normalize_scientific_text as _normalize_scientific_text,
        render_scientific_text as _render_scientific_text,
    )
    from app.services.paperbase_service import (
        FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
        KnowledgeBaseConversation,
        KnowledgeBaseEvidence,
        KnowledgeBaseTurnMetadata,
        PaperBaseService,
    )
else:
    from components.answer_sections import split_answer_sections as _split_answer_sections
    from components.evidence_preview import build_evidence_preview
    from components.scientific_text import (
        normalize_scientific_text as _normalize_scientific_text,
        render_scientific_text as _render_scientific_text,
    )
    from services.paperbase_service import (
        FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
        KnowledgeBaseConversation,
        KnowledgeBaseEvidence,
        KnowledgeBaseTurnMetadata,
        PaperBaseService,
    )


def _avatar_html(kind: str) -> str:
    """返回稳定的内联 SVG 头像，避免不同系统 emoji 造成布局跳动。"""
    if kind == "user":
        return (
            "<div class='pb-avatar pb-avatar-user' aria-label='用户头像'>"
            "<svg viewBox='0 0 24 24' aria-hidden='true'>"
            "<circle cx='12' cy='8' r='4.1' fill='currentColor'/>"
            "<path d='M4.8 20c.45-4.2 3.05-6.3 7.2-6.3s6.75 2.1 7.2 6.3' "
            "fill='currentColor'/></svg></div>"
        )
    return (
        "<div class='pb-avatar pb-avatar-assistant' aria-label='PaperBase 头像'>"
        "<svg viewBox='0 0 28 28' aria-hidden='true'>"
        "<rect class='sq-a' x='4' y='4' width='10' height='13' rx='1.6'/>"
        "<rect class='sq-b' x='9' y='7' width='10' height='13' rx='1.6'/>"
        "<rect class='sq-c' x='14' y='10' width='10' height='13' rx='1.6'/>"
        "</svg></div>"
    )


def _render_conversation(service: PaperBaseService, conversation_id: str) -> None:
    """恢复完整历史，并按右侧用户 / 左侧 PaperBase 的聊天布局展示。"""
    try:
        turns = service.get_conversation_turns(conversation_id)
    except Exception as error:  # noqa: BLE001 - 会话失效时安全降级。
        st.warning(f"当前会话暂时无法读取：{_friendly_error(error)}")
        return

    if not turns:
        st.caption("这是一个新会话，输入问题后即可开始跨论文检索。")
        return

    for turn in turns:
        message_time = _conversation_timestamp(turn.created_at)

        # 用户消息：名称/时间放在气泡上方，头像与气泡属于同一行。
        user_spacer, user_group, user_avatar = st.columns(
            [0.35, 0.59, 0.06], gap="small", vertical_alignment="top"
        )
        user_spacer.empty()
        with user_group:
            st.markdown(
                "<div class='pb-message-header pb-user-header'>"
                f"<span>你</span><span class='pb-message-time'>{escape(message_time)}</span></div>",
                unsafe_allow_html=True,
            )
            with st.container(border=False, key=f"kb-user-bubble-{turn.turn_id}"):
                _render_scientific_text(turn.user_query)
        with user_avatar:
            st.markdown(_avatar_html("user"), unsafe_allow_html=True)

        # PaperBase 消息：头像在左，名称/时间位于回答卡片上方。
        assistant_avatar, assistant_group, assistant_spacer = st.columns(
            [0.06, 0.84, 0.10], gap="small", vertical_alignment="top"
        )
        with assistant_avatar:
            st.markdown(_avatar_html("assistant"), unsafe_allow_html=True)
        with assistant_group:
            st.markdown(
                "<div class='pb-message-header pb-assistant-header'>"
                "<span>PaperBase</span>"
                f"<span class='pb-message-time'>{escape(message_time)}</span></div>",
                unsafe_allow_html=True,
            )
            with st.container(border=False, key=f"kb-assistant-bubble-{turn.turn_id}"):
                _render_answer_body(turn.assistant_answer)
                is_unresolved = turn.resolution_status == "unresolved"
                if is_unresolved:
                    st.info("这是后端根据当前上下文返回的澄清提示，请补充明确的论文或主题。")
                metadata = service.get_conversation_turn_metadata(turn)
                _render_answer_status(metadata, turn.assistant_answer, is_unresolved=is_unresolved)
                audit = service.get_conversation_turn_audit(turn)
                _render_agentic_audit(
                    audit,
                    audit_json=(
                        service.export_conversation_turn_audit(turn)
                        if audit is not None
                        else None
                    ),
                    turn_id=turn.turn_id,
                )
                evidence = service.get_conversation_turn_evidence(turn)
                _render_evidence(
                    evidence,
                    conversation_id=conversation_id,
                    turn_id=turn.turn_id,
                    citations=_citation_ids(turn.assistant_answer),
                )
        assistant_spacer.empty()

def _render_answer_body(answer: str) -> None:
    """把后端固定排版的答案拆成更易读的直接回答、说明和理解层级。"""
    sections = _split_answer_sections(answer)
    if sections["direct"]:
        _render_answer_section("1. 直接回答", sections["direct"])
    if sections["explanation"]:
        _render_answer_section("2. 论文中的依据与推导", sections["explanation"])
    if sections["interpretation"]:
        _render_answer_section("3. 阅读理解", sections["interpretation"], muted=True)
    if sections["coverage"]:
        _render_answer_section("4. 覆盖提示", sections["coverage"], muted=True)


def _render_answer_section(title: str, body: str, *, muted: bool = False) -> None:
    """渲染一个小标题和普通正文，避免整段回答都呈现为粗体标题。"""
    class_name = "pb-answer-section pb-answer-section-muted" if muted else "pb-answer-section"
    st.markdown(
        f"<div class='{class_name}'><div class='pb-answer-section-title'>"
        f"{escape(title)}</div></div>",
        unsafe_allow_html=True,
    )
    _render_scientific_text(body)



def _render_answer_status(
    metadata: KnowledgeBaseTurnMetadata | None,
    answer: str,
    *,
    is_unresolved: bool,
) -> None:
    """把后端降级状态转成用户可读提示，不展示内部检索过程。"""
    if metadata is None:
        return
    if metadata.insufficient_evidence and not is_unresolved and "证据不足" not in answer:
        st.warning("当前论文证据不足以支持完整回答，请尝试补充更具体的问题。")
    if metadata.coverage_note and "证据覆盖说明" not in answer:
        _render_answer_section("4. 覆盖提示", metadata.coverage_note, muted=True)


def _render_agentic_audit(
    audit: AgenticAuditSnapshot | None,
    *,
    audit_json: str | None,
    turn_id: str,
) -> None:
    """显示可持久化的研究过程，不暴露 Prompt、隐藏推理或原始 Graph State。"""
    if audit is None:
        return
    status_text, status_level = _audit_status_message(audit)
    if status_level == "success":
        st.success(status_text)
    elif status_level == "warning":
        st.warning(status_text)
    else:
        st.error(status_text)

    with st.expander("查看研究过程与归因", expanded=False):
        _render_audit_summary(audit)
        _render_requirement_audit(audit)
        _render_candidate_audit(audit)
        _render_paper_research_audit(audit)
        _render_research_timeline(audit)
        _render_rejected_actions(audit)
        _render_claim_audit(audit)
        _render_advanced_diagnostics(audit)
        if audit_json:
            st.download_button(
                "导出脱敏审计 JSON",
                data=audit_json,
                file_name=f"paperbase-agentic-audit-{turn_id}.json",
                mime="application/json",
                key=f"kb-audit-download-{turn_id}",
            )


def _render_audit_summary(audit: AgenticAuditSnapshot) -> None:
    """展示研究结论状态与实际预算使用情况。"""
    st.markdown("#### 研究摘要")
    scope_label = _SCOPE_LABELS.get(audit.scope or "", audit.scope or "未记录")
    coverage_label = (
        "已完成"
        if audit.coverage_complete is True
        else "尚未完全覆盖"
        if audit.coverage_complete is False
        else "未记录"
    )
    st.markdown(
        f"范围：**{scope_label}** · 覆盖：**{coverage_label}** · "
        f"迭代：**{audit.budget.agent_iterations}** · "
        f"工具调用：**{audit.budget.tool_calls}** · "
        f"检索查询：**{audit.budget.retrieval_queries}** · "
        f"LLM 调用：**{audit.budget.llm_calls}**"
    )
    if audit.clarification:
        st.info(f"需要补充的信息：{audit.clarification}")


def _render_candidate_audit(audit: AgenticAuditSnapshot) -> None:
    """按处置结果展示论文元数据筛选，明确哪些论文没有进入正文检索。"""
    if not audit.candidates:
        return
    st.markdown("#### 论文筛选")
    for disposition in ("candidate", "uncertain", "excluded"):
        items = tuple(
            item for item in audit.candidates if item.disposition == disposition
        )
        if not items:
            continue
        st.markdown(f"**{_CANDIDATE_LABELS.get(disposition, disposition)}（{len(items)}）**")
        for item in items:
            st.markdown(f"- {item.title}：{item.reason}")


def _render_requirement_audit(audit: AgenticAuditSnapshot) -> None:
    """展示最终回答应满足的 Requirement 及其当前证据状态。"""
    if not audit.requirements:
        return
    st.markdown("#### 回答需求完成度")
    for item in audit.requirements:
        required = "是" if item.required else "否"
        evidence = ", ".join(item.evidence_ids) if item.evidence_ids else "—"
        st.markdown(
            f"**{item.requirement_id}**：{item.description}  "
            f"  \n必需：{required} · 状态：{item.status} · 支持证据：{evidence}"
        )


def _render_paper_research_audit(audit: AgenticAuditSnapshot) -> None:
    """逐论文展示动态 Query、Coverage 与尚未解决的研究缺口。"""
    if not audit.papers:
        return
    st.markdown("#### 逐论文研究")
    for paper in audit.papers:
        status = _PAPER_STATUS_LABELS.get(paper.status, paper.status)
        with st.container(border=True):
            st.markdown(f"**{paper.title}** · {status}")
            if paper.rationale:
                st.caption(paper.rationale)
            if paper.queries_tried:
                st.markdown("检索 Query：")
                for query in paper.queries_tried:
                    st.code(query, language=None)
            _render_tag_line("已覆盖目标", paper.covered_targets)
            _render_tag_line("未覆盖目标", paper.remaining_targets)
            _render_tag_line("涉及章节", paper.sections_seen)


def _render_research_timeline(audit: AgenticAuditSnapshot) -> None:
    """用确定性格式展示 Tool 执行和筛选结果，不再请求 LLM 总结。"""
    if not audit.steps:
        return
    st.markdown("#### 研究时间线")
    for step in audit.steps:
        status = _TOOL_STATUS_LABELS.get(step.tool_status, step.tool_status)
        st.markdown(
            f"**第 {step.iteration} 步 · {step.action_name}** — {status}；"
            f"新增 {step.added_count} 条，复用 {step.reused_count} 条"
        )
        for attempt in step.attempts:
            attempt_status = _TOOL_STATUS_LABELS.get(attempt.status, attempt.status)
            suffix = f"；错误：{attempt.error_code}" if attempt.error_code else ""
            st.markdown(
                f"- {attempt.title}：`{attempt.query}` → {attempt_status}{suffix}"
            )
            _render_tag_line("Requirement", attempt.requirement_ids)
            _render_tag_line("Lexical terms", attempt.lexical_terms)
        if step.action_arguments:
            st.caption("Tool 参数：")
            st.json(step.action_arguments, expanded=False)
        if step.filter_total:
            st.caption(
                "Evidence Filter："
                f"候选 {step.filter_total}，接纳 {step.filter_accepted}，"
                f"排除 {step.filter_rejected}，降权 {step.filter_deprioritized}"
            )
            if step.filter_reasons:
                st.caption(
                    "筛选原因："
                    + "、".join(
                        f"{_FILTER_REASON_LABELS.get(item.reason, item.reason)} × {item.count}"
                        for item in step.filter_reasons
                    )
                )
        if step.cache_hits or step.cache_misses or step.cache_writes or step.cache_errors:
            st.caption(
                "分析缓存："
                f"命中 {step.cache_hits}，未命中 {step.cache_misses}，"
                f"写入 {step.cache_writes}，错误 {step.cache_errors}"
            )


def _render_rejected_actions(audit: AgenticAuditSnapshot) -> None:
    if not audit.rejected_actions:
        return
    st.markdown("#### 被运行时校验拒绝的动作")
    for item in audit.rejected_actions:
        st.markdown(f"**第 {item.iteration} 步 · {item.action_name}** — 已拒绝")
        st.caption(f"{item.rejection_code}：{item.rejection_reason}")
        st.caption(f"阶段：{item.rejection_stage} · 恢复动作：{item.recovery_action}")
        _render_tag_line("Requirement", item.requirement_ids)
        if item.blockers:
            st.json({"blockers": item.blockers}, expanded=False)
        st.json(item.action_arguments, expanded=False)


def _render_claim_audit(audit: AgenticAuditSnapshot) -> None:
    """展示 Evidence Digest 与 Claim 级归属，防止跨论文错误归因。"""
    qualified = tuple(item for item in audit.evidence if item.selected or item.cited)
    if not qualified:
        return
    st.markdown("#### 结论与证据归因")
    for item in qualified:
        cited = " · 回答已引用" if item.cited else ""
        with st.container(border=True):
            st.markdown(f"**{item.evidence_id} · {item.paper_title}{cited}**")
            if item.digest:
                st.markdown(item.digest)
            _render_tag_line("覆盖目标", item.covered_targets)
            for claim in item.claims:
                relation = _RELATION_LABELS.get(claim.relation, claim.relation)
                ownership = _OWNERSHIP_LABELS.get(claim.ownership, claim.ownership)
                st.markdown(
                    f"- **{claim.claim_id}**：{claim.statement}  "
                    f"  \n  关系：{relation} · 归属：{ownership}"
                )
                if claim.rationale:
                    st.caption(f"判定理由：{claim.rationale}")
                if claim.source_excerpt:
                    st.caption(f"原文锚点：{claim.source_excerpt}")


def _render_advanced_diagnostics(audit: AgenticAuditSnapshot) -> None:
    """默认折叠技术诊断；记录数值和状态，不显示任何调用正文。"""
    if not audit.llm_calls and not audit.error_codes:
        return
    with st.container(border=True):
        st.markdown("##### 高级诊断")
        if audit.error_codes:
            st.markdown("错误代码：" + "、".join(audit.error_codes))
        for call in audit.llm_calls:
            outcome = "成功" if call.success else "失败"
            tokens = []
            if call.prompt_tokens is not None:
                tokens.append(f"输入 {call.prompt_tokens}")
            if call.completion_tokens is not None:
                tokens.append(f"输出 {call.completion_tokens}")
            token_text = " · Token " + "/".join(tokens) if tokens else ""
            detail = call.error_stage or call.finish_reason or "未记录结束原因"
            st.markdown(
                f"- {call.node} 第 {call.attempt} 次：{outcome} · {detail}{token_text}"
            )
            if call.recovery_action:
                st.caption(f"恢复动作：{call.recovery_action}")


def _render_tag_line(label: str, values: tuple[str, ...]) -> None:
    """把短列表统一渲染成单行，空列表不制造视觉噪声。"""
    if values:
        translated = tuple(_TARGET_LABELS.get(item, item) for item in values)
        st.markdown(f"{label}：" + "、".join(translated))


def _audit_status_message(audit: AgenticAuditSnapshot) -> tuple[str, str]:
    """将终止原因转换成稳定的用户提示和严重级别。"""
    reason = _STOP_REASON_MESSAGES.get(
        audit.stop_reason,
        "研究任务已经结束，可展开查看本轮完成的步骤和证据。",
    )
    if audit.run_status == "completed":
        return f"研究任务已完成。{reason}", "success"
    if audit.run_status in {"partial", "needs_clarification", "no_evidence"}:
        return f"研究任务未完全覆盖问题。{reason}", "warning"
    return f"研究任务未能完成。{reason}", "error"


_STOP_REASON_MESSAGES = {
    "agent_finished": "Agent 已完成证据收集和回答生成。",
    "budget_exhausted": "已达到本轮迭代、工具或查询预算，现有结果可能只是部分回答。",
    "scope_unresolved": "尚未确定需要研究的论文范围，请补充论文名称或选择论文。",
    "no_available_actions": "现有信息下已没有新的安全检索动作。",
    "invalid_action_limit": "Agent 连续提出未通过运行时校验的动作，系统已停止以避免重复调用。",
    "fatal_error": "运行时发生不可恢复错误，已保留完成的证据和步骤。",
}

_SCOPE_LABELS = {
    "single_paper": "单篇论文",
    "selected_papers": "指定论文",
    "knowledge_base": "整个知识库",
}

_CANDIDATE_LABELS = {
    "candidate": "候选论文",
    "uncertain": "待验证论文",
    "excluded": "已排除论文",
}

_PAPER_STATUS_LABELS = {
    "unvisited": "尚未检索",
    "candidate": "候选",
    "positive": "已找到支持证据",
    "no_support_found": "未找到支持证据",
    "uncertain": "证据仍不确定",
    "excluded": "已在元数据筛选阶段排除",
}

_TOOL_STATUS_LABELS = {
    "ok": "成功",
    "empty": "没有找到新证据",
    "partial": "部分成功",
    "error": "失败",
    "unavailable": "不可用",
}

_RELATION_LABELS = {
    "supports": "支持",
    "contradicts": "反驳",
    "background_only": "仅作背景",
    "uncertain": "不确定",
    "irrelevant": "无关",
}

_OWNERSHIP_LABELS = {
    "this_paper": "本文工作",
    "prior_work": "既有工作",
    "general_background": "一般背景",
    "unknown": "归属未知",
}

_TARGET_LABELS = {
    "paper_own_fact": "论文本身事实",
    "method": "方法",
    "result": "结果",
    "claim": "核心主张",
    "prior_work": "相关工作",
    "field_overview": "领域概览",
    "reference": "参考文献",
    "general": "一般信息",
}

_FILTER_REASON_LABELS = {
    "below_relevance_threshold": "相关度低于阈值",
    "duplicate_fingerprint": "证据内容重复",
    "overlapping_source_window": "来源窗口高度重叠",
    "table_fragment": "表格碎片信息不足",
    "deprioritized_table_fragment": "表格碎片已降权",
    "rescued_result_table": "结果表格保守保留",
}


def _render_evidence(
    evidence: tuple[KnowledgeBaseEvidence, ...],
    *,
    conversation_id: str,
    turn_id: str,
    citations: tuple[str, ...],
) -> None:
    """渲染可折叠 Evidence 卡片，并在当前卡片原地切换全文。"""
    if not evidence:
        return
    with st.expander(f"查看证据（{len(evidence)}）", expanded=False):
        for index, item in enumerate(evidence):
            is_cited = item.evidence_id in citations
            label = "参考文献" if item.kind == "bibliography" else "论文证据"
            cited_mark = " · 回答引用" if is_cited else ""
            with st.container(border=True):
                st.markdown(
                    f"<div class='pb-evidence-id'>{escape(item.evidence_id)} · "
                    f"{escape(label)}{escape(cited_mark)}</div>",
                    unsafe_allow_html=True,
                )
                meta_parts = [f"论文：{item.paper_title}"]
                if item.section:
                    meta_parts.append(f"章节：{item.section}")
                page_text = _format_pages(item)
                if page_text:
                    meta_parts.append(f"页码：{page_text}")
                st.markdown(
                    f"<div class='pb-evidence-meta'>{escape(' · '.join(meta_parts))}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='pb-evidence-text-label'>文献条目</div>"
                    if item.kind == "bibliography"
                    else "<div class='pb-evidence-text-label'>证据原文</div>",
                    unsafe_allow_html=True,
                )
                state_key = _evidence_state_key(conversation_id, turn_id, item.evidence_id)
                expanded = bool(st.session_state.get(state_key, False))
                preview, truncated = build_evidence_preview(item.text)
                _render_scientific_text(item.text if expanded else preview)
                if truncated:
                    # Streamlit 按钮触发 rerun；状态 key 同时包含会话、回合和 Evidence ID，
                    # 所以展开一条证据不会影响其它消息中的同名 E1/R1。
                    if st.button(
                        "收起" if expanded else "展开全文",
                        key=f"{state_key}::toggle",
                        type="secondary",
                    ):
                        st.session_state[state_key] = not expanded
                        st.rerun()
            if index < len(evidence) - 1:
                st.markdown("<div class='pb-evidence-gap'></div>", unsafe_allow_html=True)


def _evidence_state_key(conversation_id: str, turn_id: str, evidence_id: str) -> str:
    """生成每条证据独立的展开状态 key，避免跨会话/跨回合串状态。"""
    return f"kb_evidence_expanded::{conversation_id}::{turn_id}::{evidence_id}"



def _ensure_active_conversation(
    service: PaperBaseService,
    conversations: tuple[KnowledgeBaseConversation, ...],
) -> str | None:
    """校验 rerun 后保存的 active ID，防止旧会话或其它 scope 串入当前页面。"""
    active_id = st.session_state.kb_conversation_id
    known_ids = {item.conversation_id for item in conversations}
    if active_id is None:
        return None
    if active_id in known_ids:
        return active_id
    # 列表没有该 ID 时再读取一次元数据，区分“列表为空”和“scope 不匹配”。
    try:
        record = service.get_conversation(active_id)
        if (
            record.scope_type != "knowledge_base"
            or record.scope_id != FORMAL_KNOWLEDGE_BASE_SCOPE_ID
        ):
            raise ValueError("Conversation scope does not match the current knowledge base.")
    except Exception:
        st.session_state.kb_conversation_id = None
        st.warning("当前知识库会话已失效，请点击“新建会话”继续。")
        return None
    return active_id


def _conversation_label(conversation: KnowledgeBaseConversation) -> str:
    """给第二栏会话卡片显示标题、轮数和轻量时间，不显示内部 ID。"""
    timestamp = _conversation_timestamp(conversation.updated_at)
    if conversation.turn_count:
        meta = f"{conversation.turn_count} 轮 · {timestamp}"
    else:
        meta = f"尚未开始 · {timestamp}"
    return f"{conversation.title}\n{meta}"


def _conversation_name(conversation: KnowledgeBaseConversation) -> str:
    """返回主对话 Header 使用的单行会话名称。"""
    suffix = f"（{conversation.turn_count} 轮）" if conversation.turn_count else ""
    return f"{conversation.title}{(' ' + suffix) if suffix else ''}"


def _conversation_timestamp(value: str) -> str:
    """把 UTC ISO 时间压缩成卡片可读的本地时间或日期。"""
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        local_time = timestamp.astimezone()
    except (TypeError, ValueError, OverflowError):
        return ""
    now = datetime.now().astimezone()
    if local_time.date() == now.date():
        return local_time.strftime("%H:%M")
    if (now.date() - local_time.date()).days == 1:
        return "昨天"
    return local_time.strftime("%m-%d")


def _citation_ids(answer: str) -> tuple[str, ...]:
    """从已保存答案正文提取已出现的 E#/REF#，仅用于标记对应 Evidence。"""
    seen: set[str] = set()
    citations: list[str] = []
    for citation in re.findall(r"\[((?:E|REF)\d+)\]", answer):
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return tuple(citations)


def _format_pages(item: KnowledgeBaseEvidence) -> str | None:
    """页码缺失时返回 None，从 UI 中优雅省略 None。"""
    if item.page_start is None and item.page_end is None:
        return None
    if item.page_start is not None and item.page_end is not None:
        return (
            str(item.page_start)
            if item.page_start == item.page_end
            else f"{item.page_start}–{item.page_end}"
        )
    return str(item.page_start or item.page_end)


def _friendly_error(error: Exception) -> str:
    """把后端异常转换成不暴露 traceback、检索内部细节的提示。"""
    message = str(error).strip()
    if "Conversation" in message or "conversation" in message:
        return "当前会话不存在或已失效，请新建会话后重试。"
    if "scope" in message.lower():
        return "当前会话不属于这个知识库，请新建会话后重试。"
    if "问题不能为空" in message:
        return "问题不能为空，请输入后再发送。"
    if "暂停的研究任务" in message or "尚未完成" in message:
        return "当前会话已有未完成的研究任务，请先继续或放弃恢复入口。"
    if "Unknown research thread" in message or "没有可继续" in message:
        return "研究检查点不存在或已经失效，可以放弃恢复入口后重新提问。"
    # 模型路径、SQLite 路径和供应商原始报错不进入主 UI，避免暴露工程内部信息。
    return "知识库服务暂时不可用，请稍后重试。"


def render_context_panel(service: PaperBaseService) -> str | None:
    """渲染统一 App Shell 的第二栏：Knowledge Base Context Panel。"""
    # 论文和会话列表在第二栏读取；它们仍然通过 Service 访问后端。
    try:
        papers = service.list_knowledge_base_papers()
    except Exception as error:  # noqa: BLE001 - 元数据失败时保留聊天工作台空状态。
        papers = ()
        st.warning(f"暂时无法读取已入库论文：{_friendly_error(error)}")
    try:
        conversations = service.list_kb_conversations()
    except Exception as error:  # noqa: BLE001 - 会话列表失败时保留主对话区。
        conversations = ()
        st.error(f"暂时无法读取会话列表：{_friendly_error(error)}")

    active_conversation_id = _ensure_active_conversation(service, conversations)
    return _render_context_content(
        service,
        papers=papers,
        conversations=conversations,
        active_conversation_id=active_conversation_id,
    )


def _render_context_content(
    service: PaperBaseService,
    *,
    papers: tuple,
    conversations: tuple[KnowledgeBaseConversation, ...],
    active_conversation_id: str | None,
) -> str | None:
    """绘制 Knowledge Base 的会话、论文和新建会话控件。"""
    st.markdown(
        "<div class='pb-kb-context-heading'><div class='pb-pane-kicker'>当前工作区</div>"
        "<div class='pb-pane-title'>知识库</div>"
        "<div class='pb-pane-subtitle'>跨论文检索与问答</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='pb-context-stat'>已入库论文 · {len(papers)} 篇</div>",
        unsafe_allow_html=True,
    )
    st.radio(
        "问答模式",
        ("快速问答", "Agentic Research"),
        key="kb_research_mode",
        horizontal=True,
        help="快速问答执行一次检索；Agentic Research 会动态规划、分步检索并检查论文覆盖度。",
    )
    if st.session_state.get("kb_research_mode") == "Agentic Research":
        st.checkbox("检索整个知识库", key="kb_scope_all")
        if not st.session_state.get("kb_scope_all", True):
            st.multiselect(
                "指定论文",
                options=[paper.paper_id for paper in papers],
                format_func=lambda paper_id: next(
                    (
                        paper.display_title
                        for paper in papers
                        if paper.paper_id == paper_id
                    ),
                    paper_id,
                ),
                key="kb_selected_paper_ids",
            )
    with st.container(key="kb-new-chat"):
        if st.button(
            "＋  新建会话",
            type="primary",
            use_container_width=True,
            key="kb_new_chat_panel",
        ):
            try:
                record = service.create_kb_conversation()
            except Exception as error:  # noqa: BLE001 - 只向用户显示可读错误。
                st.error(_friendly_error(error))
            else:
                st.session_state.kb_conversation_id = record.conversation_id
                st.rerun()

    st.markdown(
        "<div class='pb-panel-divider'></div><div class='pb-context-section-title'>会话列表</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="kb-conversation-list"):
        if conversations:
            for conversation in conversations:
                is_active = conversation.conversation_id == active_conversation_id
                if st.button(
                    _conversation_label(conversation),
                    key=f"kb_conversation_{conversation.conversation_id}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.kb_conversation_id = conversation.conversation_id
                    st.rerun()
        else:
            st.caption("还没有会话。点击上方按钮开始。")

    # 已入库论文属于知识库上下文而非主聊天内容；保留折叠交互，避免在会话
    # 较多时抢占第二栏的可用高度。
    with st.container(key="kb-paper-library"):
        with st.expander(f"已入库论文（{len(papers)}）", expanded=True):
            if papers:
                for paper in papers:
                    st.markdown(
                        f"<div class='pb-paper-item'><div class='pb-paper-item-title'>"
                        f"{escape(paper.display_title)}</div><div class='pb-paper-item-meta'>"
                        f"{paper.total_chunk_count} 个分块</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("正式知识库中还没有论文。")
    return active_conversation_id


def render_main_panel(service: PaperBaseService) -> None:
    """渲染统一 App Shell 的第三栏：当前会话、历史和 Composer。"""
    try:
        conversations = service.list_kb_conversations()
    except Exception as error:  # noqa: BLE001 - 主面板安全降级。
        conversations = ()
        st.error(f"暂时无法读取会话列表：{_friendly_error(error)}")
    active_conversation_id = _ensure_active_conversation(service, conversations)
    conversation_by_id = {item.conversation_id: item for item in conversations}
    active_title = (
        _conversation_name(conversation_by_id[active_conversation_id])
        if active_conversation_id in conversation_by_id
        else None
    )
    _render_chat_panel(service, active_conversation_id, conversation_title=active_title)


def render(service: PaperBaseService) -> None:
    """兼容旧调用方；正式入口由 App Shell 分别渲染两个 panel。"""
    render_context_panel(service)
    render_main_panel(service)


def _render_chat_panel(
    service: PaperBaseService,
    conversation_id: str | None,
    *,
    conversation_title: str | None,
) -> None:
    """渲染第三栏主对话区和固定底部输入框。"""
    title = escape(conversation_title or "知识库对话")
    mode = st.session_state.get("kb_research_mode", "快速问答")
    mode_subtitle = (
        "Agentic Research · 动态规划、逐篇检索与可恢复执行"
        if mode == "Agentic Research"
        else "Knowledge Base · 跨论文检索、比较并追问"
    )
    st.markdown(
        "<div class='pb-chat-topline'><div><div class='pb-chat-title'>"
        f"{title}</div><div class='pb-chat-subtitle'>"
        f"{mode_subtitle}</div></div>"
        "<div class='pb-chat-status'>历史自动恢复</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='pb-chat-divider'></div>", unsafe_allow_html=True)
    pending = None
    if conversation_id:
        try:
            pending = service.get_pending_agentic_task(conversation_id)
        except Exception as error:  # noqa: BLE001 - 恢复入口失败不阻塞历史阅读。
            st.warning(f"暂时无法读取研究任务状态：{_friendly_error(error)}")
    if pending is not None and conversation_id is not None:
        status_label = {
            "running": "上次运行可能意外中断",
            "interrupted": "研究任务已暂停",
            "recovering": "研究任务正在恢复",
        }.get(pending.status, "研究任务尚未完成")
        st.warning(f"{status_label}：{pending.user_query}")
        resume_column, discard_column = st.columns(2)
        with resume_column:
            resume_clicked = st.button(
                "从检查点继续",
                type="primary",
                use_container_width=True,
                key=f"kb-resume-{pending.thread_id}",
            )
        with discard_column:
            discard_clicked = st.button(
                "放弃恢复入口",
                type="secondary",
                use_container_width=True,
                key=f"kb-discard-{pending.thread_id}",
            )
        if resume_clicked:
            try:
                with st.status("正在恢复研究任务…", expanded=True) as progress:
                    completed = False
                    for event in service.resume_knowledge_base_agentic(conversation_id):
                        progress.update(label=event.message)
                        if event.kind == "completed":
                            completed = True
                    progress.update(
                        label="研究任务已完成" if completed else "研究任务再次暂停",
                        state="complete" if completed else "error",
                    )
            except Exception as error:  # noqa: BLE001 - 保留恢复入口供再次处理。
                st.error(_friendly_error(error))
            else:
                st.rerun()
        if discard_clicked:
            try:
                service.discard_pending_agentic_task(conversation_id)
            except Exception as error:  # noqa: BLE001 - 不泄漏 checkpoint 路径。
                st.error(_friendly_error(error))
            else:
                st.rerun()
    # 只保留 Streamlit 原生 height container 作为唯一的消息滚动层。
    with st.container(height=640, border=False, key="kb-message-scroll"):
        if conversation_id:
            _render_conversation(service, conversation_id)
        else:
            st.markdown(
                "<div class='pb-empty-chat'><div class='pb-empty-title'>从一个新会话开始</div>"
                "<div class='pb-empty-copy'>在左侧创建会话，然后向知识库提问。</div></div>",
                unsafe_allow_html=True,
            )

    # Composer 在消息容器外部，因此历史消息滚动时不会把输入框带走。
    query = ""
    submitted = False
    with st.form("kb_composer", clear_on_submit=True, border=False):
        input_column, send_column = st.columns((1, 0.09), gap="small", vertical_alignment="center")
        with input_column:
            query = st.text_input(
                "向知识库提问",
                placeholder="向知识库提问，例如：ESDTW 和 DTW 有什么区别？",
                label_visibility="collapsed",
                disabled=conversation_id is None or pending is not None,
            )
        with send_column:
            submitted = st.form_submit_button(
                "➤",
                type="primary",
                use_container_width=True,
                disabled=conversation_id is None or pending is not None,
            )
    if submitted and query.strip() and conversation_id:
        try:
            if mode == "Agentic Research":
                with st.status("研究任务正在启动…", expanded=True) as progress:
                    completed = False
                    for event in service.stream_knowledge_base_agentic(
                        query,
                        conversation_id,
                        selected_paper_ids=tuple(
                            st.session_state.get("kb_selected_paper_ids", [])
                        ),
                        knowledge_base_all=bool(
                            st.session_state.get("kb_scope_all", True)
                        ),
                    ):
                        progress.update(label=event.message)
                        if event.kind == "completed":
                            completed = True
                    if completed:
                        progress.update(label="研究任务已完成", state="complete")
                    else:
                        progress.update(label="研究任务已暂停", state="error")
            else:
                with st.spinner("正在检索论文并生成回答…"):
                    service.ask_knowledge_base(query, conversation_id)
        except Exception as error:  # noqa: BLE001 - 页面不直接暴露 traceback。
            st.error(_friendly_error(error))
        else:
            st.rerun()
