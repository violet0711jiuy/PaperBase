"""Knowledge Base 页面：三栏工作台、多会话聊天和用户可读 Evidence。"""

from __future__ import annotations

from datetime import datetime
from html import escape
import re

import streamlit as st

if __package__ == "app.pages":
    from app.services.paperbase_service import (
        FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
        KnowledgeBaseConversation,
        KnowledgeBaseEvidence,
        KnowledgeBaseTurnMetadata,
        PaperBaseService,
    )
else:
    from services.paperbase_service import (
        FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
        KnowledgeBaseConversation,
        KnowledgeBaseEvidence,
        KnowledgeBaseTurnMetadata,
        PaperBaseService,
    )


def _render_conversation(service: PaperBaseService, conversation_id: str) -> None:
    """从 ConversationStore 读取并渲染完整 User/Assistant 历史。

    每一轮消息都是一组普通的 ``st.columns``：用户是“留白 / 气泡 / 头像”，
    PaperBase 是“头像 / 气泡 /
    留白”。这样头像只由一个 Streamlit markdown 控件绘制，不需要隐藏或
    重排 Streamlit 内置头像。
    """
    try:
        turns = service.get_conversation_turns(conversation_id)
    except Exception as error:  # noqa: BLE001 - 会话被删除或 scope 错误时安全降级。
        st.warning(f"当前会话暂时无法读取：{_friendly_error(error)}")
        return

    if not turns:
        st.caption("这是一个新会话，输入问题后即可开始跨论文检索。")
        return

    for turn in turns:
        message_time = _conversation_timestamp(turn.created_at)

        # 用户消息：右侧气泡，头像单独放在最右列。
        user_spacer, user_bubble, user_avatar = st.columns(
            [0.42, 0.52, 0.06], gap="small", vertical_alignment="top"
        )
        user_spacer.empty()
        with user_bubble:
            with st.container(border=True, key=f"kb-user-bubble-{turn.turn_id}"):
                st.markdown(
                    "<div class='pb-message-header pb-user-header'>"
                    f"<span>你</span><span class='pb-message-time'>{escape(message_time)}</span></div>",
                    unsafe_allow_html=True,
                )
                _render_scientific_text(turn.user_query)
        with user_avatar:
            st.markdown(
                "<div class='pb-chat-avatar pb-user-avatar'>👤</div>",
                unsafe_allow_html=True,
            )

        # PaperBase 消息：左侧头像，中间回答气泡，右侧留白。
        assistant_avatar, assistant_bubble, assistant_spacer = st.columns(
            [0.055, 0.80, 0.145], gap="small", vertical_alignment="top"
        )
        with assistant_avatar:
            st.markdown(
                "<div class='pb-chat-avatar pb-assistant-avatar'>📚</div>",
                unsafe_allow_html=True,
            )
        with assistant_bubble:
            with st.container(border=True, key=f"kb-assistant-bubble-{turn.turn_id}"):
                st.markdown(
                    "<div class='pb-message-header pb-assistant-header'>"
                    "<span>PaperBase</span>"
                    f"<span class='pb-message-time'>{escape(message_time)}</span></div>",
                    unsafe_allow_html=True,
                )
                _render_answer_body(turn.assistant_answer)
                is_unresolved = turn.resolution_status == "unresolved"
                if is_unresolved:
                    st.info("这是后端根据当前上下文返回的澄清提示，请补充明确的论文或主题。")
                metadata = service.get_conversation_turn_metadata(turn)
                _render_answer_status(metadata, turn.assistant_answer, is_unresolved=is_unresolved)
                # Evidence 从 Service 的快照读取，不根据 [E#]/[R#] 重新检索。
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


def _render_scientific_text(text: str) -> None:
    r"""统一渲染普通文字和 KaTeX 科学公式。

    Streamlit Markdown 原生支持 ``$...$``/``$$...$$``。后端有时使用 LaTeX
    的 ``\(...\)``/``\[...\]`` 写法，这里只标准化定界符，不改动公式内容。
    未加定界符的复杂度表达式 ``O(N^2)`` 也会被包成行内公式。
    """
    st.markdown(_normalize_scientific_text(text))


def _normalize_scientific_text(text: str) -> str:
    """把常见 LaTeX 定界符转换为 Streamlit Markdown/KaTeX 可识别形式。"""
    normalized = text or ""
    normalized = re.sub(r"\\\((.*?)\\\)", r"$\1$", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", normalized, flags=re.DOTALL)
    # 只处理没有处在 $ 定界符中的常见复杂度表达式，避免重复包裹已有公式。
    normalized = re.sub(
        r"(?<![$\\])\bO\([^\n)]{1,80}\)",
        lambda match: f"${match.group(0)}$",
        normalized,
    )
    return normalized


def _split_answer_sections(answer: str) -> dict[str, str]:
    """识别现有后端答案中的小标题；无标题的降级答案保持完整显示。"""
    sections = {"direct": "", "explanation": "", "interpretation": "", "coverage": ""}
    # 兼容历史答案把下一个 ``###`` 标题拼在同一行的情况；先把它规范到新行，
    # 这样前端不会把 Markdown 标记原样暴露给用户。
    normalized_answer = re.sub(r"[ \t]+###\s+", "\n### ", answer.strip())
    matches = list(re.finditer(r"(?m)^###\s+(.+?)\s*$", normalized_answer))
    if not matches:
        sections["direct"] = normalized_answer
        return sections

    heading_map = {
        "直接回答": "direct",
        "论文中的依据与推导": "explanation",
        "如何理解": "interpretation",
        "证据覆盖说明": "coverage",
        "direct answer": "direct",
        "evidence explanation": "explanation",
        "reading interpretation": "interpretation",
        "coverage note": "coverage",
    }
    unknown_parts: list[str] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        body_start = match.end()
        body_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(normalized_answer)
        )
        body = normalized_answer[body_start:body_end].strip()
        section_key = heading_map.get(heading.casefold(), heading_map.get(heading))
        if section_key is None:
            unknown_parts.append(f"{heading}\n{body}" if body else heading)
        elif body:
            sections[section_key] = (
                f"{sections[section_key]}\n\n{body}" if sections[section_key] else body
            )
    if unknown_parts:
        unknown_text = "\n\n".join(unknown_parts)
        sections["direct"] = (
            f"{sections['direct']}\n\n{unknown_text}" if sections["direct"] else unknown_text
        )
    return sections


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
                preview, truncated = _truncate_evidence(item.text)
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


def _truncate_evidence(text: str, limit: int = 620) -> tuple[str, bool]:
    """默认预览最多两个完整句子；没有句号时再使用安全字符上限。"""
    normalized = " ".join((text or "").split())
    # 中英文论文原文常混用标点；优先保留完整句子，避免在术语中间截断。
    pieces = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*|(?<=[.!?])\s+", normalized) if part.strip()]
    if pieces:
        preview = " ".join(pieces[:2]).strip()
        if preview and len(pieces) > 2:
            return f"{preview}…", True
        if preview and len(normalized) <= limit:
            return normalized, False
        if preview and len(preview) <= limit:
            return f"{preview}…", True
    if len(normalized) <= limit:
        return normalized, False
    return f"{normalized[:limit].rstrip()}…", True


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
    """从已保存答案正文提取已出现的 E#/R#，仅用于标记对应 Evidence。"""
    seen: set[str] = set()
    citations: list[str] = []
    for citation in re.findall(r"\[([ER]\d+)\]", answer):
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
        "<div class='pb-pane-kicker'>当前工作区</div>"
        "<div class='pb-pane-title'>知识库</div>"
        "<div class='pb-pane-subtitle'>跨论文检索与问答</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='pb-context-stat'><span>▤</span>"
        f"{len(papers)} 篇论文</div>",
        unsafe_allow_html=True,
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
        "<div class='pb-panel-divider'></div><div class='pb-context-section-title'>会话</div>",
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

    with st.expander(f"已入库论文（{len(papers)}）", expanded=False):
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
    st.markdown(
        "<div class='pb-chat-topline'><div><div class='pb-chat-title'>"
        f"{title}</div><div class='pb-chat-subtitle'>"
        "Knowledge Base · 跨论文检索、比较并追问</div></div>"
        "<div class='pb-chat-status'>◷&nbsp; 历史自动恢复</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='pb-chat-divider'></div>", unsafe_allow_html=True)
    # 只保留 Streamlit 原生 height container 作为唯一的消息滚动层。
    with st.container(border=False, key="kb-message-scroll"):
        if conversation_id:
            _render_conversation(service, conversation_id)
        else:
            st.markdown(
                "<div class='pb-empty-chat'><div class='pb-empty-icon'>✦</div>"
                "<div class='pb-empty-title'>从一个新会话开始</div>"
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
                placeholder="📎  向知识库提问，例如：ESDTW 和 DTW 有什么区别？",
                label_visibility="collapsed",
                disabled=conversation_id is None,
            )
        with send_column:
            submitted = st.form_submit_button(
                "➤",
                type="primary",
                use_container_width=True,
                disabled=conversation_id is None,
            )
    if submitted and query.strip() and conversation_id:
        try:
            with st.spinner("正在检索论文并生成回答…"):
                service.ask_knowledge_base(query, conversation_id)
        except Exception as error:  # noqa: BLE001 - 页面不直接暴露 traceback。
            st.error(_friendly_error(error))
        else:
            st.rerun()
