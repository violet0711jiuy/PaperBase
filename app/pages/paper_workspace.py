"""Paper Workspace 页面。

本模块只提供统一 App Shell 中的第二栏和第三栏内容：第二栏负责选择
staging workspace、上传入口和当前 workspace 状态；第三栏负责论文摘要
和 Paper Overview、Explain Section、Ask This Paper。
"""

from __future__ import annotations

from html import escape

import streamlit as st

from paperbase.explain_section.service import ExplainSectionError
from paperbase.overview.service import PaperOverviewError
from paperbase.staging.sections import WorkspaceSectionError

if __package__ == "app.pages":
    from app.components.scientific_text import render_scientific_text
    from app.pages.knowledge_base import (
        _avatar_html,
        _citation_ids,
        _conversation_timestamp,
        _friendly_error,
        _render_answer_body,
        _render_answer_status,
        _render_evidence,
    )
    from app.services.paperbase_service import (
        PaperBaseService,
        PaperOverviewArtifactError,
        PaperUploadError,
        SectionExplanationArtifactError,
        WorkspaceSummary,
        WorkspaceActionError,
    )
    from app.components.explain_section import render_section_explanation
    from app.components.overview import render_paper_overview
    from app.components.section_tree import render_section_tree
    from app.state import (
        activate_workspace,
        clear_deleted_workspace,
        set_workspace_conversation,
    )
else:
    from components.scientific_text import render_scientific_text
    from pages.knowledge_base import (
        _avatar_html,
        _citation_ids,
        _conversation_timestamp,
        _friendly_error,
        _render_answer_body,
        _render_answer_status,
        _render_evidence,
    )
    from services.paperbase_service import (
        PaperBaseService,
        PaperOverviewArtifactError,
        PaperUploadError,
        SectionExplanationArtifactError,
        WorkspaceSummary,
        WorkspaceActionError,
    )
    from components.explain_section import render_section_explanation
    from components.overview import render_paper_overview
    from components.section_tree import render_section_tree
    from state import activate_workspace, clear_deleted_workspace, set_workspace_conversation


def render_context_panel(service: PaperBaseService) -> str | None:
    """绘制统一 App Shell 第二栏的 Workspace Context Panel。"""
    st.markdown(
        "<div class='pb-paper-context-heading'><div class='pb-pane-kicker'>当前工作区</div>"
        "<div class='pb-pane-title'>论文工作区</div>"
        "<div class='pb-pane-subtitle'>单篇论文阅读、解析与加入知识库</div></div>",
        unsafe_allow_html=True,
    )

    # workspace 列表来自 Service，页面不直接打开 staging 文件。
    try:
        workspaces = service.list_workspaces()
    except Exception as error:  # noqa: BLE001 - 页面显示可读错误。
        st.error(f"暂时无法读取论文工作区：{error}")
        workspaces = ()
    workspace_by_id = {workspace.workspace_id: workspace for workspace in workspaces}
    current_workspace_id = st.session_state.active_workspace_id
    # staging 目录被外部清理后，失效 ID 不再显示为“当前论文”。
    if current_workspace_id not in workspace_by_id:
        activate_workspace(st.session_state, None)
        current_workspace_id = None

    # 无论文时仍显示上传卡片作为空状态，但不默认展开 Streamlit 的大型
    # file uploader；用户点击后才展开，列表也能在首屏保持可见。
    upload_expanded = bool(st.session_state.get("upload_panel_expanded", False))
    # 上传入口保持在工作区面板顶部；没有论文时默认展开，有论文时默认收起，
    # 以免占据会话/工作区列表的阅读空间。
    with st.container(key="paper-upload-panel"):
        if current_workspace_id is None:
            st.markdown(
                "<div class='pb-upload-icon' aria-hidden='true'>"
                "<svg viewBox='0 0 24 24'><path d='M12 16V5m0 0-4 4m4-4 4 4'/>"
                "<path d='M5 14.5A4.5 4.5 0 0 0 5.5 23h12A4.5 4.5 0 0 0 18 14.03'/>"
                "</svg></div>"
                "<div class='pb-upload-panel-title'>上传论文</div>"
                "<div class='pb-upload-panel-copy'>支持 PDF 格式，或拖拽文件到此处</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='pb-upload-compact'><strong>添加新论文</strong>"
                "<span>上传 PDF 创建新的临时阅读工作区</span></div>",
                unsafe_allow_html=True,
            )
        upload_label = "收起上传区域" if upload_expanded else "＋ 上传新论文"
        if st.button(upload_label, type="primary", use_container_width=True, key="paper_upload_toggle"):
            st.session_state.upload_panel_expanded = not upload_expanded
            st.rerun()
        if upload_expanded:
            uploaded_pdf = st.file_uploader(
                "上传论文",
                type=["pdf"],
                help="上传后将创建独立 staging 工作区，不会自动加入正式知识库。",
                key="paper_upload",
            )
            if uploaded_pdf is not None:
                st.caption(f"已选择：{uploaded_pdf.name} · {uploaded_pdf.size / 1024 / 1024:.2f} MB")
                if st.button(
                    "解析并创建论文工作区",
                    type="primary",
                    use_container_width=True,
                    key="paper_upload_submit",
                ):
                    try:
                        # 页面只交付文件名和原始字节；Service 负责临时副本、后端
                        # staging pipeline 以及最终 workspace manifest 的读取。
                        with st.spinner("正在解析 PDF、生成分块并创建论文工作区…"):
                            workspace = service.create_workspace_from_pdf_upload(
                                filename=uploaded_pdf.name,
                                content=uploaded_pdf.getvalue(),
                            )
                    except PaperUploadError as error:
                        st.error(str(error))
                    except Exception:  # noqa: BLE001 - 不向页面暴露 parser/model traceback。
                        st.error("论文工作区创建失败，请稍后重试。")
                    else:
                        # 新 workspace 成功发布后立即切换到它；state.py 会清理旧章节
                        # 选择并恢复新论文自己的 Ask This Paper conversation scope。
                        activate_workspace(st.session_state, workspace.workspace_id)
                        st.session_state.upload_panel_expanded = False
                        st.rerun()

    st.markdown("<div class='pb-context-section-title'>最近论文工作区</div>", unsafe_allow_html=True)
    with st.container(key="paper-workspace-list"):
        if workspaces:
            for workspace in workspaces:
                is_active = workspace.workspace_id == current_workspace_id
                if st.button(
                    _workspace_card_label(workspace),
                    key=f"paper_workspace_{workspace.workspace_id}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    # 只在卡片点击时切换 scope；state.py 同时恢复该论文自己的会话。
                    activate_workspace(st.session_state, workspace.workspace_id)
                    st.rerun()
        else:
            st.caption("尚未发现 staging 论文。")

    active_workspace = workspace_by_id.get(current_workspace_id)
    if active_workspace is not None:
        _render_workspace_status_card(active_workspace)
    return current_workspace_id


def render_main_panel(service: PaperBaseService) -> None:
    """绘制统一 App Shell 第三栏的 Paper Main Panel。"""
    action_notice = st.session_state.pop("workspace_action_notice", None)
    if isinstance(action_notice, str) and action_notice:
        st.success(action_notice)
    workspace_id = st.session_state.active_workspace_id
    if not workspace_id:
        st.markdown(
            "<div class='pb-empty-panel'><div class='pb-empty-title'>选择一篇论文开始</div>"
            "<div class='pb-empty-copy'>在左侧工作区面板中选择已有的 staging 论文。</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Service 负责读取 workspace 视图模型；不存在时给出空状态而不是 traceback。
    try:
        workspace = service.get_workspace(workspace_id)
    except Exception as error:  # noqa: BLE001 - 页面安全降级。
        st.error(f"暂时无法读取当前论文：{error}")
        return
    if workspace is None:
        st.warning("当前 staging 工作区已不存在，请重新选择论文。")
        return

    _render_header(service, workspace)
    # Tab 使用稳定的文字标签和轻量图标；三个业务入口与原有状态/Service
    # 完全相同，仅调整为与工作台一致的视觉层级。
    overview_tab, explain_tab, ask_tab = st.tabs(
        ("论文概览", "解释章节", "询问本文")
    )
    with overview_tab:
        with st.container(
            height=720, border=False, key=f"paper-overview-scroll-{workspace.workspace_id}"
        ):
            _render_overview(service, workspace)
    with explain_tab:
        _render_explain_section(service, workspace)
    with ask_tab:
        _render_ask_this_paper(service, workspace)


def render(service: PaperBaseService) -> None:
    """兼容旧调用方；正式入口使用 App Shell 分别调用两个 panel。"""
    render_context_panel(service)
    render_main_panel(service)


def _workspace_option(workspace: WorkspaceSummary) -> str:
    """把 workspace 视图模型格式化为选择器中的用户可读标题。"""
    return workspace.display_title


def _workspace_card_label(workspace: WorkspaceSummary) -> str:
    """为第二栏工作区卡片生成标题、解析规模和状态，不展示内部 ID。"""
    status = "已加入知识库" if workspace.added_to_kb else "已解析"
    size = f"{workspace.section_count} 个章节 · {workspace.total_chunk_count} 个分块"
    return f"{workspace.display_title}\n{status} · {size}"


def _render_workspace_status_card(workspace: WorkspaceSummary) -> None:
    """显示当前论文的稳定摘要，不读取内部路径或 paper_id。"""
    status = "已加入" if workspace.added_to_kb else "暂存中"
    st.markdown(
        "<div class='pb-workspace-status-card'>"
        "<div class='pb-workspace-status-title'>论文状态</div>"
        f"<div class='pb-workspace-status-row'><span>状态</span><strong>{status}</strong></div>"
        f"<div class='pb-workspace-status-row'><span>章节数量</span><strong>{workspace.section_count}</strong></div>"
        f"<div class='pb-workspace-status-row'><span>分块数量</span><strong>{workspace.total_chunk_count}</strong></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_header(service: PaperBaseService, workspace: WorkspaceSummary) -> None:
    """绘制论文标题以及加入/删除当前临时论文的受控操作。"""
    title_column, action_column = st.columns((0.76, 0.24), gap="small")
    with title_column:
        st.markdown(
            "<div class='pb-paper-header'><div class='pb-paper-title'>"
            f"{escape(workspace.display_title)}</div>"
            "<div class='pb-paper-workspace-subtitle'>"
            "Paper Workspace · 单篇论文阅读、解释与问答</div></div>",
            unsafe_allow_html=True,
        )
    with action_column:
        _render_workspace_actions(service, workspace)


def _render_workspace_actions(service: PaperBaseService, workspace: WorkspaceSummary) -> None:
    """将 promotion 和 staging 删除动作集中在论文 Header 的状态位置。"""
    workspace_id = workspace.workspace_id
    with st.container(key=f"paper-workspace-actions-{workspace_id}"):
        if workspace.added_to_kb:
            st.markdown(
                "<div class='pb-paper-status-chip is-added'>✓ 已加入知识库</div>",
                unsafe_allow_html=True,
            )
        elif st.button(
            "＋ 加入知识库",
            type="primary",
            use_container_width=True,
            key=f"paper-add-to-kb-{workspace_id}",
        ):
            try:
                with st.spinner("正在加入正式知识库…"):
                    result = service.add_workspace_to_knowledge_base(workspace_id)
            except WorkspaceActionError as error:
                st.error(str(error))
            else:
                if result.status == "already_exists":
                    st.session_state.workspace_action_notice = "该论文已存在于正式知识库。"
                else:
                    st.session_state.workspace_action_notice = "论文已加入知识库。"
                st.rerun()

        if st.button(
            "删除该论文",
            type="secondary",
            use_container_width=True,
            key=f"paper-delete-{workspace_id}",
        ):
            st.session_state.workspace_delete_confirmation_id = workspace_id
            st.rerun()

        if st.session_state.get("workspace_delete_confirmation_id") != workspace_id:
            return
        st.warning("删除会移除当前临时工作区及其本地产物；已加入正式知识库的论文不会被删除。")
        confirm_column, cancel_column = st.columns(2, gap="small")
        with confirm_column:
            confirmed = st.button(
                "确认删除",
                type="primary",
                use_container_width=True,
                key=f"paper-delete-confirm-{workspace_id}",
            )
        with cancel_column:
            cancelled = st.button(
                "取消",
                use_container_width=True,
                key=f"paper-delete-cancel-{workspace_id}",
            )
        if cancelled:
            st.session_state.workspace_delete_confirmation_id = None
            st.rerun()
        if not confirmed:
            return
        try:
            service.delete_workspace(workspace_id)
        except WorkspaceActionError as error:
            st.error(str(error))
            return
        clear_deleted_workspace(st.session_state, workspace_id)
        st.session_state.workspace_action_notice = "临时论文工作区已删除。"
        st.rerun()

def _render_overview(service: PaperBaseService, workspace: WorkspaceSummary) -> None:
    """读取已保存概览；只有用户点击按钮时才触发一次生成。"""
    try:
        overview = service.get_paper_overview(workspace.workspace_id)
    except PaperOverviewArtifactError:
        st.error("论文概览文件损坏或无法读取，请重新生成。")
        _render_generate_button(service, workspace.workspace_id, label="重新生成论文概览")
        return
    except Exception:  # noqa: BLE001 - 页面显示可读恢复提示。
        st.error("论文概览暂时无法读取，请稍后重试。")
        _render_generate_button(service, workspace.workspace_id, label="重新生成论文概览")
        return

    if overview is None:
        st.markdown("**尚未生成论文概览**")
        st.caption("PaperBase 将基于论文结构提取研究问题、方法、贡献、数据集、实验与主要结果。")
        _render_generate_button(service, workspace.workspace_id, label="生成论文概览")
        return

    render_paper_overview(
        overview,
        workspace_id=workspace.workspace_id,
        source_loader=service.get_paper_overview_sources,
    )


def _render_generate_button(
    service: PaperBaseService, workspace_id: str, *, label: str
) -> None:
    """把耗时的 LLM 调用绑定到明确按钮，避免 rerun 时重复生成。"""
    if not st.button(label, type="primary", key=f"paper-overview-generate-{workspace_id}"):
        return
    try:
        with st.spinner("正在分析论文结构并生成概览…"):
            service.generate_paper_overview(workspace_id)
    except PaperOverviewError:
        st.error("论文概览生成失败，请稍后重试。")
        return
    except Exception:  # noqa: BLE001 - 不把 API/网络 traceback 暴露给用户。
        st.error("论文概览生成失败，请检查模型配置后重试。")
        return
    st.success("论文概览已生成。")
    st.rerun()


def _render_explain_section(service: PaperBaseService, workspace: WorkspaceSummary) -> None:
    """在 Explain Tab 内渲染 Section Tree 和章节解释双栏。"""
    try:
        sections = service.get_section_tree(workspace.workspace_id)
    except Exception:  # noqa: BLE001 - 坏工作区显示可读错误而不是 traceback。
        st.error("当前论文目录无法读取，请重新选择工作区。")
        return

    section_by_id = {section.section_id: section for section in sections}
    selected_id = st.session_state.get("selected_section_id")
    if selected_id not in section_by_id:
        if selected_id is not None:
            st.session_state.selected_section_id = None
            st.warning("当前章节已不存在，请重新选择章节。")
        selected_id = None

    # 两栏先提供一个稳定的 Streamlit 滚动容器；layout.css 会把它调整为
    # “视口高度 - Header/Tabs” 的剩余高度，避免 24 个章节把整页撑长。
    tree_column, explanation_column = st.columns([0.24, 0.76], gap="small")
    with tree_column:
        with st.container(height=650, border=False, key="explain-tree-scroll"):
            clicked_id = render_section_tree(sections, selected_id)
    if clicked_id is not None:
        st.session_state.selected_section_id = clicked_id
        st.rerun()

    with explanation_column:
        with st.container(height=650, border=False, key="explain-reader-scroll"):
            if selected_id is None:
                _render_explain_empty_state()
                return
            selected_section = section_by_id[selected_id]
            _render_selected_explanation(
                service,
                workspace_id=workspace.workspace_id,
                selected_section=selected_section,
                direct_children=tuple(
                    section
                    for section in sections
                    if section.parent_section_id == selected_section.section_id
                ),
            )


def _render_explain_empty_state() -> None:
    """首次进入 Explain Tab 时不触发任何 LLM 调用。"""
    st.markdown(
        "<div class='pb-explain-empty'><div class='pb-empty-title'>选择一个章节开始</div>"
        "<div class='pb-empty-copy'>从左侧论文目录选择章节。<br>"
        "选择大章节可查看整体结构，选择具体小节可获得详细解释。</div></div>",
        unsafe_allow_html=True,
    )


def _render_selected_explanation(
    service: PaperBaseService,
    *,
    workspace_id: str,
    selected_section: object,
    direct_children: tuple[object, ...],
) -> None:
    """恢复当前章节 artifact，或显示按真实 has_children 决定的生成入口。"""
    try:
        explanation = service.get_section_explanation(
            workspace_id, selected_section.section_id
        )
    except SectionExplanationArtifactError:
        st.error("章节解释文件损坏或无法读取，请重新生成。")
        _render_section_generate_button(
            service, workspace_id, selected_section, direct_children
        )
        return
    except WorkspaceSectionError:
        st.session_state.selected_section_id = None
        st.warning("当前章节已不存在，请重新选择章节。")
        return
    except Exception:  # noqa: BLE001 - 失效章节由上层目录状态处理。
        st.error("当前章节解释暂时无法读取，请重新选择章节。")
        return

    if explanation is not None:
        render_section_explanation(
            explanation,
            workspace_id=workspace_id,
            section=selected_section,
            direct_children=direct_children,
            source_loader=service.get_section_explanation_sources,
        )
        return

    is_parent = bool(selected_section.has_children)
    st.markdown(
        f"<div class='pb-explain-title'>{selected_section.title}</div>"
        f"<div class='pb-explain-mode'>{'章节概览' if is_parent else '详细解释'}</div>",
        unsafe_allow_html=True,
    )
    if not selected_section.has_content:
        st.info("该章节没有足够正文内容可供解释。")
        return
    if is_parent:
        st.caption("PaperBase 将结合本章及其子节结构生成整体概览。")
        label = "生成章节概览"
    else:
        st.caption("PaperBase 将仅基于该章节内容进行解释。")
        label = "解释该章节"
    _render_section_generate_button(
        service, workspace_id, selected_section, direct_children, label=label
    )


def _render_section_generate_button(
    service: PaperBaseService,
    workspace_id: str,
    section: object,
    direct_children: tuple[object, ...],
    *,
    label: str | None = None,
) -> None:
    """只在用户点击时调用 Explain backend，并在成功后从 artifact 恢复。"""
    is_parent = bool(section.has_children)
    button_label = label or ("重新生成章节概览" if is_parent else "重新解释该章节")
    if not st.button(
        button_label,
        type="primary",
        key=f"paper-explain-generate-{workspace_id}-{section.section_id}",
    ):
        return
    loading_text = (
        "正在分析章节及子节结构…" if is_parent else "正在解释当前小节…"
    )
    try:
        with st.spinner(loading_text):
            service.generate_section_explanation(workspace_id, section.section_id)
    except ExplainSectionError:
        st.error("章节解释生成失败，请稍后重试。")
        return
    except Exception:  # noqa: BLE001 - 不暴露模型供应商原始错误。
        st.error("章节解释生成失败，请稍后重试。")
        return
    st.rerun()


def _render_ask_this_paper(service: PaperBaseService, workspace: WorkspaceSummary) -> None:
    """渲染当前 staging 论文专属的多会话问答。"""
    st.markdown(
        "<div class='pb-ask-title'>询问本文</div>"
        "<div class='pb-ask-subtitle'>仅基于当前论文回答</div>"
        f"<div class='pb-ask-paper'>当前论文：{escape(workspace.display_title)}</div>",
        unsafe_allow_html=True,
    )
    try:
        conversations = service.list_paper_conversations(workspace.workspace_id)
    except Exception as error:  # noqa: BLE001 - 页面显示可读错误。
        conversations = ()
        st.error(f"暂时无法读取本文会话：{_paper_friendly_error(error)}")

    active_id = _ensure_paper_active_conversation(
        service, workspace.workspace_id, conversations
    )
    conversation_by_id = {item.conversation_id: item for item in conversations}
    selector_column, new_column = st.columns((0.72, 0.28), gap="small")
    with selector_column:
        if conversations:
            options = [item.conversation_id for item in conversations]
            selected_index = options.index(active_id) if active_id in options else 0
            selected_id = st.selectbox(
                "当前会话",
                options=options,
                index=selected_index,
                format_func=lambda item: _paper_conversation_label(
                    conversation_by_id[item]
                ),
                key=f"paper-conversation-selector-{workspace.workspace_id}",
            )
            if selected_id != active_id:
                set_workspace_conversation(
                    st.session_state,
                    workspace_id=workspace.workspace_id,
                    conversation_id=selected_id,
                )
                st.rerun()
        else:
            st.caption("尚未创建本文会话。")
    with new_column:
        if st.button(
            "＋ 新建会话",
            type="primary",
            use_container_width=True,
            key=f"paper-new-conversation-{workspace.workspace_id}",
        ):
            try:
                record = service.create_paper_conversation(workspace.workspace_id)
            except Exception as error:  # noqa: BLE001 - 不暴露底层 traceback。
                st.error(_paper_friendly_error(error))
            else:
                set_workspace_conversation(
                    st.session_state,
                    workspace_id=workspace.workspace_id,
                    conversation_id=record.conversation_id,
                )
                st.rerun()

    st.markdown("<div class='pb-ask-divider'></div>", unsafe_allow_html=True)
    with st.container(height=520, border=False, key=f"paper-message-scroll-{workspace.workspace_id}"):
        if active_id is None:
            st.markdown(
                "<div class='pb-empty-chat'><div class='pb-empty-title'>先创建一个会话</div>"
                "<div class='pb-empty-copy'>点击“＋ 新建会话”，然后向当前论文提问。</div></div>",
                unsafe_allow_html=True,
            )
        else:
            _render_paper_conversation(service, workspace.workspace_id, active_id)

    query = ""
    submitted = False
    with st.form(
        f"paper-composer-{workspace.workspace_id}", clear_on_submit=True, border=False
    ):
        input_column, send_column = st.columns((1, 0.09), gap="small")
        with input_column:
            query = st.text_input(
                "向当前论文提问",
                placeholder="向当前论文提问，例如：这篇论文的核心方法是什么？",
                label_visibility="collapsed",
                disabled=active_id is None,
            )
        with send_column:
            submitted = st.form_submit_button(
                "➤", type="primary", use_container_width=True, disabled=active_id is None
            )

    if not submitted:
        return
    if active_id is None:
        st.warning("请先新建一个本文会话。")
        return
    if not query.strip():
        st.warning("问题不能为空，请输入后再发送。")
        return
    try:
        with st.spinner("正在检索当前论文并生成回答…"):
            service.ask_this_paper(workspace.workspace_id, active_id, query)
    except Exception as error:  # noqa: BLE001 - 主页面不显示 traceback。
        st.error(_paper_friendly_error(error))
        return
    st.rerun()


def _render_paper_conversation(
    service: PaperBaseService, workspace_id: str, conversation_id: str
) -> None:
    """恢复本文会话，并复用与 Knowledge Base 一致的左右聊天样式。"""
    try:
        turns = service.get_paper_conversation_turns(
            conversation_id, workspace_id=workspace_id
        )
    except Exception as error:  # noqa: BLE001 - scope 错误安全降级。
        st.warning(f"当前本文会话暂时无法读取：{_paper_friendly_error(error)}")
        return
    if not turns:
        st.caption("这是一个新会话，输入问题后即可开始。")
        return

    for turn in turns:
        message_time = _conversation_timestamp(turn.created_at)

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
            with st.container(border=False, key=f"paper-user-bubble-{turn.turn_id}"):
                render_scientific_text(turn.user_query)
        with user_avatar:
            st.markdown(_avatar_html("user"), unsafe_allow_html=True)

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
            with st.container(border=False, key=f"paper-assistant-bubble-{turn.turn_id}"):
                is_unresolved = turn.resolution_status == "unresolved"
                if is_unresolved:
                    st.info(turn.assistant_answer)
                else:
                    _render_answer_body(turn.assistant_answer)
                metadata = service.get_paper_conversation_turn_metadata(turn)
                _render_answer_status(
                    metadata, turn.assistant_answer, is_unresolved=is_unresolved
                )
                evidence = service.get_paper_conversation_turn_evidence(turn)
                _render_evidence(
                    evidence,
                    conversation_id=conversation_id,
                    turn_id=turn.turn_id,
                    citations=_citation_ids(turn.assistant_answer),
                )
        assistant_spacer.empty()

def _ensure_paper_active_conversation(
    service: PaperBaseService, workspace_id: str, conversations: tuple[object, ...]
) -> str | None:
    """只恢复当前 workspace 的 conversation，切换论文时绝不复用其它 scope。"""
    active_id = st.session_state.get("paper_conversation_id")
    known_ids = {item.conversation_id for item in conversations}
    if active_id is None:
        return None
    if active_id in known_ids:
        return active_id
    try:
        record = service.get_conversation(active_id)
        if record.scope_type != "workspace" or record.scope_id != workspace_id:
            raise ValueError("Paper conversation scope does not match current workspace.")
    except Exception:
        st.session_state.paper_conversation_id = None
        mapping = st.session_state.get("paper_conversation_ids_by_workspace")
        if isinstance(mapping, dict):
            mapping.pop(workspace_id, None)
        return None
    return active_id


def _paper_conversation_label(conversation: object) -> str:
    """会话选择器只显示首问、轮数和时间，不显示 conversation_id。"""
    title = str(getattr(conversation, "title", "新会话"))
    count = int(getattr(conversation, "turn_count", 0))
    timestamp = _conversation_timestamp(str(getattr(conversation, "updated_at", "")))
    return f"{title}（{count} 轮 · {timestamp}）" if count else f"{title}（新会话 · {timestamp}）"


def _paper_friendly_error(error: Exception) -> str:
    """把 Ask This Paper 异常转换成用户可读提示。"""
    message = str(error).lower()
    if "scope" in message or "conversation" in message:
        return "当前本文会话不存在或不属于这篇论文，请新建会话后重试。"
    if "问题不能为空" in str(error):
        return "问题不能为空，请输入后再发送。"
    return "当前论文问答服务暂时不可用，请稍后重试。"
