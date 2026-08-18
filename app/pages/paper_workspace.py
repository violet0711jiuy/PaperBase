"""Paper Workspace 页面。

本模块只提供统一 App Shell 中的第二栏和第三栏内容：第二栏负责选择
staging workspace、上传入口和当前 workspace 状态；第三栏负责论文摘要
和三个后续功能占位。上传、解析、Overview、Explain、Ask 的真实业务仍由
后续前端步骤接入，本轮不改后端能力。
"""

from __future__ import annotations

import streamlit as st

if __package__ == "app.pages":
    from app.services.paperbase_service import PaperBaseService, WorkspaceSummary
    from app.state import activate_workspace
else:
    from services.paperbase_service import PaperBaseService, WorkspaceSummary
    from state import activate_workspace


def render_context_panel(service: PaperBaseService) -> str | None:
    """绘制统一 App Shell 第二栏的 Workspace Context Panel。"""
    st.markdown(
        "<div class='pb-pane-kicker'>当前工作区</div>"
        "<div class='pb-pane-title'>论文工作区</div>"
        "<div class='pb-pane-subtitle'>选择 staging 论文并查看入库状态</div>",
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
    options = [None, *workspace_by_id]
    selected_index = options.index(current_workspace_id) if current_workspace_id in options else 0
    selected_workspace_id = st.selectbox(
        "选择已有的 staging 工作区",
        options=options,
        index=selected_index,
        format_func=lambda item: (
            "请选择工作区"
            if item is None
            else _workspace_option(workspace_by_id[item])
        ),
        key="paper_workspace_selector",
    )

    # 只在选择发生变化时清理章节和论文 conversation，防止不同 workspace 串状态。
    activate_workspace(st.session_state, selected_workspace_id)

    has_active_workspace = selected_workspace_id is not None
    upload_expanded = (
        bool(st.session_state.get("upload_panel_expanded", False))
        if has_active_workspace
        else True
    )
    upload_label = (
        "收起上传区域"
        if upload_expanded and has_active_workspace
        else "＋ 上传新论文"
    )
    if st.button(upload_label, use_container_width=True, key="paper_upload_toggle"):
        st.session_state.upload_panel_expanded = (
            not upload_expanded if has_active_workspace else True
        )
        st.rerun()

    if upload_expanded:
        st.file_uploader(
            "上传论文",
            type=["pdf"],
            disabled=True,
            help="PDF 上传和 staging 流程将在后续前端阶段接入。",
            key="paper_upload",
        )

    if not has_active_workspace:
        st.caption("尚未选择论文。请从上方选择已有的 staging 工作区。")
    return selected_workspace_id


def render_main_panel(service: PaperBaseService) -> None:
    """绘制统一 App Shell 第三栏的 Paper Main Panel。"""
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

    _render_header(workspace)
    overview_tab, explain_tab, ask_tab = st.tabs(("论文概览", "解释章节", "询问本文"))
    with overview_tab:
        st.info("论文概览功能将在后续前端阶段接入。")
    with explain_tab:
        st.info("章节树和章节解释功能将在后续前端阶段接入。")
    with ask_tab:
        st.info("本文范围内的多会话问答将在后续前端阶段接入。")


def render(service: PaperBaseService) -> None:
    """兼容旧调用方；正式入口使用 App Shell 分别调用两个 panel。"""
    render_context_panel(service)
    render_main_panel(service)


def _workspace_option(workspace: WorkspaceSummary) -> str:
    """把 workspace 视图模型格式化为选择器中的用户可读标题。"""
    return workspace.display_title


def _render_header(workspace: WorkspaceSummary) -> None:
    """绘制论文标题、规模和加入知识库状态。"""
    st.markdown(
        f"<div class='pb-paper-header'><div class='pb-paper-title'>{workspace.display_title}</div>"
        f"<div class='pb-paper-source'>{workspace.source_filename}</div></div>",
        unsafe_allow_html=True,
    )
    if workspace.section_count:
        paper_size = f"{workspace.section_count} 个章节 · {workspace.total_chunk_count} 个分块"
    else:
        paper_size = f"{workspace.total_chunk_count} 个分块"
    status = "✓ 已加入知识库" if workspace.added_to_kb else "尚未加入知识库"
    size_column, status_column = st.columns((3, 2), gap="small")
    with size_column:
        st.markdown(f"**{paper_size}**")
    with status_column:
        st.caption(status)

