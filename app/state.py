"""集中管理 Streamlit Session State 的小型辅助函数。

Session State 只保存当前页面需要的 ID 和选择项，不保存完整会话历史、解析对象或 chunks。
真正的会话内容仍在 ``conversations.sqlite3``，workspace 数据仍在 staging 文件中。
"""

# 允许使用新式类型标注，并延迟解析类型名称。
from __future__ import annotations

# ``MutableMapping`` 让这些函数既能接收 Streamlit 的 Session State，也能接收普通字典测试。
from typing import MutableMapping


# Knowledge Base 页面名称，Sidebar 和主入口共同使用这个稳定值。
KNOWLEDGE_BASE_PAGE = "Knowledge Base"
# Paper Workspace 页面名称，Sidebar 和主入口共同使用这个稳定值。
PAPER_WORKSPACE_PAGE = "Paper Workspace"

# 所有新会话默认使用的 UI 状态。
_DEFAULTS: dict[str, object] = {
    # 当前一级页面，首次打开时默认显示 Knowledge Base。
    "current_page": KNOWLEDGE_BASE_PAGE,
    # 当前选中的 staging workspace；没有选择时为 None。
    "active_workspace_id": None,
    # 当前选中的章节；Frontend Step 1 尚未真正使用它。
    "selected_section_id": None,
    # 正式 Knowledge Base 当前正在使用的 conversation ID。
    "kb_conversation_id": None,
    # 当前 Paper Workspace 对应的 conversation ID。
    "paper_conversation_id": None,
    # 是否展开 Paper Workspace 的上传区域；已有 workspace 默认收起。
    "upload_panel_expanded": False,
    # 按 workspace scope 保存各自的 paper conversation ID，防止切换论文时串会话。
    "paper_conversation_ids_by_workspace": {},
}


def initialize_session_state(state: MutableMapping[str, object]) -> None:
    """补齐缺失的 UI 状态，但不覆盖 Streamlit rerun 中已有的值。"""
    # 逐项检查默认字段，保证后续页面可以直接读取这些 key。
    for key, default in _DEFAULTS.items():
        # Streamlit 每次 rerun 都会重新执行脚本，因此已有值必须保留。
        if key not in state:
            # 字典默认值需要复制，避免不同 Session 共用同一个可变字典。
            state[key] = default.copy() if isinstance(default, dict) else default


def activate_workspace(state: MutableMapping[str, object], workspace_id: str | None) -> None:
    """切换当前 workspace，并只恢复该 workspace 自己的 conversation ID。"""
    # 读取切换前的 workspace，用来判断是否真的发生了切换。
    previous_workspace_id = state.get("active_workspace_id")
    # 只有 workspace 发生变化时才清理章节选择，避免普通 rerun 丢失选择。
    if previous_workspace_id != workspace_id:
        # 记录新的 workspace ID。
        state["active_workspace_id"] = workspace_id
        # 章节属于论文范围，切换论文后不能继续使用旧论文的章节。
        state["selected_section_id"] = None
        # 切换论文后把上传面板恢复为收起状态，避免占用顶部空间。
        state["upload_panel_expanded"] = False

    # 取出按 workspace 保存的 conversation 映射。
    conversations = state.get("paper_conversation_ids_by_workspace")
    # 如果外部状态被破坏或旧版本没有这个字段，就重新创建空映射。
    if not isinstance(conversations, dict):
        # 创建新的字典，保证后续赋值安全。
        conversations = {}
        # 把修复后的字典写回 Session State。
        state["paper_conversation_ids_by_workspace"] = conversations
    # 恢复当前 workspace 的会话；没有 workspace 时明确设置为 None。
    state["paper_conversation_id"] = conversations.get(workspace_id) if workspace_id else None


def set_workspace_conversation(
    state: MutableMapping[str, object], *, workspace_id: str, conversation_id: str
) -> None:
    """把一个独立会话绑定到明确的 workspace scope。"""
    # 读取 workspace 到 conversation 的映射。
    conversations = state.get("paper_conversation_ids_by_workspace")
    # 映射不存在或类型不正确时，先恢复为可写字典。
    if not isinstance(conversations, dict):
        # 创建新的映射，避免向 None 或其他对象赋值。
        conversations = {}
        # 保存修复后的映射。
        state["paper_conversation_ids_by_workspace"] = conversations
    # 只写入当前 workspace 的 conversation，不影响其他论文。
    conversations[workspace_id] = conversation_id
    # 如果这个 workspace 正在显示，把当前会话字段同步为新 ID。
    if state.get("active_workspace_id") == workspace_id:
        # 页面读取的是这个短字段，因此需要立即更新它。
        state["paper_conversation_id"] = conversation_id
