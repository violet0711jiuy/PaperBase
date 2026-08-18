"""独立于知识库 SQLite 的多轮会话持久化。"""

# 正式 Knowledge Base 的唯一 scope 标识。
# Streamlit、CLI 和后端 Service 都从这里引用，避免不同入口创建出互不兼容的会话。
FORMAL_KNOWLEDGE_BASE_SCOPE_ID = "paperbase_formal_knowledge_base"

from .store import (
    ConversationRecord,
    ConversationScopeError,
    ConversationStore,
    ConversationStoreError,
    ConversationTurn,
    format_turns_as_context,
)

__all__ = [
    "FORMAL_KNOWLEDGE_BASE_SCOPE_ID",
    "ConversationRecord",
    "ConversationScopeError",
    "ConversationStore",
    "ConversationStoreError",
    "ConversationTurn",
    "format_turns_as_context",
]
