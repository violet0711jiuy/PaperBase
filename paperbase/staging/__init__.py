"""v0.2 临时论文工作区。

这里的文件只服务于用户尚未确认入库的论文，绝不连接正式 SQLite、FTS5 或 FAISS。
"""

from .service import (
    TemporaryPaperWorkspace,
    TemporaryWorkspaceError,
    create_temporary_workspace,
    delete_temporary_workspace,
    run_temporary_workspace_stage,
)
from .sections import (
    WorkspaceChunk,
    WorkspaceSection,
    WorkspaceSectionError,
    WorkspaceSectionRepository,
    WorkspaceSectionSnapshot,
)
from .bm25 import (
    WorkspaceBM25Error,
    WorkspaceBM25Index,
    WorkspaceBM25IndexCache,
    WorkspaceBM25Match,
    workspace_bm25_fingerprint,
)

__all__ = [
    "TemporaryPaperWorkspace",
    "TemporaryWorkspaceError",
    "create_temporary_workspace",
    "delete_temporary_workspace",
    "run_temporary_workspace_stage",
    "WorkspaceChunk",
    "WorkspaceSection",
    "WorkspaceSectionError",
    "WorkspaceSectionRepository",
    "WorkspaceSectionSnapshot",
    "WorkspaceBM25Error",
    "WorkspaceBM25Index",
    "WorkspaceBM25IndexCache",
    "WorkspaceBM25Match",
    "workspace_bm25_fingerprint",
]
