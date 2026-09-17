"""现有检索器到 Agentic RetrievalService 的适配层。"""

from __future__ import annotations

from typing import Protocol, Sequence

from paperbase.database import MetadataDatabase
from paperbase.staging.sections import WorkspaceSectionSnapshot

from .contracts import PaperSearchSpec, QueryOrigin
from .hybrid_retriever import HybridRetriever, RetrievalResult
from .query_rewriter import TrustedPaperScope


class _WorkspaceRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        trusted_scope: TrustedPaperScope | None = None,
        conversation_context: Sequence[str] | None = None,
        result_limit: int | None = None,
    ) -> RetrievalResult: ...


class FormalKnowledgeBaseAdapter:
    """把正式 HybridRetriever 限定到一个已验证的正式 ``paper_id``。"""

    def __init__(self, *, database: MetadataDatabase, retriever: HybridRetriever) -> None:
        self._database = database
        self._retriever = retriever

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> RetrievalResult:
        if self._database.get_document(search.paper_id) is None:
            raise LookupError(f"Unknown formal paper_id: {search.paper_id}")
        if QueryOrigin(query_origin) is QueryOrigin.AGENT_SUPPLIED:
            raise NotImplementedError(
                "agent_supplied query execution is introduced in Phase 1 commit 4."
            )
        return self._retriever.retrieve(
            search.query,
            paper_ids=(search.paper_id,),
            result_limit=result_limit,
        )


class WorkspacePaperAdapter:
    """包装一个已装配 Workspace Retriever，并保持单论文隔离边界。"""

    def __init__(
        self,
        *,
        snapshot: WorkspaceSectionSnapshot,
        retriever: _WorkspaceRetriever,
    ) -> None:
        self._snapshot = snapshot
        self._retriever = retriever

    @property
    def workspace_id(self) -> str:
        return self._snapshot.workspace_id

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> RetrievalResult:
        if search.paper_id != self._snapshot.paper_id:
            raise LookupError(
                "Workspace paper_id mismatch: "
                f"expected {self._snapshot.paper_id}, got {search.paper_id}."
            )
        if QueryOrigin(query_origin) is QueryOrigin.AGENT_SUPPLIED:
            raise NotImplementedError(
                "agent_supplied query execution is introduced in Phase 1 commit 4."
            )
        return self._retriever.retrieve(
            search.query,
            trusted_scope=TrustedPaperScope(
                paper_id=self._snapshot.paper_id,
                paper_title=self._snapshot.paper_title,
            ),
            result_limit=result_limit,
        )
