"""现有检索器到 Agentic RetrievalService 的适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from paperbase.database import MetadataDatabase
from paperbase.staging.sections import WorkspaceSectionSnapshot

from .contracts import PaperSearchSpec, QueryOrigin
from .hybrid_retriever import HybridRetriever, RetrievalResult
from .query_rewriter import TrustedPaperScope

if TYPE_CHECKING:
    from paperbase.generation.section_expander import ExpansionResult


class _WorkspaceRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        trusted_scope: TrustedPaperScope | None = None,
        conversation_context: Sequence[str] | None = None,
        result_limit: int | None = None,
        query_origin: QueryOrigin = QueryOrigin.INITIAL_UNDERSTANDING,
        lexical_terms: Sequence[str] = (),
    ) -> RetrievalResult: ...


class _EvidenceExpander(Protocol):
    def expand(self, result: RetrievalResult) -> "ExpansionResult": ...


@dataclass(frozen=True)
class PaperRetrievalSnapshot:
    """一篇论文独立完成检索、重排和可选扩展后的内部快照。"""

    paper_id: str
    retrieval: RetrievalResult
    expansion: "ExpansionResult | None"


class FormalKnowledgeBaseAdapter:
    """把正式 HybridRetriever 限定到一个已验证的正式 ``paper_id``。"""

    def __init__(
        self,
        *,
        database: MetadataDatabase,
        retriever: HybridRetriever,
        expander: _EvidenceExpander | None = None,
    ) -> None:
        self._database = database
        self._retriever = retriever
        self._expander = expander

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> PaperRetrievalSnapshot:
        if self._database.get_document(search.paper_id) is None:
            raise LookupError(f"Unknown formal paper_id: {search.paper_id}")
        retrieval = self._retriever.retrieve(
            search.query,
            paper_ids=(search.paper_id,),
            result_limit=result_limit,
            query_origin=QueryOrigin(query_origin),
            lexical_terms=search.lexical_terms,
        )
        return PaperRetrievalSnapshot(
            paper_id=search.paper_id,
            retrieval=retrieval,
            expansion=self._expander.expand(retrieval) if self._expander is not None else None,
        )


class WorkspacePaperAdapter:
    """包装一个已装配 Workspace Retriever，并保持单论文隔离边界。"""

    def __init__(
        self,
        *,
        snapshot: WorkspaceSectionSnapshot,
        retriever: _WorkspaceRetriever,
        expander: _EvidenceExpander | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._retriever = retriever
        self._expander = expander

    @property
    def workspace_id(self) -> str:
        return self._snapshot.workspace_id

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> PaperRetrievalSnapshot:
        if search.paper_id != self._snapshot.paper_id:
            raise LookupError(
                "Workspace paper_id mismatch: "
                f"expected {self._snapshot.paper_id}, got {search.paper_id}."
            )
        retrieval = self._retriever.retrieve(
            search.query,
            trusted_scope=TrustedPaperScope(
                paper_id=self._snapshot.paper_id,
                paper_title=self._snapshot.paper_title,
            ),
            result_limit=result_limit,
            query_origin=QueryOrigin(query_origin),
            lexical_terms=search.lexical_terms,
        )
        return PaperRetrievalSnapshot(
            paper_id=search.paper_id,
            retrieval=retrieval,
            expansion=self._expander.expand(retrieval) if self._expander is not None else None,
        )
