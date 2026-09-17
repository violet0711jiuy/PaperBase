"""现有检索器到 Agentic RetrievalService 的适配层。"""

from __future__ import annotations

from paperbase.database import MetadataDatabase

from .contracts import PaperSearchSpec, QueryOrigin
from .hybrid_retriever import HybridRetriever, RetrievalResult


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
