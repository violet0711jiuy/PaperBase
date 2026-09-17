"""Phase 1 Facade → Batch → ToolResult 的端到端契约测试。"""

from __future__ import annotations

import unittest

from paperbase.retrieval import (
    PaperRetrievalSnapshot,
    PaperSearchSpec,
    PaperSearchStatus,
    QueryOrigin,
    RetrievalService,
    RetrievalSource,
    SearchPapersRequest,
    ToolStatus,
)
from paperbase.retrieval.hybrid_retriever import (
    RetrievedChunk,
    RetrievalResult,
    RetrievalSourceMatch,
)
from paperbase.retrieval.query_rewriter import QueryRewritePlan


class _ContractAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...], int | None]] = []

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> PaperRetrievalSnapshot:
        self.calls.append(
            (search.paper_id, search.query, search.lexical_terms, result_limit)
        )
        if search.paper_id == "paper-error":
            raise RuntimeError("isolated paper failure")
        chunks = () if search.paper_id == "paper-empty" else (_chunk(search),)
        return PaperRetrievalSnapshot(
            paper_id=search.paper_id,
            retrieval=RetrievalResult(
                query=search.query,
                rewrite_plan=QueryRewritePlan(
                    original_query=search.query,
                    resolved_query=search.query,
                    lexical_keywords_en=search.lexical_terms,
                ),
                reranking_status="success",
                chunks=chunks,
            ),
            expansion=None,
        )


def _chunk(search: PaperSearchSpec) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id=f"{search.paper_id}-chunk",
        vector_id=1,
        paper_id=search.paper_id,
        paper_title=search.paper_id,
        section="Method",
        content_kind="body",
        front_matter_type=None,
        section_type="content",
        page_start=1,
        page_end=1,
        raw_text=f"Evidence for query: {search.query}",
        pre_rerank_rank=1,
        rerank_score=0.8,
        fused_score=0.02,
        source_matches=(
            RetrievalSourceMatch(
                route="dense_resolved",
                query=search.query,
                rank=1,
                raw_score=0.8,
                effective_weight=1.0,
            ),
        ),
        section_id=f"{search.paper_id}-method",
    )


class AgenticRetrievalContractTests(unittest.TestCase):
    def test_heterogeneous_batch_reaches_typed_tool_result_without_cross_paper_loss(self) -> None:
        adapter = _ContractAdapter()
        service = RetrievalService(formal_adapter=adapter)
        request = SearchPapersRequest(
            query_origin=QueryOrigin.AGENT_SUPPLIED,
            searches=(
                PaperSearchSpec(
                    paper_id="paper-a",
                    query="A spatial encoder",
                    lexical_terms=("spatial encoder",),
                ),
                PaperSearchSpec(
                    paper_id="paper-b",
                    query="B temporal decoder",
                    lexical_terms=("temporal decoder",),
                ),
            ),
            top_k_per_paper=2,
        )

        result = service.search_papers(
            request,
            source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual([call[1] for call in adapter.calls], [
            "A spatial encoder",
            "B temporal decoder",
        ])
        self.assertEqual([call[3] for call in adapter.calls], [2, 2])
        outcomes = result.payload.paper_results
        self.assertEqual([outcome.paper_id for outcome in outcomes], ["paper-a", "paper-b"])
        self.assertTrue(all(outcome.status is PaperSearchStatus.OK for outcome in outcomes))
        self.assertEqual(
            outcomes[0].evidence_candidates[0].content,
            "Evidence for query: A spatial encoder",
        )
        self.assertEqual(
            outcomes[1].evidence_candidates[0].section_id,
            "paper-b-method",
        )

    def test_empty_paper_does_not_kill_successful_paper(self) -> None:
        service = RetrievalService(formal_adapter=_ContractAdapter())
        result = service.search_papers(
            SearchPapersRequest(
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                searches=(
                    PaperSearchSpec(paper_id="paper-empty", query="missing method"),
                    PaperSearchSpec(paper_id="paper-b", query="B method"),
                ),
            ),
            source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(
            [outcome.status for outcome in result.payload.paper_results],
            [PaperSearchStatus.EMPTY, PaperSearchStatus.OK],
        )

    def test_error_paper_produces_partial_and_preserves_success(self) -> None:
        service = RetrievalService(formal_adapter=_ContractAdapter())
        result = service.search_papers(
            SearchPapersRequest(
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                searches=(
                    PaperSearchSpec(paper_id="paper-error", query="broken"),
                    PaperSearchSpec(paper_id="paper-b", query="B method"),
                ),
            ),
            source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
        )

        self.assertEqual(result.status, ToolStatus.PARTIAL)
        self.assertEqual(result.payload.paper_results[0].error_code, "RuntimeError")
        self.assertEqual(
            result.payload.paper_results[1].evidence_candidates[0].paper_id,
            "paper-b",
        )


if __name__ == "__main__":
    unittest.main()
