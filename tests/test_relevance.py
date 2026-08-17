"""回答前低相关性拒答的回归测试。"""

from __future__ import annotations

import unittest

from paperbase.generation.relevance import has_insufficient_retrieval_relevance
from paperbase.retrieval.hybrid_retriever import RetrievalResult, RetrievedChunk
from paperbase.retrieval.query_rewriter import QueryRewritePlan


def _result(*, score: float | None, status: str = "success") -> RetrievalResult:
    """构造最小检索结果，避免加载向量模型。"""
    chunk = RetrievedChunk(
        rank=1,
        chunk_id="chunk-1",
        vector_id=1,
        paper_id="paper-1",
        paper_title="Paper 1",
        section="3 Method",
        content_kind="body",
        front_matter_type=None,
        section_type="content",
        page_start=1,
        page_end=1,
        raw_text="unrelated passage",
        pre_rerank_rank=1,
        rerank_score=score,
        fused_score=0.1,
        source_matches=(),
    )
    return RetrievalResult(
        query="question",
        rewrite_plan=QueryRewritePlan(original_query="question"),
        reranking_status=status,
        chunks=(chunk,),
    )


class RetrievalRelevanceTests(unittest.TestCase):
    def test_successful_low_rerank_scores_are_rejected(self) -> None:
        self.assertTrue(
            has_insufficient_retrieval_relevance(_result(score=0.00005), min_rerank_score=0.05)
        )

    def test_relevant_or_unreranked_results_are_not_rejected(self) -> None:
        self.assertFalse(
            has_insufficient_retrieval_relevance(_result(score=0.36), min_rerank_score=0.05)
        )
        self.assertFalse(
            has_insufficient_retrieval_relevance(
                _result(score=None, status="disabled"), min_rerank_score=0.05
            )
        )


if __name__ == "__main__":
    unittest.main()
