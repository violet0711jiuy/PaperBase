"""逐论文 Top-K、Reranker 和 Expansion 配额彼此独立的回归测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest

import numpy as np

from paperbase.config import ContextExpansionSettings, RerankingSettings, RetrievalSettings
from paperbase.generation.section_expander import SectionAwareNeighborExpander
from paperbase.reranking.base import RerankScore
from paperbase.retrieval import FormalKnowledgeBaseAdapter, PaperSearchSpec, QueryOrigin
from paperbase.retrieval.hybrid_retriever import HybridRetriever
from paperbase.retrieval.query_rewriter import QueryRewritePlan


class _Embedder:
    def embed_queries(self, texts: list[str], *, instruction: str) -> np.ndarray:
        _ = instruction
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _Index:
    ntotal = 6

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        _ = vectors, top_k
        return (
            np.asarray([[0.99, 0.98, 0.97, 0.96, 0.95, 0.94]], dtype=np.float32),
            np.asarray([[1, 4, 2, 5, 3, 6]], dtype=np.int64),
        )


class _IndexStore:
    def load_for_search(self, *, database: object) -> _Index:
        _ = database
        return _Index()


class _Planner:
    def plan(self, query: str, **kwargs: object) -> QueryRewritePlan:
        _ = kwargs
        return QueryRewritePlan(
            original_query=query,
            resolved_query=query,
            rewrite_status="degraded",
        )


class _Reranker:
    backend_id = "fake"
    model_id = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, passages: list[str]) -> tuple[RerankScore, ...]:
        self.calls.append((query, len(passages)))
        return tuple(
            RerankScore(input_index=index, score=float(len(passages) - index))
            for index in range(len(passages))
        )


@dataclass
class _Database:
    rows: dict[int, dict[str, object]]

    def get_document(self, paper_id: str) -> object | None:
        return {"paper_id": paper_id} if paper_id in {"paper-a", "paper-b"} else None

    def chunks_by_vector_ids(
        self, vector_ids: list[int], *, paper_ids: tuple[str, ...] | None = None
    ) -> tuple[dict[str, object], ...]:
        allowed = set(paper_ids or ("paper-a", "paper-b"))
        return tuple(
            self.rows[vector_id]
            for vector_id in vector_ids
            if self.rows[vector_id]["paper_id"] in allowed
        )

    def list_content_chunks_in_section(
        self, *, paper_id: str, section: str
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            row
            for row in self.rows.values()
            if row["paper_id"] == paper_id and row["section"] == section
        )


def _row(vector_id: int, paper_id: str, chunk_index: int) -> dict[str, object]:
    return {
        "vector_id": vector_id,
        "paper_id": paper_id,
        "paper_title": paper_id,
        "chunk_id": f"{paper_id}-chunk-{chunk_index}",
        "chunk_index": chunk_index,
        "section": "Method",
        "content_kind": "body",
        "front_matter_type": None,
        "section_type": "content",
        "page_start": chunk_index,
        "page_end": chunk_index,
        "raw_text": f"{paper_id} method evidence {chunk_index}",
        "raw_token_count": 256,
    }


class PerPaperRetrievalPipelineTests(unittest.TestCase):
    def test_each_paper_gets_its_own_rerank_top_k_and_expansion_budget(self) -> None:
        rows = {
            1: _row(1, "paper-a", 1),
            2: _row(2, "paper-a", 2),
            3: _row(3, "paper-a", 3),
            4: _row(4, "paper-b", 1),
            5: _row(5, "paper-b", 2),
            6: _row(6, "paper-b", 3),
        }
        database = _Database(rows=rows)
        reranker = _Reranker()
        retriever = HybridRetriever(
            database=database,  # type: ignore[arg-type]
            query_embedder=_Embedder(),  # type: ignore[arg-type]
            index_store=_IndexStore(),  # type: ignore[arg-type]
            settings=RetrievalSettings(
                backend="hybrid_rrf",
                query_instruction="Retrieve academic passages relevant to this question.",
                dense_top_k_per_query=3,
                fused_top_k=3,
                dense_resolved_weight=1.0,
                dense_semantic_weight=1.0,
                bm25_keywords_weight=1.0,
            ),
            query_planner=_Planner(),  # type: ignore[arg-type]
            reranker=reranker,
            reranking_settings=RerankingSettings(
                backend="fake",
                model_id="fake",
                model_path=Path(__file__).resolve(),
                device="cpu",
                batch_size=2,
                candidate_top_k=2,
                final_top_k=1,
                max_length=128,
            ),
        )
        expander = SectionAwareNeighborExpander(
            database=database,  # type: ignore[arg-type]
            settings=ContextExpansionSettings(
                enabled=False,
                neighbor_window=0,
                max_total_tokens=256,
            ),
        )
        adapter = FormalKnowledgeBaseAdapter(
            database=database,  # type: ignore[arg-type]
            retriever=retriever,
            expander=expander,
        )

        first = adapter.search(
            PaperSearchSpec(paper_id="paper-a", query="A method"),
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
        )
        second = adapter.search(
            PaperSearchSpec(paper_id="paper-b", query="B method"),
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
        )

        self.assertEqual(reranker.calls, [("A method", 2), ("B method", 2)])
        self.assertEqual(len(first.retrieval.chunks), 1)
        self.assertEqual(len(second.retrieval.chunks), 1)
        self.assertEqual(first.retrieval.chunks[0].paper_id, "paper-a")
        self.assertEqual(second.retrieval.chunks[0].paper_id, "paper-b")
        self.assertEqual(len(first.expansion.content_evidence), 1)  # type: ignore[union-attr]
        self.assertEqual(len(second.expansion.content_evidence), 1)  # type: ignore[union-attr]
        self.assertEqual(first.expansion.content_evidence[0].token_count, 256)  # type: ignore[union-attr]
        self.assertEqual(second.expansion.content_evidence[0].token_count, 256)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
