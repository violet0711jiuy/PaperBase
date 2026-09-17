"""正式知识库按 paper_id 限定 Dense/BM25 候选的契约测试。"""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from paperbase.config import RetrievalSettings
from paperbase.retrieval import FormalKnowledgeBaseAdapter, PaperSearchSpec, QueryOrigin
from paperbase.retrieval.hybrid_retriever import HybridRetriever
from paperbase.retrieval.query_rewriter import QueryRewritePlan


class _Embedder:
    def embed_queries(self, texts: list[str], *, instruction: str) -> np.ndarray:
        _ = texts, instruction
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class _Index:
    ntotal = 3

    def __init__(self) -> None:
        self.top_k_values: list[int] = []

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        _ = vectors
        self.top_k_values.append(top_k)
        return (
            np.asarray([[0.99, 0.8, 0.7]], dtype=np.float32),
            np.asarray([[1, 2, 3]], dtype=np.int64),
        )


class _IndexStore:
    def __init__(self) -> None:
        self.index = _Index()

    def load_for_search(self, *, database: object) -> _Index:
        _ = database
        return self.index


class _Planner:
    def plan(self, query: str, **kwargs: object) -> QueryRewritePlan:
        _ = kwargs
        return QueryRewritePlan(
            original_query=query,
            resolved_query=query,
            rewrite_status="degraded",
        )


class _ForbiddenPlanner:
    def plan(self, query: str, **kwargs: object) -> QueryRewritePlan:
        raise AssertionError(f"Agent query must bypass planner: {query}, {kwargs}")


@dataclass
class _Database:
    vector_calls: list[tuple[tuple[int, ...], tuple[str, ...] | None]]
    bm25_calls: list[tuple[tuple[str, ...], tuple[str, ...] | None]]

    def get_document(self, paper_id: str) -> object | None:
        return {"paper_id": paper_id} if paper_id == "paper-target" else None

    def chunks_by_vector_ids(
        self, vector_ids: list[int], *, paper_ids: tuple[str, ...] | None = None
    ) -> tuple[dict[str, object], ...]:
        self.vector_calls.append((tuple(vector_ids), paper_ids))
        rows = {
            1: _row(1, "paper-other", "other-1"),
            2: _row(2, "paper-target", "target-1"),
            3: _row(3, "paper-target", "target-2"),
        }
        allowed = set(paper_ids or ("paper-other", "paper-target"))
        return tuple(rows[item] for item in vector_ids if rows[item]["paper_id"] in allowed)

    def search_bm25_keyword_group(
        self,
        keywords: tuple[str, ...],
        *,
        top_k: int,
        paper_ids: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, object], ...]:
        _ = top_k
        self.bm25_calls.append((tuple(keywords), paper_ids))
        return ()


def _row(vector_id: int, paper_id: str, chunk_id: str) -> dict[str, object]:
    return {
        "vector_id": vector_id,
        "paper_id": paper_id,
        "paper_title": paper_id,
        "chunk_id": chunk_id,
        "section": "Method",
        "content_kind": "body",
        "front_matter_type": None,
        "section_type": "content",
        "page_start": 1,
        "page_end": 1,
        "raw_text": chunk_id,
    }


class FormalScopedRetrievalTests(unittest.TestCase):
    def _retriever(self) -> tuple[HybridRetriever, _Database, _IndexStore]:
        database = _Database(vector_calls=[], bm25_calls=[])
        index_store = _IndexStore()
        retriever = HybridRetriever(
            database=database,  # type: ignore[arg-type]
            query_embedder=_Embedder(),  # type: ignore[arg-type]
            index_store=index_store,  # type: ignore[arg-type]
            settings=RetrievalSettings(
                backend="hybrid_rrf",
                query_instruction="Retrieve paper-scoped academic evidence.",
                dense_top_k_per_query=1,
                fused_top_k=2,
                dense_resolved_weight=1.0,
                dense_semantic_weight=1.0,
                bm25_keywords_weight=1.0,
            ),
            query_planner=_Planner(),  # type: ignore[arg-type]
        )
        return retriever, database, index_store

    def test_agent_supplied_query_never_calls_planner_and_uses_explicit_terms(self) -> None:
        retriever, database, _ = self._retriever()
        retriever._query_planner = _ForbiddenPlanner()  # type: ignore[attr-defined]
        adapter = FormalKnowledgeBaseAdapter(
            database=database,  # type: ignore[arg-type]
            retriever=retriever,
        )

        result = adapter.search(
            PaperSearchSpec(
                paper_id="paper-target",
                query="target architecture details",
                lexical_terms=("Graph encoder",),
            ),
            query_origin=QueryOrigin.AGENT_SUPPLIED,
        )

        self.assertTrue(result.chunks)
        self.assertEqual(result.rewrite_plan.lexical_keywords_en, ("Graph encoder",))
        self.assertEqual(database.bm25_calls, [(("Graph encoder",), ("paper-target",))])

    def test_scoped_dense_search_scans_index_then_reranks_inside_paper(self) -> None:
        retriever, database, index_store = self._retriever()

        result = retriever.retrieve("method", paper_ids=("paper-target",))

        self.assertEqual(index_store.index.top_k_values, [3])
        self.assertEqual([chunk.chunk_id for chunk in result.chunks], ["target-1"])
        self.assertTrue(all(chunk.paper_id == "paper-target" for chunk in result.chunks))
        self.assertEqual(database.vector_calls[0][1], ("paper-target",))
        self.assertEqual(result.chunks[0].source_matches[0].rank, 1)

    def test_formal_adapter_validates_paper_id_before_retrieval(self) -> None:
        retriever, database, _ = self._retriever()
        adapter = FormalKnowledgeBaseAdapter(
            database=database,  # type: ignore[arg-type]
            retriever=retriever,
        )

        result = adapter.search(
            PaperSearchSpec(paper_id="paper-target", query="method"),
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
        )
        self.assertTrue(result.chunks)
        with self.assertRaisesRegex(LookupError, "Unknown formal paper_id"):
            adapter.search(
                PaperSearchSpec(paper_id="missing", query="method"),
                query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
            )


if __name__ == "__main__":
    unittest.main()
