"""Step 6 混合检索的无 GPU、无网络回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from paperbase.config import RetrievalSettings
from paperbase.retrieval.hybrid_retriever import HybridRetriever
from paperbase.retrieval.query_rewriter import QueryRewritePlan


def _row(chunk_id: str, vector_id: int, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "vector_id": vector_id,
        "paper_id": "paper_test",
        "paper_title": "Test paper",
        "section": "1. Introduction",
        "content_kind": "body",
        "front_matter_type": None,
        "section_type": "content",
        "page_start": 1,
        "page_end": 1,
        "raw_text": text,
    }


class _FakeIndex:
    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        # 第一个维度为 1 代表原始 Query，第二个维度为 1 代表 semantic 改写 Query。
        outputs = []
        for vector in vectors:
            outputs.append((([0.91, 0.70], [1, 2]) if vector[0] else ([0.89, 0.50], [2, 3])))
        return (
            np.asarray([item[0] for item in outputs], dtype=np.float32),
            np.asarray([item[1] for item in outputs], dtype=np.int64),
        )


class _FakeIndexStore:
    def load_for_search(self, *, database: object) -> _FakeIndex:
        return _FakeIndex()


class _FakeEmbedder:
    def embed_queries(self, texts: list[str], *, instruction: str) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if text == "原始问题" else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class _FakeRewriter:
    def rewrite(
        self,
        query: str,
        *,
        conversation_context: list[str] | None = None,
    ) -> QueryRewritePlan:
        # 测试替身不依赖历史，但必须接受正式接口的可选参数。
        _ = conversation_context
        return QueryRewritePlan(
            original_query="原始问题",
            semantic_query="改写问题",
            lexical_keywords_en=("LSTM", "wind speed prediction", "author"),
            search_bibliography=False,
        )


class _FakeDatabase:
    def __init__(self) -> None:
        self._rows = {
            1: _row("chunk_1", 1, "first"),
            2: _row("chunk_2", 2, "second"),
            3: _row("chunk_3", 3, "third"),
        }

    def chunks_by_vector_ids(self, vector_ids: list[int]) -> tuple[dict[str, object], ...]:
        return tuple(self._rows[vector_id] for vector_id in vector_ids if vector_id in self._rows)

    def search_bm25(self, query: str, *, top_k: int) -> tuple[dict[str, object], ...]:
        return ({**self._rows[1], "bm25_score": -1.0},)

    def search_bm25_keyword_group(
        self,
        keywords: tuple[str, ...],
        *,
        top_k: int,
    ) -> tuple[dict[str, object], ...]:
        self.keyword_group = keywords
        return ({**self._rows[2], "bm25_score": -2.0},)

    def search_bibliography(self, query: str, *, top_k: int) -> tuple[dict[str, object], ...]:
        raise AssertionError("普通问题不应查询 bibliography FTS5")

    def search_bibliography_keyword_group(
        self, keywords: tuple[str, ...], *, top_k: int
    ) -> tuple[dict[str, object], ...]:
        raise AssertionError("普通问题不应查询 bibliography FTS5")


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_deduplicates_routes_and_keeps_all_source_evidence(self) -> None:
        settings = RetrievalSettings(
            backend="hybrid_rrf",
            dense_top_k_per_query=2,
            bm25_original_top_k=10,
            bm25_rewrite_top_k=20,
            fused_top_k=3,
            rrf_k=10,
            dense_original_weight=1.0,
            dense_rewrite_weight=1.0,
            bm25_original_weight=0.7,
            bm25_rewrite_weight=1.0,
            query_instruction="Retrieve academic passages relevant to this question.",
        )
        retriever = HybridRetriever(
            database=_FakeDatabase(),  # type: ignore[arg-type]
            query_embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            index_store=_FakeIndexStore(),  # type: ignore[arg-type]
            settings=settings,
            query_rewriter=_FakeRewriter(),
        )

        result = retriever.retrieve("任意输入")

        self.assertEqual([chunk.chunk_id for chunk in result.chunks], ["chunk_2", "chunk_1", "chunk_3"])
        second_chunk = result.chunks[0]
        self.assertEqual(
            {source.route for source in second_chunk.source_matches},
            {"dense_original", "dense_rewrite", "bm25_rewrite"},
        )
        # 语义改写路固定只有一条，因此应保留该路完整的 1.0 权重。
        semantic_source = next(
            source for source in second_chunk.source_matches if source.route == "dense_rewrite"
        )
        self.assertEqual(semantic_source.effective_weight, 1.0)
        # 三个英文关键词必须作为一组只调用一次 BM25，而不是拆成三路。
        self.assertEqual(
            retriever._database.keyword_group,  # type: ignore[attr-defined]
            ("LSTM", "wind speed prediction", "author"),
        )


if __name__ == "__main__":
    unittest.main()
