"""Step 6 混合检索的无 GPU、无网络回归测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from paperbase.config import RerankingSettings, RetrievalSettings
from paperbase.reranking.base import RerankScore
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


def _bibliography_row(chunk_id: str, paper_id: str, text: str) -> dict[str, object]:
    """构造不进入主向量索引、但可由 bibliography FTS5 返回的参考文献测试行。"""
    return {
        **_row(chunk_id, 0, text),
        "paper_id": paper_id,
        "section": "References",
        "section_type": "bibliography",
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


class _BibliographyRewriter:
    """模拟明确引用意图，确保测试会进入 bibliography FTS5 辅助路径。"""

    def rewrite(
        self,
        query: str,
        *,
        conversation_context: list[str] | None = None,
    ) -> QueryRewritePlan:
        _ = query, conversation_context
        return QueryRewritePlan(
            original_query="original citation question",
            semantic_query="Which references concern wind speed prediction?",
            lexical_keywords_en=("wind speed prediction",),
            search_bibliography=True,
        )


class _FakeReranker:
    """不加载真实模型的重排序替身；分数故意与 RRF 顺序不同以验证重排生效。"""

    backend_id = "fake_reranker"
    model_id = "fake/reranker"

    def rerank(self, query: str, passages: list[str]) -> tuple[RerankScore, ...]:
        _ = query, passages
        # 输入正文 RRF 候选的三个分数分别为 0.10、0.95、0.50。
        return tuple(
            RerankScore(input_index=index, score=score)
            for index, score in enumerate((0.10, 0.95, 0.50)[: len(passages)])
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

    def search_bibliography(
        self, query: str, *, paper_id: str, top_k: int
    ) -> tuple[dict[str, object], ...]:
        raise AssertionError("普通问题不应查询 bibliography FTS5")

    def search_bibliography_keyword_group(
        self, keywords: tuple[str, ...], *, paper_id: str, top_k: int
    ) -> tuple[dict[str, object], ...]:
        raise AssertionError("普通问题不应查询 bibliography FTS5")


class _BibliographyFakeDatabase(_FakeDatabase):
    """记录 bibliography FTS5 的真实参数，验证它被限制在正文确定的目标论文内。"""

    def __init__(self) -> None:
        super().__init__()
        self._rows[1] = {**self._rows[1], "paper_id": "paper_target"}
        self._rows[2] = {**self._rows[2], "paper_id": "paper_target"}
        self._rows[3] = {**self._rows[3], "paper_id": "paper_other"}
        self.bibliography_calls: list[tuple[tuple[str, ...], str, int]] = []

    def search_bibliography_keyword_group(
        self, keywords: tuple[str, ...], *, paper_id: str, top_k: int
    ) -> tuple[dict[str, object], ...]:
        self.bibliography_calls.append((keywords, paper_id, top_k))
        return (
            {
                **_bibliography_row(
                    "reference_1", paper_id, "Wind speed prediction reference."
                ),
                "bm25_score": -3.0,
            },
        )

    def search_bibliography(
        self, query: str, *, paper_id: str, top_k: int
    ) -> tuple[dict[str, object], ...]:
        raise AssertionError("本测试应优先使用英文关键词组检索 bibliography FTS5")


class HybridRetrieverTests(unittest.TestCase):
    @staticmethod
    def _reranking_settings() -> RerankingSettings:
        """构造只用于测试的重排序配置，不触发真实本地模型加载。"""
        return RerankingSettings(
            backend="bge_cross_encoder",
            enabled=True,
            model_id="BAAI/bge-reranker-v2-m3",
            model_path=Path.cwd(),
            device="cuda",
            batch_size=8,
            candidate_top_k=40,
            final_top_k=2,
            max_length=1024,
            normalize_scores=True,
        )

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

    def test_bibliography_bm25_uses_keywords_and_is_scoped_to_best_content_paper(self) -> None:
        settings = RetrievalSettings(
            backend="hybrid_rrf",
            dense_top_k_per_query=2,
            bm25_original_top_k=10,
            bm25_rewrite_top_k=20,
            fused_top_k=4,
            rrf_k=10,
            dense_original_weight=1.0,
            dense_rewrite_weight=1.0,
            bm25_original_weight=0.7,
            bm25_rewrite_weight=1.0,
            query_instruction="Retrieve academic passages relevant to this question.",
        )
        database = _BibliographyFakeDatabase()
        retriever = HybridRetriever(
            database=database,  # type: ignore[arg-type]
            query_embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            index_store=_FakeIndexStore(),  # type: ignore[arg-type]
            settings=settings,
            query_rewriter=_BibliographyRewriter(),
        )

        result = retriever.retrieve("风速预测那篇论文里面的参考文献哪些也是风速预测")

        self.assertEqual(
            database.bibliography_calls,
            [(("wind speed prediction",), "paper_target", 5)],
        )
        bibliography_chunk = next(chunk for chunk in result.chunks if chunk.chunk_id == "reference_1")
        bibliography_source = bibliography_chunk.source_matches[0]
        self.assertEqual(bibliography_source.route, "bm25_bibliography")
        self.assertEqual(bibliography_source.query, "wind speed prediction")
        # 参考文献按自身 BM25 直接排在正文 RRF 结果之后，绝不贡献或参与正文 RRF 分数。
        self.assertEqual(result.chunks[-1].chunk_id, "reference_1")
        self.assertEqual(bibliography_chunk.fused_score, 0.0)
        self.assertEqual(bibliography_source.effective_weight, 0.0)

    def test_reranker_reorders_only_content_and_preserves_pre_rerank_evidence(self) -> None:
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
            reranker=_FakeReranker(),
            reranking_settings=self._reranking_settings(),
        )

        result = retriever.retrieve("任意输入")

        self.assertEqual(result.reranking_status, "success")
        self.assertEqual([chunk.chunk_id for chunk in result.chunks], ["chunk_1", "chunk_3"])
        self.assertEqual(result.chunks[0].pre_rerank_rank, 2)
        self.assertEqual(result.chunks[0].rerank_score, 0.95)
        self.assertEqual(result.chunks[1].pre_rerank_rank, 3)
        self.assertEqual(result.chunks[1].rerank_score, 0.5)

    def test_bibliography_is_direct_output_and_not_sent_to_reranker(self) -> None:
        settings = RetrievalSettings(
            backend="hybrid_rrf",
            dense_top_k_per_query=2,
            bm25_original_top_k=10,
            bm25_rewrite_top_k=20,
            fused_top_k=4,
            rrf_k=10,
            dense_original_weight=1.0,
            dense_rewrite_weight=1.0,
            bm25_original_weight=0.7,
            bm25_rewrite_weight=1.0,
            query_instruction="Retrieve academic passages relevant to this question.",
        )
        retriever = HybridRetriever(
            database=_BibliographyFakeDatabase(),  # type: ignore[arg-type]
            query_embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            index_store=_FakeIndexStore(),  # type: ignore[arg-type]
            settings=settings,
            query_rewriter=_BibliographyRewriter(),
            reranker=_FakeReranker(),
            reranking_settings=self._reranking_settings(),
        )

        result = retriever.retrieve("这篇论文有没有引用 Graph WaveNet？", result_limit=3)

        bibliography_chunk = result.chunks[-1]
        self.assertEqual(bibliography_chunk.chunk_id, "reference_1")
        self.assertIsNone(bibliography_chunk.pre_rerank_rank)
        self.assertIsNone(bibliography_chunk.rerank_score)


if __name__ == "__main__":
    unittest.main()
