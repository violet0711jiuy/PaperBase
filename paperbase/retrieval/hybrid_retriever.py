"""以固定、可解释的 RRF 规则融合稠密检索和 SQLite BM25 检索。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from paperbase.config import RetrievalSettings
from paperbase.database import MetadataDatabase
from paperbase.embedding.base import QueryEmbedder
from paperbase.indexing import FaissIndexStore

from .query_rewriter import QueryRewritePlan, QueryRewriter


@dataclass(frozen=True)
class RetrievalSourceMatch:
    """某条结果在一条具体召回路径中的证据，供调试、评估与后续重排使用。"""

    route: str
    query: str
    rank: int
    raw_score: float
    effective_weight: float


@dataclass(frozen=True)
class RetrievedChunk:
    """融合后的一个 chunk，含原文、元数据和全部召回来源。"""

    rank: int
    chunk_id: str
    vector_id: int | None
    paper_id: str
    paper_title: str
    section: str
    content_kind: str
    front_matter_type: str | None
    section_type: str
    page_start: int | None
    page_end: int | None
    raw_text: str
    fused_score: float
    source_matches: tuple[RetrievalSourceMatch, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """一次查询的完整检索快照；不包含 LLM 的最终回答。"""

    query: str
    rewrite_plan: QueryRewritePlan
    chunks: tuple[RetrievedChunk, ...]


@dataclass
class _Candidate:
    """RRF 聚合期间的内部可变状态，最终会转换为不可变公开对象。"""

    row: Any
    score: float = 0.0
    sources: list[RetrievalSourceMatch] = field(default_factory=list)


class HybridRetriever:
    """Step 6 的在线检索器。

    FAISS 在构造时由正式索引与 SQLite 映射共同校验，因此运行时不依赖 Step 4 的
    ``vectors.npy`` 和 ``records.jsonl`` 中间产物。所有通道均只产生候选，跨通道
    只按名次做加权 RRF，避免把余弦相似度与 BM25 的不同量纲直接相加。
    """

    def __init__(
        self,
        *,
        database: MetadataDatabase,
        query_embedder: QueryEmbedder,
        index_store: FaissIndexStore,
        settings: RetrievalSettings,
        query_rewriter: QueryRewriter,
    ) -> None:
        self._database = database
        self._query_embedder = query_embedder
        self._settings = settings
        self._query_rewriter = query_rewriter
        # 启动时验证索引、清单、SQLite 映射三者一致；不读取 staging 工件。
        self._index = index_store.load_for_search(database=database)

    def retrieve(
        self,
        query: str,
        *,
        conversation_context: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """运行四路召回并返回按 chunk_id 去重后的 Top-K chunk。

        1. 原始问题 Dense Top-20；2. 原始问题 BM25 Top-10；3. 一条语义改写 Dense Top-20；
        4. 英文关键词组 BM25 Top-20。RRF 聚合字典以 chunk_id 为键，因此融合期间便已去重。
        """
        # 上下文只交给改写器消解指代；原始 Dense/BM25 路径始终只使用当前用户问题。
        plan = self._query_rewriter.rewrite(
            query,
            conversation_context=conversation_context,
        )
        candidates: dict[str, _Candidate] = {}

        self._add_dense_matches(
            candidates=candidates,
            route="dense_original",
            queries=(plan.original_query,),
            group_weight=self._settings.dense_original_weight,
        )
        self._add_dense_matches(
            candidates=candidates,
            route="dense_rewrite",
            queries=(plan.semantic_query,) if plan.semantic_query is not None else (),
            group_weight=self._settings.dense_rewrite_weight,
        )
        self._add_original_bm25_matches(
            candidates=candidates,
            route="bm25_original",
            query=plan.original_query,
            top_k=self._settings.bm25_original_top_k,
            weight=self._settings.bm25_original_weight,
        )
        self._add_rewritten_bm25_matches(
            candidates=candidates,
            keywords=plan.lexical_keywords_en,
            weight=self._settings.bm25_rewrite_weight,
        )
        if plan.search_bibliography:
            self._add_bibliography_matches(
                candidates=candidates,
                query=(plan.semantic_query or plan.original_query),
                keywords=plan.lexical_keywords_en,
            )

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item.score, str(item.row["chunk_id"])),
        )
        if plan.search_bibliography:
            bibliography_limit = self._settings.query_rewrite.bibliography_top_k
            content_limit = max(self._settings.fused_top_k - bibliography_limit, 0)
            content_candidates = [
                item for item in ordered if str(item.row["section_type"]) == "content"
            ][:content_limit]
            bibliography_candidates = [
                item for item in ordered if str(item.row["section_type"]) == "bibliography"
            ][:bibliography_limit]
            ordered = sorted(
                (*content_candidates, *bibliography_candidates),
                key=lambda item: (-item.score, str(item.row["chunk_id"])),
            )
        else:
            ordered = ordered[: self._settings.fused_top_k]
        return RetrievalResult(
            query=plan.original_query,
            rewrite_plan=plan,
            chunks=tuple(
                _to_retrieved_chunk(candidate, rank=index + 1)
                for index, candidate in enumerate(ordered)
            ),
        )

    def _add_dense_matches(
        self,
        *,
        candidates: dict[str, _Candidate],
        route: str,
        queries: tuple[str, ...],
        group_weight: float,
    ) -> None:
        """批量编码原始问题或单条语义改写问题；每一路只对应一条 Dense 查询。"""
        if not queries:
            return
        vectors = np.asarray(
            self._query_embedder.embed_queries(
                list(queries), instruction=self._settings.query_instruction
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(queries):
            raise RuntimeError("Query embedder returned a matrix inconsistent with query count.")
        scores, vector_ids = self._index.search(vectors, self._settings.dense_top_k_per_query)
        all_ids = [int(vector_id) for vector_id in vector_ids.reshape(-1) if int(vector_id) > 0]
        rows_by_vector_id = {
            int(row["vector_id"]): row
            for row in self._database.chunks_by_vector_ids(all_ids)
        }
        for query_index, current_query in enumerate(queries):
            for result_index, vector_id in enumerate(vector_ids[query_index]):
                vector_id = int(vector_id)
                if vector_id < 1 or vector_id not in rows_by_vector_id:
                    continue
                self._add_candidate(
                    candidates=candidates,
                    row=rows_by_vector_id[vector_id],
                    route=route,
                    query=current_query,
                    rank=result_index + 1,
                    raw_score=float(scores[query_index][result_index]),
                    effective_weight=group_weight,
                )

    def _add_original_bm25_matches(
        self,
        *,
        candidates: dict[str, _Candidate],
        route: str,
        query: str,
        top_k: int,
        weight: float,
    ) -> None:
        """执行原始问题 BM25 Top-10；RRF 只使用排名而不混用 BM25 原始分数。"""
        rows = self._database.search_bm25(query, top_k=top_k)
        self._add_bm25_rows(
            candidates=candidates,
            rows=rows,
            route=route,
            query=query,
            weight=weight,
        )

    def _add_rewritten_bm25_matches(
        self,
        *,
        candidates: dict[str, _Candidate],
        keywords: tuple[str, ...],
        weight: float,
    ) -> None:
        """将英文关键词组作为一条 Rewritten BM25 Top-20 路径，而非三条独立查询。"""
        if not keywords:
            return
        rows = self._database.search_bm25_keyword_group(
            keywords, top_k=self._settings.bm25_rewrite_top_k
        )
        # 对外证据使用可读的 OR 文本；SQLite 内部会对每个关键词安全转义并编译为 FTS5 表达式。
        rendered_query = " OR ".join(keywords)
        self._add_bm25_rows(
            candidates=candidates,
            rows=rows,
            route="bm25_rewrite",
            query=rendered_query,
            weight=weight,
        )

    def _add_bm25_rows(
        self,
        *,
        candidates: dict[str, _Candidate],
        rows: tuple[Any, ...],
        route: str,
        query: str,
        weight: float,
    ) -> None:
        """把一条 BM25 路径返回的候选按名次加入 RRF，保证该路径只贡献一次权重。"""
        for result_index, row in enumerate(rows):
            self._add_candidate(
                candidates=candidates,
                row=row,
                route=route,
                query=query,
                rank=result_index + 1,
                raw_score=float(row["bm25_score"]),
                effective_weight=weight,
            )

    def _add_bibliography_matches(
        self,
        *,
        candidates: dict[str, _Candidate],
        query: str,
        keywords: tuple[str, ...],
    ) -> None:
        """将参考文献作为少量辅助候选，绝不进入主 Dense/正文 BM25 索引。"""
        rows = (
            self._database.search_bibliography_keyword_group(
                keywords,
                top_k=self._settings.query_rewrite.bibliography_top_k,
            )
            if keywords
            else self._database.search_bibliography(
                query,
                top_k=self._settings.query_rewrite.bibliography_top_k,
            )
        )
        self._add_bm25_rows(
            candidates=candidates,
            rows=rows,
            route="bm25_bibliography",
            query=query,
            # 作为辅助证据采用较低权重，避免少量 bibliography 候选压过正文证据。
            weight=0.35,
        )

    def _add_candidate(
        self,
        *,
        candidates: dict[str, _Candidate],
        row: Any,
        route: str,
        query: str,
        rank: int,
        raw_score: float,
        effective_weight: float,
    ) -> None:
        """将一个通道命中并入 chunk 级 RRF 分数，并完整保留其来源。"""
        chunk_id = str(row["chunk_id"])
        candidate = candidates.setdefault(chunk_id, _Candidate(row=row))
        candidate.score += effective_weight / (self._settings.rrf_k + rank)
        candidate.sources.append(
            RetrievalSourceMatch(
                route=route,
                query=query,
                rank=rank,
                raw_score=raw_score,
                effective_weight=effective_weight,
            )
        )


def _to_retrieved_chunk(candidate: _Candidate, *, rank: int) -> RetrievedChunk:
    """将 SQLite 行转换为 API 稳定的数据类，避免上层依赖 sqlite3.Row。"""
    row = candidate.row
    return RetrievedChunk(
        rank=rank,
        chunk_id=str(row["chunk_id"]),
        vector_id=int(row["vector_id"]) if row["vector_id"] is not None else None,
        paper_id=str(row["paper_id"]),
        paper_title=str(row["paper_title"]),
        section=str(row["section"]),
        content_kind=str(row["content_kind"]),
        front_matter_type=(
            str(row["front_matter_type"]) if row["front_matter_type"] is not None else None
        ),
        section_type=str(row["section_type"]),
        page_start=int(row["page_start"]) if row["page_start"] is not None else None,
        page_end=int(row["page_end"]) if row["page_end"] is not None else None,
        raw_text=str(row["raw_text"]),
        fused_score=candidate.score,
        source_matches=tuple(candidate.sources),
    )
