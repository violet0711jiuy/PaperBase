"""Ask This Paper 的工作区隔离混合检索器。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from paperbase.config import RerankingSettings, RetrievalSettings
from paperbase.embedding.base import QueryEmbedder
from paperbase.reranking import Reranker
from paperbase.retrieval.hybrid_retriever import (
    RetrievedChunk,
    RetrievalResult,
    RetrievalSourceMatch,
)
from paperbase.retrieval.query_rewriter import QueryPlanner, TrustedPaperScope
from paperbase.staging.bm25 import WorkspaceBM25Index, WorkspaceBM25IndexCache
from paperbase.staging.sections import WorkspaceChunk, WorkspaceSectionSnapshot


class WorkspaceRetrievalError(RuntimeError):
    """临时索引、向量映射或本论文检索数据不一致时抛出。"""


class WorkspaceVectorIndex(Protocol):
    """工作区 FAISS 的窄接口，测试可注入替身而无需加载真实 FAISS。"""

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """返回与 FAISS 相同形状的 ``scores, vector_ids``。"""


@dataclass(frozen=True)
class LoadedWorkspaceVectorIndex:
    """已验证的 Temporary FAISS 与 ``vector_id -> chunk_id`` 映射。"""

    index: Any
    vector_to_chunk_id: dict[int, str]
    dimension: int

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """只接受二维 float32 查询向量，避免把异常结果静默送进 RRF。"""
        if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise WorkspaceRetrievalError("Query embeddings do not match the temporary FAISS index.")
        if top_k < 1:
            raise WorkspaceRetrievalError("Dense top_k must be positive.")
        scores, vector_ids = self.index.search(np.ascontiguousarray(vectors), top_k)
        return np.asarray(scores), np.asarray(vector_ids)


def load_workspace_vector_index(snapshot: WorkspaceSectionSnapshot) -> LoadedWorkspaceVectorIndex:
    """读取且交叉验证当前 workspace 的独立 FAISS 与 manifest，不接触正式索引。"""
    index_path = snapshot.root_dir / "index" / "paper.faiss"
    manifest_path = snapshot.root_dir / "index" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkspaceRetrievalError("Cannot read temporary FAISS manifest.") from error
    if not isinstance(manifest, dict) or manifest.get("included_section_type") != "content":
        raise WorkspaceRetrievalError("Temporary FAISS manifest is not a content-only index.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise WorkspaceRetrievalError("Temporary FAISS manifest has no vector records.")
    try:
        dimension = int(manifest["dimension"])
        vector_to_chunk_id = {
            int(record["vector_id"]): str(record["chunk_id"])
            for record in records
            if isinstance(record, dict)
        }
    except (KeyError, TypeError, ValueError) as error:
        raise WorkspaceRetrievalError("Temporary FAISS manifest record is invalid.") from error
    if len(vector_to_chunk_id) != len(records) or min(vector_to_chunk_id, default=0) < 1:
        raise WorkspaceRetrievalError("Temporary FAISS manifest vector IDs are invalid.")
    expected_chunk_ids = {
        chunk.chunk_id for chunk in snapshot.chunks if chunk.section_type == "content"
    }
    if set(vector_to_chunk_id.values()) != expected_chunk_ids:
        raise WorkspaceRetrievalError("Temporary FAISS manifest does not match workspace content chunks.")
    try:
        import faiss

        index = faiss.deserialize_index(np.frombuffer(index_path.read_bytes(), dtype=np.uint8))
        stored_ids = {int(value) for value in faiss.vector_to_array(index.id_map)}
    except Exception as error:
        raise WorkspaceRetrievalError("Cannot load temporary FAISS index.") from error
    if int(index.d) != dimension or int(index.ntotal) != len(vector_to_chunk_id) or stored_ids != set(vector_to_chunk_id):
        raise WorkspaceRetrievalError("Temporary FAISS index does not match its manifest.")
    return LoadedWorkspaceVectorIndex(
        index=index,
        vector_to_chunk_id=vector_to_chunk_id,
        dimension=dimension,
    )


@dataclass
class _Candidate:
    """工作区 RRF 的内部候选；语义与正式 HybridRetriever 保持一致。"""

    chunk: WorkspaceChunk
    vector_id: int | None
    score: float = 0.0
    rerank_score: float | None = None
    sources: list[RetrievalSourceMatch] = field(default_factory=list)


class WorkspaceHybridRetriever:
    """只在一个 Temporary Workspace 内运行 Dense + BM25 + RRF + Reranker。"""

    def __init__(
        self,
        *,
        snapshot: WorkspaceSectionSnapshot,
        query_embedder: QueryEmbedder,
        vector_index: WorkspaceVectorIndex,
        vector_to_chunk_id: dict[int, str],
        content_bm25: WorkspaceBM25Index,
        bibliography_bm25: WorkspaceBM25Index | None,
        settings: RetrievalSettings,
        query_planner: QueryPlanner,
        reranker: Reranker | None = None,
        reranking_settings: RerankingSettings | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._query_embedder = query_embedder
        self._vector_index = vector_index
        self._vector_to_chunk_id = dict(vector_to_chunk_id)
        self._content_bm25 = content_bm25
        self._bibliography_bm25 = bibliography_bm25
        self._settings = settings
        self._query_planner = query_planner
        self._reranker = reranker
        self._reranking_settings = reranking_settings
        self._content_by_id = {
            chunk.chunk_id: chunk
            for chunk in snapshot.chunks
            if chunk.section_type == "content"
        }

    @classmethod
    def from_workspace(
        cls,
        *,
        snapshot: WorkspaceSectionSnapshot,
        query_embedder: QueryEmbedder,
        settings: RetrievalSettings,
        query_planner: QueryPlanner,
        reranker: Reranker | None,
        reranking_settings: RerankingSettings | None,
        bm25_cache: WorkspaceBM25IndexCache,
    ) -> "WorkspaceHybridRetriever":
        """装配当前论文的三项本地检索工件；不创建 SQLite 或重新 embedding。"""
        vector_index = load_workspace_vector_index(snapshot)
        bibliography_bm25: WorkspaceBM25Index | None = None
        try:
            bibliography_bm25 = bm25_cache.get_or_build(
                snapshot, section_type="bibliography"
            )
        except ValueError:
            # 没有 references 是合法论文状态；仅在引用问题时自然返回空辅助结果。
            bibliography_bm25 = None
        return cls(
            snapshot=snapshot,
            query_embedder=query_embedder,
            vector_index=vector_index,
            vector_to_chunk_id=vector_index.vector_to_chunk_id,
            content_bm25=bm25_cache.get_or_build(snapshot),
            bibliography_bm25=bibliography_bm25,
            settings=settings,
            query_planner=query_planner,
            reranker=reranker,
            reranking_settings=reranking_settings,
        )

    def retrieve(
        self,
        query: str,
        *,
        trusted_scope: TrustedPaperScope | None = None,
        conversation_context: Sequence[str] | None = None,
        result_limit: int | None = None,
    ) -> RetrievalResult:
        """运行四路正文召回；参考文献只在明确意图时本论文内直接返回。"""
        if result_limit is not None and result_limit < 1:
            raise ValueError("result_limit must be positive when provided.")
        plan = self._query_planner.plan(
            query,
            trusted_scope=trusted_scope,
            conversation_context=conversation_context,
        )
        if plan.resolution_status == "unresolved":
            return RetrievalResult(
                query=plan.original_query,
                rewrite_plan=plan,
                reranking_status="not_run",
                chunks=(),
            )
        resolved_query = plan.resolved_query or plan.original_query
        candidates: dict[str, _Candidate] = {}
        self._add_dense_matches(
            candidates, route="dense_resolved", queries=(resolved_query,),
            weight=self._settings.dense_resolved_weight,
        )
        if plan.semantic_query_en:
            self._add_dense_matches(
                candidates, route="dense_semantic", queries=(plan.semantic_query_en,),
                weight=self._settings.dense_semantic_weight,
            )
        if plan.lexical_keywords_en:
            self._add_bm25_matches(
                candidates,
                route="bm25_keywords",
                query=" OR ".join(plan.lexical_keywords_en),
                top_k=self._settings.bm25_keywords_top_k,
                weight=self._settings.bm25_keywords_weight,
                keywords=plan.lexical_keywords_en,
            )
        ordered = sorted(candidates.values(), key=lambda item: (-item.score, item.chunk.chunk_id))
        reranked, reranking_status = self._rerank(
            resolved_query,
            ordered,
        )
        bibliography = self._retrieve_bibliography(plan) if plan.search_bibliography else ()
        output_limit = result_limit or self._default_output_limit(include_bibliography=bool(bibliography))
        content_limit = max(0, output_limit - len(bibliography))
        selected = (*reranked[:content_limit], *bibliography[:output_limit])
        return RetrievalResult(
            query=plan.original_query,
            rewrite_plan=plan,
            reranking_status=reranking_status,
            chunks=tuple(
                self._to_retrieved(candidate, rank=rank, pre_rerank=ordered)
                for rank, candidate in enumerate(selected, start=1)
            ),
        )

    def _add_dense_matches(
        self,
        candidates: dict[str, _Candidate],
        *, route: str,
        queries: tuple[str, ...],
        weight: float,
    ) -> None:
        if not queries:
            return
        vectors = np.asarray(
            self._query_embedder.embed_queries(
                list(queries), instruction=self._settings.query_instruction
            ),
            dtype=np.float32,
        )
        scores, vector_ids = self._vector_index.search(vectors, self._settings.dense_top_k_per_query)
        for query_index, current_query in enumerate(queries):
            for result_index, vector_id in enumerate(vector_ids[query_index]):
                integer_id = int(vector_id)
                chunk_id = self._vector_to_chunk_id.get(integer_id)
                chunk = self._content_by_id.get(chunk_id or "")
                if chunk is None:
                    continue
                self._add_candidate(
                    candidates,
                    chunk=chunk,
                    vector_id=integer_id,
                    route=route,
                    query=current_query,
                    rank=result_index + 1,
                    raw_score=float(scores[query_index][result_index]),
                    weight=weight,
                )

    def _add_bm25_matches(
        self,
        candidates: dict[str, _Candidate],
        *, route: str,
        query: str,
        top_k: int,
        weight: float,
        keywords: tuple[str, ...] = (),
    ) -> None:
        # 英文关键词组是正式链路中的一条 OR 路径；内存 BM25 逐词评分，仍只贡献一次 RRF 权重。
        matches = self._content_bm25.search(" ".join(keywords) if keywords else query, top_k=top_k)
        for match in matches:
            self._add_candidate(
                candidates,
                chunk=match.chunk,
                vector_id=self._vector_id_for_chunk(match.chunk.chunk_id),
                route=route,
                query=query,
                rank=match.rank,
                raw_score=match.bm25_score,
                weight=weight,
            )

    def _add_candidate(
        self,
        candidates: dict[str, _Candidate],
        *, chunk: WorkspaceChunk,
        vector_id: int | None,
        route: str,
        query: str,
        rank: int,
        raw_score: float,
        weight: float,
    ) -> None:
        candidate = candidates.setdefault(chunk.chunk_id, _Candidate(chunk=chunk, vector_id=vector_id))
        candidate.score += weight / (self._settings.rrf_k + rank)
        candidate.sources.append(
            RetrievalSourceMatch(
                route=route, query=query, rank=rank, raw_score=raw_score, effective_weight=weight
            )
        )

    def _rerank(
        self, query: str, candidates: list[_Candidate]
    ) -> tuple[list[_Candidate], str]:
        if not candidates:
            return [], "disabled" if self._reranker is None else "success"
        if self._reranker is None or self._reranking_settings is None or not self._reranking_settings.enabled:
            return candidates[: self._settings.fused_top_k], "disabled"
        selected = candidates[: self._reranking_settings.candidate_top_k]
        try:
            scores = self._reranker.rerank(query, [_rerank_passage(item.chunk, self._title) for item in selected])
            if {score.input_index for score in scores} != set(range(len(selected))):
                raise ValueError("Reranker result does not cover all workspace candidates.")
            for score in scores:
                selected[score.input_index].rerank_score = score.score
            reranked = sorted(selected, key=lambda item: (-(item.rerank_score or 0.0), -item.score, item.chunk.chunk_id))
            return reranked[: self._reranking_settings.final_top_k], "success"
        except Exception:
            return candidates[: self._settings.fused_top_k], "fallback"

    def _retrieve_bibliography(self, plan: object) -> tuple[_Candidate, ...]:
        if self._bibliography_bm25 is None:
            return ()
        keywords = tuple(getattr(plan, "lexical_keywords_en", ()))
        query = " ".join(keywords) if keywords else str(
            getattr(plan, "semantic_query_en", None)
            or getattr(plan, "resolved_query", None)
            or getattr(plan, "original_query")
        )
        matches = self._bibliography_bm25.search(query, top_k=self._settings.query_rewrite.bibliography_top_k)
        return tuple(
            _Candidate(
                chunk=match.chunk,
                vector_id=None,
                sources=[RetrievalSourceMatch(
                    route="bm25_bibliography", query=query, rank=match.rank,
                    raw_score=match.bm25_score, effective_weight=0.0,
                )],
            )
            for match in matches
        )

    def _to_retrieved(
        self, candidate: _Candidate, *, rank: int, pre_rerank: Sequence[_Candidate]
    ) -> RetrievedChunk:
        chunk = candidate.chunk
        is_content = chunk.section_type == "content"
        pre_rank = next((index for index, item in enumerate(pre_rerank, start=1) if item is candidate), None)
        return RetrievedChunk(
            rank=rank,
            chunk_id=chunk.chunk_id,
            vector_id=candidate.vector_id,
            paper_id=self._snapshot.paper_id,
            paper_title=chunk.paper_title or self._title,
            section=chunk.section or "",
            content_kind=chunk.content_kind,
            front_matter_type=chunk.front_matter_type,
            section_type=chunk.section_type,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            raw_text=chunk.raw_text,
            pre_rerank_rank=pre_rank if is_content else None,
            rerank_score=candidate.rerank_score if is_content else None,
            fused_score=candidate.score,
            source_matches=tuple(candidate.sources),
        )

    @property
    def _title(self) -> str:
        return self._snapshot.paper_title or "论文标题未识别"

    def _vector_id_for_chunk(self, chunk_id: str) -> int | None:
        return next((vector_id for vector_id, item_id in self._vector_to_chunk_id.items() if item_id == chunk_id), None)

    def _default_output_limit(self, *, include_bibliography: bool) -> int:
        base = self._reranking_settings.final_top_k if self._reranking_settings else self._settings.fused_top_k
        return base + (self._settings.query_rewrite.bibliography_top_k if include_bibliography else 0)


def _rerank_passage(chunk: WorkspaceChunk, paper_title: str) -> str:
    """与正式 Reranker 相同的紧凑 passage 格式，不把 embedding_text 送入 Cross-Encoder。"""
    lines = [f"Paper title: {paper_title}"]
    if chunk.section:
        lines.append(f"Section: {chunk.section}")
    lines.append(f"Passage: {chunk.raw_text}")
    return "\n".join(lines)
