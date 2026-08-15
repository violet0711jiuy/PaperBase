"""以固定、可解释的 RRF 规则融合稠密检索和 SQLite BM25 检索。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from paperbase.config import RerankingSettings, RetrievalSettings
from paperbase.database import MetadataDatabase
from paperbase.embedding.base import QueryEmbedder
from paperbase.indexing import FaissIndexStore
from paperbase.reranking import Reranker

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
    # 进入 Step 7 前在正文 RRF 队列中的名次；bibliography 直出候选为 None。
    pre_rerank_rank: int | None
    # Cross-Encoder 相关性分数；未重排或 bibliography 直出候选为 None。
    rerank_score: float | None
    fused_score: float
    source_matches: tuple[RetrievalSourceMatch, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """一次查询的完整检索快照；不包含 LLM 的最终回答。"""

    query: str
    rewrite_plan: QueryRewritePlan
    # reranking 状态：success、disabled 或 fallback；失败时正文仍按 RRF 顺序输出。
    reranking_status: str
    chunks: tuple[RetrievedChunk, ...]


@dataclass
class _Candidate:
    """RRF 聚合期间的内部可变状态，最终会转换为不可变公开对象。"""

    row: Any
    score: float = 0.0
    # Step 7 Cross-Encoder 分数；None 表示未参与正文重排序。
    rerank_score: float | None = None
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
        reranker: Reranker | None = None,
        reranking_settings: RerankingSettings | None = None,
    ) -> None:
        self._database = database
        self._query_embedder = query_embedder
        self._settings = settings
        self._query_rewriter = query_rewriter
        self._reranker = reranker
        self._reranking_settings = reranking_settings
        # 启动时验证索引、清单、SQLite 映射三者一致；不读取 staging 工件。
        self._index = index_store.load_for_search(database=database)

    def retrieve(
        self,
        query: str,
        *,
        conversation_context: Sequence[str] | None = None,
        result_limit: int | None = None,
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
        if result_limit is not None and result_limit < 1:
            raise ValueError("result_limit must be positive when provided.")
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
        bibliography_candidates: tuple[_Candidate, ...] = ()
        if plan.search_bibliography:
            # 先从正文混合召回中定位“这篇论文”的最强候选，再把参考文献检索限制在该论文内。
            # 无法定位目标论文时宁可不查 bibliography，也不能退化为全库宽泛主题词匹配。
            target_paper_id = _select_bibliography_target_paper_id(candidates)
            bibliography_candidates = self._collect_bibliography_candidates(
                query=(plan.semantic_query or plan.original_query),
                keywords=plan.lexical_keywords_en,
                target_paper_id=target_paper_id,
            )

        rrf_ordered_content = sorted(
            candidates.values(),
            key=lambda item: (-item.score, str(item.row["chunk_id"])),
        )
        reranking_status = "disabled"
        ordered_content, reranking_status = self._rerank_content_candidates(
            query=(plan.semantic_query or plan.original_query),
            candidates=rrf_ordered_content,
        )
        if plan.search_bibliography:
            # bibliography 候选不参与正文 RRF 或 Cross-Encoder；正文和参考文献分别按自身排序后直接拼接。
            # --top-k 显式指定时它表示总数；未指定时默认输出正文 final_top_k 加参考文献配额。
            output_limit = result_limit or self._default_output_limit(include_bibliography=True)
            direct_bibliography = bibliography_candidates[
                : min(self._settings.query_rewrite.bibliography_top_k, output_limit)
            ]
            content_output_limit = output_limit - len(direct_bibliography)
            ordered = [
                *ordered_content[:content_output_limit],
                *direct_bibliography,
            ]
        else:
            output_limit = result_limit or self._default_output_limit(include_bibliography=False)
            content_output_limit = output_limit
            ordered = ordered_content[:content_output_limit]
        return RetrievalResult(
            query=plan.original_query,
            rewrite_plan=plan,
            reranking_status=reranking_status,
            chunks=tuple(
                _to_retrieved_chunk(
                    candidate,
                    rank=index + 1,
                    pre_rerank_rank=(
                        _find_pre_rerank_rank(rrf_ordered_content, candidate)
                        if str(candidate.row["section_type"]) == "content"
                        else None
                    ),
                    rerank_score=_find_rerank_score(candidate),
                )
                for index, candidate in enumerate(ordered)
            ),
        )

    def _rerank_content_candidates(
        self,
        *,
        query: str,
        candidates: list[_Candidate],
    ) -> tuple[list[_Candidate], str]:
        """只对正文 RRF 候选进行 Cross-Encoder 重排；模型异常时无声退回原 RRF 顺序。"""
        if self._reranker is None:
            return candidates, "disabled"
        if self._reranking_settings is None or not self._reranking_settings.enabled:
            return candidates, "disabled"
        candidate_limit = min(len(candidates), self._reranking_settings.candidate_top_k)
        rerank_candidates = candidates[:candidate_limit]
        passages = [_build_rerank_passage(candidate.row) for candidate in rerank_candidates]
        try:
            scores = self._reranker.rerank(query, passages)
            if len(scores) != len(rerank_candidates):
                raise ValueError("Reranker score count does not match candidate count.")
            scores_by_index = {item.input_index: item.score for item in scores}
            if set(scores_by_index) != set(range(len(rerank_candidates))):
                raise ValueError("Reranker returned invalid candidate indexes.")
            # 将分数放入临时属性前先保留 RRF 原分，便于按分数相同的情况稳定回退到原排序。
            for input_index, candidate in enumerate(rerank_candidates):
                candidate.rerank_score = scores_by_index[input_index]
            reranked = sorted(
                rerank_candidates,
                key=lambda item: (-item.rerank_score, -item.score, str(item.row["chunk_id"])),
            )
            return [*reranked, *candidates[candidate_limit:]], "success"
        except Exception:
            # Reranker 是可选增强环节；本地模型缺失、显存不足等不能阻断已可靠运行的 Step 6 召回。
            return candidates, "fallback"

    def _default_output_limit(self, *, include_bibliography: bool) -> int:
        """未指定 CLI --top-k 时，按 Step 7 配置决定正文与参考文献的默认输出配额。"""
        if self._reranking_settings is None or not self._reranking_settings.enabled:
            content_limit = self._settings.fused_top_k
        else:
            content_limit = self._reranking_settings.final_top_k
        if include_bibliography:
            return content_limit + self._settings.query_rewrite.bibliography_top_k
        return content_limit

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

    def _collect_bibliography_candidates(
        self,
        *,
        query: str,
        keywords: tuple[str, ...],
        target_paper_id: str | None,
    ) -> tuple[_Candidate, ...]:
        """返回目标论文内按 BM25 排序的参考文献候选，不把它们加入正文 RRF。"""
        if target_paper_id is None:
            return ()
        # 与正文 Rewritten BM25 一致：关键词组合为一条 OR 查询，并如实写入调试来源字段。
        rendered_query = " OR ".join(keywords) if keywords else query
        rows = (
            self._database.search_bibliography_keyword_group(
                keywords,
                paper_id=target_paper_id,
                top_k=self._settings.query_rewrite.bibliography_top_k,
            )
            if keywords
            else self._database.search_bibliography(
                query,
                paper_id=target_paper_id,
                top_k=self._settings.query_rewrite.bibliography_top_k,
            )
        )
        return tuple(
            _Candidate(
                row=row,
                # 参考文献不参加 RRF；保留 0 明确表示 fused_score 对它不适用。
                score=0.0,
                sources=[
                    RetrievalSourceMatch(
                        route="bm25_bibliography",
                        query=rendered_query,
                        rank=result_index + 1,
                        raw_score=float(row["bm25_score"]),
                        # 0 表示此来源直接输出而非加入 RRF 权重计算。
                        effective_weight=0.0,
                    )
                ],
            )
            for result_index, row in enumerate(rows)
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


def _to_retrieved_chunk(
    candidate: _Candidate,
    *,
    rank: int,
    pre_rerank_rank: int | None,
    rerank_score: float | None,
) -> RetrievedChunk:
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
        pre_rerank_rank=pre_rerank_rank,
        rerank_score=rerank_score,
        fused_score=candidate.score,
        source_matches=tuple(candidate.sources),
    )


def _find_pre_rerank_rank(
    rrf_ordered_candidates: Sequence[_Candidate],
    candidate: _Candidate,
) -> int:
    """返回正文候选进入 Cross-Encoder 之前的 RRF 名次，从 1 开始。"""
    for index, item in enumerate(rrf_ordered_candidates, start=1):
        if item is candidate:
            return index
    raise ValueError("Content candidate is absent from the pre-rerank RRF ordering.")


def _find_rerank_score(candidate: _Candidate) -> float | None:
    """只暴露正文候选的 Cross-Encoder 分数；参考文献直出候选不具有该分数。"""
    return candidate.rerank_score


def _build_rerank_passage(row: Any) -> str:
    """为 Cross-Encoder 生成紧凑、可读的正文输入，不复用 embedding_text 的检索模板。"""
    title = " ".join(str(row["paper_title"] or "").split())
    section = " ".join(str(row["section"] or "").split())
    raw_text = " ".join(str(row["raw_text"] or "").split())
    if not raw_text:
        raise ValueError("Cannot rerank a chunk without raw_text.")
    context_lines = []
    if title:
        context_lines.append(f"Paper title: {title}")
    if section:
        context_lines.append(f"Section: {section}")
    context_lines.append(f"Passage: {raw_text}")
    return "\n".join(context_lines)


def _select_bibliography_target_paper_id(candidates: dict[str, _Candidate]) -> str | None:
    """从正文融合候选中选择证据最强的论文，作为 bibliography FTS5 的唯一检索范围。

    这里使用 chunk 的既有 RRF 分数而非新增一套主题分类器：同一分数下按 paper_id 稳定排序，
    不会因数据库返回顺序变化而更换目标论文。后续若支持用户显式选择论文，可直接以显式 paper_id 覆盖本函数。
    """
    scores_by_paper_id: dict[str, float] = {}
    for candidate in candidates.values():
        if str(candidate.row["section_type"]) != "content":
            continue
        paper_id = str(candidate.row["paper_id"])
        scores_by_paper_id[paper_id] = scores_by_paper_id.get(paper_id, 0.0) + candidate.score
    if not scores_by_paper_id:
        return None
    # 多个正文 chunk 共同支持同一篇论文时，累积证据强度，而不是只看某一条 chunk 的最高分。
    return min(scores_by_paper_id, key=lambda paper_id: (-scores_by_paper_id[paper_id], paper_id))
