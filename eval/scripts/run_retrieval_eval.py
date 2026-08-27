"""使用 PaperBase 正式 HybridRetriever 运行 Full Retrieval Baseline。

评测器只负责读取 Golden、观察正式检索调用、计算确定性指标并写报告；Dense、
BM25、Weighted RRF、Query Rewrite 与 Reranker 均由产品现有实现执行。本脚本不会
修改 Golden Dataset、正式检索参数或生产知识库。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
# 直接执行 ``python eval/scripts/...py`` 时，Python 只把脚本目录加入模块路径；
# 显式加入项目根目录，保证命令无需安装 PaperBase 包也能复用正式模块。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperbase.config import AppSettings, default_config_path, load_settings
from paperbase.database import MetadataDatabase
from paperbase.generation.section_expander import (
    ExpansionResult,
    SectionAwareNeighborExpander,
)
from paperbase.retrieval.hybrid_retriever import RetrievalResult
from paperbase.retrieval.service import create_hybrid_retriever


DEFAULT_DATASET = ROOT / "eval" / "datasets" / "golden_dataset_v1_2.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "results" / "retrieval_full"
ROUTE_TO_STAGE = {
    "dense_resolved": "original_dense",
    "dense_semantic": "rewritten_dense",
    "bm25_keywords": "rewritten_bm25",
}


class RetrievalEvaluationError(RuntimeError):
    """表示数据、配置或正式知识库不满足离线评测前置条件。"""


@dataclass
class RetrievalTrace:
    """保存一次正式 retrieve 调用中可无侵入观察到的阶段结果和耗时。"""

    route_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    rrf_results: list[dict[str, Any]] = field(default_factory=list)
    reranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    bibliography_results: list[dict[str, Any]] = field(default_factory=list)
    rewrite_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    dense_route_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_retrieval_latency_ms: float = 0.0

    def reset(self) -> None:
        """为下一条 Case 分配新容器，避免清空已写入上一条结果的排名快照。"""
        # 不能对旧列表调用 clear：Case 结果会保留这些列表的引用，原地清空会让此前
        # 已正确计算指标的 Case 丢失 RRF/reranker trace。
        self.route_results = {}
        self.rrf_results = []
        self.reranked_candidates = []
        self.bibliography_results = []
        self.rewrite_latency_ms = 0.0
        self.embedding_latency_ms = 0.0
        self.dense_route_latency_ms = 0.0
        self.bm25_latency_ms = 0.0
        self.rerank_latency_ms = 0.0
        self.total_retrieval_latency_ms = 0.0

    def latency_payload(self) -> dict[str, float | None]:
        """按设计文档字段输出耗时；不可独立观察的 fusion 明确保留为 null。"""
        dense_without_embedding = max(
            0.0, self.dense_route_latency_ms - self.embedding_latency_ms
        )
        return {
            "rewrite_latency_ms": _rounded(self.rewrite_latency_ms),
            "embedding_latency_ms": _rounded(self.embedding_latency_ms),
            "dense_latency_ms": _rounded(dense_without_embedding),
            "bm25_latency_ms": _rounded(self.bm25_latency_ms),
            # 正式实现的 Python 排序位于两个可观察方法之间，无法在不改产品代码时单独计时。
            "fusion_latency_ms": None,
            "rerank_latency_ms": _rounded(self.rerank_latency_ms),
            "total_retrieval_latency_ms": _rounded(self.total_retrieval_latency_ms),
        }


class _TimedPlanner:
    """透明代理正式 Query Planner，仅累计 plan 调用耗时。"""

    def __init__(self, target: object, trace: RetrievalTrace) -> None:
        self._target = target
        self._trace = trace

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        """原样转发 plan 参数和返回值，不改变 Query Rewrite 行为。"""
        started = perf_counter()
        try:
            return self._target.plan(*args, **kwargs)
        finally:
            self._trace.rewrite_latency_ms += _elapsed_ms(started)


class _TimedEmbedder:
    """透明代理正式 Query Embedder，仅累计查询向量生成耗时。"""

    def __init__(self, target: object, trace: RetrievalTrace) -> None:
        self._target = target
        self._trace = trace

    def embed_queries(self, *args: Any, **kwargs: Any) -> Any:
        """原样执行正式 embed_queries，并记录真实模型调用时间。"""
        started = perf_counter()
        try:
            return self._target.embed_queries(*args, **kwargs)
        finally:
            self._trace.embedding_latency_ms += _elapsed_ms(started)

    def __getattr__(self, name: str) -> Any:
        """其余模型属性继续由正式 Embedder 提供。"""
        return getattr(self._target, name)


class _TimedReranker:
    """透明代理正式 Cross-Encoder，仅累计 rerank 调用耗时。"""

    def __init__(self, target: object, trace: RetrievalTrace) -> None:
        self._target = target
        self._trace = trace

    def rerank(self, *args: Any, **kwargs: Any) -> Any:
        """原样执行正式 reranker，任何异常仍交由产品 fallback 逻辑处理。"""
        started = perf_counter()
        try:
            return self._target.rerank(*args, **kwargs)
        finally:
            self._trace.rerank_latency_ms += _elapsed_ms(started)

    def __getattr__(self, name: str) -> Any:
        """向配置记录暴露正式 reranker 的 model_id/backend_id。"""
        return getattr(self._target, name)


class ProductionRetrieverObserver:
    """在不复制检索逻辑的前提下，为正式 HybridRetriever 增加评测观察点。"""

    def __init__(self, retriever: object) -> None:
        self.retriever = retriever
        self.trace = RetrievalTrace()
        self._install_proxies()

    def _install_proxies(self) -> None:
        """替换实例级依赖为透明计时代理，并包装已有私有阶段边界。"""
        self.retriever._query_planner = _TimedPlanner(  # type: ignore[attr-defined]
            self.retriever._query_planner, self.trace  # type: ignore[attr-defined]
        )
        self.retriever._query_embedder = _TimedEmbedder(  # type: ignore[attr-defined]
            self.retriever._query_embedder, self.trace  # type: ignore[attr-defined]
        )
        if self.retriever._reranker is not None:  # type: ignore[attr-defined]
            self.retriever._reranker = _TimedReranker(  # type: ignore[attr-defined]
                self.retriever._reranker, self.trace  # type: ignore[attr-defined]
            )

        original_dense = self.retriever._add_dense_matches  # type: ignore[attr-defined]
        original_bm25 = self.retriever._add_rewritten_bm25_matches  # type: ignore[attr-defined]
        original_rerank = self.retriever._rerank_content_candidates  # type: ignore[attr-defined]
        original_bibliography = self.retriever._collect_bibliography_candidates  # type: ignore[attr-defined]

        def traced_dense(*args: Any, **kwargs: Any) -> Any:
            """记录整条 Dense route 的耗时，结果仍由正式方法写入 candidates。"""
            started = perf_counter()
            try:
                return original_dense(*args, **kwargs)
            finally:
                self.trace.dense_route_latency_ms += _elapsed_ms(started)

        def traced_bm25(*args: Any, **kwargs: Any) -> Any:
            """记录正式 rewritten BM25 route 耗时。"""
            started = perf_counter()
            try:
                return original_bm25(*args, **kwargs)
            finally:
                self.trace.bm25_latency_ms += _elapsed_ms(started)

        def traced_rerank(*args: Any, **kwargs: Any) -> Any:
            """在正式 Reranker 前保存 RRF 队列，并在返回后保存完整候选顺序。"""
            candidates = kwargs.get("candidates")
            if candidates is None and len(args) >= 2:
                candidates = args[1]
            candidate_list = list(candidates or [])
            self.trace.rrf_results = [
                _candidate_to_record(candidate, rank)
                for rank, candidate in enumerate(candidate_list, 1)
            ]
            self.trace.route_results = _route_rankings(candidate_list)
            result = original_rerank(*args, **kwargs)
            ordered, _status = result
            self.trace.reranked_candidates = [
                _candidate_to_record(candidate, rank)
                for rank, candidate in enumerate(ordered, 1)
            ]
            return result

        def traced_bibliography(*args: Any, **kwargs: Any) -> Any:
            """记录 bibliography FTS5 的真实候选与耗时，不加入正文 RRF。"""
            started = perf_counter()
            try:
                result = original_bibliography(*args, **kwargs)
                self.trace.bibliography_results = [
                    _candidate_to_record(candidate, rank)
                    for rank, candidate in enumerate(result, 1)
                ]
                return result
            finally:
                self.trace.bm25_latency_ms += _elapsed_ms(started)

        self.retriever._add_dense_matches = traced_dense  # type: ignore[attr-defined]
        self.retriever._add_rewritten_bm25_matches = traced_bm25  # type: ignore[attr-defined]
        self.retriever._rerank_content_candidates = traced_rerank  # type: ignore[attr-defined]
        self.retriever._collect_bibliography_candidates = traced_bibliography  # type: ignore[attr-defined]

    def retrieve(self, query: str) -> tuple[RetrievalResult, RetrievalTrace]:
        """不注入论文范围，只用原始 Query 调用一次正式 retrieve。"""
        self.trace.reset()
        started = perf_counter()
        try:
            # Context-free Benchmark 禁止把 Golden paper_id 传给检索器或 Query Planner。
            result = self.retriever.retrieve(query)
        finally:
            self.trace.total_retrieval_latency_ms = _elapsed_ms(started)
        return result, self.trace


def _elapsed_ms(started: float) -> float:
    """把 perf_counter 的时间差转换为毫秒。"""
    return (perf_counter() - started) * 1000.0


def _rounded(value: float) -> float:
    """统一将耗时和指标保留三位小数，兼顾可读性与复算精度。"""
    return round(float(value), 3)


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    """兼容 sqlite3.Row 与测试字典，安全读取候选元数据字段。"""
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return default


def _candidate_to_record(candidate: object, rank: int) -> dict[str, Any]:
    """把正式 Retriever 内部候选转换为不含正文的可审计排名记录。"""
    row = candidate.row
    section_type = str(_row_value(row, "section_type", ""))
    direct_source = candidate.sources[0] if candidate.sources else None
    is_direct_bibliography = section_type == "bibliography" and direct_source is not None
    page_start = _row_value(row, "page_start")
    page_end = _row_value(row, "page_end")
    return {
        "chunk_id": str(_row_value(row, "chunk_id", "")),
        "rank": rank,
        "score": _rounded(
            direct_source.raw_score if is_direct_bibliography else candidate.score
        ),
        "score_type": "bm25" if is_direct_bibliography else "weighted_rrf",
        "fused_score": _rounded(candidate.score),
        "rerank_score": (
            _rounded(candidate.rerank_score)
            if candidate.rerank_score is not None
            else None
        ),
        "paper_id": str(_row_value(row, "paper_id", "")),
        "section": _row_value(row, "section"),
        "section_type": section_type,
        "page": {"start": page_start, "end": page_end},
        "page_start": page_start,
        "page_end": page_end,
    }


def _route_rankings(candidates: Sequence[object]) -> dict[str, list[dict[str, Any]]]:
    """从正式 Candidate 的 source_matches 还原各真实召回路径排名。"""
    rankings: dict[str, list[dict[str, Any]]] = {stage: [] for stage in ROUTE_TO_STAGE.values()}
    for candidate in candidates:
        base = _candidate_to_record(candidate, 0)
        for source in candidate.sources:
            stage = ROUTE_TO_STAGE.get(source.route)
            if stage is None:
                continue
            item = dict(base)
            item.update(
                {
                    "rank": int(source.rank),
                    "score": _rounded(source.raw_score),
                    "score_type": (
                        "dense_similarity" if source.route.startswith("dense_") else "bm25"
                    ),
                    "query": source.query,
                    "route": source.route,
                    "rrf_weight": source.effective_weight,
                }
            )
            rankings[stage].append(item)
    for items in rankings.values():
        items.sort(key=lambda item: (item["rank"], item["chunk_id"]))
    return rankings


def _retrieved_to_record(chunk: object) -> dict[str, Any]:
    """把正式公开 RetrievedChunk 转为最终结果记录。"""
    direct_bibliography_score = (
        chunk.source_matches[0].raw_score
        if chunk.section_type == "bibliography" and chunk.source_matches
        else None
    )
    score = (
        chunk.rerank_score
        if chunk.rerank_score is not None
        else direct_bibliography_score
        if direct_bibliography_score is not None
        else chunk.fused_score
    )
    return {
        "chunk_id": chunk.chunk_id,
        "rank": chunk.rank,
        "score": _rounded(score),
        "score_type": (
            "reranker"
            if chunk.rerank_score is not None
            else "bm25"
            if direct_bibliography_score is not None
            else "weighted_rrf"
        ),
        "fused_score": _rounded(chunk.fused_score),
        "rerank_score": (
            _rounded(chunk.rerank_score) if chunk.rerank_score is not None else None
        ),
        "pre_rerank_rank": chunk.pre_rerank_rank,
        "paper_id": chunk.paper_id,
        "section": chunk.section,
        "section_type": chunk.section_type,
        "page": {"start": chunk.page_start, "end": chunk.page_end},
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }


def load_golden(path: Path) -> list[dict[str, Any]]:
    """严格读取 Golden JSONL；任何坏行都会终止，避免静默改变评测分母。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RetrievalEvaluationError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise RetrievalEvaluationError(
                    f"Golden row {line_number} must be a JSON object"
                )
            records.append(value)
    ids = [record.get("id") for record in records]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RetrievalEvaluationError(f"Duplicate Golden IDs: {duplicates}")
    return records


def ground_truth_ids(record: Mapping[str, Any]) -> list[str]:
    """仅按 Golden relevant_evidence 顺序提取人工审核后的 Ground Truth IDs。"""
    result: list[str] = []
    for evidence in record.get("relevant_evidence", []):
        for chunk_id in evidence.get("chunk_ids", []):
            value = str(chunk_id)
            if value not in result:
                result.append(value)
    return result


def validate_ground_truth(
    database_path: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """仅用 paper_id 核对 Ground Truth chunk 所属论文，不做相似度 remap。"""
    all_ids = sorted({chunk_id for record in records for chunk_id in ground_truth_ids(record)})
    with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows: list[sqlite3.Row] = []
        for offset in range(0, len(all_ids), 900):
            batch = all_ids[offset : offset + 900]
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    "SELECT chunk_id, paper_id FROM chunks "
                    f"WHERE chunk_id IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        chunk_papers = {str(row["chunk_id"]): str(row["paper_id"]) for row in rows}
    invalid: dict[str, set[str]] = {}
    reasons: dict[str, str] = {}
    for record in records:
        case_id = str(record["id"])
        expected_paper = str(record["paper_id"])
        bad: set[str] = set()
        messages: list[str] = []
        for chunk_id in ground_truth_ids(record):
            actual_paper = chunk_papers.get(chunk_id)
            if actual_paper is None:
                bad.add(chunk_id)
                messages.append(f"missing chunk_id: {chunk_id}")
            elif actual_paper != expected_paper:
                bad.add(chunk_id)
                messages.append(
                    f"chunk {chunk_id} belongs to {actual_paper}, expected {expected_paper}"
                )
        if bad:
            invalid[case_id] = bad
            reasons[case_id] = "; ".join(messages)
    return invalid, reasons


def ranking_metrics(
    ranking: Sequence[Mapping[str, Any]], ground_truth: Sequence[str], *, k: int
) -> tuple[float, float]:
    """计算指定 ranking 的 Hit@K 与 Recall@K，Ground Truth 按集合去重。"""
    expected = set(ground_truth)
    if not expected:
        return 0.0, 0.0
    retrieved = {str(item["chunk_id"]) for item in ranking[:k]}
    hit_count = len(expected & retrieved)
    return (1.0 if hit_count else 0.0), hit_count / len(expected)


def reciprocal_rank(
    ranking: Sequence[Mapping[str, Any]], ground_truth: Sequence[str]
) -> float:
    """计算第一个 Ground Truth 在完整 ranking 中的 reciprocal rank。"""
    if not ranking:
        return 0.0
    return reciprocal_rank_at_k(ranking, ground_truth, k=len(ranking))


def reciprocal_rank_at_k(
    ranking: Sequence[Mapping[str, Any]], ground_truth: Sequence[str], *, k: int
) -> float:
    """计算前 K 条中第一个 Ground Truth 的 reciprocal rank，未命中时返回 0。"""
    if k < 1:
        raise ValueError("k must be positive")
    expected = set(ground_truth)
    for index, item in enumerate(ranking[:k], 1):
        if str(item["chunk_id"]) in expected:
            return 1.0 / index
    return 0.0


def classify_content_failure(
    *,
    ground_truth: Sequence[str],
    original_dense: Sequence[Mapping[str, Any]],
    rewritten_dense: Sequence[Mapping[str, Any]],
    rewritten_bm25: Sequence[Mapping[str, Any]],
    rrf_results: Sequence[Mapping[str, Any]],
    final_results: Sequence[Mapping[str, Any]],
    reranking_status: str,
) -> tuple[str, str]:
    """仅依据 Retrieval trace 保守标记正文失败，不调用 LLM 解释原因。"""
    expected = set(ground_truth)
    final_top5 = {str(item["chunk_id"]) for item in final_results[:5]}
    if expected & final_top5:
        return "none", "最终正文 Top-5 已命中 Ground Truth。"

    rrf_ids = [str(item["chunk_id"]) for item in rrf_results]
    if not expected.intersection(rrf_ids):
        original_top5 = {
            str(item["chunk_id"]) for item in original_dense[:5]
        }
        rewritten_ids = {
            str(item["chunk_id"])
            for item in (*rewritten_dense, *rewritten_bm25)
        }
        if expected & original_top5 and not expected & rewritten_ids:
            return (
                "rewrite_drift",
                "原始 Dense Top-5 命中，但改写路径均未命中且 Ground Truth 未进入 RRF。",
            )
        return "retrieval_miss", "Ground Truth 未出现在完整 RRF 候选集合中。"

    best_pre_rank = min(index for index, chunk_id in enumerate(rrf_ids, 1) if chunk_id in expected)
    if reranking_status == "success" and best_pre_rank <= 5:
        return (
            "reranker_drop",
            f"Ground Truth 在 RRF 中最高为第 {best_pre_rank}，经 Reranker 后掉出 Top-5。",
        )
    return (
        "ranking_failure",
        f"Ground Truth 已进入 RRF（最高第 {best_pre_rank}），但最终 Top-5 未命中。",
    )


def _plan_payload(result: RetrievalResult) -> dict[str, Any]:
    """保存正式 QueryRewritePlan 的全部结构化字段并映射用户要求的名称。"""
    plan = result.rewrite_plan
    return {
        "original_query": plan.original_query,
        "resolved_query": plan.resolved_query,
        "rewritten_query": plan.semantic_query_en,
        # 当前正式 Rewriter 没有中文关键词字段，空列表代表“未实现”而非模型输出为空。
        "chinese_keywords": [],
        "english_keywords": list(plan.lexical_keywords_en),
        "semantic_query_en": plan.semantic_query_en,
        "lexical_keywords_en": list(plan.lexical_keywords_en),
        "resolution_status": plan.resolution_status,
        "semantic_status": plan.semantic_status,
        "lexical_status": plan.lexical_status,
        "rewrite_status": plan.rewrite_status,
        "validation_diagnostics": list(plan.validation_diagnostics),
        "predicted_bibliography_intent": plan.search_bibliography,
        "clarification_message": plan.clarification_message,
    }


def evaluate_case(
    *,
    record: Mapping[str, Any],
    result: RetrievalResult,
    trace: RetrievalTrace,
    invalid_ids: set[str],
    invalid_reason: str | None,
) -> dict[str, Any]:
    """把一次正式检索快照转换为 Case 级指标、阶段排名和规则化诊断。"""
    ground_truth = ground_truth_ids(record)
    final_results = [_retrieved_to_record(chunk) for chunk in result.chunks]
    final_content = [item for item in final_results if item["section_type"] == "content"]
    original_dense = trace.route_results.get("original_dense", [])
    rewritten_dense = trace.route_results.get("rewritten_dense", [])
    rewritten_bm25 = trace.route_results.get("rewritten_bm25", [])
    is_content_case = record.get("answerable") is True and not record.get(
        "expected_bibliography_intent"
    )
    is_bibliography_case = record.get("expected_bibliography_intent") is True
    invalid_ground_truth = bool(invalid_ids)

    hit5: float | None = None
    recall5: float | None = None
    recall10: float | None = None
    recall20: float | None = None
    mrr_at_5: float | None = None
    bibliography_hit5: float | None = None
    failure_type = "none"
    diagnosis = ""

    if invalid_ground_truth:
        failure_type = "invalid_ground_truth"
        diagnosis = invalid_reason or "Ground Truth 与当前 KB 不一致。"
    elif is_content_case:
        hit5, recall5 = ranking_metrics(final_content, ground_truth, k=5)
        _, recall10 = ranking_metrics(trace.rrf_results, ground_truth, k=10)
        _, recall20 = ranking_metrics(trace.rrf_results, ground_truth, k=20)
        mrr_at_5 = reciprocal_rank_at_k(final_content, ground_truth, k=5)
        failure_type, diagnosis = classify_content_failure(
            ground_truth=ground_truth,
            original_dense=original_dense,
            rewritten_dense=rewritten_dense,
            rewritten_bm25=rewritten_bm25,
            rrf_results=trace.rrf_results,
            final_results=final_content,
            reranking_status=result.reranking_status,
        )
    elif is_bibliography_case:
        bibliography_hit5, _ = ranking_metrics(
            trace.bibliography_results, ground_truth, k=5
        )
        if result.rewrite_plan.search_bibliography is False:
            failure_type = "routing_error"
            diagnosis = "正式 Query Rewriter 未开启 bibliography 检索。"
        elif bibliography_hit5 == 0:
            failure_type = "bibliography_miss"
            diagnosis = "Bibliography Top-5 未命中 Ground Truth reference chunk。"
    else:
        diagnosis = "Unanswerable Case 仅保存 Retrieval trace，不计算 Recall/MRR。"

    return {
        "case_id": record.get("id"),
        "question": record.get("question"),
        "primary_type": record.get("primary_type"),
        "tags": record.get("tags"),
        "paper_id": record.get("paper_id"),
        "answerable": record.get("answerable"),
        "expected_bibliography_intent": record.get("expected_bibliography_intent"),
        **_plan_payload(result),
        "ground_truth_chunk_ids": ground_truth,
        "invalid_ground_truth": invalid_ground_truth,
        "invalid_ground_truth_ids": sorted(invalid_ids),
        "reranking_status": result.reranking_status,
        "stage_availability": {
            "original_dense": True,
            "original_bm25": False,
            "rewritten_dense": result.rewrite_plan.semantic_query_en is not None,
            "rewritten_bm25": bool(result.rewrite_plan.lexical_keywords_en),
            "rrf_results": True,
            "reranked_results": result.reranking_status == "success",
            "chinese_keywords": False,
        },
        "original_dense": original_dense,
        # 正式链路已移除 resolved/original BM25；保留空数组防止下游误当成缺字段。
        "original_bm25": [],
        "rewritten_dense": rewritten_dense,
        "rewritten_bm25": rewritten_bm25,
        "rrf_results": trace.rrf_results,
        "reranked_results": trace.reranked_candidates,
        "bibliography_results": trace.bibliography_results,
        "final_results": final_results,
        "hit_at_5": _optional_round(hit5),
        "recall_at_5": _optional_round(recall5),
        # 正式 final_top_k=5，因此 @10/@20 明确使用 pre-rerank RRF，不混用阶段。
        "recall_at_10": _optional_round(recall10),
        "recall_at_20": _optional_round(recall20),
        # 保留旧 mrr 字段兼容已有 Case 分析；正式 summary 只使用命名明确的 mrr_at_5。
        "mrr": _optional_round(mrr_at_5),
        "mrr_at_5": _optional_round(mrr_at_5),
        "bibliography_hit_at_5": _optional_round(bibliography_hit5),
        "metric_stages": {
            "hit_at_5": "final_reranked_content",
            "recall_at_5": "final_reranked_content",
            "recall_at_10": "pre_rerank_rrf",
            "recall_at_20": "pre_rerank_rrf",
            "mrr": "final_reranked_content_top_5",
            "mrr_at_5": "final_reranked_content_top_5",
            "bibliography_hit_at_5": "bibliography_bm25",
        },
        "latency": trace.latency_payload(),
        "failure_type": failure_type,
        "diagnosis": diagnosis,
    }


def create_production_expander(settings: AppSettings) -> SectionAwareNeighborExpander:
    """使用正式数据库配置和生产类装配 Evidence Expansion，不复制扩展算法。"""
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    return SectionAwareNeighborExpander(
        database=database,
        settings=settings.context_expansion,
    )


class _UnavailableProductionExpander:
    """生产Expander装配失败时，将同一错误安全记录到每条适用Case。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def expand(self, result: RetrievalResult) -> ExpansionResult:
        """由Case级Expansion捕获器记录错误，而不是中止整轮Retrieval。"""
        raise RuntimeError(
            f"Production expander initialization failed: {type(self._error).__name__}: {self._error}"
        ) from self._error


def _deduplicate_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
    """按生产 Evidence 顺序去重 chunk ID，同时保留第一次出现的位置。"""
    seen: set[str] = set()
    result: list[str] = []
    for chunk_id in chunk_ids:
        value = str(chunk_id)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _not_applicable_expansion(reason: str) -> dict[str, Any]:
    """为 Bibliography、Unanswerable 或无效GT Case输出稳定的非适用trace。"""
    return {
        "available": False,
        "status": "not_applicable",
        "error": None,
        "reason": reason,
        "anchor_chunk_ids": [],
        "expanded_chunk_ids": [],
        "expanded_chunk_count": 0,
        "expanded_context_char_count": 0,
        "expanded_context_token_count": 0,
        "expansion_latency_ms": None,
        "per_anchor": [],
        "ground_truth_hits_after_expansion": [],
        "expanded_hit": None,
        "expanded_recall": None,
        "diagnosis": "not_applicable",
    }


def is_content_expansion_case(
    record: Mapping[str, Any], invalid_ids: set[str]
) -> bool:
    """只允许有效的可回答正文Case进入Expansion执行和指标分母。"""
    return (
        record.get("answerable") is True
        and record.get("expected_bibliography_intent") is False
        and not invalid_ids
    )


def evaluate_production_expansion(
    *,
    result: RetrievalResult,
    ground_truth: Sequence[str],
    raw_hit_at_5: float,
    expander: object,
) -> dict[str, Any]:
    """调用生产Expander，并从真实EvidenceUnit计算Expansion-aware指标。"""
    # Evaluation只审计Final Top-5。即使未来Retriever返回更多结果，也不能让
    # 第6名以后候选进入Expansion，否则会悄悄改变本指标的含义。
    top_5_result = (
        result
        if len(result.chunks) <= 5
        else RetrievalResult(
            query=result.query,
            rewrite_plan=result.rewrite_plan,
            reranking_status=result.reranking_status,
            chunks=tuple(result.chunks[:5]),
        )
    )
    anchors = [
        chunk.chunk_id
        for chunk in top_5_result.chunks
        if chunk.section_type == "content"
    ]
    started = perf_counter()
    try:
        expansion: ExpansionResult = expander.expand(top_5_result)
        latency_ms = _elapsed_ms(started)
        content_evidence = tuple(expansion.content_evidence)
        expanded_ids = _deduplicate_chunk_ids(
            [chunk_id for evidence in content_evidence for chunk_id in evidence.chunk_ids]
        )
        per_anchor: list[dict[str, Any]] = []
        for anchor in anchors:
            anchor_expanded = _deduplicate_chunk_ids(
                [
                    chunk_id
                    for evidence in content_evidence
                    if anchor in evidence.seed_chunk_ids
                    for chunk_id in evidence.chunk_ids
                ]
            )
            per_anchor.append(
                {
                    "anchor_chunk_id": anchor,
                    "expanded_chunk_ids": anchor_expanded,
                }
            )

        expected = [str(chunk_id) for chunk_id in ground_truth]
        expanded_set = set(expanded_ids)
        hits = [chunk_id for chunk_id in expected if chunk_id in expanded_set]
        expanded_hit = 1.0 if hits else 0.0
        expanded_recall = len(hits) / len(set(expected)) if expected else 0.0
        if raw_hit_at_5 == 0.0 and expanded_hit == 1.0:
            diagnosis = "expansion_recovered"
        elif expanded_recall < 1.0:
            diagnosis = "post_expansion_incomplete"
        else:
            diagnosis = "expansion_complete"

        return {
            "available": True,
            "status": "success",
            "error": None,
            "reason": None,
            "anchor_chunk_ids": anchors,
            "expanded_chunk_ids": expanded_ids,
            "expanded_chunk_count": len(expanded_ids),
            "expanded_context_char_count": sum(
                len(evidence.text) for evidence in content_evidence
            ),
            "expanded_context_token_count": sum(
                int(evidence.token_count) for evidence in content_evidence
            ),
            "expansion_latency_ms": _rounded(latency_ms),
            "per_anchor": per_anchor,
            "ground_truth_hits_after_expansion": hits,
            "expanded_hit": expanded_hit,
            "expanded_recall": _optional_round(expanded_recall),
            "diagnosis": diagnosis,
        }
    except Exception as error:  # noqa: BLE001 - Expansion失败不能改变Retrieval评测分母。
        return {
            "available": True,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "reason": "production_expansion_error",
            "anchor_chunk_ids": anchors,
            "expanded_chunk_ids": [],
            "expanded_chunk_count": 0,
            "expanded_context_char_count": 0,
            "expanded_context_token_count": 0,
            "expansion_latency_ms": _rounded(_elapsed_ms(started)),
            "per_anchor": [
                {"anchor_chunk_id": anchor, "expanded_chunk_ids": []}
                for anchor in anchors
            ],
            "ground_truth_hits_after_expansion": [],
            # 错误按未恢复计入32条固定分母，避免只统计成功调用造成指标虚高。
            "expanded_hit": 0.0,
            "expanded_recall": 0.0,
            "diagnosis": "expansion_error",
        }


def _optional_round(value: float | None) -> float | None:
    """为可选指标保留六位小数；不适用的 Case 继续输出 null。"""
    return round(value, 6) if value is not None else None


def intent_metrics(case_results: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """计算 bibliography intent 的 Accuracy、Precision、Recall 和 F1。"""
    tp = fp = tn = fn = 0
    for item in case_results:
        expected = item.get("expected_bibliography_intent") is True
        predicted = item.get("predicted_bibliography_intent") is True
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": _optional_round((tp + tn) / total if total else 0.0),
        "precision": _optional_round(precision),
        "recall": _optional_round(recall),
        "f1": _optional_round(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _macro_average(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """对非 null Case 指标计算 macro average。"""
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return _optional_round(sum(values) / len(values)) if values else None


def aggregate_ranking_metrics(
    items: Sequence[Mapping[str, Any]],
    *,
    ranking_field: str,
    cutoffs: Sequence[int],
    mrr_k: int,
) -> dict[str, float | int]:
    """对指定阶段排名计算固定 K 的 macro Hit、Recall 和 MRR。"""
    if not items:
        result: dict[str, float | int] = {"cases": 0}
        for cutoff in cutoffs:
            result[f"hit_at_{cutoff}"] = 0.0
            result[f"recall_at_{cutoff}"] = 0.0
        result[f"mrr_at_{mrr_k}"] = 0.0
        return result

    hit_values: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    recall_values: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    reciprocal_ranks: list[float] = []
    for item in items:
        ranking = item.get(ranking_field, [])
        if not isinstance(ranking, list):
            ranking = []
        ground_truth = item.get("ground_truth_chunk_ids", [])
        if not isinstance(ground_truth, list):
            ground_truth = []
        for cutoff in cutoffs:
            hit, recall = ranking_metrics(ranking, ground_truth, k=cutoff)
            hit_values[cutoff].append(hit)
            recall_values[cutoff].append(recall)
        reciprocal_ranks.append(
            reciprocal_rank_at_k(ranking, ground_truth, k=mrr_k)
        )

    result = {"cases": len(items)}
    for cutoff in cutoffs:
        result[f"hit_at_{cutoff}"] = _optional_round(
            sum(hit_values[cutoff]) / len(hit_values[cutoff])
        ) or 0.0
        result[f"recall_at_{cutoff}"] = _optional_round(
            sum(recall_values[cutoff]) / len(recall_values[cutoff])
        ) or 0.0
    result[f"mrr_at_{mrr_k}"] = _optional_round(
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    ) or 0.0
    return result


def build_slice_metrics(
    valid_content: Sequence[Mapping[str, Any]],
    *,
    values: Sequence[str],
    selector: str,
) -> dict[str, dict[str, float | int]]:
    """按 primary_type 或 tag 构造固定的 Slice Evaluation 行。"""
    slices: dict[str, dict[str, float | int]] = {}
    for value in values:
        if selector == "primary_type":
            selected = [item for item in valid_content if item.get("primary_type") == value]
        elif selector == "tag":
            selected = [
                item
                for item in valid_content
                if isinstance(item.get("tags"), list) and value in item["tags"]
            ]
        else:
            raise ValueError(f"Unsupported slice selector: {selector}")
        slices[value] = aggregate_ranking_metrics(
            selected,
            ranking_field="final_results",
            cutoffs=(5,),
            mrr_k=5,
        )
    return slices


def _expansion_slice_metrics(
    valid_content: Sequence[Mapping[str, Any]],
    *,
    values: Sequence[str],
    selector: str,
) -> dict[str, dict[str, float | int]]:
    """按类型或Hop比较原始Recall@5与Expanded Recall，不改原Slice定义。"""
    result: dict[str, dict[str, float | int]] = {}
    for value in values:
        if selector == "primary_type":
            selected = [item for item in valid_content if item.get("primary_type") == value]
        elif selector == "tag":
            selected = [
                item
                for item in valid_content
                if isinstance(item.get("tags"), list) and value in item["tags"]
            ]
        else:
            raise ValueError(f"Unsupported expansion slice selector: {selector}")
        raw_recall = _macro_average(selected, "recall_at_5") or 0.0
        expanded_values = [
            float(item.get("expansion", {}).get("expanded_recall", 0.0) or 0.0)
            for item in selected
        ]
        expanded_recall = (
            _optional_round(sum(expanded_values) / len(expanded_values)) or 0.0
            if expanded_values
            else 0.0
        )
        result[value] = {
            "cases": len(selected),
            "raw_recall_at_5": raw_recall,
            "expanded_recall": expanded_recall,
            "delta": _optional_round(expanded_recall - raw_recall) or 0.0,
        }
    return result


def build_expansion_evaluation(
    valid_content: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总生产Evidence Expansion前后质量、恢复Case、耗时和Context规模。"""
    raw_hit = _macro_average(valid_content, "hit_at_5") or 0.0
    raw_recall = _macro_average(valid_content, "recall_at_5") or 0.0
    expanded_hits = [
        float(item.get("expansion", {}).get("expanded_hit", 0.0) or 0.0)
        for item in valid_content
    ]
    expanded_recalls = [
        float(item.get("expansion", {}).get("expanded_recall", 0.0) or 0.0)
        for item in valid_content
    ]
    expanded_hit = (
        _optional_round(sum(expanded_hits) / len(expanded_hits)) or 0.0
        if expanded_hits
        else 0.0
    )
    expanded_recall = (
        _optional_round(sum(expanded_recalls) / len(expanded_recalls)) or 0.0
        if expanded_recalls
        else 0.0
    )
    recovery_ids = [
        str(item["case_id"])
        for item in valid_content
        if float(item.get("hit_at_5") or 0.0) == 0.0
        and float(item.get("expansion", {}).get("expanded_hit") or 0.0) == 1.0
    ]
    latency_values = [
        float(item["expansion"]["expansion_latency_ms"])
        for item in valid_content
        if item.get("expansion", {}).get("expansion_latency_ms") is not None
    ]
    chunk_counts = [
        int(item.get("expansion", {}).get("expanded_chunk_count", 0))
        for item in valid_content
    ]
    char_counts = [
        int(item.get("expansion", {}).get("expanded_context_char_count", 0))
        for item in valid_content
    ]
    token_counts = [
        int(item.get("expansion", {}).get("expanded_context_token_count", 0))
        for item in valid_content
    ]
    error_ids = [
        str(item["case_id"])
        for item in valid_content
        if item.get("expansion", {}).get("status") == "error"
    ]
    diagnoses = Counter(
        str(item.get("expansion", {}).get("diagnosis", "missing"))
        for item in valid_content
    )
    return {
        "cases": len(valid_content),
        "before_expansion": {
            "hit_at_5": raw_hit,
            "recall_at_5": raw_recall,
        },
        "after_expansion": {
            "expanded_hit": expanded_hit,
            "expanded_recall": expanded_recall,
        },
        "delta": {
            "hit": _optional_round(expanded_hit - raw_hit) or 0.0,
            "recall": _optional_round(expanded_recall - raw_recall) or 0.0,
        },
        "recovery_cases": {
            "count": len(recovery_ids),
            "case_ids": recovery_ids,
        },
        "avg_expansion_latency_ms": (
            _rounded(sum(latency_values) / len(latency_values))
            if latency_values
            else None
        ),
        "context_size": {
            "avg_expanded_chunks": (
                _optional_round(sum(chunk_counts) / len(chunk_counts)) or 0.0
                if chunk_counts
                else 0.0
            ),
            "max_expanded_chunks": max(chunk_counts, default=0),
            "avg_expanded_context_chars": (
                _optional_round(sum(char_counts) / len(char_counts)) or 0.0
                if char_counts
                else 0.0
            ),
            "avg_expanded_context_tokens": (
                _optional_round(sum(token_counts) / len(token_counts)) or 0.0
                if token_counts
                else 0.0
            ),
        },
        "expansion_errors": {"count": len(error_ids), "case_ids": error_ids},
        "diagnosis_counts": dict(sorted(diagnoses.items())),
        "slice_before_after": {
            "type": _expansion_slice_metrics(
                valid_content,
                values=("fact", "method", "experiment", "result", "synthesis"),
                selector="primary_type",
            ),
            "hop": _expansion_slice_metrics(
                valid_content,
                values=("single_hop", "multi_hop"),
                selector="tag",
            ),
        },
    }


def _average_latency(case_results: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """按阶段计算平均耗时；正式链路不可单测的 fusion 继续为 null。"""
    keys = (
        "rewrite_latency_ms",
        "embedding_latency_ms",
        "dense_latency_ms",
        "bm25_latency_ms",
        "fusion_latency_ms",
        "rerank_latency_ms",
        "total_retrieval_latency_ms",
    )
    result: dict[str, float | None] = {}
    for key in keys:
        values = [
            float(item["latency"][key])
            for item in case_results
            if item.get("latency", {}).get(key) is not None
        ]
        result[key] = _rounded(sum(values) / len(values)) if values else None
    return result


def build_summary(
    *,
    case_results: Sequence[Mapping[str, Any]],
    settings: AppSettings,
    invalid_reasons: Mapping[str, str],
    git_commit: str | None,
) -> dict[str, Any]:
    """保留四组主指标定义，并追加独立的Evidence Expansion诊断。"""
    valid_content = [
        item
        for item in case_results
        if item.get("answerable") is True
        and item.get("expected_bibliography_intent") is False
        and item.get("invalid_ground_truth") is False
        and item.get("retrieval_error") is None
    ]
    bibliography = [
        item for item in case_results if item.get("expected_bibliography_intent") is True
    ]
    unanswerable = [item for item in case_results if item.get("answerable") is False]
    invalid = [item for item in case_results if item.get("invalid_ground_truth") is True]
    ordinary_top5 = [
        result
        for item in valid_content
        for result in item.get("final_results", [])[:5]
    ]
    bibliography_noise = (
        sum(item.get("section_type") == "bibliography" for item in ordinary_top5)
        / len(ordinary_top5)
        if ordinary_top5
        else 0.0
    )
    failure_counts = Counter(
        str(item.get("failure_type", "unknown"))
        for item in case_results
        if item.get("failure_type") != "none"
    )
    retrieval = settings.retrieval
    reranking = settings.reranking
    average_latency = _average_latency(case_results)
    overall_metrics = aggregate_ranking_metrics(
        valid_content,
        ranking_field="final_results",
        cutoffs=(5,),
        mrr_k=5,
    )
    stage_comparison = {
        "original_dense": {
            "label": "Original Dense",
            **aggregate_ranking_metrics(
                valid_content,
                ranking_field="original_dense",
                cutoffs=(5, 10, 20),
                mrr_k=20,
            ),
        },
        "rrf_fusion": {
            "label": "RRF Fusion",
            **aggregate_ranking_metrics(
                valid_content,
                ranking_field="rrf_results",
                cutoffs=(5, 10, 20),
                mrr_k=20,
            ),
        },
        "reranker": {
            "label": "Reranker",
            **aggregate_ranking_metrics(
                valid_content,
                ranking_field="reranked_results",
                cutoffs=(5, 10, 20),
                mrr_k=20,
            ),
        },
    }
    intent = intent_metrics(case_results)
    expansion_evaluation = build_expansion_evaluation(valid_content)
    return {
        "experiment": {
            "name": "Full Retrieval Baseline",
            "query_context_mode": "context_free",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "dataset": str(DEFAULT_DATASET.resolve()),
        },
        "counts": {
            "total_cases": len(case_results),
            "content_answerable_cases": len(valid_content),
            "bibliography_cases": len(bibliography),
            "unanswerable_cases": len(unanswerable),
            "invalid_ground_truth_cases": len(invalid),
            "retrieval_error_cases": sum(
                item.get("retrieval_error") is not None for item in case_results
            ),
        },
        "overall_retrieval": {
            "hit_at_5": overall_metrics["hit_at_5"],
            "recall_at_5": overall_metrics["recall_at_5"],
            "mrr_at_5": overall_metrics["mrr_at_5"],
            "avg_retrieval_latency_ms": average_latency[
                "total_retrieval_latency_ms"
            ],
        },
        "retrieval_stage_comparison": stage_comparison,
        "slice_evaluation": {
            "type": build_slice_metrics(
                valid_content,
                values=("fact", "method", "experiment", "result", "synthesis"),
                selector="primary_type",
            ),
            "hop": build_slice_metrics(
                valid_content,
                values=("single_hop", "multi_hop"),
                selector="tag",
            ),
        },
        "bibliography": {
            "intent_accuracy": intent["accuracy"],
            "intent_precision": intent["precision"],
            "intent_recall": intent["recall"],
            "intent_f1": intent["f1"],
            "bibliography_hit_at_5": _macro_average(
                bibliography, "bibliography_hit_at_5"
            ),
            "bibliography_noise_rate_at_5": _optional_round(bibliography_noise),
        },
        "evidence_expansion_evaluation": expansion_evaluation,
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "failed_case_ids": [
            str(item["case_id"])
            for item in case_results
            if item.get("failure_type") != "none"
        ],
        "invalid_ground_truth": dict(invalid_reasons),
        "production_configuration": {
            "config_path": str(settings.config_path),
            "retrieval_backend": retrieval.backend,
            "embedding_backend": settings.embedding.backend,
            "embedding_model": settings.embedding.model_id,
            "reranker_backend": reranking.backend,
            "reranker_model": reranking.model_id,
            "query_rewrite_enabled": retrieval.query_rewrite.enabled,
            "dense_top_k_per_query": retrieval.dense_top_k_per_query,
            "original_bm25_available": False,
            "bm25_keywords_top_k": retrieval.bm25_keywords_top_k,
            "fused_top_k": retrieval.fused_top_k,
            "rrf_k": retrieval.rrf_k,
            "rrf_weights": {
                "dense_resolved": retrieval.dense_resolved_weight,
                "dense_semantic": retrieval.dense_semantic_weight,
                "bm25_keywords": retrieval.bm25_keywords_weight,
            },
            "rerank_candidate_top_k": reranking.candidate_top_k,
            "rerank_final_top_k": reranking.final_top_k,
            "bibliography_top_k": retrieval.query_rewrite.bibliography_top_k,
            "trusted_paper_scope_used": False,
            "context_expansion_enabled": settings.context_expansion.enabled,
            "context_expansion_neighbor_window": settings.context_expansion.neighbor_window,
            "context_expansion_max_total_tokens": settings.context_expansion.max_total_tokens,
        },
        "trace_limitations": [
            "Context-free Benchmark 只把 Golden paper_id 用于 Ground Truth 校验与评分，不向检索器注入 trusted scope。",
            "当前正式链路没有 original/resolved BM25 路径；对应字段 available=false。",
            "当前正式 Query Rewriter 不输出中文关键词；chinese_keywords 为空且 available=false。",
            "fusion 排序没有独立公开计时边界；fusion_latency_ms 为 null，并包含在 total 中。",
            "Overall 与 Slice 使用最终产品 Top-5；Stage Comparison 使用各阶段保存的独立 ranking。",
            "Expanded Hit/Recall 是独立Diagnostic，不写回或替换现有Retrieval主指标。",
        ],
    }


def retrieval_error_case(
    record: Mapping[str, Any], error: Exception, invalid_ids: set[str], invalid_reason: str | None
) -> dict[str, Any]:
    """检索异常时保留 Case 和 Ground Truth，使分母变化不会被静默隐藏。"""
    return {
        "case_id": record.get("id"),
        "question": record.get("question"),
        "primary_type": record.get("primary_type"),
        "tags": record.get("tags"),
        "paper_id": record.get("paper_id"),
        "answerable": record.get("answerable"),
        "expected_bibliography_intent": record.get("expected_bibliography_intent"),
        "predicted_bibliography_intent": None,
        "ground_truth_chunk_ids": ground_truth_ids(record),
        "invalid_ground_truth": bool(invalid_ids),
        "invalid_ground_truth_ids": sorted(invalid_ids),
        "original_dense": [],
        "original_bm25": [],
        "rewritten_dense": [],
        "rewritten_bm25": [],
        "rrf_results": [],
        "reranked_results": [],
        "bibliography_results": [],
        "final_results": [],
        "hit_at_5": None,
        "recall_at_5": None,
        "recall_at_10": None,
        "recall_at_20": None,
        "mrr": None,
        "mrr_at_5": None,
        "bibliography_hit_at_5": None,
        "expansion": _not_applicable_expansion("retrieval_error"),
        "latency": {},
        "failure_type": "invalid_ground_truth" if invalid_ids else "retrieval_error",
        "diagnosis": invalid_reason or str(error),
        "retrieval_error": f"{type(error).__name__}: {error}",
    }


def render_failures(case_results: Sequence[Mapping[str, Any]]) -> str:
    """生成只依据 trace 的失败 Case Markdown，不让 LLM补写原因。"""
    failures = [item for item in case_results if item.get("failure_type") != "none"]
    lines = ["# Full Retrieval Baseline Failures", "", f"Failure cases: **{len(failures)}**", ""]
    if not failures:
        lines.append("No failed cases.")
        return "\n".join(lines) + "\n"
    for item in failures:
        is_bibliography = item.get("expected_bibliography_intent") is True
        top5 = (
            item.get("bibliography_results", [])[:5]
            if is_bibliography
            else item.get("final_results", [])[:5]
        )
        ranking_label = "Bibliography Top-5" if is_bibliography else "Final Top-5"
        lines.extend(
            [
                f"## {item['case_id']} — {item['failure_type']}",
                "",
                f"- Question: {item['question']}",
                f"- Ground Truth: `{json.dumps(item.get('ground_truth_chunk_ids', []), ensure_ascii=False)}`",
                f"- {ranking_label}: `{json.dumps([row.get('chunk_id') for row in top5], ensure_ascii=False)}`",
                f"- Diagnosis: {item.get('diagnosis') or '无可确定诊断。'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """原子写出 Case JSONL，避免长时间评测后留下半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    """原子写出格式化 JSON summary。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """原子写出 Markdown 文本报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    """按固定表格分组写成长格式 CSV，避免不同阶段指标被压在同一行。"""
    rows: list[dict[str, Any]] = []
    overall = summary["overall_retrieval"]
    for label, key in (
        ("Hit@5", "hit_at_5"),
        ("Recall@5", "recall_at_5"),
        ("MRR@5", "mrr_at_5"),
        ("Avg Retrieval Latency (ms)", "avg_retrieval_latency_ms"),
    ):
        rows.append({"section": "Overall Retrieval", "row": label, "result": overall[key]})

    for item in summary["retrieval_stage_comparison"].values():
        rows.append(
            {
                "section": "Retrieval Stage Comparison",
                "row": item["label"],
                **{key: value for key, value in item.items() if key not in {"label", "cases"}},
            }
        )
    for slice_name, item in summary["slice_evaluation"]["type"].items():
        rows.append(
            {
                "section": "Slice Evaluation - Type",
                "row": slice_name,
                **item,
            }
        )
    for slice_name, item in summary["slice_evaluation"]["hop"].items():
        rows.append(
            {
                "section": "Slice Evaluation - Hop",
                "row": slice_name,
                **item,
            }
        )
    bibliography = summary["bibliography"]
    for label, key in (
        ("Intent Accuracy", "intent_accuracy"),
        ("Intent Precision", "intent_precision"),
        ("Intent Recall", "intent_recall"),
        ("Intent F1", "intent_f1"),
        ("Bibliography Hit@5", "bibliography_hit_at_5"),
        ("Bibliography Noise Rate@5", "bibliography_noise_rate_at_5"),
    ):
        rows.append({"section": "Bibliography", "row": label, "result": bibliography[key]})

    fieldnames = (
        "section",
        "row",
        "cases",
        "result",
        "hit_at_5",
        "recall_at_5",
        "mrr_at_5",
        "hit_at_10",
        "recall_at_10",
        "hit_at_20",
        "recall_at_20",
        "mrr_at_20",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown_metric(value: Any) -> str:
    """把 JSON 指标渲染为稳定 Markdown 文本，null 使用短横线表示。"""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """原样渲染四组主表，并在其后追加独立Expansion表。"""
    overall = summary["overall_retrieval"]
    lines = [
        "# Full Retrieval Baseline Summary",
        "",
        "## Overall Retrieval",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Hit@5 | {_markdown_metric(overall['hit_at_5'])} |",
        f"| Recall@5 | {_markdown_metric(overall['recall_at_5'])} |",
        f"| MRR@5 | {_markdown_metric(overall['mrr_at_5'])} |",
        "| Avg Retrieval Latency | "
        f"{_markdown_metric(overall['avg_retrieval_latency_ms'])} ms |",
        "",
        "## Retrieval Stage Comparison",
        "",
        "| Stage | Hit@5 | Recall@5 | Hit@10 | Recall@10 | Hit@20 | Recall@20 | MRR@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["retrieval_stage_comparison"].values():
        lines.append(
            f"| {item['label']} | {_markdown_metric(item['hit_at_5'])} | "
            f"{_markdown_metric(item['recall_at_5'])} | "
            f"{_markdown_metric(item['hit_at_10'])} | "
            f"{_markdown_metric(item['recall_at_10'])} | "
            f"{_markdown_metric(item['hit_at_20'])} | "
            f"{_markdown_metric(item['recall_at_20'])} | "
            f"{_markdown_metric(item['mrr_at_20'])} |"
        )

    lines.extend(
        [
            "",
            "## Slice Evaluation",
            "",
            "| Type | Cases | Hit@5 | Recall@5 | MRR@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["slice_evaluation"]["type"].items():
        lines.append(
            f"| {name} | {item['cases']} | {_markdown_metric(item['hit_at_5'])} | "
            f"{_markdown_metric(item['recall_at_5'])} | "
            f"{_markdown_metric(item['mrr_at_5'])} |"
        )
    lines.extend(
        [
            "",
            "| Hop | Cases | Hit@5 | Recall@5 | MRR@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["slice_evaluation"]["hop"].items():
        lines.append(
            f"| {name} | {item['cases']} | {_markdown_metric(item['hit_at_5'])} | "
            f"{_markdown_metric(item['recall_at_5'])} | "
            f"{_markdown_metric(item['mrr_at_5'])} |"
        )

    bibliography = summary["bibliography"]
    lines.extend(
        [
            "",
            "## Bibliography",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Intent Accuracy | {_markdown_metric(bibliography['intent_accuracy'])} |",
            f"| Intent Precision | {_markdown_metric(bibliography['intent_precision'])} |",
            f"| Intent Recall | {_markdown_metric(bibliography['intent_recall'])} |",
            f"| Intent F1 | {_markdown_metric(bibliography['intent_f1'])} |",
            "| Bibliography Hit@5 | "
            f"{_markdown_metric(bibliography['bibliography_hit_at_5'])} |",
            "| Bibliography Noise Rate@5 | "
            f"{_markdown_metric(bibliography['bibliography_noise_rate_at_5'])} |",
        ]
    )

    expansion = summary["evidence_expansion_evaluation"]
    before = expansion["before_expansion"]
    after = expansion["after_expansion"]
    delta = expansion["delta"]
    context_size = expansion["context_size"]
    lines.extend(
        [
            "",
            "## Evidence Expansion Evaluation",
            "",
            "| Metric | Before Expansion | After Expansion | Delta |",
            "|---|---:|---:|---:|",
            f"| Hit | {_markdown_metric(before['hit_at_5'])} | "
            f"{_markdown_metric(after['expanded_hit'])} | {_markdown_metric(delta['hit'])} |",
            f"| Recall | {_markdown_metric(before['recall_at_5'])} | "
            f"{_markdown_metric(after['expanded_recall'])} | {_markdown_metric(delta['recall'])} |",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Expansion Recovery Cases | {expansion['recovery_cases']['count']} |",
            "| Avg Expansion Latency | "
            f"{_markdown_metric(expansion['avg_expansion_latency_ms'])} ms |",
            f"| Avg Expanded Chunks | {_markdown_metric(context_size['avg_expanded_chunks'])} |",
            f"| Max Expanded Chunks | {context_size['max_expanded_chunks']} |",
            "| Avg Expanded Context Chars | "
            f"{_markdown_metric(context_size['avg_expanded_context_chars'])} |",
            "| Avg Expanded Context Tokens | "
            f"{_markdown_metric(context_size['avg_expanded_context_tokens'])} |",
            f"| Expansion Errors | {expansion['expansion_errors']['count']} |",
            "",
            "### Expansion Recall by Type",
            "",
            "| Slice | Cases | Raw Recall@5 | Expanded Recall | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in expansion["slice_before_after"]["type"].items():
        lines.append(
            f"| {name} | {item['cases']} | {_markdown_metric(item['raw_recall_at_5'])} | "
            f"{_markdown_metric(item['expanded_recall'])} | {_markdown_metric(item['delta'])} |"
        )
    lines.extend(
        [
            "",
            "### Expansion Recall by Hop",
            "",
            "| Slice | Cases | Raw Recall@5 | Expanded Recall | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in expansion["slice_before_after"]["hop"].items():
        lines.append(
            f"| {name} | {item['cases']} | {_markdown_metric(item['raw_recall_at_5'])} | "
            f"{_markdown_metric(item['expanded_recall'])} | {_markdown_metric(item['delta'])} |"
        )
    recovery_ids = expansion["recovery_cases"]["case_ids"]
    lines.extend(
        [
            "",
            "Expansion Recovery Case IDs: "
            f"`{json.dumps(recovery_ids, ensure_ascii=False)}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def git_commit(root: Path) -> str | None:
    """读取当前 git commit；非 Git 环境或命令不可用时返回 null。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析评测路径与可选 Case 过滤参数；不提供任何生产参数覆盖项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="仅用于调试的 Case ID 过滤；可重复传入，正式 Baseline 不应使用。",
    )
    return parser.parse_args(argv)


def run_evaluation(
    args: argparse.Namespace,
    *,
    retriever_factory: Callable[..., object] = create_hybrid_retriever,
    expander_factory: Callable[[AppSettings], object] = create_production_expander,
    progress: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """装配一次正式 Retriever，逐 Case 运行并写出全部评测产物。"""
    dataset_path = args.dataset.resolve()
    settings = load_settings(args.config)
    records = load_golden(dataset_path)
    if args.case_id:
        selected = set(args.case_id)
        records = [record for record in records if record.get("id") in selected]
        missing = sorted(selected - {str(record.get("id")) for record in records})
        if missing:
            raise RetrievalEvaluationError(f"Unknown --case-id values: {missing}")
    invalid, invalid_reasons = validate_ground_truth(settings.database.path, records)

    progress("正在装配正式 HybridRetriever（首次加载本地模型可能需要一些时间）……")
    observer = ProductionRetrieverObserver(
        retriever_factory(config_path=settings.config_path)
    )
    # 与线上AnswerService一致：使用正式SectionAwareNeighborExpander和同一份配置。
    try:
        expander = expander_factory(settings)
    except Exception as error:  # noqa: BLE001 - Expansion装配失败也不能中止Retrieval Eval。
        expander = _UnavailableProductionExpander(error)
    case_results: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        case_id = str(record["id"])
        progress(f"[{index}/{len(records)}] {case_id}")
        invalid_ids = invalid.get(case_id, set())
        try:
            result, trace = observer.retrieve(str(record["question"]))
            case_result = evaluate_case(
                record=record,
                result=result,
                trace=trace,
                invalid_ids=invalid_ids,
                invalid_reason=invalid_reasons.get(case_id),
            )
            is_content_case = is_content_expansion_case(record, invalid_ids)
            if is_content_case:
                case_result["expansion"] = evaluate_production_expansion(
                    result=result,
                    ground_truth=case_result["ground_truth_chunk_ids"],
                    raw_hit_at_5=float(case_result["hit_at_5"] or 0.0),
                    expander=expander,
                )
            else:
                reason = (
                    "bibliography_case"
                    if record.get("expected_bibliography_intent") is True
                    else "unanswerable_case"
                    if record.get("answerable") is False
                    else "invalid_ground_truth"
                )
                case_result["expansion"] = _not_applicable_expansion(reason)
            case_result["retrieval_error"] = None
        except Exception as error:  # noqa: BLE001 - 单 Case 失败必须落盘，不能改变评测分母。
            case_result = retrieval_error_case(
                record, error, invalid_ids, invalid_reasons.get(case_id)
            )
        case_results.append(case_result)

    summary = build_summary(
        case_results=case_results,
        settings=settings,
        invalid_reasons=invalid_reasons,
        git_commit=git_commit(ROOT),
    )
    # 覆盖自定义 dataset 路径，避免 summary 固定显示默认文件。
    summary["experiment"]["dataset"] = str(dataset_path)
    output_dir = args.output_dir.resolve()
    write_jsonl_atomic(output_dir / "case_results.jsonl", case_results)
    write_json_atomic(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_text_atomic(output_dir / "summary.md", render_summary_markdown(summary))
    write_text_atomic(output_dir / "failures.md", render_failures(case_results))
    return case_results, summary


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：运行 Full Baseline 并打印最终核心指标。"""
    args = parse_args(argv)
    try:
        _cases, summary = run_evaluation(args)
    except (OSError, sqlite3.Error, RetrievalEvaluationError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    content = summary["overall_retrieval"]
    bibliography = summary["bibliography"]
    print("Full Retrieval Baseline 完成")
    print(f"Hit@5: {content['hit_at_5']}")
    print(f"Recall@5: {content['recall_at_5']}")
    print(f"MRR@5: {content['mrr_at_5']}")
    print(f"Avg Retrieval Latency: {content['avg_retrieval_latency_ms']} ms")
    print(f"Bibliography Intent F1: {bibliography['intent_f1']}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0 if summary["counts"]["retrieval_error_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
