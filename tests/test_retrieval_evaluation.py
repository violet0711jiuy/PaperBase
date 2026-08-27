"""Full Retrieval Baseline 确定性指标与失败分类测试。"""

from __future__ import annotations

from paperbase.config import load_settings
from paperbase.generation.section_expander import (
    EvidenceUnit,
    ExpansionResult,
    SectionAwareNeighborExpander,
)
from paperbase.retrieval.hybrid_retriever import RetrievedChunk, RetrievalResult
from paperbase.retrieval.query_rewriter import QueryRewritePlan

from eval.scripts.run_retrieval_eval import (
    DEFAULT_DATASET,
    ProductionRetrieverObserver,
    RetrievalTrace,
    aggregate_ranking_metrics,
    build_expansion_evaluation,
    build_slice_metrics,
    classify_content_failure,
    create_production_expander,
    evaluate_production_expansion,
    intent_metrics,
    is_content_expansion_case,
    ranking_metrics,
    reciprocal_rank,
    reciprocal_rank_at_k,
)


def test_expansion_trace_deduplicates_ids_and_recovers_raw_miss() -> None:
    """真实Evidence对象展开后应去重，并把raw miss→expanded hit标为recovered。"""
    result = _retrieval_result(("anchor_1", "anchor_2"))
    original_anchors = tuple(chunk.chunk_id for chunk in result.chunks)
    expander = _StaticExpander(
        _expansion_result(
            seed_ids=("anchor_1", "anchor_2"),
            chunk_ids=("anchor_1", "gt_1", "gt_1", "anchor_2"),
        )
    )

    trace = evaluate_production_expansion(
        result=result,
        ground_truth=("gt_1", "gt_2"),
        raw_hit_at_5=0.0,
        expander=expander,
    )

    assert trace["expanded_chunk_ids"] == ["anchor_1", "gt_1", "anchor_2"]
    assert trace["expanded_hit"] == 1.0
    assert trace["expanded_recall"] == 0.5
    assert trace["diagnosis"] == "expansion_recovered"
    assert trace["ground_truth_hits_after_expansion"] == ["gt_1"]
    assert tuple(chunk.chunk_id for chunk in result.chunks) == original_anchors


def test_expansion_receives_only_final_top_5_anchors() -> None:
    """Expansion输入必须严格止于Final Top-5，不能利用更深排名候选。"""
    result = _retrieval_result(tuple(f"anchor_{index}" for index in range(1, 8)))
    expander = _StaticExpander(
        _expansion_result(
            seed_ids=("anchor_1",),
            chunk_ids=("anchor_1",),
        )
    )

    trace = evaluate_production_expansion(
        result=result,
        ground_truth=("anchor_1",),
        raw_hit_at_5=1.0,
        expander=expander,
    )

    assert trace["anchor_chunk_ids"] == [
        "anchor_1",
        "anchor_2",
        "anchor_3",
        "anchor_4",
        "anchor_5",
    ]
    assert expander.received_anchor_ids == tuple(trace["anchor_chunk_ids"])


def test_partial_expansion_is_diagnosed_without_changing_raw_metrics() -> None:
    """已有raw hit但Expansion仍缺GT时应标incomplete，原主指标计算保持不变。"""
    result = _retrieval_result(("gt_1",))
    trace = evaluate_production_expansion(
        result=result,
        ground_truth=("gt_1", "gt_2"),
        raw_hit_at_5=1.0,
        expander=_StaticExpander(
            _expansion_result(seed_ids=("gt_1",), chunk_ids=("gt_1",))
        ),
    )
    case = _case("fact", "single_hop", gt="gt_1", rank=1)
    baseline = aggregate_ranking_metrics(
        [case], ranking_field="final_results", cutoffs=(5,), mrr_k=5
    )
    case["expansion"] = trace

    assert trace["diagnosis"] == "post_expansion_incomplete"
    assert aggregate_ranking_metrics(
        [case], ranking_field="final_results", cutoffs=(5,), mrr_k=5
    ) == baseline


def test_expansion_error_is_recorded_without_raising() -> None:
    """生产Expansion异常只能降低Expansion指标，不能中止整次Retrieval Eval。"""
    trace = evaluate_production_expansion(
        result=_retrieval_result(("anchor_1",)),
        ground_truth=("gt_1",),
        raw_hit_at_5=0.0,
        expander=_FailingExpander(),
    )

    assert trace["status"] == "error"
    assert trace["expanded_hit"] == 0.0
    assert trace["expanded_recall"] == 0.0
    assert trace["diagnosis"] == "expansion_error"


def test_expansion_metrics_exclude_bibliography_and_unanswerable_cases() -> None:
    """Bibliography和Unanswerable不得进入32条正文Expansion分母。"""
    assert is_content_expansion_case(
        {"answerable": True, "expected_bibliography_intent": False}, set()
    )
    assert not is_content_expansion_case(
        {"answerable": True, "expected_bibliography_intent": True}, set()
    )
    assert not is_content_expansion_case(
        {"answerable": False, "expected_bibliography_intent": False}, set()
    )


def test_expansion_summary_computes_gain_context_size_and_slices() -> None:
    """Expansion汇总应正确计算Hit/Recall增益、规模与type/hop切片。"""
    recovered = _expansion_case(
        "case_1", raw_hit=0.0, raw_recall=0.0, expanded_hit=1.0, expanded_recall=0.5
    )
    complete = _expansion_case(
        "case_2", raw_hit=1.0, raw_recall=1.0, expanded_hit=1.0, expanded_recall=1.0
    )

    summary = build_expansion_evaluation([recovered, complete])

    assert summary["after_expansion"] == {
        "expanded_hit": 1.0,
        "expanded_recall": 0.75,
    }
    assert summary["delta"] == {"hit": 0.5, "recall": 0.25}
    assert summary["recovery_cases"] == {"count": 1, "case_ids": ["case_1"]}
    assert summary["context_size"]["avg_expanded_chunks"] == 4.0
    assert summary["context_size"]["max_expanded_chunks"] == 5
    assert summary["slice_before_after"]["type"]["fact"]["expanded_recall"] == 0.75


def test_eval_factory_uses_production_section_aware_expander() -> None:
    """Evaluation必须装配生产Expander类，不能使用eval目录内的模拟算法。"""
    expander = create_production_expander(load_settings())

    assert isinstance(expander, SectionAwareNeighborExpander)


def test_context_free_observer_does_not_pass_golden_scope_to_retriever() -> None:
    """Context-free 评测只能传 Query，不能把 Golden paper_id 变成隐藏上下文。"""
    retriever = _ContextFreeRetriever()
    observer = ProductionRetrieverObserver(retriever)

    result, _trace = observer.retrieve("D²STGNN如何学习动态空间依赖？")

    assert result == "context-free-result"
    assert retriever.received_query == "D²STGNN如何学习动态空间依赖？"
    assert DEFAULT_DATASET.name == "golden_dataset_v1_2.jsonl"


def test_trace_reset_does_not_clear_previous_case_rankings() -> None:
    """复用模型进入下一条 Case 时，上一条已保存的阶段排名必须保持不变。"""
    trace = RetrievalTrace(
        route_results={"original_dense": [_item("dense_1")]},
        rrf_results=[_item("rrf_1")],
        reranked_candidates=[_item("reranked_1")],
        bibliography_results=[_item("reference_1")],
    )
    previous_rrf = trace.rrf_results
    previous_reranked = trace.reranked_candidates
    previous_bibliography = trace.bibliography_results

    trace.reset()

    assert previous_rrf == [_item("rrf_1")]
    assert previous_reranked == [_item("reranked_1")]
    assert previous_bibliography == [_item("reference_1")]
    assert trace.rrf_results == []


def test_ranking_metrics_support_multiple_ground_truth_chunks() -> None:
    """Recall 分母必须使用全部人工 Ground Truth，不能把 single_hop 简化为一块。"""
    ranking = [_item("other"), _item("gt_2"), _item("gt_1")]

    assert ranking_metrics(ranking, ["gt_1", "gt_2"], k=2) == (1.0, 0.5)
    assert ranking_metrics(ranking, ["gt_1", "gt_2"], k=3) == (1.0, 1.0)
    assert reciprocal_rank(ranking, ["gt_1", "gt_2"]) == 0.5
    assert reciprocal_rank_at_k(ranking, ["gt_1"], k=2) == 0.0
    assert reciprocal_rank_at_k(ranking, ["gt_1"], k=3) == 1 / 3


def test_fixed_stage_and_slice_metrics_use_the_requested_cutoffs() -> None:
    """Stage 固定输出 @5/@10/@20，Slice 固定输出最终 Top-5 指标。"""
    cases = [
        _case("fact", "single_hop", gt="gt_1", rank=1),
        _case("method", "multi_hop", gt="gt_2", rank=6),
    ]

    stage = aggregate_ranking_metrics(
        cases,
        ranking_field="rrf_results",
        cutoffs=(5, 10, 20),
        mrr_k=20,
    )
    slices = build_slice_metrics(
        cases,
        values=("fact", "method"),
        selector="primary_type",
    )

    assert stage == {
        "cases": 2,
        "hit_at_5": 0.5,
        "recall_at_5": 0.5,
        "hit_at_10": 1.0,
        "recall_at_10": 1.0,
        "hit_at_20": 1.0,
        "recall_at_20": 1.0,
        "mrr_at_20": 0.583333,
    }
    assert slices["fact"] == {
        "cases": 1,
        "hit_at_5": 1.0,
        "recall_at_5": 1.0,
        "mrr_at_5": 1.0,
    }
    assert slices["method"] == {
        "cases": 1,
        "hit_at_5": 0.0,
        "recall_at_5": 0.0,
        "mrr_at_5": 0.0,
    }


def test_failure_classification_distinguishes_reranker_drop_and_ranking_failure() -> None:
    """RRF Top-5 被挤出才叫 reranker_drop，较深候选未进 Top-5 属于 ranking_failure。"""
    reranker_drop, _ = classify_content_failure(
        ground_truth=["gt"],
        original_dense=[_item("gt")],
        rewritten_dense=[],
        rewritten_bm25=[],
        rrf_results=[_item("gt"), _item("other")],
        final_results=[_item("other")],
        reranking_status="success",
    )
    ranking_failure, _ = classify_content_failure(
        ground_truth=["gt"],
        original_dense=[],
        rewritten_dense=[_item("gt")],
        rewritten_bm25=[],
        rrf_results=[*[_item(f"other_{index}") for index in range(6)], _item("gt")],
        final_results=[_item("other")],
        reranking_status="success",
    )

    assert reranker_drop == "reranker_drop"
    assert ranking_failure == "ranking_failure"


def test_retrieval_miss_and_conservative_rewrite_drift_require_trace_evidence() -> None:
    """只有原始 Dense 明确命中且改写路径全失效时，才保守标记 rewrite_drift。"""
    drift, _ = classify_content_failure(
        ground_truth=["gt"],
        original_dense=[_item("gt")],
        rewritten_dense=[_item("other")],
        rewritten_bm25=[],
        rrf_results=[_item("other")],
        final_results=[_item("other")],
        reranking_status="success",
    )
    miss, _ = classify_content_failure(
        ground_truth=["gt"],
        original_dense=[_item("other")],
        rewritten_dense=[],
        rewritten_bm25=[],
        rrf_results=[_item("other")],
        final_results=[_item("other")],
        reranking_status="success",
    )

    assert drift == "rewrite_drift"
    assert miss == "retrieval_miss"


def test_bibliography_intent_metrics_use_all_binary_cases() -> None:
    """Intent 四项指标由确定性混淆矩阵计算，不交给 LLM Judge。"""
    cases = [
        _intent(True, True),
        _intent(True, False),
        _intent(False, True),
        _intent(False, False),
    ]

    assert intent_metrics(cases) == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }


def _item(chunk_id: str) -> dict[str, str]:
    """构造只含指标计算所需 chunk_id 的最小排名项。"""
    return {"chunk_id": chunk_id}


class _ContextFreeRetriever:
    """只接受 query 的最小 Retriever；若 Observer 传 scope，测试会直接 TypeError。"""

    def __init__(self) -> None:
        self._query_planner = object()
        self._query_embedder = object()
        self._reranker = None
        self.received_query: str | None = None

    def _add_dense_matches(self, *args: object, **kwargs: object) -> None:
        """提供 Observer 安装 trace 所需的 Dense 阶段边界。"""

    def _add_rewritten_bm25_matches(self, *args: object, **kwargs: object) -> None:
        """提供 Observer 安装 trace 所需的 BM25 阶段边界。"""

    def _rerank_content_candidates(self, *args: object, **kwargs: object) -> tuple[list[object], str]:
        """提供 Observer 安装 trace 所需的 Reranker 阶段边界。"""
        return [], "success"

    def _collect_bibliography_candidates(self, *args: object, **kwargs: object) -> list[object]:
        """提供 Observer 安装 trace 所需的 Bibliography 阶段边界。"""
        return []

    def retrieve(self, query: str) -> str:
        """严格只接收 Query，以验证评测器没有传入任何隐藏范围。"""
        self.received_query = query
        return "context-free-result"


def _intent(expected: bool, predicted: bool) -> dict[str, bool]:
    """构造一个 bibliography intent 二分类结果。"""
    return {
        "expected_bibliography_intent": expected,
        "predicted_bibliography_intent": predicted,
    }


def _case(primary_type: str, hop: str, *, gt: str, rank: int) -> dict[str, object]:
    """构造带同一份 RRF/final 排名的最小正文 Case。"""
    ranking = [_item(f"other_{index}") for index in range(1, rank)] + [_item(gt)]
    return {
        "primary_type": primary_type,
        "tags": [hop],
        "ground_truth_chunk_ids": [gt],
        "rrf_results": ranking,
        "final_results": ranking[:5],
    }


def _retrieval_result(chunk_ids: tuple[str, ...]) -> RetrievalResult:
    """构造可直接交给生产Expander接口的最小Final RetrievalResult。"""
    chunks = tuple(
        RetrievedChunk(
            rank=index,
            chunk_id=chunk_id,
            vector_id=index,
            paper_id="paper_1",
            paper_title="Test Paper",
            section="3.2 Method",
            content_kind="body",
            front_matter_type=None,
            section_type="content",
            page_start=3,
            page_end=3,
            raw_text=f"content for {chunk_id}",
            pre_rerank_rank=index,
            rerank_score=1.0,
            fused_score=1.0,
            source_matches=(),
        )
        for index, chunk_id in enumerate(chunk_ids, start=1)
    )
    return RetrievalResult(
        query="test query",
        rewrite_plan=QueryRewritePlan(original_query="test query"),
        reranking_status="success",
        chunks=chunks,
    )


def _expansion_result(
    *, seed_ids: tuple[str, ...], chunk_ids: tuple[str, ...]
) -> ExpansionResult:
    """构造生产ExpansionResult形状的正文Evidence，供指标测试复用。"""
    return ExpansionResult(
        content_evidence=(
            EvidenceUnit(
                evidence_id="E1",
                kind="content",
                paper_id="paper_1",
                paper_title="Test Paper",
                section="3.2 Method",
                page_start=3,
                page_end=3,
                seed_chunk_ids=seed_ids,
                chunk_ids=chunk_ids,
                text="expanded evidence",
                token_count=10,
            ),
        ),
        bibliography_evidence=(),
    )


class _StaticExpander:
    """返回固定生产数据结构，并记录Evaluation传入的Final锚点。"""

    def __init__(self, expansion: ExpansionResult) -> None:
        self._expansion = expansion
        self.received_anchor_ids: tuple[str, ...] = ()

    def expand(self, result: RetrievalResult) -> ExpansionResult:
        """保存输入锚点后返回固定Expansion结果。"""
        self.received_anchor_ids = tuple(chunk.chunk_id for chunk in result.chunks)
        return self._expansion


class _FailingExpander:
    """模拟生产Expansion异常，验证单Case失败不会中止整次评测。"""

    def expand(self, result: RetrievalResult) -> ExpansionResult:
        """稳定抛出异常，使错误路径可确定地回归测试。"""
        raise RuntimeError("expansion failed")


def _expansion_case(
    case_id: str,
    *,
    raw_hit: float,
    raw_recall: float,
    expanded_hit: float,
    expanded_recall: float,
) -> dict[str, object]:
    """构造一个带Expansion trace的正文Case，用于汇总与切片测试。"""
    recovered = raw_hit == 0.0 and expanded_hit == 1.0
    complete = expanded_recall == 1.0
    return {
        "case_id": case_id,
        "primary_type": "fact",
        "tags": ["single_hop", "easy"],
        "hit_at_5": raw_hit,
        "recall_at_5": raw_recall,
        "expansion": {
            "status": "success",
            "expanded_hit": expanded_hit,
            "expanded_recall": expanded_recall,
            "expanded_chunk_count": 3 if recovered else 5,
            "expanded_context_char_count": 100 if recovered else 200,
            "expanded_context_token_count": 20 if recovered else 40,
            "expansion_latency_ms": 2.0 if recovered else 4.0,
            "diagnosis": (
                "expansion_recovered"
                if recovered
                else "expansion_complete"
                if complete
                else "post_expansion_incomplete"
            ),
        },
    }
