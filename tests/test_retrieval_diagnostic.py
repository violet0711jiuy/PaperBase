"""Retrieval Diagnostic Report 的确定性回归测试。"""

from __future__ import annotations

from pathlib import Path

from eval.scripts.build_retrieval_diagnostic import (
    build_report,
    build_cross_paper_diagnostics,
    build_ground_truth_rank_changes,
    build_reranker_comparison,
    build_route_diagnostics,
    build_rrf_curve,
    validate_trace_structure,
)


def test_route_diagnostics_distinguish_overall_and_available_only() -> None:
    """Rewrite 路径缺失应降低 Overall，但不能污染 Available-only 路径质量。"""
    available = _case("case_1", gt="gt_1")
    available["rewritten_dense"] = [_ranked("gt_1", 1, "paper_1")]
    available["stage_availability"]["rewritten_dense"] = True
    unavailable = _case("case_2", gt="gt_2")

    diagnostics = build_route_diagnostics([available, unavailable])

    rewritten = diagnostics["rewritten_dense"]
    assert rewritten["available_cases"] == 1
    assert rewritten["availability_rate"] == 0.5
    assert rewritten["overall"]["hit_at_5"] == 0.5
    assert rewritten["when_available"]["hit_at_5"] == 1.0
    assert diagnostics["original_bm25"]["implemented"] is False
    assert diagnostics["original_bm25"]["overall"] is None


def test_rrf_curve_and_reranker_delta_use_the_same_cutoff() -> None:
    """Reranker 提升必须与同 K 的 RRF 比较，不能混合 Top-5 和 Top-20。"""
    case = _case("case_1", gt="gt_1")
    case["rrf_results"] = [
        *[_ranked(f"other_{index}", index, "paper_1") for index in range(1, 7)],
        _ranked("gt_1", 7, "paper_1"),
    ]
    case["reranked_results"] = [
        _ranked("gt_1", 1, "paper_1"),
        *[_ranked(f"other_{index}", index + 1, "paper_1") for index in range(1, 7)],
    ]

    curve = build_rrf_curve([case])
    comparison = build_reranker_comparison([case])

    assert curve["at_5"] == {"hit": 0.0, "recall": 0.0}
    assert curve["at_10"] == {"hit": 1.0, "recall": 1.0}
    assert comparison["at_5"]["delta"]["hit_at_5"] == 1.0
    assert comparison["at_10"]["delta"]["hit_at_10"] == 0.0


def test_ground_truth_rank_changes_keep_chunk_level_movements() -> None:
    """每个 Ground Truth chunk 都必须保留独立的 pre/post rank，而非只保存 Case 最佳值。"""
    case = _case("case_1", gt="gt_1")
    case["ground_truth_chunk_ids"] = ["gt_1", "gt_2"]
    case["rrf_results"] = [_ranked("gt_1", 3, "paper_1")]
    case["reranked_results"] = [_ranked("gt_1", 1, "paper_1")]
    case["final_results"] = [_ranked("gt_1", 1, "paper_1")]

    changes = build_ground_truth_rank_changes([case])

    assert changes["summary"]["movement_counts"] == {
        "missing_both": 1,
        "promoted": 1,
    }
    first, second = changes["chunk_rank_changes"]
    assert first["rank_delta_post_minus_pre"] == -2
    assert first["final_top_5_rank"] == 1
    assert second["rrf_rank"] is None
    assert second["movement"] == "missing_both"


def test_cross_paper_noise_and_trace_validation_are_explicit() -> None:
    """跨论文噪声按目标 paper_id 判断，必需阶段为空时报告 trace 不完整。"""
    case = _case("case_1", gt="gt_1")
    case["rrf_results"] = [
        _ranked("gt_1", 1, "paper_1"),
        _ranked("other", 2, "paper_2"),
    ]
    noise = build_cross_paper_diagnostics([case])

    assert noise["rrf_at_40"]["cross_paper_count"] == 1
    assert noise["rrf_at_40"]["cross_paper_rate"] == 0.5
    assert validate_trace_structure([case])["valid"] is True

    case["reranked_results"] = []
    validation = validate_trace_structure([case])
    assert validation["valid"] is False
    assert validation["empty_required_rankings"] == {
        "case_1": ["reranked_results"]
    }


def test_diagnostic_report_contains_independent_expansion_evaluation() -> None:
    """Diagnostic必须复用Expansion trace输出独立指标，不能覆盖原Retrieval指标。"""
    case = _case("case_1", gt="gt_1")

    report = build_report([case], source_path=Path("case_results.jsonl"))

    expansion = report["evidence_expansion_evaluation"]
    assert expansion["cases"] == 1
    assert expansion["before_expansion"]["hit_at_5"] == 1.0
    assert expansion["after_expansion"]["expanded_hit"] == 1.0


def _case(case_id: str, *, gt: str) -> dict[str, object]:
    """构造包含 Diagnostic 所需字段的最小正文 Case。"""
    ranking = [_ranked(gt, 1, "paper_1")]
    return {
        "case_id": case_id,
        "primary_type": "fact",
        "tags": ["single_hop", "easy"],
        "paper_id": "paper_1",
        "answerable": True,
        "expected_bibliography_intent": False,
        "invalid_ground_truth": False,
        "retrieval_error": None,
        "rewrite_status": "degraded",
        "ground_truth_chunk_ids": [gt],
        "stage_availability": {
            "original_dense": True,
            "original_bm25": False,
            "rewritten_dense": False,
            "rewritten_bm25": False,
            "rrf_results": True,
            "reranked_results": True,
        },
        "original_dense": ranking,
        "original_bm25": [],
        "rewritten_dense": [],
        "rewritten_bm25": [],
        "rrf_results": ranking,
        "reranked_results": ranking,
        "final_results": ranking,
        "hit_at_5": 1.0,
        "recall_at_5": 1.0,
        "expansion": {
            "available": True,
            "status": "success",
            "error": None,
            "anchor_chunk_ids": [gt],
            "expanded_chunk_ids": [gt],
            "expanded_chunk_count": 1,
            "expanded_context_char_count": 20,
            "expanded_context_token_count": 5,
            "expansion_latency_ms": 1.0,
            "per_anchor": [
                {"anchor_chunk_id": gt, "expanded_chunk_ids": [gt]}
            ],
            "ground_truth_hits_after_expansion": [gt],
            "expanded_hit": 1.0,
            "expanded_recall": 1.0,
            "diagnosis": "expansion_complete",
        },
    }


def _ranked(chunk_id: str, rank: int, paper_id: str) -> dict[str, object]:
    """构造一个带显式排名和论文范围的阶段结果。"""
    return {"chunk_id": chunk_id, "rank": rank, "paper_id": paper_id}
