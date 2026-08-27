"""Query Planner Audit 的自动指标与人工字段回归测试。"""

from __future__ import annotations

from eval.scripts.build_query_planner_audit import (
    build_manual_review_row,
    build_planner_metrics,
)


def test_planner_metrics_count_status_coverage_and_latency() -> None:
    """Success、Partial、Degraded、Semantic、Keyword 和延迟使用同一分母。"""
    records = [
        _case("case_success", "partial", semantic="English query", keywords=[]),
        _case("case_success_keywords", "success", semantic="Query", keywords=["Model"]),
        _case("case_degraded", "partial", semantic=None, keywords=["ESDTW"]),
        _case("case_empty", "degraded", semantic=None, keywords=[]),
    ]

    metrics = build_planner_metrics(records)

    assert metrics["success_rate"] == 0.25
    assert metrics["partial_rate"] == 0.5
    assert metrics["degraded_rate"] == 0.25
    assert metrics["semantic_query_coverage"] == {"cases": 2, "total": 4, "rate": 0.5}
    assert metrics["lexical_keyword_coverage"] == {"cases": 2, "total": 4, "rate": 0.5}
    assert metrics["success_cases_with_lexical_keywords"] == {
        "cases": 1,
        "success_total": 1,
        "rate": 1.0,
    }
    assert metrics["avg_rewrite_latency_ms"] == 20.0


def test_manual_review_exposes_new_statuses_and_diagnostics() -> None:
    """人工审计行必须暴露拆分状态和程序诊断，不再套用修复前结论。"""
    record = _case(
        "method_007",
        "partial",
        semantic=None,
        keywords=["ESDTW", "DTW", "LEDTW", "shapeDTW"],
    )
    record["semantic_status"] = "unavailable"
    record["lexical_status"] = "valid_fallback"
    record["validation_diagnostics"] = ["rewrite_error:LLMRequestError"]

    review = build_manual_review_row(record)

    assert review["semantic_status"] == "unavailable"
    assert review["lexical_status"] == "valid_fallback"
    assert review["validation_diagnostics"] == ["rewrite_error:LLMRequestError"]
    assert review["entities_lost"] == []
    assert review["noise_introduced"] == []
    assert review["malformed_or_contaminated"] is False


def _case(
    case_id: str,
    status: str,
    *,
    semantic: str | None,
    keywords: list[str],
) -> dict[str, object]:
    """构造 Planner Audit 自动指标所需的最小 Case trace。"""
    return {
        "case_id": case_id,
        "rewrite_status": status,
        "original_query": "Original query",
        "resolved_query": "Resolved query",
        "semantic_query_en": semantic,
        "lexical_keywords_en": keywords,
        "latency": {"rewrite_latency_ms": 10.0 * (len(case_id) % 4 + 1)},
    }
