"""Phase 4 多论文结果的确定性质量指标，不调用 LLM 评分。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .contracts import ResearchRunResult, ResearchRunStatus


@dataclass(frozen=True)
class AgenticEvaluationReport:
    """用于回归测试与运行诊断的可解释指标集合。"""

    citation_validity: float
    positive_paper_coverage: float
    evidence_balance: float
    coverage_complete: bool
    fabricated_citations: tuple[str, ...]
    passed: bool


def evaluate_research_result(result: ResearchRunResult) -> AgenticEvaluationReport:
    """检查引用存在性、多论文覆盖与最终证据分配是否失衡。"""
    selected_ids = {record.evidence_id for record in result.selected_evidence}
    citations = result.answer.citations if result.answer is not None else ()
    fabricated = tuple(item for item in citations if item not in selected_ids)
    citation_validity = (
        (len(citations) - len(fabricated)) / len(citations)
        if citations
        else (1.0 if result.answer and result.answer.insufficient_evidence else 0.0)
    )

    positive_ids = set(result.coverage.positive_paper_ids) if result.coverage else set()
    selected_papers = {
        record.evidence.paper_id for record in result.selected_evidence
    }
    positive_coverage = (
        len(positive_ids & selected_papers) / len(positive_ids)
        if positive_ids
        else 1.0
    )

    counts = Counter(
        record.evidence.paper_id
        for record in result.selected_evidence
        if not positive_ids or record.evidence.paper_id in positive_ids
    )
    evidence_balance = (
        min(counts.values()) / max(counts.values()) if len(counts) > 1 else 1.0
    )
    coverage_complete = bool(result.coverage and result.coverage.coverage_complete)
    passed = (
        result.status in {ResearchRunStatus.COMPLETED, ResearchRunStatus.PARTIAL}
        and citation_validity == 1.0
        and positive_coverage == 1.0
        and coverage_complete
    )
    return AgenticEvaluationReport(
        citation_validity=citation_validity,
        positive_paper_coverage=positive_coverage,
        evidence_balance=evidence_balance,
        coverage_complete=coverage_complete,
        fabricated_citations=fabricated,
        passed=passed,
    )
