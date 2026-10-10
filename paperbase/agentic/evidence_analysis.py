"""证据分析器共享的运行期结果契约。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisCacheStats,
    EvidenceRecord,
    LLMCallRecord,
    EvidenceAnalysisStatus,
    QueryContext,
    ResearchPlan,
)


@dataclass(frozen=True)
class EvidenceAnalysisResult:
    analyses: tuple[EvidenceAnalysis, ...]
    llm_calls: int
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()
    cache_stats: EvidenceAnalysisCacheStats = EvidenceAnalysisCacheStats()


def projection_requirement_scope(record: EvidenceRecord, plan: ResearchPlan) -> tuple[str, ...]:
    """Binding is the scope authority; Plan order makes grouping deterministic."""
    requested = set(record.requested_requirement_ids)
    return tuple(item.requirement_id for item in plan.requirements
                 if item.requirement_id in requested)


def requires_analysis(record: EvidenceRecord, plan: ResearchPlan) -> bool:
    return (record.analysis_status is EvidenceAnalysisStatus.PENDING
            or bool(set(projection_requirement_scope(record, plan))
                    - set(record.assessment.evaluated_requirement_ids)))


@dataclass(frozen=True)
class EvidenceAnalysisRequest:
    question: str
    evidence: tuple[EvidenceRecord, ...]
    context: QueryContext
    plan: ResearchPlan
    requirement_scopes: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        ids = tuple(record.evidence_id for record in evidence)
        scopes = tuple((eid, tuple(rids)) for eid, rids in self.requirement_scopes)
        scope_ids = tuple(eid for eid, _ in scopes)
        if len(set(ids)) != len(ids) or len(set(scope_ids)) != len(scope_ids):
            raise ValueError("EvidenceAnalysisRequest requires unique Evidence IDs/scopes.")
        if set(ids) != set(scope_ids):
            raise ValueError("Each input Evidence requires exactly one projection scope.")
        known = {item.requirement_id for item in self.plan.requirements}
        by_id = dict(scopes)
        for record in evidence:
            scope = by_id[record.evidence_id]
            if not scope or len(scope) != len(set(scope)) or not set(scope).issubset(known):
                raise ValueError("Projection scope must be nonempty, unique and belong to Plan.")
            if set(scope) != set(projection_requirement_scope(record, self.plan)):
                raise ValueError("Projection scope must equal Evidence-local binding intersect Plan.")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "requirement_scopes", tuple(
            (record.evidence_id, projection_requirement_scope(record, self.plan))
            for record in evidence
        ))

    def scope_for(self, evidence_id: str) -> tuple[str, ...]:
        for eid, scope in self.requirement_scopes:
            if eid == evidence_id:
                return scope
        raise KeyError(evidence_id)


class EvidenceAnalyzer(Protocol):
    def analyze(
        self, *, request: EvidenceAnalysisRequest
    ) -> EvidenceAnalysisResult: ...


class LegacyEvidenceAnalyzer(Protocol):
    def analyze(self, *, question: str, evidence: tuple[EvidenceRecord, ...]) -> EvidenceAnalysisResult: ...


class LegacyEvidenceAnalyzerAdapter:
    """Explicit construction-time compatibility; claims cannot attest a task scope."""

    def __init__(self, analyzer: LegacyEvidenceAnalyzer) -> None:
        self._analyzer = analyzer

    def analyze(self, *, request: EvidenceAnalysisRequest) -> EvidenceAnalysisResult:
        result = self._analyzer.analyze(question=request.question, evidence=request.evidence)
        return replace(result, analyses=tuple(
            replace(item, evaluated_requirement_ids=()) for item in result.analyses
        ))


def merge_analysis_groups(record: EvidenceRecord, groups: tuple[EvidenceAnalysis, ...],
                          plan: ResearchPlan) -> EvidenceAnalysis:
    """Append successful missing-scope results; failed groups attest no evaluation."""
    from .projection_reads import dedupe_projections
    from .grounding import verified_support_ids
    old = record.assessment
    successful = tuple(item for item in groups if item.status in {
        EvidenceAnalysisStatus.QUALIFIED, EvidenceAnalysisStatus.IRRELEVANT})
    if not successful:
        if old.status in {EvidenceAnalysisStatus.QUALIFIED, EvidenceAnalysisStatus.IRRELEVANT}:
            return old
        return EvidenceAnalysis(record.evidence_id, EvidenceAnalysisStatus.FAILED,
                                error_code="EvidenceProjectionFailed")
    projections = dedupe_projections((*old.projections,
        *(projection for item in successful for projection in item.projections)))
    evaluated = set(old.evaluated_requirement_ids)
    evaluated.update(rid for item in successful for rid in item.evaluated_requirement_ids)
    desired = set(projection_requirement_scope(record, plan))
    if not evaluated.issubset(desired):
        raise ValueError("Aggregate evaluated scope exceeds Evidence-local binding.")
    candidate = EvidenceAnalysis(record.evidence_id,
        EvidenceAnalysisStatus.QUALIFIED if projections else EvidenceAnalysisStatus.IRRELEVANT,
        projections=projections,
        evaluated_requirement_ids=tuple(r.requirement_id for r in plan.requirements if r.requirement_id in evaluated),
        projection_rejections=tuple(dict.fromkeys((*old.projection_rejections,
            *(reason for item in groups for reason in item.projection_rejections)))),
    )
    if not set(verified_support_ids(record, plan.requirements)).issubset(
        verified_support_ids(replace(record, assessment=candidate), plan.requirements)):
        raise ValueError("support_regression in aggregate merged assessment.")
    return candidate
