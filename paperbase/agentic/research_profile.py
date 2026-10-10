"""从现有运行状态派生逐论文 Research Profile。"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    EvidenceAnalysisStatus,
    EvidenceRecord,
    EvidenceTarget,
    CoverageState,
    QueryContext,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
)
from .grounding import projection_facts, verified_support_ids
from .requirement_targets import targets_for_requirement_ids


@dataclass(frozen=True)
class PaperResearchProfile:
    """Decision 所需的逐论文已知事实和剩余缺口，不作为独立状态持久化。"""

    paper_id: str
    queries_tried: tuple[str, ...]
    # EvidenceTarget is a retrieval hint, not an answer-completion state.
    requested_targets: frozenset[EvidenceTarget]
    covered_targets: frozenset[EvidenceTarget]
    supported_requirement_ids: tuple[str, ...]
    sections_seen: tuple[str, ...]
    qualified_digests: tuple[str, ...]
    qualified_claims: tuple[str, ...]
    empty_search_count: int = 0
    unresolved_targets: frozenset[EvidenceTarget] = frozenset()

    @property
    def remaining_targets(self) -> frozenset[EvidenceTarget]:
        return self.unresolved_targets


def build_research_profiles(
    *,
    paper_ids: tuple[str, ...],
    plan: ResearchPlan,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...],
    default_targets: frozenset[EvidenceTarget] = frozenset(),
) -> tuple[PaperResearchProfile, ...]:
    """用 Registry 与 Observation 重算 Profile，避免维护第二份可漂移状态。"""
    # Remaining routing intent is Requirement-level across the current State,
    # not a per-paper completeness claim. Local gaps stay in support boundaries.
    globally_supported_ids = {
        rid for record in evidence for rid in verified_support_ids(record, plan.requirements)
    }
    unresolved_targets = targets_for_requirement_ids(plan, tuple(
        item.requirement_id for item in plan.requirements
        if item.requirement_id not in globally_supported_ids))
    profiles: list[PaperResearchProfile] = []
    for paper_id in dict.fromkeys(paper_ids):
        records = tuple(
            record for record in evidence if record.evidence.paper_id == paper_id
        )
        attempts = tuple(
            attempt
            for observation in observations
            for attempt in observation.search_attempts
            if attempt.paper_id == paper_id
        )
        legacy_queries = tuple(
            query
            for observation in observations
            for observed_paper_id, query in observation.queries_by_paper
            if observed_paper_id == paper_id
        )
        queries = tuple(
            dict.fromkeys(
                (
                    *(attempt.query for attempt in attempts),
                    *legacy_queries,
                    *(query for record in records for query in record.supporting_queries),
                )
            )
        )
        # Plan declares intent; wire/default/Registry target fields are telemetry.
        requested_targets = targets_for_requirement_ids(
            plan, tuple(item.requirement_id for item in plan.requirements))
        qualified = tuple(
            record
            for record in records
            if record.analysis_status is EvidenceAnalysisStatus.QUALIFIED
        )
        covered_targets = frozenset()  # deprecated compatibility telemetry
        supported_ids = {
            requirement_id
            for record in qualified
            for requirement_id in verified_support_ids(record, plan.requirements)
        }
        sections_seen = tuple(
            dict.fromkeys(
                value
                for record in records
                for value in (record.evidence.section_id, record.evidence.section)
                if value
            )
        )
        profiles.append(
            PaperResearchProfile(
                paper_id=paper_id,
                queries_tried=queries,
                requested_targets=requested_targets,
                covered_targets=covered_targets,
                unresolved_targets=unresolved_targets,
                supported_requirement_ids=tuple(
                    item.requirement_id for item in plan.requirements
                    if item.requirement_id in supported_ids
                ),
                sections_seen=sections_seen,
                qualified_digests=tuple(
                    record.digest for record in qualified if record.digest
                )[-4:],
                qualified_claims=tuple(
                    statement
                    for record in qualified
                    for statement in (
                        tuple(value for projection in record.projections for value in projection_facts(projection))
                        if record.projections
                        else tuple(claim.statement for claim in record.claims)
                    )
                )[-8:],
                empty_search_count=sum(
                    attempt.status == "empty" for attempt in attempts
                ),
            )
        )
    return tuple(profiles)


def goal_is_sufficient(
    *,
    context: QueryContext,
    plan: ResearchPlan,
    coverage: CoverageState,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...] = (),
) -> bool:
    """Formal stop gate: scope complete and every required state terminal."""
    from .requirement_state import finalize_requirement_states, required_requirements_terminal

    # Existing non-enumeration flows retain their explicit FinishAction path;
    # this automatic stop gate is the exhaustive-enumeration convergence path.
    if context.retrieval_goal is not ResearchGoal.EXHAUSTIVE_ENUMERATION:
        return False
    if not coverage.coverage_complete:
        return False
    states = finalize_requirement_states(
        plan=plan,
        evidence=evidence,
        observations=observations,
    ).states
    return required_requirements_terminal(plan, states)


def goal_signal_items(evidence: tuple[EvidenceRecord, ...]) -> frozenset[str]:
    """用于 no_goal_progress 的稳定集合，只统计用户可见的方法信息。"""
    return frozenset(
        f"{kind}:{_normalize_entity(value)}"
        for record in evidence
        if record.analysis_status is EvidenceAnalysisStatus.QUALIFIED
        for kind, values in (
            ("method", record.method_entities),
            ("family", record.method_families),
            ("baseline", record.baseline_entities),
        )
        for value in values
        if _normalize_entity(value)
    )


def _normalize_entity(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())
