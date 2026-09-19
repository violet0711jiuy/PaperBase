"""从现有运行状态派生逐论文 Research Profile。"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    EvidenceAnalysisStatus,
    EvidenceRecord,
    EvidenceTarget,
    ResearchObservation,
    ResearchPlan,
)


@dataclass(frozen=True)
class PaperResearchProfile:
    """Decision 所需的逐论文已知事实和剩余缺口，不作为独立状态持久化。"""

    paper_id: str
    queries_tried: tuple[str, ...]
    requested_targets: frozenset[EvidenceTarget]
    covered_targets: frozenset[EvidenceTarget]
    remaining_targets: frozenset[EvidenceTarget]
    requested_information: tuple[str, ...]
    addressed_information: tuple[str, ...]
    remaining_information: tuple[str, ...]
    sections_seen: tuple[str, ...]
    qualified_digests: tuple[str, ...]
    qualified_claims: tuple[str, ...]
    empty_search_count: int = 0


def build_research_profiles(
    *,
    paper_ids: tuple[str, ...],
    plan: ResearchPlan,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...],
    default_targets: frozenset[EvidenceTarget] = frozenset(),
) -> tuple[PaperResearchProfile, ...]:
    """用 Registry 与 Observation 重算 Profile，避免维护第二份可漂移状态。"""
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
        planned_targets = frozenset(
            target
            for subtask in plan.subtasks
            if not subtask.target_paper_ids or paper_id in subtask.target_paper_ids
            for target in subtask.evidence_targets
        )
        requested_targets = frozenset(
            (
                *default_targets,
                *planned_targets,
                *(target for attempt in attempts for target in attempt.evidence_targets),
                *(
                    target
                    for record in records
                    for target in record.requested_evidence_targets
                ),
            )
        )
        requested_information = tuple(
            dict.fromkeys(
                (
                    *(
                        item
                        for attempt in attempts
                        for item in attempt.missing_information
                    ),
                    *(
                        item
                        for record in records
                        for item in record.requested_missing_information
                    ),
                )
            )
        )
        qualified = tuple(
            record
            for record in records
            if record.analysis_status is EvidenceAnalysisStatus.QUALIFIED
        )
        covered_targets = frozenset(
            target for record in qualified for target in record.covered_targets
        )
        addressed_information = tuple(
            dict.fromkeys(
                item
                for record in qualified
                for item in record.addressed_information
            )
        )
        addressed_set = set(addressed_information)
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
                remaining_targets=requested_targets - covered_targets,
                requested_information=requested_information,
                addressed_information=addressed_information,
                remaining_information=tuple(
                    item
                    for item in requested_information
                    if item not in addressed_set
                ),
                sections_seen=sections_seen,
                qualified_digests=tuple(
                    record.digest for record in qualified if record.digest
                )[-4:],
                qualified_claims=tuple(
                    claim.statement
                    for record in qualified
                    for claim in record.claims
                )[-8:],
                empty_search_count=sum(
                    attempt.status == "empty" for attempt in attempts
                ),
            )
        )
    return tuple(profiles)
