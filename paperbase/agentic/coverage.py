"""从 Scope、检索历史和 Qualification 推导论文覆盖状态。"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    CorpusScopeSnapshot,
    CoverageState,
    CoverageStrategy,
    EvidenceAnalysisStatus,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    PaperAssessment,
    PaperAssessmentStatus,
    QueryContext,
    QueryScope,
    ResearchObservation,
)


@dataclass(frozen=True)
class CoveragePolicy:
    candidate_min_queries_per_paper: int = 1
    exhaustive_min_queries_per_paper: int = 2

    def __post_init__(self) -> None:
        if self.candidate_min_queries_per_paper < 1:
            raise ValueError("候选验证的单篇最少查询数必须大于零。")
        if self.exhaustive_min_queries_per_paper < 1:
            raise ValueError("穷举策略的单篇最少查询数必须大于零。")


class CoverageManager:
    """Coverage 是可重算的派生状态，不把“未找到”表达成事实不存在。"""

    def __init__(self, policy: CoveragePolicy | None = None) -> None:
        self._policy = policy or CoveragePolicy()

    def evaluate(
        self,
        *,
        context: QueryContext,
        corpus_snapshot: CorpusScopeSnapshot | None,
        candidate_selection: CandidateSelection | None = None,
        evidence: tuple[EvidenceRecord, ...],
        observations: tuple[ResearchObservation, ...],
    ) -> CoverageState:
        strategy = (
            CoverageStrategy.EXHAUSTIVE
            if context.requires_complete_coverage
            else CoverageStrategy.CANDIDATE_VERIFICATION
        )
        scope_ids = self._scope_ids(context, corpus_snapshot)
        selection = self._selection(context, scope_ids, candidate_selection)
        candidate_ids = selection.candidate_paper_ids if selection is not None else ()
        uncertain_ids = selection.uncertain_paper_ids if selection is not None else ()
        required_ids = tuple(dict.fromkeys((*candidate_ids, *uncertain_ids)))
        assessments = tuple(
            self._assess_paper(
                paper_id,
                strategy=strategy,
                shallow=paper_id in set(uncertain_ids),
                evidence=evidence,
                observations=observations,
            )
            for paper_id in required_ids
        )
        return CoverageState(
            scope_paper_ids=scope_ids,
            assessments=assessments,
            strategy=strategy,
            candidate_paper_ids=candidate_ids,
            screened_paper_ids=(selection.screened_paper_ids if selection else ()),
            uncertain_paper_ids=uncertain_ids,
            excluded_paper_ids=(selection.excluded_paper_ids if selection else ()),
            discovery_completed=(
                selection is not None
                and set(selection.screened_paper_ids) == set(scope_ids)
            ),
        )

    @staticmethod
    def _scope_ids(
        context: QueryContext, snapshot: CorpusScopeSnapshot | None
    ) -> tuple[str, ...]:
        if context.scope is QueryScope.KNOWLEDGE_BASE:
            if snapshot is None:
                raise ValueError("全库 Coverage 必须绑定 CorpusScopeSnapshot。")
            return snapshot.paper_ids
        return context.target_paper_ids

    @staticmethod
    def _selection(
        context: QueryContext,
        scope_ids: tuple[str, ...],
        selection: CandidateSelection | None,
    ) -> CandidateSelection | None:
        if selection is not None:
            if selection.scope_paper_ids != scope_ids:
                raise ValueError("Candidate Selection 与当前冻结 Scope 不一致。")
            return selection
        if context.scope is QueryScope.KNOWLEDGE_BASE:
            return None
        return CandidateSelection(
            scope_paper_ids=scope_ids,
            assessments=tuple(
                CandidateAssessment(
                    paper_id,
                    CandidateDisposition.CANDIDATE,
                    "用户显式选择的论文直接进入正文验证。",
                )
                for paper_id in scope_ids
            ),
        )

    def _assess_paper(
        self,
        paper_id: str,
        *,
        strategy: CoverageStrategy,
        shallow: bool,
        evidence: tuple[EvidenceRecord, ...],
        observations: tuple[ResearchObservation, ...],
    ) -> PaperAssessment:
        records = tuple(record for record in evidence if record.evidence.paper_id == paper_id)
        queries = tuple(
            dict.fromkeys(
                query
                for observation in observations
                if observation.action_name != "search_knowledge_base"
                for observed_paper_id, query in observation.queries_by_paper
                if observed_paper_id == paper_id
            )
        )
        direct = tuple(
            record
            for record in records
            if any(
                claim.ownership is EvidenceOwnership.THIS_PAPER
                and claim.relation
                in {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}
                for claim in record.claims
            )
        )
        minimum = (
            1
            if shallow
            else (
                self._policy.exhaustive_min_queries_per_paper
                if strategy is CoverageStrategy.EXHAUSTIVE
                else self._policy.candidate_min_queries_per_paper
            )
        )
        direct_verified_outside_discovery = any(
            any(tool != "search_knowledge_base" for tool in record.originating_tools)
            for record in direct
        )
        direct_ready = len(queries) >= minimum or (
            strategy is CoverageStrategy.CANDIDATE_VERIFICATION
            and direct_verified_outside_discovery
        )
        if direct and direct_ready:
            return PaperAssessment(
                paper_id=paper_id,
                status=PaperAssessmentStatus.POSITIVE,
                queries_tried=queries,
                evidence_ids=tuple(record.evidence_id for record in direct),
                rationale="已找到并标注为本文直接证据的内容。",
            )
        all_evidence_ids = tuple(record.evidence_id for record in records)
        if len(queries) < minimum:
            return PaperAssessment(
                paper_id=paper_id,
                status=(
                    PaperAssessmentStatus.CANDIDATE
                    if queries or records
                    else PaperAssessmentStatus.UNVISITED
                ),
                queries_tried=queries,
                evidence_ids=all_evidence_ids,
                rationale=f"尚未达到当前策略要求的 {minimum} 次论文级查询。",
            )
        uncertain = any(
            record.analysis_status
            in {EvidenceAnalysisStatus.PENDING, EvidenceAnalysisStatus.FAILED}
            or any(
                claim.relation is EvidenceRelation.UNCERTAIN
                or claim.ownership is EvidenceOwnership.UNKNOWN
                for claim in record.claims
            )
            for record in records
        )
        if uncertain:
            return PaperAssessment(
                paper_id=paper_id,
                status=PaperAssessmentStatus.UNCERTAIN,
                queries_tried=queries,
                evidence_ids=all_evidence_ids,
                rationale="已按策略检索，但现有证据仍无法确定归属或关系。",
            )
        return PaperAssessment(
            paper_id=paper_id,
            status=PaperAssessmentStatus.NO_SUPPORT_FOUND,
            queries_tried=queries,
            evidence_ids=all_evidence_ids,
            rationale="在当前查询与检索策略下未找到本文直接证据；不代表事实不存在。",
        )
