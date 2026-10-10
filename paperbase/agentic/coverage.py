"""Derive paper-level retrieval coverage without judging answer support."""

from __future__ import annotations

from .contracts import (
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    CorpusScopeSnapshot,
    CoverageState,
    CoverageStrategy,
    EvidenceRecord,
    PaperAssessment,
    PaperAssessmentStatus,
    QueryContext,
    QueryScope,
    ResearchObservation,
)


class CoverageManager:
    """Coverage 是可重算的派生状态，不把“未找到”表达成事实不存在。"""

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
            uncertain_paper_ids=uncertain_ids,
            excluded_paper_ids=(selection.excluded_paper_ids if selection else ()),
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
        # Coverage answers only whether a candidate paper was actually queried
        # and yielded raw material. Semantic answer support belongs exclusively
        # to the Grounded Answerer and its validated citations.
        if records and queries:
            return PaperAssessment(
                paper_id=paper_id,
                status=PaperAssessmentStatus.POSITIVE,
                queries_tried=queries,
                evidence_ids=tuple(record.evidence_id for record in records),
                rationale="论文级检索已返回原始 Evidence；尚未据此推断答案支持。",
            )
        all_evidence_ids = tuple(record.evidence_id for record in records)
        if not queries:
            return PaperAssessment(
                paper_id=paper_id,
                status=(
                    PaperAssessmentStatus.CANDIDATE
                    if records
                    else PaperAssessmentStatus.UNVISITED
                ),
                queries_tried=queries,
                evidence_ids=all_evidence_ids,
                rationale="尚未执行论文级正文查询。",
            )
        return PaperAssessment(
            paper_id=paper_id,
            status=PaperAssessmentStatus.NO_SUPPORT_FOUND,
            queries_tried=queries,
            evidence_ids=all_evidence_ids,
            rationale="在当前查询与检索策略下未找到本文直接证据；不代表事实不存在。",
        )
