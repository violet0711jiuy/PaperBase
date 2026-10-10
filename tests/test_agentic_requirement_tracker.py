"""Phase 4: Projection-to-RequirementState and stop-gate regression tests."""

from __future__ import annotations

from dataclasses import replace

from paperbase.agentic.audit import build_agentic_audit_snapshot
from paperbase.agentic.contracts import (
    CoverageState,
    CoverageStrategy,
    EnumerationProjection,
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceTarget,
    PaperAssessment,
    PaperAssessmentStatus,
    PaperSearchAttempt,
    ProjectedEntity,
    ProjectedEntityRole,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
    ResearchRequirement,
    RequirementSupportAnchor,
    ScopeSource,
)
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.agentic.requirement_state import (
    derive_requirement_states,
    finalize_requirement_states,
    required_requirements_researched,
    required_requirements_terminal,
    track_requirement_states,
)
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence, ToolStatus
from paperbase.retrieval.contracts import PaperSearchSpec
from paperbase.agentic.contracts import (
    GetPaperOverviewAction,
    GetSectionContextAction,
    SearchPapersAction,
)


def _context() -> QueryContext:
    return QueryContext(
        original_query="有哪些风速预测方法？",
        resolved_query="有哪些风速预测方法？",
        intent="retrieve",
        scope=QueryScope.KNOWLEDGE_BASE,
        target_paper_ids=(),
        scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
        retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="枚举方法",
        requirements=(
            ResearchRequirement("R1", "论文自身提出或实际采用的核心方法", required=True, evidence_targets=frozenset({EvidenceTarget.METHOD})),
            ResearchRequirement("R2", "用于实验比较的 baseline/comparison methods", required=True, evidence_targets=frozenset({EvidenceTarget.METHOD})),
            ResearchRequirement("R3", "正文明确讨论的 prior-work 方法或方法族", required=True, evidence_targets=frozenset({EvidenceTarget.PRIOR_WORK})),
        ),
    )


def _coverage(*, complete: bool) -> CoverageState:
    statuses = (
        PaperAssessment("P1", PaperAssessmentStatus.POSITIVE, queries_tried=("own",)),
        PaperAssessment(
            "P2",
            PaperAssessmentStatus.POSITIVE if complete else PaperAssessmentStatus.CANDIDATE,
            queries_tried=("baseline",) if complete else (),
        ),
    )
    return CoverageState(
        scope_paper_ids=("P1", "P2"),
        assessments=statuses,
        strategy=CoverageStrategy.EXHAUSTIVE,
        candidate_paper_ids=("P1", "P2"),
    )


def _record(evidence_id: str, requirement_ids: tuple[str, ...], paper_id: str = "P1") -> EvidenceRecord:
    entities = tuple(
        ProjectedEntity(
            "EMVE" if requirement_id == "R1" else "LUBE-ANN" if requirement_id == "R2" else "MVE approach",
            "method",
            ProjectedEntityRole.PROPOSED if requirement_id == "R1" else ProjectedEntityRole.BASELINE if requirement_id == "R2" else ProjectedEntityRole.PRIOR_WORK,
            EvidenceOwnership.THIS_PAPER if requirement_id == "R1" else EvidenceOwnership.PRIOR_WORK,
        )
        for requirement_id in requirement_ids
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id=paper_id,
            seed_chunk_ids=(evidence_id,), source_chunk_ids=(evidence_id,),
            content="Method evidence discusses " + ", ".join(entity.canonical_name for entity in entities), score=0.9,
        ),
        supporting_queries=("method",), first_seen_iteration=1, last_seen_iteration=1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requested_requirement_ids=requirement_ids,
        assessment=EvidenceAnalysis(
            evidence_id=evidence_id,
            status=EvidenceAnalysisStatus.QUALIFIED,
            projection=EnumerationProjection(
                evidence_id, entities, requirement_ids,
                tuple(
                    RequirementSupportAnchor(
                        requirement_id, entity.canonical_name, entity.canonical_name
                    )
                    for requirement_id, entity in zip(requirement_ids, entities)
                ),
            ),
        ),
    )


def _attempt(requirement_id: str, status: str = "ok") -> ResearchObservation:
    return ResearchObservation(
        iteration=1, action_name="search_papers", tool_status=ToolStatus.OK,
        added_evidence_ids=(), reused_evidence_ids=(),
        search_attempts=(PaperSearchAttempt("P1", "methods", requirement_ids=(requirement_id,), status=status),),
        action_arguments={"searches": [{"requirement_ids": [requirement_id]}]},
    )


def test_r2_collects_only_actual_projection_supporting_evidence() -> None:
    states = derive_requirement_states(
        plan=_plan(), evidence=(_record("E8", ("R2",)), _record("E10", ("R2",), "P2")),
        observations=(_attempt("R2"),), context=_context(), coverage=_coverage(complete=True),
    )
    r2 = next(item for item in states if item.requirement_id == "R2")
    assert r2.status.value == "satisfied"  # legacy name for support_found
    assert r2.evidence_ids == ("E8", "E10")


def test_scope_controls_terminal_state_not_projection_membership() -> None:
    states = derive_requirement_states(
        plan=_plan(), evidence=(_record("E8", ("R2",)),), observations=(_attempt("R2"),),
        context=_context(), coverage=_coverage(complete=False),
    )
    assert next(item for item in states if item.requirement_id == "R2").status.value == "satisfied"


def test_no_support_uncertain_and_pending_are_deterministic() -> None:
    complete = _coverage(complete=True)
    no_support = derive_requirement_states(plan=_plan(), evidence=(), observations=(_attempt("R2", "empty"),), context=_context(), coverage=complete)
    assert next(item for item in no_support if item.requirement_id == "R2").status.value == "in_progress"
    failed = replace(
        _record("Efailed", ("R3",)),
        assessment=EvidenceAnalysis("Efailed", EvidenceAnalysisStatus.FAILED, error_code="projection_failed"),
    )
    uncertain = derive_requirement_states(plan=_plan(), evidence=(failed,), observations=(_attempt("R3"),), context=_context(), coverage=complete)
    assert next(item for item in uncertain if item.requirement_id == "R3").status.value == "in_progress"
    assert next(item for item in no_support if item.requirement_id == "R1").status.value == "pending"


def test_finish_finalization_distinguishes_support_miss_and_uncertainty() -> None:
    failed = replace(
        _record("Efailed", ("R3",)),
        assessment=EvidenceAnalysis("Efailed", EvidenceAnalysisStatus.FAILED, error_code="projection_failed"),
    )
    observations = (_attempt("R1"), _attempt("R2"), _attempt("R3"))
    progress = derive_requirement_states(
        plan=_plan(), evidence=(_record("E1", ("R1",)), failed), observations=observations,
        context=_context(), coverage=_coverage(complete=True),
    )
    assert required_requirements_researched(_plan(), progress)
    finalized = finalize_requirement_states(
        plan=_plan(), evidence=(_record("E1", ("R1",)), failed), observations=observations,
    ).states
    assert {item.requirement_id: item.status.value for item in finalized} == {
        "R1": "satisfied", "R2": "no_support_found", "R3": "uncertain",
    }


def test_budget_stop_does_not_claim_no_support_research_was_completed() -> None:
    states = finalize_requirement_states(
        plan=_plan(), evidence=(), observations=(_attempt("R2"),),
        research_complete=False,
    ).states
    assert next(item for item in states if item.requirement_id == "R2").status.value == "uncertain"


def test_legacy_terminal_state_is_not_reopened_by_progress_derivation() -> None:
    from paperbase.agentic.contracts import RequirementState, RequirementStatus

    states = derive_requirement_states(
        plan=_plan(), evidence=(), observations=(), context=_context(), coverage=_coverage(complete=False),
        previous_states=(RequirementState("R2", RequirementStatus.NO_SUPPORT_FOUND),),
    )
    assert next(item for item in states if item.requirement_id == "R2").status.value == "no_support_found"


def test_in_progress_requirement_can_trigger_new_search_and_audit_has_reason() -> None:
    plan = _plan()
    coverage = _coverage(complete=True)
    tracking = track_requirement_states(
        plan=plan, evidence=(_record("E8", ("R2",)),), observations=(_attempt("R2"),), context=_context(), coverage=coverage,
    )
    assert not required_requirements_terminal(plan, tracking.states)  # R1/R3 remain pending.
    action = SearchPapersAction((PaperSearchSpec("P1", "Find baselines", requirement_ids=("R2",)),))
    guard = BudgetAndGuardTracker(AgenticResearchSettings()).validate_action(
        action, context=_context(), plan=plan, snapshot=None, requirement_states=tracking.states,
    )
    assert guard.code != "InvalidRequirementBinding"
    result = type("Result", (), {
        "status": "completed", "stop_reason": "agent_finished", "thread_id": None,
        "query_context": _context(), "plan": plan, "evidence": (_record("E8", ("R2",)),),
        "observations": (_attempt("R2"),), "coverage": coverage, "candidate_selection": None,
        "selected_evidence": (), "answer": None, "budget_usage": None,
        "requirement_states": tracking.states, "llm_diagnostics": (), "error_codes": (), "clarification": None,
    })()
    audit = build_agentic_audit_snapshot(result)
    assert next(item for item in audit.requirements if item.requirement_id == "R2").reason


def test_tool_autonomy_allows_overview_section_follow_up_and_multi_requirement_action() -> None:
    """Purpose binding permits orientation and localization, not only search."""
    context = QueryContext(
        original_query="method", resolved_query="method", intent="retrieve",
        scope=QueryScope.SINGLE_PAPER, target_paper_ids=("P1",),
        scope_source=ScopeSource.UI_SELECTION, retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    plan = _plan()
    tracker = BudgetAndGuardTracker(AgenticResearchSettings())
    pending = derive_requirement_states(plan=plan, evidence=(), observations=())
    overview = tracker.validate_action(
        GetPaperOverviewAction("P1", requirement_ids=("R2",)),
        context=context, plan=plan, snapshot=None, requirement_states=pending,
    )
    assert overview.accepted
    section = tracker.validate_action(
        GetSectionContextAction("P1", "method", requirement_ids=("R2",)),
        context=context, plan=plan, snapshot=None, requirement_states=pending,
        known_section_pairs=frozenset({("P1", "method")}),
    )
    assert section.accepted
    broad = tracker.validate_action(
        SearchPapersAction((PaperSearchSpec("P1", "methods and comparisons", requirement_ids=("R1", "R2")),)),
        context=context, plan=plan, snapshot=None, requirement_states=pending,
    )
    assert broad.accepted


def test_paper_coverage_marks_projection_support_as_positive() -> None:
    from paperbase.agentic.coverage import CoverageManager

    context = QueryContext(
        original_query="method", resolved_query="method", intent="retrieve",
        scope=QueryScope.SINGLE_PAPER, target_paper_ids=("P1",),
        scope_source=ScopeSource.UI_SELECTION, retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    coverage = CoverageManager().evaluate(
        context=context,
        corpus_snapshot=None,
        candidate_selection=None,
        evidence=(_record("E8", ("R2",)),),
        observations=(_attempt("R2"),),
    )
    assert coverage.assessments[0].status is PaperAssessmentStatus.POSITIVE
    assert coverage.assessments[0].evidence_ids == ("E8",)
