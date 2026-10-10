"""Semantic authority and retrieval-hint regressions; no model calls."""

import json

from paperbase.agentic.answering import build_completion_view, generate_agentic_answer
from paperbase.generation.answer_generator import AnswerGenerationOutcome
from paperbase.generation.answer_coverage import AnswerRequirementCoverage
from paperbase.agentic.contracts import (
    BudgetUsage, EvidenceAnalysis, EvidenceAnalysisStatus, EvidenceRecord, EvidenceTarget,
    Intent, QueryContext, QueryScope, ResearchGoal, ScopeSource,
    MechanismProjection, RequirementState, RequirementStatus, RequirementSupportAnchor,
    ResearchPlan, ResearchRequirement,
)
from paperbase.agentic.grounding import support_rejection
from paperbase.agentic.decision import DecisionState, _state_prompt
from paperbase.agentic.requirement_coverage import critical_anchors, derive_requirement_coverage, source_has_anchor
from paperbase.agentic.requirement_state import derive_requirement_states, finalize_requirement_states
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


def _plan(description: str) -> ResearchPlan:
    return ResearchPlan("explain mechanism", (
        ResearchRequirement("R1", description, frozenset({EvidenceTarget.METHOD})),
    ))


def _record(evidence_id: str, source: str, summary: str) -> EvidenceRecord:
    anchor = RequirementSupportAnchor("R1", source, summary)
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT, paper_id="P", section="Method",
            seed_chunk_ids=(evidence_id,), source_chunk_ids=(evidence_id,),
            content=source,
        ),
        supporting_queries=("method",), first_seen_iteration=1, last_seen_iteration=1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requested_requirement_ids=("R1",),
        assessment=EvidenceAnalysis(
            evidence_id, EvidenceAnalysisStatus.QUALIFIED,
            projection=MechanismProjection(
                evidence_id, "mechanism", supported_requirement_ids=("R1",),
                proposed_requirement_ids=("R1",), support_anchors=(anchor,),
            ),
        ),
    )


def test_integrity_rejects_fabricated_excerpt() -> None:
    plan = _plan("解释模型的工作机制")
    record = _record("E1", "The model has a documented mechanism in the source text.", "documented mechanism")
    assert support_rejection(
        requirement=plan.requirements[0], record=record,
        anchor=RequirementSupportAnchor("R1", "A fabricated mechanism is present.", "fabricated"),
    ) == "source_anchor_not_found"


def test_projection_text_does_not_create_requirement_condition() -> None:
    plan = _plan("解释模型的工作机制")
    source = "The model has a documented mechanism in the source text."
    record = _record("E1", source, "The model has a documented mechanism and NaN in the source text")
    anchor = RequirementSupportAnchor("R1", source, "The model has a documented mechanism and NaN in the source text")
    assert support_rejection(
        requirement=plan.requirements[0], record=record, anchor=anchor,
    ) is None


def test_critical_identifiers_exclude_ordinary_prose() -> None:
    assert critical_anchors("LSTM-EMVE、MLE-QV、MPIWcapt、C_i 如何结合机制并输出") == (
        "LSTM-EMVE", "MLE-QV", "MPIWcapt", "C_i"
    )
    assert critical_anchors("解释模型的机制、结合方式和输出结果") == ()
    assert critical_anchors("A / B / C 共同作用") == ("A", "B", "C")
    assert critical_anchors("使用someVariable和snake_case") == ("someVariable", "snake_case")
    assert source_has_anchor("The LSTM EMVE model predicts wind speed.", "LSTM-EMVE")
    assert not source_has_anchor("The LSTMEMVE model predicts wind speed.", "LSTM-EMVE")


def test_lexical_coverage_is_a_hint_not_a_support_or_selection_gate() -> None:
    plan = _plan("TechA、TechB、TechC 如何结合")
    first = _record("E1", "TechA has a documented mechanism in the source text.",
                    "TechA TechB TechC have a documented mechanism in the source text")
    second = _record("E2", "TechB has a documented mechanism in the source text.",
                     "TechB has a documented mechanism in the source text")
    coverage = derive_requirement_coverage(plan.requirements[0], (first, second))
    assert coverage.covered_anchors == ("TechA", "TechB")
    assert coverage.missing_anchors == ("TechC",)
    assert coverage.supporting_evidence_ids == ("E1", "E2")
    progress = derive_requirement_states(plan=plan, evidence=(first, second), observations=())
    assert progress[0].status is RequirementStatus.SATISFIED
    final = finalize_requirement_states(plan=plan, evidence=(first, second), observations=()).states
    assert final[0].status is RequirementStatus.SATISFIED
    assert final[0].evidence_ids == ("E1", "E2")
    assert {record.evidence_id for record in FinalEvidenceSelector().select(
        plan=plan, evidence=(first, second), requirement_states=final
    )} == {"E1", "E2"}
    stale = (RequirementState("R1", RequirementStatus.SATISFIED, ("E1",)),)
    replay = finalize_requirement_states(
        plan=plan, evidence=(first, second), observations=(), previous_states=stale,
    ).states
    assert replay[0].status is RequirementStatus.SATISFIED


def test_decision_observes_missing_anchors_without_prescribing_action() -> None:
    plan = _plan("TechA、TechB 如何结合")
    first = _record("E1", "TechA has a documented mechanism in the source text.",
                    "TechA has a documented mechanism in the source text")
    context = QueryContext(
        original_query="mechanism", resolved_query="mechanism", intent=Intent.EXPLAIN,
        scope=QueryScope.SINGLE_PAPER, target_paper_ids=("P",),
        scope_source=ScopeSource.UI_SELECTION, retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    state = DecisionState(context, plan, (first,), (), BudgetUsage())
    payload = json.loads(_state_prompt(state, []))
    progress = payload["requirements"][0]["lexical_hints"]
    assert progress["surface_missing"] == ["TechB"]
    assert "covered" not in progress and "missing" not in progress


def test_multi_evidence_coverage_and_selector_keep_all_contributors() -> None:
    plan = _plan("TechA、TechB、TechC 如何结合")
    records = tuple(
        _record(f"E{index}", f"{anchor} has a documented mechanism in the source text.",
                f"{anchor} has a documented mechanism in the source text")
        for index, anchor in enumerate(("TechA", "TechB", "TechC"), 1)
    )
    coverage = derive_requirement_coverage(plan.requirements[0], records)
    assert coverage.complete and coverage.missing_anchors == ()
    final = finalize_requirement_states(plan=plan, evidence=records, observations=()).states
    assert final[0].status is RequirementStatus.SATISFIED
    selected = FinalEvidenceSelector().select(plan=plan, evidence=records, requirement_states=final)
    assert {item.evidence_id for item in selected} == {"E1", "E2", "E3"}
    view = build_completion_view(plan=plan, states=final, selected=selected)
    class Answerer:
        def __init__(self, citations: tuple[str, ...], coverage_status: str) -> None:
            self.citations = citations
            self.coverage_status = coverage_status

        def generate(self, **kwargs: object) -> AnswerGenerationOutcome:
            return AnswerGenerationOutcome(
                status="success", answer="A mechanism is described.",
                citations=self.citations, insufficient_evidence=False,
                requirement_coverage=(AnswerRequirementCoverage(
                    "R1", self.coverage_status, self.citations
                ),),
            )

    partial = generate_agentic_answer(
        generator=Answerer(("E1", "E2"), "partial"), query="mechanism", evidence=selected,
        completion_view=view, plan=plan,
    )
    assert partial.partial_answer and partial.overall_status == "partial"
    complete = generate_agentic_answer(
        generator=Answerer(("E1", "E2", "E3"), "complete"), query="mechanism", evidence=selected,
        completion_view=view, plan=plan,
    )
    assert complete.overall_status == "complete" and not complete.insufficient_evidence


def test_no_anchor_requirement_keeps_existing_completion() -> None:
    plan = _plan("解释模型的工作机制")
    record = _record("E1", "The model has a documented mechanism in the source text.",
                     "The model has a documented mechanism in the source text")
    coverage = derive_requirement_coverage(plan.requirements[0], (record,))
    assert coverage.critical_anchors == () and coverage.complete
    final = finalize_requirement_states(plan=plan, evidence=(record,), observations=()).states
    assert final[0].status is RequirementStatus.SATISFIED
