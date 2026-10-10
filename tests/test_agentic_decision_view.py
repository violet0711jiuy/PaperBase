from __future__ import annotations

from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceRecord,
    EvidenceTarget,
    GroundedAnswerAttempt,
    GroundedCitation,
    Intent,
    QueryContext,
    QueryScope,
    RequirementAnswer,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
    ResearchRequirement,
    ScopeSource,
)
from paperbase.agentic.decision import DecisionState
from paperbase.agentic.decision_view import CompactDecisionLimits, build_compact_decision_view
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence, ToolStatus


def _context() -> QueryContext:
    return QueryContext(
        "Explain the method", "Explain the method", Intent.EXPLAIN,
        QueryScope.SINGLE_PAPER, ("P1",), ScopeSource.UI_SELECTION,
        ResearchGoal.FIND_EVIDENCE, frozenset({EvidenceTarget.METHOD}),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan("Explain", (
        ResearchRequirement("R1", "Explain MPIWcapt", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain C_i", frozenset({EvidenceTarget.METHOD})),
    ))


def record(number: int = 1, *, iteration: int = 1, eid: str | None = None) -> EvidenceRecord:
    source = RetrievedEvidence(
        EvidenceSourceType.CONTENT, "P1", (f"c{number}",), (f"c{number}",),
        f"FULL RAW BODY {number} SHOULD NOT BE COPIED INTO DECISION " + "x" * 1200,
        paper_title="Paper", section="Method", section_id="sec_method",
    )
    return EvidenceRecord(eid or f"E{number}", source, (f"query {number}",), iteration, iteration)


def _state(*, evidence=(), grounded_answer=None, observations=()) -> DecisionState:
    return DecisionState(
        context=_context(), plan=_plan(), evidence=tuple(evidence),
        observations=tuple(observations), budget_usage=BudgetUsage(),
        grounded_answer=grounded_answer, max_agent_iterations=8,
    )


def test_decision_view_is_atomic_and_does_not_repeat_full_evidence_bodies() -> None:
    view = build_compact_decision_view(_state(evidence=(record(),))).to_dict()
    serialized = str(view)
    assert "FULL RAW BODY" in serialized  # latest delta has one bounded preview
    assert "x" * 1200 not in serialized
    assert len(view["latest_tool_evidence"][0]["source_preview"]) <= 601
    assert view["evidence_catalog"][0]["evidence_id"] == "E1"
    assert "source_preview" not in view["evidence_catalog"][0]
    assert "supports" not in serialized and "support_found" not in serialized


def test_decision_sees_validated_answer_and_only_the_remaining_gap() -> None:
    attempt = GroundedAnswerAttempt((
        RequirementAnswer("R1", "MPIWcapt averages captured intervals.", (
            GroundedCitation("E1", ("S1",), "MPIWcapt evidence."),
        )),
        RequirementAnswer("R2", None, remaining_gap="Need the definition of C_i."),
    ), ("E1",))
    requirements = build_compact_decision_view(
        _state(evidence=(record(),), grounded_answer=attempt)
    ).to_dict()["requirements"]
    assert requirements[0]["accepted_answer"]["text"].startswith("MPIWcapt")
    assert requirements[1]["accepted_answer"]["remaining_gap"] == "Need the definition of C_i."


def test_compact_retry_limits_reduce_catalog_and_recent_history() -> None:
    observations = tuple(
        ResearchObservation(i, "search_papers", ToolStatus.OK, (), (),
                            action_arguments={"requirement_ids": ["R1"]})
        for i in range(1, 6)
    )
    state = _state(evidence=tuple(record(i, iteration=i) for i in range(1, 9)),
                   observations=observations)
    normal = build_compact_decision_view(state)
    compact = build_compact_decision_view(
        state, limits=CompactDecisionLimits(
            support_entries_per_requirement=1, recent_observations=1,
            recent_queries=1, recent_actions=1, sections=1,
            support_scope_chars=80, unresolved_aspects=1,
            unresolved_aspect_chars=40, entity_hints=1, research_leads=1,
        )
    )
    assert compact.metrics["decision_view_chars"] < normal.metrics["decision_view_chars"]
    assert compact.metrics["number_of_recent_research_events"] == 1
