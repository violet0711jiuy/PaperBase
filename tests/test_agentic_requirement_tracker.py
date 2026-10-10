from __future__ import annotations

from paperbase.agentic.contracts import (
    EvidenceTarget,
    RequirementState,
    RequirementStatus,
    ResearchObservation,
    ResearchPlan,
    ResearchRequirement,
)
from paperbase.agentic.requirement_state import finalize_requirement_states, track_requirement_states
from paperbase.retrieval.contracts import ToolStatus


def _plan() -> ResearchPlan:
    return ResearchPlan("Explain", (
        ResearchRequirement("R1", "Explain A", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain B", frozenset({EvidenceTarget.METHOD})),
    ))


def test_retrieval_targeting_marks_in_progress_but_never_satisfied() -> None:
    observation = ResearchObservation(
        1, "search_papers", ToolStatus.OK, ("E1",), (),
        action_arguments={"requirement_ids": ["R1"]},
    )
    states = track_requirement_states(plan=_plan(), observations=(observation,)).states
    assert [state.status for state in states] == [
        RequirementStatus.IN_PROGRESS, RequirementStatus.PENDING,
    ]


def test_only_validated_terminal_answer_state_is_preserved() -> None:
    previous = (RequirementState("R1", RequirementStatus.SATISFIED, ("E1",)),)
    states = track_requirement_states(
        plan=_plan(), observations=(), previous_states=previous,
    ).states
    assert states[0] == previous[0]
    assert states[1].status is RequirementStatus.PENDING


def test_research_end_without_grounded_answer_is_not_success() -> None:
    observation = ResearchObservation(
        1, "search_papers", ToolStatus.OK, ("E1",), (),
        action_arguments={"requirement_ids": ["R1"]},
    )
    states = finalize_requirement_states(
        plan=_plan(), evidence=(), observations=(observation,), research_complete=True,
    ).states
    assert states[0].status is RequirementStatus.NO_SUPPORT_FOUND
    assert states[1].status is RequirementStatus.PENDING
