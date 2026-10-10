from __future__ import annotations

import pytest

from paperbase.agentic.checkpoint_serde import (
    AGENTIC_CHECKPOINT_NAMESPACE,
    checkpoint_thread_id,
    dump_contract,
    load_contract,
)
from paperbase.agentic.contracts import (
    AgentAction,
    EvidenceTarget,
    GroundedAnswerAttempt,
    GroundedCitation,
    RequirementAnswer,
    ResearchPlan,
    ResearchRequirement,
    StopResearchAction,
    TryAnswerAction,
)
from paperbase.agentic.graph import _require_current_semantic_authority


def test_current_grounded_answer_contract_round_trips() -> None:
    attempt = GroundedAnswerAttempt((
        RequirementAnswer(
            "R1",
            "The method uses a capture indicator.",
            (GroundedCitation("E1", ("S1",), "C_i is a capture indicator."),),
        ),
    ), ("E1",))
    payload = dump_contract(attempt, GroundedAnswerAttempt)
    assert load_contract(payload, GroundedAnswerAttempt) == attempt


def test_current_plan_and_terminal_actions_round_trip() -> None:
    plan = ResearchPlan("Explain", (
        ResearchRequirement("R1", "Explain the method", frozenset({EvidenceTarget.METHOD})),
    ))
    assert load_contract(dump_contract(plan, ResearchPlan), ResearchPlan) == plan
    for action in (TryAnswerAction("evidence available"), StopResearchAction("no useful action")):
        assert load_contract(dump_contract(action, AgentAction), AgentAction) == action


def test_checkpoint_namespace_isolated_from_projector_architecture() -> None:
    assert checkpoint_thread_id("abc") == f"{AGENTIC_CHECKPOINT_NAMESPACE}:abc"
    assert "grounded-answer" in AGENTIC_CHECKPOINT_NAMESPACE
    _require_current_semantic_authority({"semantic_authority_version": "raw-evidence-grounded-answer-v1"})
    with pytest.raises(ValueError, match="legacy Projector checkpoints"):
        _require_current_semantic_authority({"semantic_authority_version": "projector-v4"})
