"""Phase 1 Requirement schema and compatibility contracts."""

from __future__ import annotations

import pytest

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceTarget,
    RequirementState,
    RequirementStatus,
    ResearchPlan,
    ResearchRequirement,
    ResearchRunResult,
    ResearchRunStatus,
    StopReason,
)


def test_research_plan_accepts_requirement_first_core_schema() -> None:
    requirement = ResearchRequirement(
        requirement_id="R1",
        description="识别论文中提出或采用的风速预测方法",
        required=True,
        evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )

    plan = ResearchPlan(goal="总结风速预测方法", requirements=(requirement,))

    assert not hasattr(plan, "subtasks")
    assert plan.requirements == (requirement,)
    assert requirement.required is True
    assert not hasattr(requirement, "priority")


def test_research_plan_rejects_duplicate_requirement_ids() -> None:
    requirement = ResearchRequirement(
        "R1", "方法", evidence_targets=frozenset({EvidenceTarget.METHOD}), required=True
    )
    with pytest.raises(ValueError, match="requirement IDs must be unique"):
        ResearchPlan(goal="q", requirements=(requirement, requirement))


def test_requirement_state_round_trips_through_checkpoint_dto() -> None:
    states = (
        RequirementState("R1", RequirementStatus.SATISFIED, ("E2",)),
        RequirementState("R2", RequirementStatus.NO_SUPPORT_FOUND),
        RequirementState("R3", RequirementStatus.UNCERTAIN),
    )

    dto = dump_contract(states, tuple[RequirementState, ...])

    assert dto == [
        {"requirement_id": "R1", "status": "satisfied", "evidence_ids": ["E2"]},
        {"requirement_id": "R2", "status": "no_support_found", "evidence_ids": []},
        {"requirement_id": "R3", "status": "uncertain", "evidence_ids": []},
    ]
    assert load_contract(dto, tuple[RequirementState, ...]) == states


def test_legacy_requirement_priority_payload_defaults_required_and_old_plan_loads() -> None:
    legacy_plan = {
        "goal": "legacy goal",
        "subtasks": [],
        "requirements": [
            {
                "requirement_id": "R1",
                "description": "旧要求",
                "priority": "optional",
                "evidence_targets": ["method"],
            }
        ],
    }

    restored = load_contract(legacy_plan, ResearchPlan)

    assert restored.requirements[0].required is False
    assert not hasattr(restored.requirements[0], "priority")
    # Old checkpoint data did not contain requirement_states; absence is valid.
    assert not hasattr(restored, "subtasks")


def test_legacy_research_result_without_requirement_states_still_loads() -> None:
    plan = ResearchPlan(
        goal="legacy result",
        requirements=(
            ResearchRequirement(
                "R1", "方法", required=True,
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
            ),
        ),
    )
    result = ResearchRunResult(
        status=ResearchRunStatus.COMPLETED,
        stop_reason=StopReason.AGENT_FINISHED,
        query_context=None,
        plan=plan,
        evidence=(),
        observations=(),
        traces=(),
        budget_usage=BudgetUsage(),
    )
    legacy_payload = dump_contract(result, ResearchRunResult)
    legacy_payload.pop("requirement_states")

    restored = load_contract(legacy_payload, ResearchRunResult)

    assert restored.requirement_states == ()
    assert restored.plan == plan
