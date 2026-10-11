"""Requirement ID Runtime domain projection for the Decision wire schema."""

from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from paperbase.agentic.contracts import (
    RequirementState,
    RequirementStatus,
    SearchReferencesAction,
)
from paperbase.agentic.decision import (
    ActionDomainValidationError,
    _decision_json_schema,
    _runtime_action_capabilities,
    _validate_action_references,
)
from paperbase.agentic.decision_view import build_compact_decision_view
from test_agentic_p0_action_control import _state


BRANCHES = (
    "SearchSpecDraft",
    "SearchKnowledgeBaseDraft",
    "SearchReferencesDraft",
    "GetSectionContextDraft",
    "GetPaperOverviewDraft",
)


def _reference_decision(requirement_ids: list[str]) -> dict[str, object]:
    return {
        "decision": {
            "action": "search_references",
            "paper_id": "A",
            "query": "2009 wind forecasting review",
            "lexical_terms": ["2009", "wind forecasting"],
            "requirement_ids": requirement_ids,
        }
    }


def _paper_decision(requirement_ids: list[str]) -> dict[str, object]:
    return {
        "decision": {
            "action": "search_papers",
            "searches": [
                {
                    "paper_id": "A",
                    "query": "method",
                    "lexical_terms": ["method"],
                    "requirement_ids": requirement_ids,
                }
            ],
        }
    }


def test_all_requirement_bound_branches_use_runtime_requirement_enum():
    state = _state()
    schema = _decision_json_schema(state)

    assert _runtime_action_capabilities(state).valid_requirement_ids == ("R1", "R2")
    for branch_name in BRANCHES:
        requirement_ids = schema["$defs"][branch_name]["properties"]["requirement_ids"]
        assert requirement_ids["type"] == "array"
        assert requirement_ids["minItems"] == 1
        assert requirement_ids["maxItems"] == 1
        assert requirement_ids["items"]["enum"] == ["R1", "R2"]


@pytest.mark.parametrize("invalid_id", ["]R1", "]", "R3"])
def test_wire_schema_rejects_ids_outside_runtime_domain(invalid_id):
    schema = _decision_json_schema(_state())
    validator = Draft202012Validator(schema)

    assert list(validator.iter_errors(_reference_decision([invalid_id])))
    assert list(validator.iter_errors(_paper_decision([invalid_id])))


@pytest.mark.parametrize("requirement_ids", [["R1"], ["R2"]])
def test_wire_schema_accepts_each_legal_requirement_binding(requirement_ids):
    schema = _decision_json_schema(_state())
    validator = Draft202012Validator(schema)

    assert not list(validator.iter_errors(_reference_decision(requirement_ids)))
    assert not list(validator.iter_errors(_paper_decision(requirement_ids)))


def test_wire_schema_rejects_multi_requirement_action_binding():
    schema = _decision_json_schema(_state())
    validator = Draft202012Validator(schema)

    assert list(validator.iter_errors(_reference_decision(["R1", "R2"])))
    assert list(validator.iter_errors(_paper_decision(["R1", "R2"])))


def test_empty_runtime_requirement_domain_exposes_stop_only():
    state = replace(
        _state(),
        requirement_states=(
            RequirementState("R1", RequirementStatus.NO_SUPPORT_FOUND),
            RequirementState("R2", RequirementStatus.UNCERTAIN),
        ),
    )
    schema = _decision_json_schema(state)
    decision = schema["properties"]["decision"]

    assert _runtime_action_capabilities(state).valid_requirement_ids == ()
    assert decision["oneOf"] == [{"$ref": "#/$defs/StopResearchDraft"}]
    assert decision["discriminator"]["mapping"] == {
        "stop_research": "#/$defs/StopResearchDraft",
    }
    validator = Draft202012Validator(schema)
    assert not list(
        validator.iter_errors(
            {"decision": {"action": "stop_research", "reason": "No active requirements."}}
        )
    )
    assert list(validator.iter_errors(_reference_decision(["R1"])))


def test_local_requirement_validator_remains_defense_in_depth():
    state = _state()
    valid = SearchReferencesAction("A", "review", requirement_ids=("R1", "R2"))
    invalid = SearchReferencesAction("A", "review", requirement_ids=("]R1",))

    _validate_action_references(valid, state)
    with pytest.raises(ActionDomainValidationError) as raised:
        _validate_action_references(invalid, state)
    assert raised.value.code == "InvalidRequirementId"
    assert raised.value.actual_summary == "unknown_id=]R1"


def test_schema_and_view_builders_are_pure_and_do_not_call_llm(monkeypatch):
    import paperbase.agentic.decision as decision

    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic builder called an LLM")

    monkeypatch.setattr(decision, "complete_json_with_diagnostics", forbidden)
    state = _state()
    before = deepcopy(state)
    runtime_before = _runtime_action_capabilities(state)

    _decision_json_schema(state)
    build_compact_decision_view(state)

    assert state == before
    assert _runtime_action_capabilities(state) == runtime_before
