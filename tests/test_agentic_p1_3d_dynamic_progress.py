"""P1.3d zero-LLM replay against the saved P1.3c Decision State."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from paperbase.agentic.decision import (
    DecisionState,
    LLMDecisionAgent,
    _decision_json_schema,
    _runtime_action_capabilities,
)
from paperbase.agentic.decision_view import (
    _has_targeted_support_progress,
    build_compact_decision_view,
    build_recent_research_events,
    compute_no_progress_signal,
)
from test_agentic_p1_3c_structured_decision import real_client


ROOT = Path(__file__).resolve().parents[1]
P1_3C_INPUTS = (
    ROOT
    / "storage"
    / "evaluation"
    / "p1_3c_strict_decision_qwen38_20261002"
    / "decision_inputs.jsonl"
)


def _saved_state(iteration: int) -> DecisionState:
    rows = [
        json.loads(line)
        for line in P1_3C_INPUTS.read_text(encoding="utf-8").splitlines()
    ]
    payload = next(row for row in rows if row["iteration"] == iteration)
    return TypeAdapter(DecisionState).validate_python(payload["full_decision_state"])


def _search_decision(
    paper_id: str, query: str, *, requirement_id: str = "R2"
) -> dict[str, object]:
    return {
        "decision": {
            "action": "search_papers",
            "searches": [
                {
                    "paper_id": paper_id,
                    "query": query,
                    "lexical_terms": ["materially different"],
                    "requirement_ids": [requirement_id],
                }
            ],
        }
    }


def test_saved_iteration_4_raw_queries_are_not_a_schema_dedup_domain():
    state = _saved_state(4)
    before = deepcopy(state)
    runtime_before = _runtime_action_capabilities(state)
    schema = _decision_json_schema(state)
    assert runtime_before.unavailable_paper_queries == ()
    assert "allOf" not in schema["$defs"]["SearchSpecDraft"]
    paper_id = runtime_before.search_paper_ids[0]
    previous_query = state.observations[0].action_arguments["searches"][0]["query"]
    # Same raw query, materially changed lexical focus: Schema permits structure.
    legal = _search_decision(paper_id, previous_query)
    assert not list(Draft202012Validator(schema).iter_errors(legal))
    assert _runtime_action_capabilities(state) == runtime_before
    assert state == before


def test_saved_iteration_4_targeted_progress_is_requirement_directed():
    state = _saved_state(4)
    events = build_recent_research_events(state)
    iteration_3 = next(event for event in events if event["iteration"] == 3)

    assert iteration_3["requirement_ids"] == ("R2",)
    assert iteration_3["newly_supported_requirement_ids"] == ("R1",)
    assert not _has_targeted_support_progress(iteration_3)

    signal = compute_no_progress_signal(events)
    assert signal["consecutive_no_targeted_support_actions"] == 2
    assert signal["last_targeted_support_iteration"] == 1
    assert {
        name: signal[name]
        for name in (
            "consecutive_no_support_actions",
            "consecutive_no_new_evidence_actions",
            "recent_section_reads_without_support",
            "recent_duplicate_rejections",
            "last_new_support_iteration",
            "last_new_evidence_iteration",
        )
    } == {
        "consecutive_no_support_actions": 0,
        "consecutive_no_new_evidence_actions": 0,
        "recent_section_reads_without_support": 0,
        "recent_duplicate_rejections": 0,
        "last_new_support_iteration": 3,
        "last_new_evidence_iteration": 3,
    }

    view = build_compact_decision_view(state).payload
    assert "no_progress_signal" not in view
    assert all("request" in event and "outcome" in event
               for event in view["recent_research_events"])


def test_saved_iteration_4_wire_schema_contains_structural_domain_only():
    state = _saved_state(4)
    paper_id = state.context.target_paper_ids[0]
    legal = _search_decision(
        paper_id,
        "Which training objective combines point accuracy and interval calibration?",
    )
    client, calls = real_client([json.dumps(legal)])

    LLMDecisionAgent(client).decide(state)

    assert len(calls) == 1
    wire_schema = calls[0]["response_format"]["json_schema"]["schema"]
    runtime = _runtime_action_capabilities(state)
    assert "allOf" not in wire_schema["$defs"]["SearchSpecDraft"]
    assert wire_schema["$defs"]["SearchSpecDraft"]["properties"]["paper_id"]["enum"] == list(runtime.search_paper_ids)
    assert set(wire_schema["properties"]["decision"]["discriminator"]["mapping"]) == set(runtime.available_tools)


def test_schema_and_view_builders_do_not_call_llm(monkeypatch):
    import paperbase.agentic.decision as decision

    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic builder called an LLM")

    monkeypatch.setattr(decision, "complete_json_with_diagnostics", forbidden)
    state = _saved_state(4)
    _decision_json_schema(state)
    build_compact_decision_view(state)
