from __future__ import annotations

from dataclasses import replace

from paperbase.evaluation.contracts import (
    AnswerTrace, CaseTrace, EvidenceTrace, RequirementTrace,
)
from paperbase.evaluation.diagnostic import evaluate_agentic_case
from paperbase.evaluation.run_agentic_e1 import safe_model_summary
from paperbase.evaluation.trace import trace_from_result
from test_agentic_audit import _result


def _gold(answerable=True):
    return {
        "id": "method_002", "primary_type": "method", "answerable": answerable,
        "relevant_evidence": ([{"chunk_ids": ["chunk_gold"]}] if answerable else []),
    }


def _trace(*, input_seen=True, cited=True, insufficient=False):
    evidence = EvidenceTrace(
        "E1", "P1", ("chunk_gold",), presented_to_answerer=input_seen,
        cited_requirement_ids=(("R1",) if cited else ()), selected=cited, cited=cited,
    )
    requirement = RequirementTrace(
        "R1", "Explain", True,
        "satisfied" if cited else "in_progress", cited,
        ("E1",) if cited else (), None if cited else "Need a grounded answer.",
    )
    answer = AnswerTrace(True, insufficient, not cited, ("E1",) if cited else (), "success")
    return CaseTrace(
        "completed", "agent_finished", (evidence,), (), (requirement,),
        ("E1",) if cited else (), ("chunk_gold",) if cited else (), answer,
        counters=(("tool_calls", 0), ("retrieval_queries", 0),
                  ("agent_iterations", 1), ("llm_calls", 1)),
        guard_rejection_codes=(), validation_error_count=0,
        llm_nodes=(("decide_action", True),), decision_calls=1,
    )


def test_grounded_funnel_is_retrieved_registered_answer_input_cited() -> None:
    row = evaluate_agentic_case(_gold(), _trace())
    funnel = row.evidence_funnel
    assert funnel.gold_registered_chunk_ids == ("chunk_gold",)
    assert funnel.gold_answer_input_chunk_ids == ("chunk_gold",)
    assert funnel.gold_cited_chunk_ids == ("chunk_gold",)
    assert row.primary_failure_stage is None


def test_gold_in_answer_input_but_not_cited_is_grounding_failure() -> None:
    row = evaluate_agentic_case(_gold(), _trace(cited=False))
    assert row.primary_failure_stage == "grounding"
    assert "gold_answer_input_not_cited" in row.failure_labels
    assert "required_requirement_gap" in row.failure_labels


def test_registered_gold_hidden_from_answerer_is_distinct_failure() -> None:
    row = evaluate_agentic_case(_gold(), _trace(input_seen=False, cited=False))
    assert row.primary_failure_stage == "answer_input"
    assert "gold_registered_not_shown_to_answerer" in row.failure_labels


def test_current_public_result_trace_uses_citation_only_selection() -> None:
    result = replace(_result(), answer_input_evidence_ids=("E1",))
    trace = trace_from_result(result)
    assert trace.evidence[0].presented_to_answerer is True
    assert trace.evidence[0].cited_requirement_ids == ("R1",)
    assert trace.selected_evidence_ids == ("E1",)


def test_manifest_model_summary_never_serializes_secrets() -> None:
    summary = safe_model_summary({
        "provider": "mock", "model": "test", "temperature": 0,
        "api_key": "SECRET", "base_url": "https://example.test/v1",
    })
    assert "SECRET" not in str(summary)
    assert summary["endpoint_host"] == "example.test"
