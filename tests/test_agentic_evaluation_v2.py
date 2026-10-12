from __future__ import annotations

from dataclasses import replace

from paperbase.evaluation.contracts import (
    AnswerTrace, CaseTrace, EvidenceTrace, RequirementTrace,
)
from paperbase.evaluation.diagnostic import evaluate_agentic_case
from paperbase.evaluation.run_agentic_e1 import safe_model_summary
from paperbase.evaluation.trace import trace_from_result, trace_from_artifacts
from paperbase.agentic.audit import build_agentic_audit_snapshot
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


def test_audited_alternative_path_counts_without_changing_exact_funnel() -> None:
    golden = {
        "id": "synthesis_001",
        "primary_type": "synthesis",
        "answerable": True,
        "relevant_evidence": [{"chunk_ids": ["canonical"]}],
        "evidence_alternatives": {"canonical": [["alternative_a", "alternative_b"]]},
    }
    evidence = EvidenceTrace(
        "E1", "P1", ("alternative_a", "alternative_b"),
        presented_to_answerer=True, cited_requirement_ids=("R1",),
        selected=True, cited=True,
    )
    trace = replace(
        _trace(), evidence=(evidence,),
        selected_source_chunk_ids=("alternative_a", "alternative_b"),
    )

    row = evaluate_agentic_case(golden, trace)

    assert row.retrieval.cumulative_gold_retrieval_recall == 0.0
    assert row.retrieval.cumulative_evidence_path_coverage == 1.0
    assert row.evidence_funnel.gold_cited_chunk_ids == ()
    assert row.evidence_funnel.path_covered_gold_cited_chunk_ids == ("canonical",)
    assert row.answer.gold_citation_coverage == 0.0
    assert row.answer.gold_citation_path_coverage == 1.0
    assert "retrieval_miss" not in row.failure_labels


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


def test_current_evidence_coverage_audit_roundtrips_to_offline_metrics_without_independent_prose():
    from paperbase.evaluation.run_agentic_eval import _result_payload
    result = replace(_result(), answer_input_evidence_ids=("E1",))
    audit = build_agentic_audit_snapshot(result).to_dict()
    restored = trace_from_artifacts(_result_payload(result), audit)
    assert restored.requirements[0].answer_present is True
    assert restored.requirements[0].citation_evidence_ids == ("E1",)
    assert restored.evidence[0].cited_requirement_ids == ("R1",)


def test_global_citation_is_not_lost_when_coverage_metadata_failed():
    from paperbase.agentic.contracts import RequirementCoverage
    result = _result()
    answer = replace(result.answer, requirement_coverage=(RequirementCoverage("R1", (), "Need coverage reassessment."),))
    trace = trace_from_result(replace(result, answer=answer))
    assert trace.evidence[0].cited is True
    assert trace.evidence[0].cited_requirement_ids == ()
    assert trace.requirements[0].answer_present is False


def test_manifest_model_summary_never_serializes_secrets() -> None:
    summary = safe_model_summary({
        "provider": "mock", "model": "test", "temperature": 0,
        "api_key": "SECRET", "base_url": "https://example.test/v1",
    })
    assert "SECRET" not in str(summary)
    assert summary["endpoint_host"] == "example.test"
