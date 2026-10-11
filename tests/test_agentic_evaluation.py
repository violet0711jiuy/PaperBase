from paperbase.evaluation.agentic import (
    aggregate_action_control_metrics,
    aggregate_case_metrics,
    covered_gold_evidence_ids,
    evaluate_case_metrics,
    evidence_path_coverage,
    metrics_by_type,
    ranking_metrics,
)
from paperbase.evaluation.run_agentic_eval import _load_completed
import json

def test_ranking_metrics_use_exact_canonical_ids_only():
    assert ranking_metrics(["x", "g2", "g1"], ["g1", "g2"]) == {"hit_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 0.5}
    assert ranking_metrics(["x"], ["g1"])["reciprocal_rank"] == 0.0


def test_evidence_path_coverage_keeps_exact_recall_separate() -> None:
    golden = {
        "relevant_evidence": [{"chunk_ids": ["g1", "g2"]}],
        "evidence_alternatives": {"g2": [["a", "b"], ["c"]]},
    }

    assert ranking_metrics(["g1", "a", "b"], ["g1", "g2"])["recall_at_5"] == 0.5
    assert covered_gold_evidence_ids(golden, ["g1", "a", "b"]) == ("g1", "g2")
    assert evidence_path_coverage(golden, ["g1", "a", "b"]) == 1.0
    assert evidence_path_coverage(golden, ["g1", "a"]) == 0.5
    assert evidence_path_coverage(golden, ["g1", "c"]) == 1.0


def test_evidence_path_coverage_is_na_without_canonical_gold() -> None:
    assert evidence_path_coverage({"relevant_evidence": []}, ["a"]) is None

def test_case_metrics_exclude_rejected_actions_and_classify_states():
    golden = {"id":"a", "question":"q", "primary_type":"fact", "answerable":True, "relevant_evidence":[{"chunk_ids":["g"]}]}
    audit = {"budget":{"tool_calls":2,"retrieval_queries":2,"llm_calls":3}, "steps":[{"added_count":1}, {"added_count":0}], "rejected_actions":[{"rejection_stage":"budget"}]}
    row = evaluate_case_metrics(golden, ranked_chunk_ids=["g"], run_status="completed", stop_reason="agent_finished", audit=audit, selected_source_chunk_ids=["g"], answer={})
    assert row["useful_action_rate"] == 0.5 and row["gold_evidence_coverage"] == 1.0
    assert "local_budget_exhausted" in row["failure_labels"]

def test_na_and_unanswerable_are_not_averaged_as_zero():
    unanswerable = {"id":"u", "question":"q", "primary_type":"unanswerable", "answerable":False, "relevant_evidence":[]}
    row = evaluate_case_metrics(unanswerable, ranked_chunk_ids=[], run_status="no_evidence", stop_reason="agent_finished", audit={}, answer={"insufficient_evidence":True})
    metrics = aggregate_case_metrics([row])
    assert row["hit_at_5"] is None and row["abstention_correct"] is True
    assert metrics["retrieval"]["hit_at_5"] is None


def test_agentic_only_metrics_skip_standalone_retriever() -> None:
    golden = {"id": "b", "question": "q", "primary_type": "bibliography", "answerable": True,
              "relevant_evidence": [{"chunk_ids": ["g"]}]}
    row = evaluate_case_metrics(
        golden, ranked_chunk_ids=None, run_status="completed", stop_reason="agent_finished",
        audit={}, agent_retrieved_source_chunk_ids=["g", "x"],
        selected_source_chunk_ids=["g"], answer={"insufficient_evidence": False},
    )
    assert row["hit_at_5"] is None
    assert row["retrieved_chunk_ids"] is None
    assert row["agent_retrieved_gold_coverage"] == 1.0
    assert row["agent_retrieved_evidence_path_coverage"] == 1.0
    assert "retrieval_miss" not in row["failure_labels"]
    assert aggregate_case_metrics([row])["agent_retrieval"]["gold_evidence_coverage"] == 1.0
    assert aggregate_case_metrics([row])["agent_retrieval"]["evidence_path_coverage"] == 1.0


def test_answer_coverage_partial_is_not_counted_as_complete_success() -> None:
    golden = {"id": "partial", "question": "q", "primary_type": "method",
              "answerable": True, "relevant_evidence": [{"chunk_ids": ["g"]}]}
    row = evaluate_case_metrics(
        golden, ranked_chunk_ids=None, run_status="completed",
        stop_reason="agent_finished", audit={}, selected_source_chunk_ids=["g"],
        answer={"overall_status": "partial", "partial_answer": True,
                "requirement_coverage": [{"requirement_id": "R1", "status": "partial", "evidence_ids": ["E1"]}]},
    )
    assert row["completed_answerable"] is False
    assert row["evaluation_outcome"] == "partial"
    assert row["answer_requirement_coverage"][0]["status"] == "partial"


def test_resume_keeps_evaluation_modes_separate(tmp_path) -> None:
    path = tmp_path / "case_results.jsonl"
    path.write_text(
        json.dumps({"id": "old", "evaluation_mode": "combined"}) + "\n" +
        json.dumps({"id": "agent", "evaluation_mode": "agentic_only"}) + "\n",
        encoding="utf-8",
    )
    assert set(_load_completed(path, evaluation_mode="agentic_only")) == {"agent"}
    assert set(_load_completed(path, evaluation_mode="combined")) == {"old"}

def test_completion_false_no_support_breakdown_and_failure_taxonomy():
    golden = {"id":"f", "question":"q", "primary_type":"method", "answerable":True, "relevant_evidence":[{"chunk_ids":["g1", "g2"]}]}
    row = evaluate_case_metrics(golden, ranked_chunk_ids=["g1"], run_status="partial", stop_reason="invalid_action_limit", audit={"budget":{}, "steps":[]}, answer={"insufficient_evidence":True})
    assert row["hit_at_5"] == 1.0 and row["recall_at_5"] == 0.5
    assert row["completed_answerable"] is False and row["false_no_support"] is True
    assert {"invalid_action_stop", "false_no_support"}.issubset(row["failure_labels"])
    assert metrics_by_type([row])["method"]["n"] == 1


def test_action_control_metrics_use_only_public_audit_fields():
    row = {"id": "case-1", "stop_reason": "invalid_action_limit", "tool_calls": 0}
    audit = {
        "validation_errors": [
            {"stage": "domain_validation", "code": "DuplicateAction", "field_path": "section_id"}
        ],
        "steps": [
            {
                "iteration": 3,
                "action_name": "get_section_context",
                "action_arguments": {"paper_id": "A", "section_id": "new"},
                "action_canonicalization": {
                    "changed": True,
                    "operations": [{"operation": "deduplicated", "count": 2}],
                },
            }
        ],
        "rejected_actions": [
            {
                "iteration": 2,
                "action_name": "get_section_context",
                "action_arguments": {"paper_id": "A", "section_id": "old"},
                "rejection_code": "DuplicateAction",
            }
        ],
    }

    metrics = aggregate_action_control_metrics(((row, audit),))

    assert metrics["zero_tool_invalid_action"]["case_ids"] == ["case-1"]
    assert metrics["validation_error_breakdown"][0]["code"] == "DuplicateAction"
    assert metrics["canonicalization_summary"]["operation_counts"] == {"deduplicated": 2}
    assert metrics["guard_rejection_breakdown"] == {"DuplicateAction": 1}
    assert metrics["duplicate_recovery"]["guard_recoveries_with_materially_different_executed_action"] == 1
