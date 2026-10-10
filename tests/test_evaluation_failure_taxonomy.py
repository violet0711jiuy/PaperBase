"""Failure attribution from saved IDs, states, and safe audit fields only."""

import sqlite3

from paperbase.evaluation.agentic import evaluate_case_metrics
from paperbase.evaluation.reclassify_agentic_eval import _recover_section_context_provenance


def case(*gold: str, answerable: bool = True) -> dict[str, object]:
    return {
        "id": "case", "question": "question", "primary_type": "bibliography",
        "answerable": answerable,
        "relevant_evidence": [{"chunk_ids": list(gold)}] if gold else [],
    }


def codes(row: dict[str, object]) -> set[str]:
    return {item["code"] for item in row["failure_diagnostics"]}


def test_sufficient_evidence_answer_domain_failure_is_not_false_no_support() -> None:
    audit = {
        "requirements": [
            {"requirement_id": "R1", "required": True, "status": "satisfied"},
            {"requirement_id": "R2", "required": True, "status": "satisfied"},
        ],
        "evidence": [{
            "evidence_id": "E1", "analysis_status": "qualified", "selected": True,
            "verified_supports_requirement_ids": ["R1", "R2"],
        }],
        "llm_calls": [{"node": "generate_answer", "error_stage": "domain_validation"}],
        "validation_errors": [{
            "node": "generate_answer", "stage": "domain_validation",
            "code": "ANSWER_UNKNOWN_CITATION", "field_path": "answer",
        }],
    }
    row = evaluate_case_metrics(
        case("g"), ranked_chunk_ids=None, run_status="completed",
        stop_reason="agent_finished", audit=audit,
        agent_retrieved_source_chunk_ids=["g"],
        evidence_source_chunk_ids={"E1": ["g"]},
        selected_source_chunk_ids=["g"],
        answer={"status": "fallback", "insufficient_evidence": True},
    )
    assert {"ANSWER_CITATION_INVALID", "ANSWER_GENERATION_FAILED"} <= codes(row)
    assert row["legacy_false_no_support"] is True
    assert row["false_no_support"] is False
    assert row["evaluation_outcome"] == "failure"
    assert "false_no_support" not in row["failure_labels"]


def test_gold_retrieval_miss_is_distinct_from_projection_failure() -> None:
    row = evaluate_case_metrics(
        case("g1", "g2"), ranked_chunk_ids=None, run_status="partial",
        stop_reason="budget_exhausted", audit={},
        agent_retrieved_source_chunk_ids=["g1"],
        answer={"insufficient_evidence": True},
    )
    assert {"RETRIEVAL_MISS", "BUDGET_EXHAUSTED"} <= codes(row)
    assert row["false_no_support"] is True


def test_gold_miss_does_not_relabel_sufficient_selected_support_answer_failure() -> None:
    audit = {
        "requirements": [{"requirement_id": "R1", "required": True,
                          "status": "satisfied"}],
        "evidence": [{"evidence_id": "E1", "selected": True,
                      "verified_supports_requirement_ids": ["R1"]}],
        "llm_calls": [{"node": "generate_answer", "error_stage": "domain_validation"}],
    }
    row = evaluate_case_metrics(
        case("g1", "g2"), ranked_chunk_ids=None, run_status="completed",
        stop_reason="agent_finished", audit=audit,
        agent_retrieved_source_chunk_ids=["g1"],
        selected_source_chunk_ids=["g1"],
        answer={"status": "fallback", "insufficient_evidence": True},
    )
    assert "RETRIEVAL_MISS" in codes(row)
    assert row["false_no_support"] is False


def test_gold_projection_failure_no_support_and_integrity_rejection() -> None:
    audit = {"evidence": [
        {"evidence_id": "E1", "analysis_status": "failed"},
        {"evidence_id": "E2", "analysis_status": "irrelevant"},
        {"evidence_id": "E3", "analysis_status": "qualified",
         "projection_rejections": ["support_rejected:R1:source_anchor_not_found"]},
    ]}
    row = evaluate_case_metrics(
        case("g1", "g2", "g3"), ranked_chunk_ids=None, run_status="completed",
        stop_reason="agent_finished", audit=audit,
        agent_retrieved_source_chunk_ids=["g1", "g2", "g3"],
        evidence_source_chunk_ids={"E1": ["g1"], "E2": ["g2"], "E3": ["g3"]},
        answer={"insufficient_evidence": True},
    )
    assert {"PROJECTOR_FAILED", "PROJECTOR_NO_SUPPORT",
            "INTEGRITY_INVALID"} <= codes(row)
    assert row["false_no_support"] is True
    assert all(item["stage"] in {
        "retrieval", "projection", "integrity", "selection", "control",
        "answer_generation", "answer_validation",
    } for item in row["failure_diagnostics"])
    assert next(item for item in row["failure_diagnostics"]
                if item["code"] == "INTEGRITY_INVALID")["requirement_ids"] == ["R1"]


def test_verified_support_not_selected_and_invalid_action_stop() -> None:
    audit = {
        "requirements": [{"requirement_id": "R1", "required": True, "status": "in_progress"}],
        "evidence": [{"evidence_id": "E1", "selected": False,
                      "verified_supports_requirement_ids": ["R1"]}],
    }
    row = evaluate_case_metrics(
        case("g"), ranked_chunk_ids=None, run_status="partial",
        stop_reason="invalid_action_limit", audit=audit,
        agent_retrieved_source_chunk_ids=["g"],
        answer={"insufficient_evidence": True},
    )
    assert {"VERIFIED_SUPPORT_NOT_SELECTED", "INVALID_ACTION_STOP"} <= codes(row)


def test_correct_unanswerable_abstention_has_no_false_no_support() -> None:
    row = evaluate_case_metrics(
        case(answerable=False), ranked_chunk_ids=None, run_status="no_evidence",
        stop_reason="agent_finished", audit={},
        answer={"status": "insufficient_evidence", "insufficient_evidence": True},
    )
    assert row["evaluation_outcome"] == "correct_abstention"
    assert row["false_no_support"] is None
    assert row["legacy_false_no_support"] is None


def test_old_section_provenance_replay_requires_saved_chunk_match(tmp_path) -> None:
    database = tmp_path / "paperbase.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE chunks (chunk_id TEXT, paper_id TEXT, section_id TEXT, "
            "chunk_index INTEGER, page_start INTEGER, page_end INTEGER)"
        )
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            [("g1", "p", "s", 1, 12, 12), ("g2", "p", "s", 2, 12, 13)],
        )
    audit = {
        "steps": [
            {"action_name": "search_papers", "added_count": 1},
            {"action_name": "get_section_context", "added_count": 1,
             "action_arguments": {"paper_id": "p", "section_id": "s"}},
        ],
        "evidence": [
            {"evidence_id": "E1", "paper_id": "p", "page_start": 1, "page_end": 1},
            {"evidence_id": "E2", "paper_id": "p", "page_start": 12, "page_end": 13},
        ],
    }
    assert _recover_section_context_provenance(
        database=database,
        result={"agent_retrieved_source_chunk_ids": ["g1", "g2", "other"]},
        audit=audit,
    ) == {"E2": ["g1", "g2"]}
    assert _recover_section_context_provenance(
        database=database,
        result={"agent_retrieved_source_chunk_ids": ["g1", "other"]},
        audit=audit,
    ) == {}
