"""Answer validation diagnostics preserve rules, never model output."""

import json
from types import SimpleNamespace

import pytest

from paperbase.agentic.audit import build_agentic_audit_snapshot
from paperbase.config import AnswerGenerationSettings
from paperbase.generation.answer_generator import (
    AnswerGenerationResult, GroundedAnswerGenerator,
    _normalize_and_validate_answer, AnswerDomainValidationError,
)
from paperbase.generation.section_expander import EvidenceUnit


class Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = json.dumps(payload, ensure_ascii=False)

    def complete_json(self, **_: object) -> str:
        return self.payload


def evidence() -> tuple[EvidenceUnit, ...]:
    return (EvidenceUnit(
        evidence_id="E1", kind="content", paper_id="p", paper_title="p",
        section="s", page_start=1, page_end=1, seed_chunk_ids=("c",),
        chunk_ids=("c",), text="A source fact.", token_count=10,
    ),)


def draft(**overrides: object) -> dict[str, object]:
    value = {
        "direct_answer": "A source fact. [E1]", "evidence_explanation": "Evidence. [E1]",
        "reading_interpretation": None, "citations": ["E1"],
        "insufficient_evidence": False,
    }
    value.update(overrides)
    return value


def outcome_for(value: dict[str, object]):
    return GroundedAnswerGenerator(
        settings=AnswerGenerationSettings(), client=Client(value),
    ).generate(query="Question?", evidence=evidence())


@pytest.mark.parametrize(("value", "code", "field"), (
    (draft(direct_answer="Unknown source. [E9]"), "ANSWER_UNKNOWN_CITATION", "answer"),
    (draft(direct_answer="Uncited answer", evidence_explanation="Uncited explanation",
           partial_answer=True), "ANSWER_PARTIAL_WITHOUT_CITATION", "answer"),
    (draft(direct_answer="Uncited answer", evidence_explanation="Uncited explanation"),
     "ANSWER_COMPLETE_WITHOUT_CITATION", "answer"),
    (draft(direct_answer="   "), "ANSWER_REQUIRED_CONTENT_BLANK", "direct_answer"),
    (draft(evidence_explanation="   "), "ANSWER_REQUIRED_CONTENT_BLANK", "evidence_explanation"),
))
def test_each_existing_answer_domain_rule_has_safe_diagnostic(
    value: dict[str, object], code: str, field: str,
) -> None:
    outcome = outcome_for(value)
    assert outcome.status == "fallback"
    call = outcome.llm_diagnostics[0]
    assert call.error_stage == "domain_validation"
    assert call.recovery_action == "fallback"
    diagnostic = call.validation_diagnostics[0]
    assert diagnostic.error_code == code
    assert diagnostic.field_path == field
    assert diagnostic.validator_name == "answer_domain_validator"
    assert diagnostic.safe_reason and diagnostic.expected_constraint
    assert diagnostic.recovery_action == "fallback"


def test_answer_schema_failure_is_distinct_and_visible_in_audit() -> None:
    value = draft()
    del value["reading_interpretation"]
    outcome = outcome_for(value)
    diagnostic = outcome.llm_diagnostics[0].validation_diagnostics[0]
    assert outcome.status == "fallback"
    assert diagnostic.error_stage == "schema_validation"
    assert diagnostic.error_code == "ANSWER_SCHEMA_REQUIRED_FIELD"
    assert diagnostic.field_path == "reading_interpretation"
    snapshot = build_agentic_audit_snapshot(SimpleNamespace(
        status="completed", stop_reason="agent_finished", thread_id="answer-test",
        query_context=None, candidate_selection=None, coverage=None, plan=None,
        evidence=(), selected_evidence=(), observations=(), budget_usage=None,
        answer=outcome, llm_diagnostics=outcome.llm_diagnostics,
        error_codes=(), clarification=None,
    ))
    assert snapshot.validation_errors[0].code == "ANSWER_SCHEMA_REQUIRED_FIELD"
    assert snapshot.validation_errors[0].stage == "schema_validation"
    assert snapshot.validation_errors[0].recovery_action == "fallback"


def test_answer_domain_rule_code_is_visible_in_audit() -> None:
    outcome = outcome_for(draft(direct_answer="Unlisted evidence. [E9]"))
    snapshot = build_agentic_audit_snapshot(SimpleNamespace(
        status="completed", stop_reason="agent_finished", thread_id="answer-domain-test",
        query_context=None, candidate_selection=None, coverage=None, plan=None,
        evidence=(), selected_evidence=(), observations=(), budget_usage=None,
        answer=outcome, llm_diagnostics=outcome.llm_diagnostics,
        error_codes=(), clarification=None,
    ))
    item = snapshot.validation_errors[0]
    assert item.code == item.rule_code == "ANSWER_UNKNOWN_CITATION"
    assert item.field_path == "answer"


def test_empty_composed_answer_rule_is_identifiable() -> None:
    with pytest.raises(AnswerDomainValidationError) as caught:
        _normalize_and_validate_answer(
            AnswerGenerationResult(answer=" ", insufficient_evidence=False),
            allowed_evidence_ids={"E1"},
        )
    assert caught.value.code == "ANSWER_BLANK"
