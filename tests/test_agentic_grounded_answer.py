from __future__ import annotations

import json
from copy import deepcopy

import pytest
from paperbase.agentic.contracts import EvidenceRecord, EvidenceTarget, ResearchPlan, ResearchRequirement
from paperbase.agentic.grounded_answer import LLMGroundedAnswerer, GroundedAnswerAttemptsExhausted, render_grounded_answer
from paperbase.llm.client import LLMCompletion
from paperbase.prompts.grounded_answer import GROUNDED_ANSWER_SYSTEM_PROMPT
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


class QueueClient:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMCompletion(json.dumps(response, ensure_ascii=False))


def plan():
    return ResearchPlan("Explain the method", (
        ResearchRequirement("R1", "Explain width", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain coverage", frozenset({EvidenceTarget.METHOD})),
    ))


def evidence(index=1):
    raw = RetrievedEvidence(EvidenceSourceType.CONTENT, "P1", (f"chunk_{index}",),
        (f"chunk_{index}",), "Width is compressed only for captured intervals.\n"
        "Equation: loss = width + calibration.\nCoverage penalizes outside distance.",
        paper_title="Paper", section="Method")
    return EvidenceRecord(f"E{index}", raw, ("width",), 1, index, requested_requirement_ids=("R1",))


def item(rid, ids=("E1",), gap=None):
    return {"requirement_id": rid, "evidence_ids": list(ids), "remaining_gap": gap}


def complete_payload():
    return {"answer": "Width and coverage work together through width compression and calibration. [E1]",
            "citations": ["E1"], "requirements": [item("R1"), item("R2")]}


def run(client, records=None):
    return LLMGroundedAnswerer(client).answer(question="How do width and coverage work together?",
                                             plan=plan(), evidence=records if records is not None else (evidence(),))


def test_intact_evidence_is_read_once_without_spans_query_binding_or_duplicate_text():
    client = QueueClient(complete_payload())
    result = run(client)
    prompt = json.loads(client.calls[0]["user_prompt"])
    row = prompt["evidence"][0]
    assert row["content"] == evidence().evidence.content
    assert "source_spans" not in row and "retrieved_by_queries" not in row
    assert "requested_requirement_ids" not in row
    assert result.complete and result.requirements[1].evidence_ids == ("E1",)
    assert result.answer == complete_payload()["answer"]
    assert set(client.calls[0]["json_schema"]["properties"]) == {"answer", "citations", "requirements"}


def test_renderer_preserves_coherent_prose_instead_of_requirement_paragraphs():
    result = run(QueueClient(complete_payload()))
    rendered = render_grounded_answer(result, plan())
    assert rendered.answer == complete_payload()["answer"]
    assert rendered.requirement_coverage == result.requirements
    assert not hasattr(rendered, "grounded_requirements")


def test_prompt_requires_question_first_and_coverage_of_actual_written_answer():
    assert "先围绕整个原问题组织一份连贯的 answer" in GROUNDED_ANSWER_SYSTEM_PROMPT
    assert "正文实际没写的内容仍是缺口" in GROUNDED_ANSWER_SYSTEM_PROMPT
    assert "分别提到输入与输出不能推出二者之间的机制" in GROUNDED_ANSWER_SYSTEM_PROMPT


def test_partial_coverage_keeps_answer_and_exposes_precise_gap():
    payload = complete_payload()
    payload["requirements"][1]["remaining_gap"] = "How does calibration change the interval?"
    result = run(QueueClient(payload))
    assert result.has_grounded_answer and not result.complete
    assert result.requirements[1].status == "partial"
    assert result.open_gaps == (("R2", "How does calibration change the interval?"),)
    assert render_grounded_answer(result, plan()).partial_answer


def test_unanswered_coverage_is_not_complete_even_with_other_cited_answer():
    payload = complete_payload()
    payload["requirements"][1] = item("R2", (), "Need calibration evidence.")
    result = run(QueueClient(payload))
    assert not result.complete and result.requirements[1].status == "unsupported"


def test_coverage_repair_preserves_prose_and_valid_sibling_requests_only_failed_item():
    payload = complete_payload()
    payload["requirements"][1]["evidence_ids"] = ["E99"]
    client = QueueClient(payload, {"requirements": [item("R2")]})
    result = run(client)
    repair = json.loads(client.calls[1]["user_prompt"])
    assert repair["validated_answer"] == {"answer": payload["answer"], "citations": ["E1"]}
    assert [r["requirement_id"] for r in repair["requirements"]] == ["R2"]
    assert set(client.calls[1]["json_schema"]["properties"]) == {"requirements"}
    assert result.answer == payload["answer"] and result.complete
    assert [d.success for d in result.llm_diagnostics] == [False, True]
    diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
    assert diagnostic.error_code == "ANSWER_COVERAGE_REFERENCES_INVALID"
    assert diagnostic.requirement_id == "R2" and diagnostic.recovery_action == "repair_coverage"
    assert result.llm_diagnostics[0].recovery_action == diagnostic.recovery_action


def test_coverage_repair_cannot_rewrite_validated_prose():
    payload = complete_payload()
    payload["requirements"][1]["evidence_ids"] = ["E99"]
    result = run(QueueClient(payload, {"answer": "Injected answer", "requirements": [item("R2")]}))
    assert result.answer == payload["answer"]
    assert result.requirements[0].complete and not result.requirements[1].complete


@pytest.mark.parametrize("response", [RuntimeError("transport"), {"requirements": [item("R2", ("E99",))]}])
def test_failed_metadata_repair_keeps_valid_prose_and_sibling_without_fatal(response):
    payload = complete_payload()
    payload["requirements"][1]["evidence_ids"] = ["E99"]
    result = run(QueueClient(payload, response))
    assert result.answer == payload["answer"] and result.requirements[0].complete
    assert not result.requirements[1].complete and result.requirements[1].remaining_gap
    assert len(result.llm_diagnostics) == 2


@pytest.mark.parametrize("partition", [[], [item("R2")], [item("R2"), item("R1")], [item("R1"), item("R1")]])
def test_missing_extra_reordered_or_duplicate_partition_never_yields_complete(partition):
    payload = complete_payload()
    payload["requirements"] = partition
    client = QueueClient(payload, {"requirements": [item("R1"), item("R2")]})
    # An empty list fails schema before prose validation and requires full regeneration.
    if not partition:
        client.responses[1] = complete_payload()
    result = run(client)
    assert result.complete and len(client.calls) == 2


@pytest.mark.parametrize("change", [
    {"citations": ["E99"]}, {"citations": ["E1", "E1"]},
    {"answer": "Uncited factual prose"}, {"answer": "Fake citation [E99]"},
    {"answer": None}, {"citations": []},
])
def test_invalid_question_wide_references_retry_whole_draft_and_remain_auditable(change):
    payload = complete_payload()
    payload.update(change)
    client = QueueClient(payload, complete_payload())
    result = run(client)
    assert result.complete and len(client.calls) == 2
    assert "validated_answer" not in json.loads(client.calls[1]["user_prompt"])
    assert [d.success for d in result.llm_diagnostics] == [False, True]


def test_exhausted_invalid_prose_does_not_publish_unsupported_text():
    payload = complete_payload()
    payload["answer"] = "Invented prose [E99]"
    with pytest.raises(GroundedAnswerAttemptsExhausted) as caught:
        run(QueueClient(payload, deepcopy(payload)))
    assert len(caught.value.diagnostics) == 2


def test_uncited_lack_of_evidence_prose_normalizes_to_gaps_without_publishing():
    payload = {"answer": "The evidence does not explain this.", "citations": [],
               "requirements": [item("R1", (), "Need width mechanism."), item("R2", (), "Need coverage mechanism.")]}
    client = QueueClient(payload)
    result = run(client)
    assert result.answer is None and result.citations == () and not result.complete
    assert len(client.calls) == 1
    assert "does not explain" not in render_grounded_answer(result, plan()).answer


def test_no_evidence_exposes_gaps_without_calling_provider():
    client = QueueClient()
    result = run(client, ())
    assert result.answer is None and len(result.open_gaps) == 2
    assert client.calls == []


def test_all_registered_evidence_reaches_answer_without_old_count_cap():
    records = tuple(evidence(i) for i in range(1, 13))
    client = QueueClient(complete_payload())
    result = run(client, records)
    payload = json.loads(client.calls[0]["user_prompt"])
    assert len(payload["evidence"]) == 12
    assert payload["evidence"][-1]["content"] == records[-1].evidence.content
    assert result.citations == ("E1",)


def test_optional_requirements_do_not_create_completion_obligations():
    from dataclasses import replace
    one = replace(plan(), requirements=(plan().requirements[0], replace(plan().requirements[1], required=False)))
    payload = complete_payload()
    payload["requirements"] = payload["requirements"][:1]
    result = LLMGroundedAnswerer(QueueClient(payload)).answer(question="Explain width", plan=one, evidence=(evidence(),))
    assert result.complete and render_grounded_answer(result, one).overall_status == "complete"


def test_reference_metadata_uses_reference_identity_not_content_evidence_identity():
    from dataclasses import replace
    record = replace(evidence(), evidence_id="REF1")
    payload = complete_payload()
    payload["answer"] = "The reference metadata identifies the cited work. [REF1]"
    payload["citations"] = ["REF1"]
    payload["requirements"] = [item("R1", ("REF1",)), item("R2", ("REF1",))]
    assert run(QueueClient(payload), (record,)).citations == ("REF1",)
