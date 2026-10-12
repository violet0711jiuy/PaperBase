from __future__ import annotations

import json

from paperbase.agentic.contracts import (
    EvidenceRecord,
    EvidenceTarget,
    ResearchPlan,
    ResearchRequirement,
)
from paperbase.agentic.grounded_answer import LLMGroundedAnswerer
from paperbase.llm.client import LLMCompletion
from paperbase.prompts.grounded_answer import GROUNDED_ANSWER_SYSTEM_PROMPT
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


class QueueClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        return LLMCompletion(json.dumps(self.responses.pop(0), ensure_ascii=False))


def plan():
    return ResearchPlan("Explain the method", (
        ResearchRequirement("R1", "Explain MPIWcapt", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain C_i", frozenset({EvidenceTarget.METHOD})),
    ))


def evidence():
    raw = RetrievedEvidence(
        EvidenceSourceType.CONTENT,
        "P1",
        ("chunk_0030",),
        ("chunk_0030", "chunk_0031", "chunk_0032"),
        "MPIWcapt averages the widths only for intervals that capture the true value. "
        "This avoids prematurely shrinking failed intervals. "
        "C_i is the indicator that the ith interval captures the target.",
        paper_title="Paper",
        section="Method",
    )
    # The Evidence was found for R1.  Grounded Answer is intentionally allowed
    # to use it for R2 as well.
    return EvidenceRecord("E1", raw, ("MPIWcapt",), 1, 1, requested_requirement_ids=("R1",))


def complete_payload():
    return {"answers": [
        {
            "requirement_id": "R1",
            "answer": "MPIWcapt averages only successfully covering intervals.",
            "citations": [{"evidence_id": "E1", "source_span_ids": ["S1", "S2"]}],
            "remaining_gap": None,
        },
        {
            "requirement_id": "R2",
            "answer": "C_i indicates whether interval i captures the target.",
            "citations": [{"evidence_id": "E1", "source_span_ids": ["S3"]}],
            "remaining_gap": None,
        },
    ]}


def test_raw_evidence_can_answer_requirements_outside_original_query_binding():
    client = QueueClient(complete_payload())
    result = LLMGroundedAnswerer(client).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )
    assert result.complete
    assert [item.requirement_id for item in result.requirements] == ["R1", "R2"]
    assert result.requirements[1].citations[0].source_excerpt.startswith("C_i is")
    assert len(client.calls) == 1


def test_prompt_requires_direct_support_for_relational_completion():
    assert "分别提到输入与输出不能推出二者之间的机制" in GROUNDED_ANSWER_SYSTEM_PROMPT
    assert "列出模型具有哪些输出，不等于解释这些输出如何计算" in GROUNDED_ANSWER_SYSTEM_PROMPT


def test_missing_requirement_is_repaired_against_the_same_raw_evidence():
    incomplete = {"answers": [complete_payload()["answers"][1]]}
    client = QueueClient(incomplete, complete_payload())
    result = LLMGroundedAnswerer(client).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )
    assert result.complete
    assert len(client.calls) == 2
    repair_payload = json.loads(client.calls[1]["user_prompt"])
    assert repair_payload["validation_errors"][0]["code"] == "ANSWER_REQUIREMENT_PARTITION_INVALID"
    assert repair_payload["evidence"][0]["evidence_id"] == "E1"


def test_invalid_requirement_is_repaired_without_regenerating_valid_sibling():
    invalid = complete_payload()
    invalid["answers"][0]["citations"] = []
    repaired_r1 = {"answers": [complete_payload()["answers"][0]]}
    client = QueueClient(invalid, repaired_r1)
    result = LLMGroundedAnswerer(client).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )

    assert result.complete
    assert len(client.calls) == 2
    repair_payload = json.loads(client.calls[1]["user_prompt"])
    assert [item["requirement_id"] for item in repair_payload["requirements"]] == ["R1"]
    assert "accepted_answers" not in repair_payload


def test_uncited_missing_evidence_prose_is_normalized_to_unanswered_gap():
    partial = complete_payload()
    partial["answers"][0] = {
        "requirement_id": "R1",
        "answer": "The current evidence does not explain MPIWcapt.",
        "citations": [],
        "remaining_gap": "Need the definition and role of MPIWcapt.",
    }
    client = QueueClient(partial)

    result = LLMGroundedAnswerer(client).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )

    assert result.requirements[0].answer is None
    assert result.requirements[0].citations == ()
    assert result.requirements[0].remaining_gap == "Need the definition and role of MPIWcapt."
    assert result.requirements[1].complete
    assert len(client.calls) == 1


def test_previously_accepted_requirement_is_preserved_without_model_repetition():
    accepted_payload = {"answers": [complete_payload()["answers"][0]]}
    accepted = LLMGroundedAnswerer(QueueClient(accepted_payload)).answer(
        question="How does the method work?",
        plan=ResearchPlan("Explain MPIWcapt", (plan().requirements[0],)),
        evidence=(evidence(),),
    ).requirements[0]
    client = QueueClient({"answers": [complete_payload()["answers"][1]]})

    result = LLMGroundedAnswerer(client).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),),
        accepted_answers=(accepted,),
    )

    prompt = json.loads(client.calls[0]["user_prompt"])
    assert [item["requirement_id"] for item in prompt["requirements"]] == ["R2"]
    assert "accepted_answers" not in prompt
    assert result.requirements == (accepted, result.requirements[1])
    assert result.complete


def test_named_requirement_term_must_appear_in_cited_source_spans():
    invalid = complete_payload()
    invalid["answers"][1] = {
        "requirement_id": "R2",
        "answer": "C_i is inferred from the interval-width behavior.",
        "citations": [{"evidence_id": "E1", "source_span_ids": ["S1"]}],
        "remaining_gap": "The cited span does not explicitly define C_i.",
    }
    repaired = {"answers": [{
        "requirement_id": "R2",
        "answer": None,
        "citations": [],
        "remaining_gap": "Need an explicit definition of C_i.",
    }]}
    client = QueueClient(invalid, repaired)

    result = LLMGroundedAnswerer(client).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )

    assert result.requirements[0].complete
    assert result.requirements[1].answer is None
    repair_payload = json.loads(client.calls[1]["user_prompt"])
    assert [item["requirement_id"] for item in repair_payload["requirements"]] == ["R2"]
    assert repair_payload["validation_errors"][0]["code"] == "ANSWER_REQUIREMENT_TERM_UNANCHORED"


def test_ordered_non_contiguous_spans_become_separate_exact_citations():
    payload = complete_payload()
    payload["answers"][0]["citations"] = [
        {"evidence_id": "E1", "source_span_ids": ["S1", "S3"]}
    ]
    result = LLMGroundedAnswerer(QueueClient(payload)).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )

    citations = result.requirements[0].citations
    assert [item.source_span_ids for item in citations] == [("S1",), ("S3",)]
    assert citations[0].source_excerpt.startswith("MPIWcapt")
    assert citations[1].source_excerpt.startswith("C_i")


def test_unanswered_requirement_must_expose_a_gap():
    partial = complete_payload()
    partial["answers"][1] = {
        "requirement_id": "R2", "answer": None, "citations": [],
        "remaining_gap": "The current Evidence does not define C_i.",
    }
    result = LLMGroundedAnswerer(QueueClient(partial)).answer(
        question="How does the method work?", plan=plan(), evidence=(evidence(),)
    )
    assert not result.complete
    assert result.open_gaps == (("R2", "The current Evidence does not define C_i."),)


def test_answerer_receives_every_registered_evidence_record_without_old_cap():
    records = tuple(
        EvidenceRecord(
            f"E{index}",
            RetrievedEvidence(
                EvidenceSourceType.CONTENT,
                "P1",
                (f"chunk_{index:04d}",),
                (f"chunk_{index:04d}",),
                f"Evidence record {index} contains a distinct source fact.",
                paper_title="Paper",
                section="Method",
            ),
            (f"query {index}",),
            1,
            index,
            requested_requirement_ids=("R1",),
        )
        for index in range(1, 13)
    )
    one_requirement_plan = ResearchPlan(
        "Find the decisive fact",
        (ResearchRequirement(
            "R1", "Report the fact from record 12",
            frozenset({EvidenceTarget.METHOD}),
        ),),
    )
    client = QueueClient({"answers": [{
        "requirement_id": "R1",
        "answer": "Record 12 contains the decisive fact.",
        "citations": [{"evidence_id": "E12", "source_span_ids": ["S1"]}],
        "remaining_gap": None,
    }]})

    result = LLMGroundedAnswerer(client).answer(
        question="What is the decisive fact?",
        plan=one_requirement_plan,
        evidence=records,
    )

    prompt = json.loads(client.calls[0]["user_prompt"])
    assert len(prompt["evidence"]) == 12
    assert prompt["evidence"][-1]["evidence_id"] == "E12"
    assert result.evidence_ids == ("E12",)
