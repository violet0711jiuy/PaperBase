"""Requirement-complete answers grounded directly in the raw Evidence Pool."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Protocol, Sequence

from pydantic import ValidationError

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCallRecord,
    LLMCompletion,
    complete_json_with_diagnostics,
    structured_client_view,
)
from paperbase.llm.diagnostics import failure_record, success_record
from paperbase.generation.answer_coverage import AnswerRequirementCoverage
from paperbase.prompts.grounded_answer import (
    GROUNDED_ANSWER_PROMPT_VERSION,
    GROUNDED_ANSWER_SYSTEM_PROMPT,
    build_grounded_answer_prompt,
)

from .contracts import (
    EvidenceRecord,
    GroundedAnswerAttempt,
    GroundedCitation,
    RequirementAnswer,
    ResearchAnswer,
    ResearchPlan,
)
from .schemas import GroundedAnswerDraft
from .source_spans import (
    SourceSpanResolutionError,
    resolve_source_span_ids,
    source_spans_payload,
)


class GroundedAnswerValidationError(ValueError):
    def __init__(self, code: str, field_path: str, safe_reason: str) -> None:
        super().__init__(safe_reason)
        self.code = code
        self.field_path = field_path
        self.safe_reason = safe_reason


class GroundedAnswerer(Protocol):
    def answer(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        evidence: Sequence[EvidenceRecord],
        accepted_answers: Sequence[RequirementAnswer] = (),
    ) -> GroundedAnswerAttempt: ...


@dataclass(frozen=True)
class EvidencePack:
    records: tuple[EvidenceRecord, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    def prompt_payload(self) -> tuple[dict[str, object], ...]:
        return tuple(_evidence_payload(record) for record in self.records)


class EvidencePackBuilder:
    """Expose the complete, deduplicated raw Evidence Pool to the Answerer.

    The Registry is the lossless boundary.  A count cap here would recreate the
    old semantic bottleneck: retrieval could find a decisive chunk while the
    answer model never receives it.  Decision has its own compact catalog; the
    semantic author intentionally sees every registered record.
    """

    def build(self, evidence: Sequence[EvidenceRecord]) -> EvidencePack:
        return EvidencePack(tuple(evidence))


class LLMGroundedAnswerer:
    """The only Agentic semantic answer author; citations are locally restored."""

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        max_attempts: int = 2,
        temperature: float = 0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        self._client = structured_client_view(
            client,
            temperature=temperature,
            structured_output_mode="json_schema",
        )
        disable_thinking = getattr(self._client, "with_thinking_disabled", None)
        if callable(disable_thinking):
            self._client = disable_thinking()
        self._max_attempts = max_attempts
        self._pack_builder = EvidencePackBuilder()

    def answer(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        evidence: Sequence[EvidenceRecord],
        accepted_answers: Sequence[RequirementAnswer] = (),
    ) -> GroundedAnswerAttempt:
        requirements = tuple(
            (item.requirement_id, item.description)
            for item in plan.requirements if item.required
        )
        if not requirements:
            raise GroundedAnswerValidationError(
                "ANSWER_NO_REQUIREMENTS", "requirements", "No required Requirement is available."
            )
        pack = self._pack_builder.build(evidence)
        if not pack.records:
            return GroundedAnswerAttempt(
                requirements=tuple(
                    RequirementAnswer(rid, None, remaining_gap="当前尚未检索到可用于回答的证据。")
                    for rid, _ in requirements
                ),
                evidence_ids=(),
            )

        errors: list[dict[str, object]] = []
        diagnostics: list[LLMCallRecord] = []
        schema = _response_schema(requirements, pack)
        for attempt in range(1, self._max_attempts + 1):
            completion: LLMCompletion | None = None
            prompt = build_grounded_answer_prompt(
                question=question,
                requirements=requirements,
                evidence=pack.prompt_payload(),
                accepted_answers=tuple(_accepted_payload(item) for item in accepted_answers),
                validation_errors=errors,
            )
            try:
                completion = complete_json_with_diagnostics(
                    self._client,
                    system_prompt=GROUNDED_ANSWER_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    json_schema=schema,
                    schema_name="paperbase_grounded_requirement_answer_v1",
                )
                draft = GroundedAnswerDraft.model_validate_json(completion.content)
                result = _convert_grounded_answer(
                    draft,
                    requirements=requirements,
                    pack=pack,
                    accepted_answers=tuple(accepted_answers),
                    diagnostics=tuple(diagnostics),
                )
                diagnostics.append(success_record(
                    node="grounded_answer", attempt=attempt, completion=completion
                ))
                return replace(result, llm_diagnostics=tuple(diagnostics))
            except Exception as error:
                can_retry = attempt < self._max_attempts
                diagnostics.append(failure_record(
                    node="grounded_answer",
                    attempt=attempt,
                    error=error,
                    completion=completion,
                    can_retry=can_retry,
                ))
                errors = [_safe_error(error)]
                if not can_retry:
                    raise
        raise RuntimeError("Grounded Answer attempts exhausted.")


def render_grounded_answer(attempt: GroundedAnswerAttempt, plan: ResearchPlan) -> ResearchAnswer:
    """Deterministically render validated Requirement answers; add no facts."""
    descriptions = {item.requirement_id: item.description for item in plan.requirements}
    sections: list[str] = []
    gaps: list[str] = []
    citations: list[str] = []
    coverage: list[AnswerRequirementCoverage] = []
    for item in attempt.requirements:
        evidence_ids = tuple(dict.fromkeys(citation.evidence_id for citation in item.citations))
        for evidence_id in evidence_ids:
            if evidence_id not in citations:
                citations.append(evidence_id)
        if item.answer is not None:
            markers = "".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
            sections.append(
                f"### {descriptions.get(item.requirement_id, item.requirement_id)}\n"
                f"{item.answer} {markers}".rstrip()
            )
            status = "complete" if item.remaining_gap is None else "partial"
        else:
            status = "unsupported"
        if item.remaining_gap is not None:
            gaps.append(f"{item.requirement_id}: {item.remaining_gap}")
        coverage.append(AnswerRequirementCoverage(item.requirement_id, status, evidence_ids))
    if not sections:
        body = "当前检索到的论文证据不足以回答该问题。"
        overall = "insufficient"
    else:
        body = "\n\n".join(sections)
        overall = "complete" if attempt.complete else "partial"
    note = "；".join(gaps) if gaps else None
    if note:
        body += "\n\n### 仍需补充的证据\n" + note
    return ResearchAnswer(
        status="success" if overall != "insufficient" else "insufficient_evidence",
        answer=body,
        citations=tuple(citations),
        insufficient_evidence=overall == "insufficient",
        partial_answer=overall == "partial",
        coverage_note=note,
        requirement_coverage=tuple(coverage),
        grounded_requirements=attempt.requirements,
        overall_status=overall,
        llm_diagnostics=attempt.llm_diagnostics,
    )


def _evidence_payload(record: EvidenceRecord) -> dict[str, object]:
    source = record.evidence
    return {
        "evidence_id": record.evidence_id,
        "kind": source.source_type.value,
        "paper_id": source.paper_id,
        "paper_title": source.paper_title or source.paper_id,
        "section": source.section,
        "pages": [source.page_start, source.page_end],
        "source_chunk_ids": list(source.source_chunk_ids),
        "retrieved_by_queries": list(record.supporting_queries),
        "source_spans": source_spans_payload(source.content),
    }


def _accepted_payload(answer: RequirementAnswer) -> dict[str, object]:
    return {
        "requirement_id": answer.requirement_id,
        "answer": answer.answer,
        "citations": [
            {
                "evidence_id": citation.evidence_id,
                "source_span_ids": list(citation.source_span_ids),
            }
            for citation in answer.citations
        ],
        "remaining_gap": answer.remaining_gap,
    }


def _response_schema(requirements, pack: EvidencePack) -> dict[str, object]:
    schema = GroundedAnswerDraft.model_json_schema()
    ids = [rid for rid, _ in requirements]
    schema["title"] = GROUNDED_ANSWER_PROMPT_VERSION
    answers = schema["properties"]["answers"]
    answers["minItems"] = answers["maxItems"] = len(ids)
    answer_def = schema["$defs"]["GroundedRequirementAnswerDraft"]
    answer_def["properties"]["requirement_id"] = {"type": "string", "enum": ids}
    citation_def = schema["$defs"]["GroundedCitationDraft"]
    citation_def["properties"]["evidence_id"] = {
        "type": "string",
        "enum": list(pack.evidence_ids),
    }
    return schema


def _convert_grounded_answer(
    draft: GroundedAnswerDraft,
    *,
    requirements: tuple[tuple[str, str], ...],
    pack: EvidencePack,
    accepted_answers: tuple[RequirementAnswer, ...],
    diagnostics: tuple[LLMCallRecord, ...],
) -> GroundedAnswerAttempt:
    expected = tuple(rid for rid, _ in requirements)
    actual = tuple(item.requirement_id for item in draft.answers)
    if actual != expected:
        raise GroundedAnswerValidationError(
            "ANSWER_REQUIREMENT_PARTITION_INVALID",
            "answers.requirement_id",
            "answers must contain every required Requirement exactly once and in Plan order.",
        )
    by_id = {record.evidence_id: record for record in pack.records}
    converted: list[RequirementAnswer] = []
    for index, item in enumerate(draft.answers):
        answer = item.answer.strip() if item.answer is not None else None
        gap = item.remaining_gap.strip() if item.remaining_gap is not None else None
        if answer and not item.citations:
            raise GroundedAnswerValidationError(
                "ANSWER_CITATION_REQUIRED", f"answers[{index}].citations",
                "Every answer must cite at least one raw Evidence span.",
            )
        if not answer and (item.citations or not gap):
            raise GroundedAnswerValidationError(
                "ANSWER_GAP_REQUIRED", f"answers[{index}]",
                "An unanswered Requirement must provide remaining_gap and no citations.",
            )
        citations: list[GroundedCitation] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for citation_index, citation in enumerate(item.citations):
            record = by_id.get(citation.evidence_id)
            if record is None:
                raise GroundedAnswerValidationError(
                    "ANSWER_UNKNOWN_EVIDENCE",
                    f"answers[{index}].citations[{citation_index}].evidence_id",
                    "Citation Evidence must belong to the supplied Evidence Pack.",
                )
            span_ids = tuple(citation.source_span_ids)
            identity = (citation.evidence_id, span_ids)
            if identity in seen:
                raise GroundedAnswerValidationError(
                    "ANSWER_DUPLICATE_CITATION",
                    f"answers[{index}].citations[{citation_index}]",
                    "Duplicate citations are not allowed.",
                )
            seen.add(identity)
            try:
                excerpt = resolve_source_span_ids(record.evidence.content, span_ids)
            except SourceSpanResolutionError as error:
                raise GroundedAnswerValidationError(
                    f"ANSWER_{error.code.upper()}",
                    f"answers[{index}].citations[{citation_index}].source_span_ids",
                    "Citation span IDs must be known, ordered, contiguous and locally bounded.",
                ) from error
            citations.append(GroundedCitation(citation.evidence_id, span_ids, excerpt))
        converted.append(RequirementAnswer(item.requirement_id, answer, tuple(citations), gap))
    accepted_by_id = {item.requirement_id: item for item in accepted_answers}
    for index, item in enumerate(converted):
        accepted = accepted_by_id.get(item.requirement_id)
        if accepted is not None and item != accepted:
            raise GroundedAnswerValidationError(
                "ANSWER_ACCEPTED_RESULT_CHANGED",
                f"answers[{index}]",
                "Previously accepted Requirement answers and citations must be returned unchanged.",
            )
    cited_evidence_ids = tuple(dict.fromkeys(
        citation.evidence_id
        for answer in converted
        for citation in answer.citations
    ))
    return GroundedAnswerAttempt(tuple(converted), cited_evidence_ids, diagnostics)


def _safe_error(error: Exception) -> dict[str, object]:
    if isinstance(error, GroundedAnswerValidationError):
        return {
            "code": error.code,
            "field_path": error.field_path,
            "instruction": error.safe_reason,
        }
    if isinstance(error, ValidationError):
        first = error.errors()[0] if error.errors() else {}
        return {
            "code": "ANSWER_SCHEMA_INVALID",
            "field_path": ".".join(str(item) for item in first.get("loc", ())) or "root",
            "instruction": "Return the complete object required by the supplied JSON Schema.",
        }
    if isinstance(error, json.JSONDecodeError):
        return {"code": "ANSWER_JSON_INVALID", "field_path": "root", "instruction": "Return valid JSON."}
    return {"code": "ANSWER_INVALID", "field_path": "root", "instruction": "Repair the answer contract."}
