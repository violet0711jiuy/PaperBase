"""Requirement-complete answers grounded directly in the raw Evidence Pool."""

from __future__ import annotations

from dataclasses import dataclass
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
from .schemas import GroundedAnswerDraft, GroundedRequirementAnswerDraft
from .source_spans import (
    SourceSpanResolutionError,
    partition_source_span_ids,
    resolve_source_span_ids,
    source_spans_payload,
)
from .requirement_terms import explicit_requirement_terms, term_present


class GroundedAnswerValidationError(ValueError):
    def __init__(self, code: str, field_path: str, safe_reason: str) -> None:
        super().__init__(safe_reason)
        self.code = code
        self.field_path = field_path
        self.safe_reason = safe_reason


class GroundedAnswerAttemptsExhausted(RuntimeError):
    """All provider attempts failed, with safe diagnostics retained."""

    def __init__(self, cause: Exception, diagnostics: Sequence[LLMCallRecord]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.diagnostics = tuple(diagnostics)


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

        required_by_id = dict(requirements)
        accepted_by_id: dict[str, RequirementAnswer] = {}
        for item in accepted_answers:
            if item.requirement_id not in required_by_id or not item.complete:
                raise GroundedAnswerValidationError(
                    "ANSWER_ACCEPTED_RESULT_INVALID",
                    "accepted_answers",
                    "Only complete answers for current required Requirements may be preserved.",
                )
            if item.requirement_id in accepted_by_id:
                raise GroundedAnswerValidationError(
                    "ANSWER_ACCEPTED_RESULT_DUPLICATE",
                    "accepted_answers",
                    "Accepted Requirement answers must be unique.",
                )
            accepted_by_id[item.requirement_id] = item

        pending = tuple(
            item for item in requirements if item[0] not in accepted_by_id
        )
        if not pending:
            return _attempt_from_answers(requirements, accepted_by_id)

        errors: list[dict[str, object]] = []
        diagnostics: list[LLMCallRecord] = []
        for attempt in range(1, self._max_attempts + 1):
            completion: LLMCompletion | None = None
            schema = _response_schema(pending, pack)
            prompt = build_grounded_answer_prompt(
                question=question,
                requirements=pending,
                evidence=pack.prompt_payload(),
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
                    if accepted_by_id:
                        _fill_unresolved_answers(accepted_by_id, pending)
                        return _attempt_from_answers(
                            requirements, accepted_by_id, tuple(diagnostics)
                        )
                    raise GroundedAnswerAttemptsExhausted(error, diagnostics) from error
                continue

            expected = tuple(rid for rid, _ in pending)
            actual = tuple(item.requirement_id for item in draft.answers)
            if actual != expected:
                error = GroundedAnswerValidationError(
                    "ANSWER_REQUIREMENT_PARTITION_INVALID",
                    "answers.requirement_id",
                    "answers must contain every requested Requirement exactly once and in order.",
                )
                can_retry = attempt < self._max_attempts
                diagnostics.append(failure_record(
                    node="grounded_answer", attempt=attempt, error=error,
                    completion=completion, can_retry=can_retry,
                ))
                errors = [_safe_error(error)]
                if not can_retry:
                    if accepted_by_id:
                        _fill_unresolved_answers(accepted_by_id, pending)
                        return _attempt_from_answers(
                            requirements, accepted_by_id, tuple(diagnostics)
                        )
                    raise GroundedAnswerAttemptsExhausted(error, diagnostics) from error
                continue

            item_errors: list[tuple[str, GroundedAnswerValidationError]] = []
            for index, item in enumerate(draft.answers):
                try:
                    accepted_by_id[item.requirement_id] = _convert_requirement_answer(
                        item,
                        index=index,
                        requirement_description=required_by_id[item.requirement_id],
                        pack=pack,
                    )
                except GroundedAnswerValidationError as error:
                    item_errors.append((item.requirement_id, error))

            if not item_errors:
                diagnostics.append(success_record(
                    node="grounded_answer", attempt=attempt, completion=completion
                ))
                return _attempt_from_answers(
                    requirements, accepted_by_id, tuple(diagnostics)
                )

            failed_ids = {rid for rid, _ in item_errors}
            pending = tuple(item for item in pending if item[0] in failed_ids)
            can_retry = attempt < self._max_attempts
            diagnostics.append(failure_record(
                node="grounded_answer", attempt=attempt,
                error=item_errors[0][1], completion=completion,
                can_retry=can_retry,
            ))
            errors = [
                {"requirement_id": rid, **_safe_error(error)}
                for rid, error in item_errors
            ]
            if not can_retry:
                _fill_unresolved_answers(accepted_by_id, pending)
                return _attempt_from_answers(
                    requirements, accepted_by_id, tuple(diagnostics)
                )
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


def _convert_requirement_answer(
    item: GroundedRequirementAnswerDraft,
    *,
    index: int,
    requirement_description: str,
    pack: EvidencePack,
) -> RequirementAnswer:
    """Validate one Requirement without discarding valid siblings.

    A model sometimes writes an explanation of missing evidence in ``answer``
    while also returning an explicit gap.  With no citations that prose cannot
    become a published claim, so it is deterministically reduced to the gap.
    """
    answer = item.answer.strip() if item.answer is not None else None
    gap = item.remaining_gap.strip() if item.remaining_gap is not None else None
    if answer and not item.citations:
        if gap:
            return RequirementAnswer(item.requirement_id, None, remaining_gap=gap)
        raise GroundedAnswerValidationError(
            "ANSWER_CITATION_REQUIRED", f"answers[{index}].citations",
            "Every answer must cite at least one raw Evidence span.",
        )
    if not answer and (item.citations or not gap):
        raise GroundedAnswerValidationError(
            "ANSWER_GAP_REQUIRED", f"answers[{index}]",
            "An unanswered Requirement must provide remaining_gap and no citations.",
        )

    by_id = {record.evidence_id: record for record in pack.records}
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
        try:
            runs = partition_source_span_ids(
                record.evidence.content, citation.source_span_ids
            )
        except SourceSpanResolutionError as error:
            raise GroundedAnswerValidationError(
                f"ANSWER_{error.code.upper()}",
                f"answers[{index}].citations[{citation_index}].source_span_ids",
                "Citation span IDs must be known, ordered and locally bounded.",
            ) from error
        for span_ids in runs:
            identity = (citation.evidence_id, span_ids)
            if identity in seen:
                raise GroundedAnswerValidationError(
                    "ANSWER_DUPLICATE_CITATION",
                    f"answers[{index}].citations[{citation_index}]",
                    "Duplicate citations are not allowed.",
                )
            seen.add(identity)
            excerpt = resolve_source_span_ids(record.evidence.content, span_ids)
            citations.append(GroundedCitation(citation.evidence_id, span_ids, excerpt))
    if len(citations) > 8:
        raise GroundedAnswerValidationError(
            "ANSWER_CITATIONS_TOO_FRAGMENTED",
            f"answers[{index}].citations",
            "A Requirement answer may use at most eight contiguous citation anchors.",
        )
    protected_terms = explicit_requirement_terms(requirement_description)
    cited_text = "\n".join(citation.source_excerpt for citation in citations)
    if answer and protected_terms and not any(
        term_present(term, cited_text) for term in protected_terms
    ):
        raise GroundedAnswerValidationError(
            "ANSWER_REQUIREMENT_TERM_UNANCHORED",
            f"answers[{index}].citations",
            "A named symbol or method in the Requirement must occur in the cited source spans.",
        )
    return RequirementAnswer(item.requirement_id, answer, tuple(citations), gap)


def _fill_unresolved_answers(
    answers: dict[str, RequirementAnswer],
    pending: Sequence[tuple[str, str]],
) -> None:
    for requirement_id, _ in pending:
        answers.setdefault(
            requirement_id,
            RequirementAnswer(
                requirement_id,
                None,
                remaining_gap=(
                    "当前 Requirement 的模型输出未通过 source-span 引用校验；"
                    "需要基于现有 Evidence 聚焦重试。"
                ),
            ),
        )


def _attempt_from_answers(
    requirements: Sequence[tuple[str, str]],
    answers: dict[str, RequirementAnswer],
    diagnostics: tuple[LLMCallRecord, ...] = (),
) -> GroundedAnswerAttempt:
    ordered = tuple(answers[rid] for rid, _ in requirements)
    cited_evidence_ids = tuple(dict.fromkeys(
        citation.evidence_id
        for answer in ordered
        for citation in answer.citations
    ))
    return GroundedAnswerAttempt(ordered, cited_evidence_ids, diagnostics)


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
