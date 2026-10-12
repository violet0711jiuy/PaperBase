"""A coherent question-wide answer from intact, cumulative Evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Protocol, Sequence

from pydantic import ValidationError

from paperbase.llm.client import (
    ChatCompletionClient, LLMCallRecord, LLMCompletion, ValidationDiagnostic,
    complete_json_with_diagnostics, structured_client_view,
)
from paperbase.llm.diagnostics import classify_llm_error, failure_record, success_record
from paperbase.prompts.grounded_answer import (
    GROUNDED_ANSWER_PROMPT_VERSION, GROUNDED_ANSWER_SYSTEM_PROMPT,
    build_grounded_answer_prompt,
)
from .contracts import EvidenceRecord, GroundedAnswerAttempt, RequirementCoverage, ResearchAnswer, ResearchPlan
from .schemas import CoverageRepairDraft, GroundedAnswerDraft, RequirementCoverageDraft


class GroundedAnswerValidationError(ValueError):
    def __init__(self, code: str, field_path: str, safe_reason: str) -> None:
        super().__init__(safe_reason)
        self.code, self.field_path, self.safe_reason = code, field_path, safe_reason


class GroundedAnswerAttemptsExhausted(RuntimeError):
    def __init__(self, cause: Exception, diagnostics: Sequence[LLMCallRecord]) -> None:
        super().__init__(str(cause))
        self.cause, self.diagnostics = cause, tuple(diagnostics)


class GroundedAnswerer(Protocol):
    def answer(self, *, question: str, plan: ResearchPlan,
               evidence: Sequence[EvidenceRecord]) -> GroundedAnswerAttempt: ...


@dataclass(frozen=True)
class EvidencePack:
    records: tuple[EvidenceRecord, ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.records)

    def prompt_payload(self) -> tuple[dict[str, object], ...]:
        return tuple(_evidence_payload(record) for record in self.records)


class EvidencePackBuilder:
    """Expose every registered Evidence once, without truncation or span slicing."""
    def build(self, evidence: Sequence[EvidenceRecord]) -> EvidencePack:
        records = tuple(evidence)
        if len({record.evidence_id for record in records}) != len(records):
            raise ValueError("Evidence Pack IDs must be unique.")
        return EvidencePack(records)


class LLMGroundedAnswerer:
    """The semantic author; local validation checks references, not entailment."""
    def __init__(self, client: ChatCompletionClient, *, max_attempts: int = 2,
                 temperature: float = 0) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        self._client = structured_client_view(client, temperature=temperature,
                                              structured_output_mode="json_schema")
        disable_thinking = getattr(self._client, "with_thinking_disabled", None)
        if callable(disable_thinking):
            self._client = disable_thinking()
        self._max_attempts = max_attempts
        self._pack_builder = EvidencePackBuilder()

    def answer(self, *, question: str, plan: ResearchPlan,
               evidence: Sequence[EvidenceRecord]) -> GroundedAnswerAttempt:
        requirements = tuple((item.requirement_id, item.description)
                             for item in plan.requirements if item.required)
        if not requirements:
            raise GroundedAnswerValidationError("ANSWER_NO_REQUIREMENTS", "requirements",
                                                "No required Requirement is available.")
        pack = self._pack_builder.build(evidence)
        if not pack.records:
            return GroundedAnswerAttempt(None, (), tuple(
                RequirementCoverage(rid, (), "当前尚未检索到可用于回答的证据。")
                for rid, _ in requirements))

        pending = requirements
        body: str | None = None
        citations: tuple[str, ...] = ()
        body_validated = False
        coverage: dict[str, RequirementCoverage] = {}
        errors: list[dict[str, object]] = []
        diagnostics: list[LLMCallRecord] = []
        for attempt in range(1, self._max_attempts + 1):
            completion: LLMCompletion | None = None
            current_errors: list[dict[str, object]] = []
            repair = body_validated
            try:
                completion = complete_json_with_diagnostics(
                    self._client, system_prompt=GROUNDED_ANSWER_SYSTEM_PROMPT,
                    user_prompt=build_grounded_answer_prompt(
                        question=question, requirements=pending,
                        evidence=pack.prompt_payload(), validation_errors=errors,
                        validated_answer={"answer": body, "citations": list(citations)} if repair else None,
                    ),
                    json_schema=_response_schema(pending, pack, repair=repair, citations=citations),
                    schema_name="paperbase_grounded_coverage_repair_v2" if repair else "paperbase_grounded_answer_v2",
                )
                draft = (CoverageRepairDraft if repair else GroundedAnswerDraft).model_validate_json(completion.content)
                if not repair:
                    body, citations = _validate_body(draft, pack)
                    body_validated = True
                expected = tuple(rid for rid, _ in pending)
                actual = tuple(item.requirement_id for item in draft.requirements)
                if actual != expected:
                    raise GroundedAnswerValidationError("ANSWER_REQUIREMENT_PARTITION_INVALID", "requirements",
                        "Return every requested Requirement exactly once and in order.")
                failed = []
                for index, item in enumerate(draft.requirements):
                    try:
                        coverage[item.requirement_id] = _validate_coverage(item, citations, index=index)
                    except GroundedAnswerValidationError as error:
                        failed.append((item.requirement_id, error))
                if failed:
                    pending = tuple(item for item in pending if item[0] in {rid for rid, _ in failed})
                    current_errors = [{"requirement_id": rid, **_safe_error(error)} for rid, error in failed]
                    raise failed[0][1]
                diagnostics.append(success_record(node="grounded_answer", attempt=attempt, completion=completion))
                return GroundedAnswerAttempt(body, citations, tuple(coverage[rid] for rid, _ in requirements), tuple(diagnostics))
            except Exception as error:
                retry = attempt < self._max_attempts
                errors = current_errors or [_safe_error(error)]
                stage = classify_llm_error(error)
                validation = tuple(ValidationDiagnostic(
                    node="grounded_answer", attempt=attempt, error_stage=stage,
                    error_code=str(row["code"]), field_path=str(row["field_path"]),
                    validator_name="whole_evidence_answer_contract",
                    safe_reason=str(row["instruction"]), expected_constraint=str(row["instruction"]),
                    actual_summary="Provider output failed the current answer/coverage contract.",
                    recovery_action=("repair_coverage" if body_validated else "regenerate_answer") if retry else None,
                    requirement_id=row.get("requirement_id"),
                ) for row in errors) if stage in {"invalid_json", "schema_validation", "domain_validation"} else ()
                recovery = ("repair_coverage" if body_validated else "regenerate_answer") if retry else None
                diagnostics.append(replace(
                    failure_record(node="grounded_answer", attempt=attempt, error=error,
                                   completion=completion, can_retry=retry, validation_diagnostics=validation),
                    recovery_action=recovery,
                ))
                if not retry:
                    if not body_validated:
                        raise GroundedAnswerAttemptsExhausted(error, diagnostics) from error
                    # Valid prose and coverage siblings are never discarded by a metadata error.
                    for rid, _ in requirements:
                        coverage.setdefault(rid, RequirementCoverage(rid, (),
                            "当前回答的覆盖字段未通过校验；需要基于累计 Evidence 重新检查此项。"))
                    return GroundedAnswerAttempt(body, citations, tuple(coverage[rid] for rid, _ in requirements), tuple(diagnostics))
        raise RuntimeError("Grounded Answer attempts exhausted.")


def _validate_body(draft: GroundedAnswerDraft, pack: EvidencePack) -> tuple[str | None, tuple[str, ...]]:
    body = draft.answer.strip() if draft.answer is not None else None
    citations = tuple(draft.citations)
    if len(citations) != len(set(citations)):
        raise GroundedAnswerValidationError("ANSWER_DUPLICATE_CITATION", "citations", "Citation IDs must be unique.")
    if set(citations) - set(pack.evidence_ids):
        raise GroundedAnswerValidationError("ANSWER_UNKNOWN_EVIDENCE", "citations", "Citations must belong to the supplied Evidence Pool.")
    if body and not citations:
        if all(item.remaining_gap and not item.evidence_ids for item in draft.requirements):
            return None, ()  # Uncited lack-of-evidence prose may only survive as explicit gaps.
        raise GroundedAnswerValidationError("ANSWER_CITATION_REQUIRED", "citations", "Answer prose needs Evidence references.")
    if not body and citations:
        raise GroundedAnswerValidationError("ANSWER_EMPTY_WITH_CITATIONS", "answer", "An absent answer cannot cite Evidence.")
    markers = tuple(dict.fromkeys(re.findall(r"\[((?:E|REF)\d+)\]", body or "")))
    if set(markers) != set(citations):
        raise GroundedAnswerValidationError("ANSWER_CITATION_MARKERS_MISMATCH", "answer",
            "Use [E1] markers beside factual prose; marker IDs and citations must match exactly.")
    return body, citations


def _validate_coverage(item: RequirementCoverageDraft, citations: tuple[str, ...], *, index: int) -> RequirementCoverage:
    path = f"requirements[{index}]"
    ids = tuple(item.evidence_ids)
    if len(ids) != len(set(ids)) or set(ids) - set(citations):
        raise GroundedAnswerValidationError("ANSWER_COVERAGE_REFERENCES_INVALID", path + ".evidence_ids",
            "Coverage IDs must be unique and reference Evidence actually cited in the answer.")
    gap = item.remaining_gap.strip() if item.remaining_gap is not None else None
    if not ids and not gap:
        raise GroundedAnswerValidationError("ANSWER_GAP_REQUIRED", path + ".remaining_gap",
            "An uncovered Requirement must describe its remaining gap.")
    return RequirementCoverage(item.requirement_id, ids, gap)


def _response_schema(requirements, pack: EvidencePack, *, repair=False, citations=()) -> dict[str, object]:
    schema = (CoverageRepairDraft if repair else GroundedAnswerDraft).model_json_schema()
    schema["title"] = GROUNDED_ANSWER_PROMPT_VERSION + ("-coverage-repair" if repair else "")
    ids = [rid for rid, _ in requirements]
    schema["properties"]["requirements"].update(minItems=len(ids), maxItems=len(ids))
    definition = schema["$defs"]["RequirementCoverageDraft"]["properties"]
    definition["requirement_id"] = {"type": "string", "enum": ids}
    allowed = list(citations if repair else pack.evidence_ids)
    definition["evidence_ids"].update(uniqueItems=True, maxItems=len(allowed))
    if allowed:
        definition["evidence_ids"]["items"] = {"type": "string", "enum": allowed}
    if not repair:
        schema["properties"]["citations"].update(uniqueItems=True, maxItems=len(pack.records),
            items={"type": "string", "enum": list(pack.evidence_ids)})
    return schema


def render_grounded_answer(attempt: GroundedAnswerAttempt, plan: ResearchPlan) -> ResearchAnswer:
    """Keep the author's coherent prose; render only an explicit gap footer."""
    expected = tuple(item.requirement_id for item in plan.requirements if item.required)
    if tuple(item.requirement_id for item in attempt.requirements) != expected:
        raise GroundedAnswerValidationError("ANSWER_REQUIREMENT_PARTITION_INVALID", "requirements",
                                            "Coverage must match the current required plan.")
    overall = "complete" if attempt.complete else "partial" if attempt.has_grounded_answer else "insufficient"
    note = "；".join(f"{rid}: {gap}" for rid, gap in attempt.open_gaps) or None
    body = attempt.answer or "当前检索到的论文证据不足以回答该问题。"
    if note:
        body += "\n\n### 仍需补充的证据\n" + note
    return ResearchAnswer(
        status="success" if overall != "insufficient" else "insufficient_evidence",
        answer=body, citations=attempt.citations, insufficient_evidence=overall == "insufficient",
        partial_answer=overall == "partial", coverage_note=note,
        requirement_coverage=attempt.requirements, overall_status=overall,
        llm_diagnostics=attempt.llm_diagnostics,
    )


def _evidence_payload(record: EvidenceRecord) -> dict[str, object]:
    source = record.evidence
    return {"evidence_id": record.evidence_id, "kind": source.source_type.value,
            "paper_id": source.paper_id, "paper_title": source.paper_title or source.paper_id,
            "section": source.section, "pages": [source.page_start, source.page_end],
            "source_chunk_ids": list(source.source_chunk_ids), "content": source.content}


def _safe_error(error: Exception) -> dict[str, object]:
    if isinstance(error, GroundedAnswerValidationError):
        return {"code": error.code, "field_path": error.field_path, "instruction": error.safe_reason}
    if isinstance(error, ValidationError):
        first = error.errors()[0] if error.errors() else {}
        return {"code": "ANSWER_SCHEMA_INVALID", "field_path": ".".join(map(str, first.get("loc", ()))) or "root",
                "instruction": "Return the complete object required by the supplied JSON Schema."}
    return {"code": "ANSWER_INVALID", "field_path": "root", "instruction": "Repair the answer contract."}
