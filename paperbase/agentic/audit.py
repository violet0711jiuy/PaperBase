"""Public, content-safe audit projection for Grounded Answer research runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Mapping


AUDIT_SCHEMA_VERSION = "grounded-answer-v2"


@dataclass(frozen=True)
class CandidateAuditItem:
    paper_id: str
    title: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class SearchAttemptAudit:
    paper_id: str
    title: str
    query: str
    lexical_terms: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    status: str = ""
    error_code: str | None = None


@dataclass(frozen=True)
class FilterReasonAudit:
    reason: str
    count: int


@dataclass(frozen=True)
class ResearchStepAudit:
    iteration: int
    action_name: str
    tool_status: str
    added_count: int
    reused_count: int
    attempts: tuple[SearchAttemptAudit, ...] = ()
    action_arguments: dict[str, object] = field(default_factory=dict)
    requirement_ids: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    filter_total: int = 0
    filter_accepted: int = 0
    filter_rejected: int = 0
    filter_deprioritized: int = 0
    filter_reasons: tuple[FilterReasonAudit, ...] = ()
    canonical_effective_requests: tuple[dict[str, object], ...] = ()
    request_identities: tuple[str, ...] = ()
    request_executions: tuple[dict[str, object], ...] = ()
    request_contract_version: str = "legacy"


@dataclass(frozen=True)
class RejectedActionAudit:
    iteration: int
    action_name: str
    action_arguments: dict[str, object]
    requirement_ids: tuple[str, ...]
    rejection_code: str
    rejection_reason: str
    rejection_stage: str
    recovery_action: str
    blockers: tuple[object, ...] = ()


@dataclass(frozen=True)
class PaperResearchAudit:
    paper_id: str
    title: str
    status: str
    queries_tried: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    rationale: str = ""
    covered_targets: tuple[str, ...] = ()
    remaining_targets: tuple[str, ...] = ()
    sections_seen: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceDigestAudit:
    evidence_id: str
    paper_id: str
    paper_title: str
    source_type: str
    section_id: str | None
    section: str | None
    source_chunk_ids: tuple[str, ...]
    supporting_queries: tuple[str, ...]
    requested_requirement_ids: tuple[str, ...]
    selected: bool
    cited: bool
    source_preview: str = ""


@dataclass(frozen=True)
class LLMCallAudit:
    node: str
    attempt: int
    success: bool
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    content_chars: int = 0
    reasoning_chars: int = 0
    error_stage: str | None = None
    recovery_action: str | None = None
    decision_iteration: int | None = None
    decision_view_metrics: dict[str, int | float] | None = None


@dataclass(frozen=True)
class ValidationErrorAudit:
    node: str
    attempt: int
    error_stage: str
    error_code: str
    field_path: str
    validator_name: str
    safe_reason: str
    expected_constraint: str
    actual_summary: str
    recovery_action: str | None = None
    requirement_id: str | None = None


@dataclass(frozen=True)
class BudgetAudit:
    agent_iterations: int = 0
    tool_calls: int = 0
    retrieval_queries: int = 0
    llm_calls: int = 0
    invalid_actions: int = 0
    queries_per_paper: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RequirementAudit:
    requirement_id: str
    description: str
    required: bool
    status: str
    evidence_ids: tuple[str, ...] = ()
    remaining_gap: str | None = None


@dataclass(frozen=True)
class AgenticAuditSnapshot:
    schema_version: str
    thread_id: str | None
    run_status: str
    stop_reason: str
    scope: str | None
    coverage_complete: bool | None
    clarification: str | None
    candidates: tuple[CandidateAuditItem, ...]
    papers: tuple[PaperResearchAudit, ...]
    steps: tuple[ResearchStepAudit, ...]
    evidence: tuple[EvidenceDigestAudit, ...]
    budget: BudgetAudit
    llm_calls: tuple[LLMCallAudit, ...]
    validation_errors: tuple[ValidationErrorAudit, ...] = ()
    requirements: tuple[RequirementAudit, ...] = ()
    rejected_actions: tuple[RejectedActionAudit, ...] = ()
    error_codes: tuple[str, ...] = ()
    planning_mode: str | None = None
    selected_evidence_ids: tuple[str, ...] = ()
    cited_evidence_ids: tuple[str, ...] = ()
    answer_input_evidence_ids: tuple[str, ...] = ()
    answer_status: str | None = None
    answer_overall_status: str | None = None
    answer_requirement_coverage: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is not None:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "AgenticAuditSnapshot":
        payload = json.loads(text)
        if payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise ValueError("Unsupported Agentic audit schema version.")
        return _snapshot_from_dict(payload)


def build_agentic_audit_snapshot(
    result: object,
    *,
    paper_titles: Mapping[str, str] | None = None,
) -> AgenticAuditSnapshot:
    titles = dict(paper_titles or {})
    records = tuple(getattr(result, "evidence", ()) or ())
    selected = tuple(getattr(result, "selected_evidence", ()) or ())
    selected_ids = tuple(dict.fromkeys(
        str(getattr(item, "evidence_id", "")) for item in selected
        if str(getattr(item, "evidence_id", ""))
    ))
    selected_id_set = set(selected_ids)
    answer = getattr(result, "answer", None)
    grounded = tuple(getattr(answer, "requirement_coverage", ()) or ())
    answer_by_requirement = {item.requirement_id: item for item in grounded}
    cited_ids = tuple(getattr(answer, "citations", ()) or ())
    cited_id_set = set(cited_ids)

    for record in (*records, *selected):
        source = getattr(record, "evidence", None)
        paper_id = str(getattr(source, "paper_id", "") or "")
        title = getattr(source, "paper_title", None)
        if paper_id and isinstance(title, str) and title.strip():
            titles.setdefault(paper_id, " ".join(title.split()))

    plan = getattr(result, "plan", None)
    requirements = tuple(getattr(plan, "requirements", ()) or ())
    states = {item.requirement_id: item for item in tuple(getattr(result, "requirement_states", ()) or ())}
    requirement_audits = []
    for requirement in requirements:
        state = states.get(requirement.requirement_id)
        grounded_answer = answer_by_requirement.get(requirement.requirement_id)
        requirement_audits.append(RequirementAudit(
            requirement_id=requirement.requirement_id,
            description=requirement.description,
            required=requirement.required,
            status=_enum_text(getattr(state, "status", "pending")),
            evidence_ids=tuple(getattr(state, "evidence_ids", ()) or ()),
            remaining_gap=getattr(grounded_answer, "remaining_gap", None),
        ))

    observations = tuple(getattr(result, "observations", ()) or ())
    steps = tuple(_step_audit(item, titles) for item in observations)
    coverage = getattr(result, "coverage", None)
    assessments = tuple(getattr(coverage, "assessments", ()) or ())
    papers = tuple(_paper_audit(item, records, titles) for item in assessments)
    candidate_selection = getattr(result, "candidate_selection", None)
    candidates = tuple(
        CandidateAuditItem(
            item.paper_id,
            titles.get(item.paper_id, item.paper_id),
            _enum_text(item.disposition),
            item.reason,
        )
        for item in tuple(getattr(candidate_selection, "assessments", ()) or ())
    )
    evidence = tuple(
        _evidence_audit(record, titles, selected_id_set, cited_id_set)
        for record in records
    )
    calls = tuple(_llm_call(item) for item in tuple(getattr(result, "llm_diagnostics", ()) or ()))
    validation = tuple(
        _validation_error(item)
        for call in tuple(getattr(result, "llm_diagnostics", ()) or ())
        for item in tuple(getattr(call, "validation_diagnostics", ()) or ())
    )
    rejections = tuple(_rejection_audit(item) for item in tuple(getattr(result, "action_rejections", ()) or ()))
    requirement_coverage = tuple(
        {
            "requirement_id": item.requirement_id,
            "status": item.status,
            "evidence_ids": tuple(item.evidence_ids),
            "remaining_gap": item.remaining_gap,
        }
        for item in tuple(getattr(answer, "requirement_coverage", ()) or ())
    )
    return AgenticAuditSnapshot(
        schema_version=AUDIT_SCHEMA_VERSION,
        thread_id=getattr(result, "thread_id", None),
        run_status=_enum_text(getattr(result, "status", "failed")),
        stop_reason=_enum_text(getattr(result, "stop_reason", "fatal_error")),
        scope=_enum_text(getattr(getattr(result, "query_context", None), "scope", None), none=True),
        coverage_complete=(bool(getattr(coverage, "coverage_complete")) if coverage is not None else None),
        clarification=getattr(result, "clarification", None),
        candidates=candidates,
        papers=papers,
        steps=steps,
        evidence=evidence,
        budget=_budget_audit(getattr(result, "budget_usage", None)),
        llm_calls=calls,
        validation_errors=validation,
        requirements=tuple(requirement_audits),
        rejected_actions=rejections,
        error_codes=tuple(getattr(result, "error_codes", ()) or ()),
        planning_mode=_enum_text(getattr(plan, "planning_mode", None), none=True),
        selected_evidence_ids=selected_ids,
        cited_evidence_ids=cited_ids,
        answer_input_evidence_ids=tuple(
            getattr(result, "answer_input_evidence_ids", ()) or ()
        ),
        answer_status=getattr(answer, "status", None),
        answer_overall_status=getattr(answer, "overall_status", None),
        answer_requirement_coverage=requirement_coverage,
    )


def _step_audit(observation: object, titles: Mapping[str, str]) -> ResearchStepAudit:
    attempts = tuple(
        SearchAttemptAudit(
            item.paper_id,
            titles.get(item.paper_id, item.paper_id),
            item.query,
            tuple(item.lexical_terms),
            tuple(item.requirement_ids),
            _enum_text(item.status),
            item.error_code,
        )
        for item in tuple(getattr(observation, "search_attempts", ()) or ())
    )
    report = getattr(observation, "evidence_filter_report", None)
    counts: dict[str, int] = {}
    for decision in tuple(getattr(report, "decisions", ()) or ()):
        for reason in tuple(getattr(decision, "reasons", ()) or ()):
            counts[reason] = counts.get(reason, 0) + 1
    return ResearchStepAudit(
        iteration=int(getattr(observation, "iteration", 0)),
        action_name=str(getattr(observation, "action_name", "")),
        tool_status=_enum_text(getattr(observation, "tool_status", "")),
        added_count=len(tuple(getattr(observation, "added_evidence_ids", ()) or ())),
        reused_count=len(tuple(getattr(observation, "reused_evidence_ids", ()) or ())),
        attempts=attempts,
        action_arguments=dict(getattr(observation, "action_arguments", {}) or {}),
        requirement_ids=_requirement_ids(getattr(observation, "action_arguments", {}) or {}),
        error_codes=tuple(getattr(observation, "error_codes", ()) or ()),
        filter_total=int(getattr(report, "total_candidates", 0) or 0),
        filter_accepted=int(getattr(report, "accepted_candidates", 0) or 0),
        filter_rejected=int(getattr(report, "rejected_candidates", 0) or 0),
        filter_deprioritized=int(getattr(report, "deprioritized_candidates", 0) or 0),
        filter_reasons=tuple(FilterReasonAudit(key, value) for key, value in sorted(counts.items())),
        canonical_effective_requests=tuple(
            getattr(observation, "canonical_effective_requests", ()) or ()
        ),
        request_identities=tuple(getattr(observation, "request_identities", ()) or ()),
        request_executions=tuple(getattr(observation, "request_executions", ()) or ()),
        request_contract_version=str(
            getattr(observation, "request_contract_version", "legacy") or "legacy"
        ),
    )


def _paper_audit(item, records, titles) -> PaperResearchAudit:
    paper_records = tuple(record for record in records if record.evidence.paper_id == item.paper_id)
    targets = sorted({
        target.value
        for record in paper_records
        for target in record.requested_evidence_targets
    })
    sections = tuple(dict.fromkeys(
        record.evidence.section_id for record in paper_records if record.evidence.section_id
    ))
    return PaperResearchAudit(
        item.paper_id,
        titles.get(item.paper_id, item.paper_id),
        _enum_text(item.status),
        tuple(item.queries_tried),
        tuple(item.evidence_ids),
        item.rationale,
        tuple(targets),
        (),
        sections,
    )


def _evidence_audit(record, titles, selected_ids, cited_ids) -> EvidenceDigestAudit:
    source = record.evidence
    preview = " ".join(source.content.split())[:500]
    return EvidenceDigestAudit(
        record.evidence_id,
        source.paper_id,
        titles.get(source.paper_id, source.paper_title or source.paper_id),
        _enum_text(getattr(source, "source_type", "")),
        getattr(source, "section_id", None),
        getattr(source, "section", None),
        tuple(getattr(source, "source_chunk_ids", ()) or ()),
        tuple(getattr(record, "supporting_queries", ()) or ()),
        tuple(getattr(record, "requested_requirement_ids", ()) or ()),
        record.evidence_id in selected_ids,
        record.evidence_id in cited_ids,
        preview,
    )


def _llm_call(item: object) -> LLMCallAudit:
    return LLMCallAudit(
        node=str(getattr(item, "node", "unknown")),
        attempt=int(getattr(item, "attempt", 1)),
        success=bool(getattr(item, "success", False)),
        finish_reason=getattr(item, "finish_reason", None),
        prompt_tokens=getattr(item, "prompt_tokens", None),
        completion_tokens=getattr(item, "completion_tokens", None),
        content_chars=int(getattr(item, "content_chars", 0) or 0),
        reasoning_chars=int(getattr(item, "reasoning_chars", 0) or 0),
        error_stage=getattr(item, "error_stage", None),
        recovery_action=getattr(item, "recovery_action", None),
        decision_iteration=getattr(item, "decision_iteration", None),
        decision_view_metrics=getattr(item, "decision_view_metrics", None),
    )


def _validation_error(item: object) -> ValidationErrorAudit:
    return ValidationErrorAudit(**{
        name: getattr(item, name)
        for name in ValidationErrorAudit.__dataclass_fields__
    })


def _rejection_audit(item: object) -> RejectedActionAudit:
    return RejectedActionAudit(
        iteration=int(getattr(item, "iteration", 0)),
        action_name=str(getattr(item, "action_name", "")),
        action_arguments=dict(getattr(item, "action_arguments", {}) or {}),
        requirement_ids=tuple(getattr(item, "requirement_ids", ()) or ()),
        rejection_code=str(getattr(item, "rejection_code", "")),
        rejection_reason=str(getattr(item, "rejection_reason", "")),
        rejection_stage=str(getattr(item, "rejection_stage", "")),
        recovery_action=str(getattr(item, "recovery_action", "replan_action")),
        blockers=tuple(getattr(item, "blockers", ()) or ()),
    )


def _budget_audit(value: object | None) -> BudgetAudit:
    if value is None:
        return BudgetAudit()
    return BudgetAudit(**{
        name: getattr(value, name, field.default)
        for name, field in BudgetAudit.__dataclass_fields__.items()
    })


def _requirement_ids(arguments: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    direct = arguments.get("requirement_ids", ())
    if isinstance(direct, (list, tuple)):
        values.extend(item for item in direct if isinstance(item, str))
    searches = arguments.get("searches", ())
    if isinstance(searches, list):
        for search in searches:
            if isinstance(search, Mapping):
                ids = search.get("requirement_ids", ())
                if isinstance(ids, (list, tuple)):
                    values.extend(item for item in ids if isinstance(item, str))
    return tuple(dict.fromkeys(values))


def _enum_text(value: object, *, none: bool = False) -> str | None:
    if value is None:
        return None if none else ""
    return str(getattr(value, "value", value))


def _snapshot_from_dict(payload: dict[str, object]) -> AgenticAuditSnapshot:
    values = dict(payload)
    values["candidates"] = tuple(CandidateAuditItem(**item) for item in values.get("candidates", ()))
    values["papers"] = tuple(PaperResearchAudit(**_tupleize(item, {"queries_tried", "evidence_ids", "covered_targets", "remaining_targets", "sections_seen"})) for item in values.get("papers", ()))
    values["steps"] = tuple(_step_from_dict(item) for item in values.get("steps", ()))
    values["evidence"] = tuple(EvidenceDigestAudit(**_tupleize(item, {"source_chunk_ids", "supporting_queries", "requested_requirement_ids"})) for item in values.get("evidence", ()))
    values["budget"] = BudgetAudit(**_tupleize(values.get("budget", {}), {"queries_per_paper"}))
    values["llm_calls"] = tuple(LLMCallAudit(**item) for item in values.get("llm_calls", ()))
    values["validation_errors"] = tuple(ValidationErrorAudit(**item) for item in values.get("validation_errors", ()))
    requirement_rows = []
    for item in values.get("requirements", ()):
        row = _tupleize(item, {"evidence_ids"})
        requirement_rows.append(RequirementAudit(**row))
    values["requirements"] = tuple(requirement_rows)
    values["rejected_actions"] = tuple(RejectedActionAudit(**_tupleize(item, {"requirement_ids", "blockers"})) for item in values.get("rejected_actions", ()))
    for name in ("error_codes", "selected_evidence_ids", "cited_evidence_ids",
                 "answer_input_evidence_ids", "answer_requirement_coverage"):
        values[name] = tuple(values.get(name, ()))
    values["answer_requirement_coverage"] = tuple(
        _tupleize(item, {"evidence_ids"}) for item in values["answer_requirement_coverage"]
    )
    return AgenticAuditSnapshot(**values)


def _step_from_dict(item):
    values = _tupleize(item, {
        "requirement_ids", "error_codes", "filter_reasons", "attempts",
        "canonical_effective_requests", "request_identities", "request_executions",
    })
    values["attempts"] = tuple(SearchAttemptAudit(**_tupleize(value, {"lexical_terms", "requirement_ids"})) for value in values["attempts"])
    values["filter_reasons"] = tuple(FilterReasonAudit(**value) for value in values["filter_reasons"])
    return ResearchStepAudit(**values)


def _tupleize(item, names):
    values = dict(item)
    for name in names:
        values[name] = tuple(values.get(name, ()))
    return values
