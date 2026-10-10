"""Agentic Research 的安全、版本化用户审计快照。

这里的对象是运行时领域状态到产品展示层之间的稳定边界。快照只保存用户理解
研究过程所需的事实，不序列化 Prompt、隐藏推理、原始 LangGraph State、向量分数
或 chunk 内部标识。
"""

from __future__ import annotations

from .projection_reads import iter_projections

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import Mapping

from .grounding import verified_support_ids
from .requirement_coverage import derive_requirement_coverage
from .research_profile import build_research_profiles
from .requirement_state import (
    finalize_requirement_states, track_requirement_states,
)


AUDIT_SCHEMA_VERSION = 1


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
    status: str
    lexical_terms: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
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
    action_arguments: dict[str, object] = field(default_factory=dict)
    raw_action_arguments: dict[str, object] = field(default_factory=dict)
    canonical_effective_requests: tuple[dict[str, object], ...] = ()
    request_identities: tuple[str, ...] = ()
    request_executions: tuple[dict[str, object], ...] = ()
    request_contract_version: str = "legacy"
    requirement_target_bindings: tuple[dict[str, object], ...] = ()
    attempts: tuple[SearchAttemptAudit, ...] = ()
    filter_total: int = 0
    filter_accepted: int = 0
    filter_rejected: int = 0
    filter_deprioritized: int = 0
    filter_reasons: tuple[FilterReasonAudit, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_errors: int = 0
    error_codes: tuple[str, ...] = ()
    requirement_progress_before: tuple[tuple[str, str], ...] = ()
    requirement_progress_after: tuple[tuple[str, str], ...] = ()
    new_supporting_evidence_count: int = 0
    newly_supported_requirement_ids: tuple[str, ...] = ()
    newly_exposed_section_ids: tuple[str, ...] = ()
    action_canonicalization: dict[str, object] | None = None


@dataclass(frozen=True)
class RejectedActionAudit:
    iteration: int
    action_name: str
    action_arguments: dict[str, object]
    requirement_ids: tuple[str, ...]
    rejection_code: str
    rejection_reason: str
    guard_name: str
    recovery_action: str
    rejection_stage: str = "guard"
    blockers: tuple[dict[str, object], ...] = ()
    action_canonicalization: dict[str, object] | None = None


@dataclass(frozen=True)
class PaperResearchAudit:
    paper_id: str
    title: str
    status: str
    rationale: str
    queries_tried: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    requested_targets: tuple[str, ...] = ()
    covered_targets: tuple[str, ...] = ()
    remaining_targets: tuple[str, ...] = ()
    requested_requirement_ids: tuple[str, ...] = ()
    covered_requirement_ids: tuple[str, ...] = ()
    remaining_requirement_ids: tuple[str, ...] = ()
    requested_information: tuple[str, ...] = ()
    addressed_information: tuple[str, ...] = ()
    remaining_information: tuple[str, ...] = ()
    sections_seen: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimAudit:
    evidence_id: str
    claim_id: str
    paper_id: str
    paper_title: str
    statement: str
    relation: str
    ownership: str
    rationale: str
    source_excerpt: str


@dataclass(frozen=True)
class EvidenceDigestAudit:
    evidence_id: str
    paper_id: str
    paper_title: str
    digest: str | None
    analysis_status: str
    selected: bool
    cited: bool
    section: str | None
    page_start: int | None
    page_end: int | None
    covered_targets: tuple[str, ...] = ()
    covered_requirement_ids: tuple[str, ...] = ()
    method_entities: tuple[str, ...] = ()
    method_families: tuple[str, ...] = ()
    baseline_entities: tuple[str, ...] = ()
    addressed_information: tuple[str, ...] = ()
    projection_type: str | None = None
    structured_projection: dict[str, object] = field(default_factory=dict)
    structured_projections: tuple[dict[str, object], ...] = ()
    evaluated_requirement_ids: tuple[str, ...] = ()
    projection_rejections: tuple[str, ...] = ()
    supports_requirement_ids: tuple[str, ...] = ()
    proposed_supports_requirement_ids: tuple[str, ...] = ()
    verified_supports_requirement_ids: tuple[str, ...] = ()
    selected_entity_or_fact_gain: tuple[str, ...] = ()
    selection_reason: str | None = None
    claims: tuple[ClaimAudit, ...] = ()


@dataclass(frozen=True)
class LLMCallAudit:
    node: str
    attempt: int
    success: bool
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_stage: str | None = None
    recovery_action: str | None = None
    validation_summary: str | None = None
    projector_input_size: int | None = None
    requested_output_tokens: int | None = None
    projector_call_mode: str | None = None
    projector_repair: bool = False
    projector_fallback_reason: str | None = None
    decision_iteration: int | None = None
    decision_view_metrics: dict[str, int | float] | None = None
    projector_pre_parse_summary: dict[str, object] | None = None
    projector_post_conversion_summary: dict[str, object] | None = None


@dataclass(frozen=True)
class ValidationErrorAudit:
    node: str
    attempt: int
    stage: str
    code: str
    field_path: str
    validator_name: str
    safe_reason: str
    expected_constraint: str
    actual_summary: str
    recovery_action: str | None = None
    rule_code: str | None = None
    requirement_id: str | None = None


@dataclass(frozen=True)
class BudgetAudit:
    agent_iterations: int = 0
    tool_calls: int = 0
    retrieval_queries: int = 0
    llm_calls: int = 0
    invalid_actions: int = 0


@dataclass(frozen=True)
class RequirementAudit:
    """User-facing requirement definition and its evidence-backed state."""

    requirement_id: str
    description: str
    required: bool
    status: str
    evidence_ids: tuple[str, ...] = ()
    reason: str | None = None
    critical_anchors: tuple[str, ...] = ()
    covered_critical_anchors: tuple[str, ...] = ()
    missing_critical_anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgenticAuditSnapshot:
    """可以稳定持久化、恢复和导出的 Agentic 用户审计视图。"""

    schema_version: int
    run_status: str
    stop_reason: str
    thread_id: str | None
    scope: str | None
    coverage_complete: bool | None
    candidates: tuple[CandidateAuditItem, ...]
    papers: tuple[PaperResearchAudit, ...]
    steps: tuple[ResearchStepAudit, ...]
    evidence: tuple[EvidenceDigestAudit, ...]
    budget: BudgetAudit
    llm_calls: tuple[LLMCallAudit, ...]
    validation_errors: tuple[ValidationErrorAudit, ...] = ()
    # Additive field kept optional so audit schema v1 snapshots made before
    # RequirementState can still be opened.
    requirements: tuple[RequirementAudit, ...] = ()
    rejected_actions: tuple[RejectedActionAudit, ...] = ()
    error_codes: tuple[str, ...] = ()
    clarification: str | None = None
    planning_mode: str | None = None
    overall_answer_status: str | None = None
    answer_requirement_coverage: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """转换为仅含 JSON 基础类型的字典。"""
        payload = asdict(self)
        # v1 readers still accept historical fields, but new snapshots do not
        # write empty aliases of the formal Requirement audit.
        for paper in payload["papers"]:
            for name in (
                "requested_targets", "requested_requirement_ids", "covered_requirement_ids",
                "remaining_requirement_ids", "requested_information",
                "addressed_information", "remaining_information",
            ):
                if not paper[name]:
                    paper.pop(name)
        for record in payload["evidence"]:
            for name in (
                "covered_requirement_ids", "supports_requirement_ids",
                "addressed_information",
            ):
                if not record[name]:
                    record.pop(name)
        payload["projector_metrics"] = projector_cost_metrics(self.llm_calls, self.evidence)
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        """使用 UTF-8 友好的稳定 JSON 表示。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "AgenticAuditSnapshot":
        """从持久化 JSON 恢复；结构损坏时显式失败，由调用层安全降级。"""
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Agentic audit snapshot must be a JSON object.")
        if payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise ValueError("Unsupported Agentic audit schema version.")
        try:
            return cls(
                schema_version=AUDIT_SCHEMA_VERSION,
                run_status=str(payload["run_status"]),
                stop_reason=str(payload["stop_reason"]),
                thread_id=_optional_string(payload.get("thread_id")),
                scope=_optional_string(payload.get("scope")),
                coverage_complete=_optional_bool(payload.get("coverage_complete")),
                candidates=tuple(
                    CandidateAuditItem(**item)
                    for item in _object_list(payload.get("candidates"))
                ),
                papers=tuple(
                    PaperResearchAudit(**_tuple_fields(item, _PAPER_TUPLE_FIELDS))
                    for item in _object_list(payload.get("papers"))
                ),
                steps=tuple(
                    _step_from_payload(item)
                    for item in _object_list(payload.get("steps"))
                ),
                evidence=tuple(
                    _evidence_from_payload(item)
                    for item in _object_list(payload.get("evidence"))
                ),
                budget=BudgetAudit(**_object(payload.get("budget"))),
                llm_calls=tuple(
                    LLMCallAudit(**item)
                    for item in _object_list(payload.get("llm_calls"))
                ),
                validation_errors=tuple(
                    ValidationErrorAudit(**item)
                    for item in _object_list(payload.get("validation_errors"))
                ),
                requirements=tuple(
                    _requirement_from_payload(item)
                    for item in _object_list(payload.get("requirements"))
                ),
                rejected_actions=tuple(
                    _rejected_action_from_payload(item)
                    for item in _object_list(payload.get("rejected_actions"))
                ),
                error_codes=_strings(payload.get("error_codes")),
                clarification=_optional_string(payload.get("clarification")),
                planning_mode=_optional_string(payload.get("planning_mode")),
                overall_answer_status=_optional_string(payload.get("overall_answer_status")),
                answer_requirement_coverage=tuple(_object_list(payload.get("answer_requirement_coverage"))),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid Agentic audit snapshot structure.") from error


def build_agentic_audit_snapshot(
    result: object,
    *,
    paper_titles: Mapping[str, str] | None = None,
) -> AgenticAuditSnapshot:
    """把 ResearchRunResult 投影为安全审计快照。

    接受 ``object`` 是为了兼容现有产品测试中的轻量替身；缺失字段按空值处理，
    真实运行结果仍会完整投影。
    """
    titles = dict(paper_titles or {})
    records = tuple(getattr(result, "evidence", ()) or ())
    selected = tuple(getattr(result, "selected_evidence", ()) or ())
    answer = getattr(result, "answer", None)
    selected_ids = {str(getattr(item, "evidence_id", "")) for item in selected}
    cited_ids = set(getattr(answer, "citations", ()) or ())
    for record in (*records, *selected):
        source = getattr(record, "evidence", None)
        paper_id = str(getattr(source, "paper_id", "") or "")
        title = getattr(source, "paper_title", None)
        if paper_id and isinstance(title, str) and title.strip():
            titles.setdefault(paper_id, " ".join(title.split()))

    candidate_items = _candidate_items(result, titles)
    observations = tuple(getattr(result, "observations", ()) or ())
    plan = getattr(result, "plan", None)
    requirements = tuple(getattr(plan, "requirements", ()) or ())
    selection_details = _selection_details(selected, requirements)
    coverage = getattr(result, "coverage", None)
    assessments = tuple(getattr(coverage, "assessments", ()) or ())
    query_context = getattr(result, "query_context", None)
    profiles_by_id: dict[str, object] = {}
    if plan is not None:
        paper_ids = tuple(
            str(getattr(item, "paper_id", ""))
            for item in assessments
            if getattr(item, "paper_id", None)
        )
        profiles_by_id = {
            item.paper_id: item
            for item in build_research_profiles(
                paper_ids=paper_ids,
                plan=plan,
                evidence=records,
                observations=observations,
                default_targets=frozenset(
                    getattr(query_context, "default_evidence_targets", ()) or ()
                ),
            )
        }
    papers = tuple(
        _paper_audit(item, profiles_by_id.get(str(getattr(item, "paper_id", ""))), titles)
        for item in assessments
    )
    known_paper_ids = {item.paper_id for item in papers}
    for candidate in candidate_items:
        if candidate.paper_id not in known_paper_ids:
            papers += (
                PaperResearchAudit(
                    paper_id=candidate.paper_id,
                    title=candidate.title,
                    status=candidate.disposition,
                    rationale=candidate.reason,
                ),
            )

    diagnostic_records = tuple(getattr(result, "llm_diagnostics", ()) or ())
    requirement_states = tuple(getattr(result, "requirement_states", ()) or ())
    required_requirement_ids = {
        str(getattr(requirement, "requirement_id", ""))
        for requirement in tuple(getattr(plan, "requirements", ()) or ())
        if bool(getattr(requirement, "required", True))
    }
    requirement_reasons: dict[str, str] = {}
    if plan is not None and not requirement_states:
        tracking = track_requirement_states(
            plan=plan,
            evidence=records,
            observations=observations,
            context=query_context,
            coverage=coverage,
        )
        requirement_states = tracking.states
        requirement_reasons = dict(tracking.reasons)
    elif plan is not None:
        finalized = finalize_requirement_states(
            plan=plan, evidence=records, observations=observations,
            research_complete=_enum_text(getattr(result, "stop_reason", "")) != "budget_exhausted",
        )
        finalized_by_id = {item.requirement_id: item for item in finalized.states}
        final_reasons = dict(finalized.reasons)
        progress_reasons = dict(track_requirement_states(
            plan=plan, evidence=records, observations=observations,
            context=query_context, coverage=coverage,
        ).reasons)
        for state_item in requirement_states:
            requirement_id = state_item.requirement_id
            if (
                requirement_id in finalized_by_id
                and state_item == finalized_by_id[requirement_id]
            ):
                requirement_reasons[requirement_id] = final_reasons[requirement_id]
            else:
                requirement_reasons[requirement_id] = progress_reasons.get(requirement_id, "state preserved from checkpoint")
    # 完整 LangGraph 结果已把 Answer 节点诊断汇总到 result；轻量替身或旧结果
    # 可能只在 answer 中携带，因此仅在顶层为空时回退，避免重复展示同一次调用。
    if not diagnostic_records and answer is not None:
        diagnostic_records = tuple(getattr(answer, "llm_diagnostics", ()) or ())
    llm_calls = tuple(_llm_call(item) for item in diagnostic_records)
    validation_errors = tuple(
        _validation_error(item)
        for record in diagnostic_records
        for item in tuple(getattr(record, "validation_diagnostics", ()) or ())
    )

    return AgenticAuditSnapshot(
        schema_version=AUDIT_SCHEMA_VERSION,
        run_status=_enum_text(getattr(result, "status", "unknown")),
        stop_reason=_enum_text(getattr(result, "stop_reason", "unknown")),
        thread_id=_optional_string(getattr(result, "thread_id", None)),
        scope=(
            _enum_text(getattr(query_context, "scope"))
            if query_context is not None and getattr(query_context, "scope", None) is not None
            else None
        ),
        coverage_complete=(
            bool(getattr(coverage, "coverage_complete")) if coverage is not None else None
        ),
        candidates=candidate_items,
        papers=papers,
        steps=tuple(_step_audit(item, titles) for item in observations),
        evidence=tuple(
            _evidence_audit(
                item,
                selected_ids,
                cited_ids,
                titles,
                selection_details,
                required_requirement_ids,
                requirements,
            )
            for item in records
        ),
        budget=_budget_audit(getattr(result, "budget_usage", None)),
        llm_calls=llm_calls,
        validation_errors=validation_errors,
        planning_mode=(
            _enum_text(getattr(plan, "planning_mode"))
            if plan is not None and getattr(plan, "planning_mode", None) is not None
            else None
        ),
        requirements=_requirement_audits(plan, requirement_states, requirement_reasons, tuple(getattr(result, "evidence", ()) or ())),
        rejected_actions=tuple(
            RejectedActionAudit(
                iteration=int(getattr(item, "iteration", 0)),
                action_name=str(getattr(item, "action_name", "unknown")),
                action_arguments=dict(getattr(item, "action_arguments", {}) or {}),
                requirement_ids=_strings(getattr(item, "requirement_ids", ())),
                rejection_code=str(getattr(item, "rejection_code", "unknown")),
                rejection_reason=str(getattr(item, "rejection_reason", "")),
                guard_name=str(getattr(item, "guard_name", "unknown")),
                recovery_action=str(getattr(item, "recovery_action", "replan_action")),
                rejection_stage=str(getattr(item, "rejection_stage", "guard")),
                blockers=_blocker_objects(getattr(item, "blockers", ())),
                action_canonicalization=_canonicalization_payload(
                    getattr(item, "action_canonicalization", None)
                ),
            )
            for item in tuple(
                getattr(result, "action_rejections", ())
                or getattr(result, "guard_rejections", ())
                or ()
            )
        ),
        error_codes=_strings(getattr(result, "error_codes", ())),
        clarification=_optional_string(getattr(result, "clarification", None)),
        overall_answer_status=_optional_string(getattr(answer, "overall_status", None)),
        answer_requirement_coverage=tuple(
            {"requirement_id": item.requirement_id, "status": item.status,
             "evidence_ids": list(item.evidence_ids)}
            for item in tuple(getattr(answer, "requirement_coverage", ()) or ())
        ),
    )


def _candidate_items(
    result: object, titles: Mapping[str, str]
) -> tuple[CandidateAuditItem, ...]:
    selection = getattr(result, "candidate_selection", None)
    return tuple(
        CandidateAuditItem(
            paper_id=str(item.paper_id),
            title=titles.get(str(item.paper_id), str(item.paper_id)),
            disposition=_enum_text(item.disposition),
            reason=str(item.reason),
        )
        for item in tuple(getattr(selection, "assessments", ()) or ())
    )


def _paper_audit(
    assessment: object,
    profile: object | None,
    titles: Mapping[str, str],
) -> PaperResearchAudit:
    paper_id = str(getattr(assessment, "paper_id", ""))
    return PaperResearchAudit(
        paper_id=paper_id,
        title=titles.get(paper_id, paper_id),
        status=_enum_text(getattr(assessment, "status", "unknown")),
        rationale=str(getattr(assessment, "rationale", "") or ""),
        queries_tried=_strings(getattr(assessment, "queries_tried", ())),
        evidence_ids=_strings(getattr(assessment, "evidence_ids", ())),
        covered_targets=_enum_strings(getattr(profile, "covered_targets", ())),
        remaining_targets=_enum_strings(getattr(profile, "remaining_targets", ())),
        sections_seen=_strings(getattr(profile, "sections_seen", ())),
    )


def _step_audit(observation: object, titles: Mapping[str, str]) -> ResearchStepAudit:
    report = getattr(observation, "evidence_filter_report", None)
    cache = getattr(observation, "evidence_analysis_cache", None)
    attempts = tuple(
        SearchAttemptAudit(
            paper_id=str(getattr(item, "paper_id", "")),
            title=titles.get(
                str(getattr(item, "paper_id", "")),
                str(getattr(item, "paper_id", "")),
            ),
            query=str(getattr(item, "query", "")),
            status=str(getattr(item, "status", "") or "unknown"),
            lexical_terms=_strings(getattr(item, "lexical_terms", ())),
            requirement_ids=_strings(getattr(item, "requirement_ids", ())),
            error_code=_optional_string(getattr(item, "error_code", None)),
        )
        for item in tuple(getattr(observation, "search_attempts", ()) or ())
    )
    reason_counts = Counter(
        str(reason)
        for decision in tuple(getattr(report, "decisions", ()) or ())
        for reason in tuple(getattr(decision, "reasons", ()) or ())
        if str(reason).strip()
    )
    return ResearchStepAudit(
        iteration=int(getattr(observation, "iteration", 0) or 0),
        action_name=str(getattr(observation, "action_name", "unknown")),
        tool_status=_enum_text(getattr(observation, "tool_status", "unknown")),
        added_count=len(tuple(getattr(observation, "added_evidence_ids", ()) or ())),
        reused_count=len(tuple(getattr(observation, "reused_evidence_ids", ()) or ())),
        action_arguments=dict(getattr(observation, "action_arguments", {}) or {}),
        raw_action_arguments=dict(getattr(observation, "raw_action_arguments", {}) or {}),
        canonical_effective_requests=tuple(getattr(observation, "canonical_effective_requests", ()) or ()),
        request_identities=tuple(getattr(observation, "request_identities", ()) or ()),
        request_executions=tuple(getattr(observation, "request_executions", ()) or ()),
        request_contract_version=getattr(observation, "request_contract_version", "legacy"),
        requirement_target_bindings=tuple(deepcopy(getattr(observation, "requirement_target_bindings", ()) or ())),
        attempts=attempts,
        filter_total=int(getattr(report, "total_candidates", 0) or 0),
        filter_accepted=int(getattr(report, "accepted_candidates", 0) or 0),
        filter_rejected=int(getattr(report, "rejected_candidates", 0) or 0),
        filter_deprioritized=int(getattr(report, "deprioritized_candidates", 0) or 0),
        filter_reasons=tuple(
            FilterReasonAudit(reason=reason, count=count)
            for reason, count in sorted(reason_counts.items())
        ),
        cache_hits=int(getattr(cache, "hits", 0) or 0),
        cache_misses=int(getattr(cache, "misses", 0) or 0),
        cache_writes=int(getattr(cache, "writes", 0) or 0),
        cache_errors=int(getattr(cache, "errors", 0) or 0),
        error_codes=_strings(getattr(observation, "error_codes", ())),
        requirement_progress_before=tuple(getattr(observation, "requirement_progress_before", ()) or ()),
        requirement_progress_after=tuple(getattr(observation, "requirement_progress_after", ()) or ()),
        new_supporting_evidence_count=int(getattr(observation, "new_supporting_evidence_count", 0) or 0),
        newly_supported_requirement_ids=_strings(getattr(observation, "newly_supported_requirement_ids", ())),
        newly_exposed_section_ids=_strings(getattr(observation, "newly_exposed_section_ids", ())),
        action_canonicalization=_canonicalization_payload(
            getattr(observation, "action_canonicalization", None)
        ),
    )


def _evidence_audit(
    record: object,
    selected_ids: set[str],
    cited_ids: set[str],
    titles: Mapping[str, str],
    selection_details: Mapping[str, tuple[tuple[str, ...], str | None]],
    required_requirement_ids: set[str],
    requirements: tuple[object, ...],
) -> EvidenceDigestAudit:
    evidence_id = str(getattr(record, "evidence_id", ""))
    source = getattr(record, "evidence", None)
    projections = iter_projections(record)
    paper_id = str(getattr(source, "paper_id", ""))
    title = titles.get(paper_id, paper_id)
    claims = tuple(
        ClaimAudit(
            evidence_id=evidence_id,
            claim_id=str(getattr(item, "claim_id", "")),
            paper_id=paper_id,
            paper_title=title,
            statement=str(getattr(item, "statement", "")),
            relation=_enum_text(getattr(item, "relation", "unknown")),
            ownership=_enum_text(getattr(item, "ownership", "unknown")),
            rationale=str(getattr(item, "rationale", "")),
            source_excerpt=str(getattr(item, "source_excerpt", "")),
        )
        for item in tuple(getattr(record, "claims", ()) or ())
    )
    return EvidenceDigestAudit(
        evidence_id=evidence_id,
        paper_id=paper_id,
        paper_title=title,
        digest=_optional_string(getattr(record, "digest", None)),
        analysis_status=_enum_text(getattr(record, "analysis_status", "unknown")),
        selected=evidence_id in selected_ids,
        cited=evidence_id in cited_ids,
        section=_optional_string(getattr(source, "section", None)),
        page_start=_optional_int(getattr(source, "page_start", None)),
        page_end=_optional_int(getattr(source, "page_end", None)),
        covered_targets=_enum_strings(getattr(record, "covered_targets", ())),
        method_entities=_strings(getattr(record, "method_entities", ())),
        method_families=_strings(getattr(record, "method_families", ())),
        baseline_entities=_strings(getattr(record, "baseline_entities", ())),
        projection_type=_projection_type(projections[0]) if len(projections) == 1 else None,
        structured_projection=_projection_payload(projections[0]) if len(projections) == 1 else {},
        structured_projections=tuple(dict(_projection_payload(p), type=_projection_type(p)) for p in iter_projections(record)),
        evaluated_requirement_ids=tuple(getattr(getattr(record, "assessment", None), "evaluated_requirement_ids", ())),
        projection_rejections=_strings(getattr(record, "projection_rejections", ())),
        proposed_supports_requirement_ids=_strings(
            tuple(dict.fromkeys(rid for p in iter_projections(record) for rid in getattr(p, "proposed_requirement_ids", ())))
        ),
        verified_supports_requirement_ids=_supported_requirement_ids(record, requirements),
        selected_entity_or_fact_gain=selection_details.get(evidence_id, ((), None))[0],
        selection_reason=_selection_reason(
            evidence_id,
            record,
            selection_details,
            required_requirement_ids,
            requirements,
        ),
        claims=claims,
    )


def _selection_reason(
    evidence_id: str,
    record: object,
    selection_details: Mapping[str, tuple[tuple[str, ...], str | None]],
    required_requirement_ids: set[str],
    requirements: tuple[object, ...],
) -> str | None:
    detail = selection_details.get(evidence_id)
    if detail is not None:
        return detail[1]
    if not set(_supported_requirement_ids(record, requirements)).intersection(required_requirement_ids):
        return "no_required_requirement_support"
    return "no_new_requirement_or_entity_gain"


def _supported_requirement_ids(
    record: object, requirements: tuple[object, ...] = ()
) -> tuple[str, ...]:
    return verified_support_ids(record, requirements)


def _selection_details(
    selected: tuple[object, ...],
    requirements: tuple[object, ...] = (),
) -> dict[str, tuple[tuple[str, ...], str]]:
    """Deterministic selector audit: no extra semantic/LLM evaluation."""
    seen: set[str] = set()
    details: dict[str, tuple[tuple[str, ...], str]] = {}
    for record in selected:
        evidence_id = str(getattr(record, "evidence_id", ""))
        names = _projection_entity_names(record)
        gain = tuple(name for name in names if _selection_entity_key(name) not in seen)
        seen.update(_selection_entity_key(name) for name in names)
        supported = _supported_requirement_ids(record, requirements)
        reasons = []
        if supported:
            reasons.append("requirement_coverage:" + ",".join(supported))
        if gain:
            reasons.append("entity_or_fact_gain:" + ",".join(gain))
        details[evidence_id] = (gain, "; ".join(reasons) or "quality_tiebreak")
    return details


def _projection_entity_names(record: object) -> tuple[str, ...]:
    names = []
    for projection in iter_projections(record):
        names.extend(str(getattr(item, "canonical_name", "")) for item in getattr(projection, "entities", ()))
        names.extend(f"{fact.fact_type}: {fact.value}" for fact in getattr(projection, "facts", ()))
        if hasattr(projection, "mechanism"):
            names.append(str(projection.mechanism))
    if names:
        return tuple(dict.fromkeys(names))
    return tuple(str(item) for item in (*tuple(getattr(record, "method_entities", ()) or ()),
        *tuple(getattr(record, "method_families", ()) or ()), *tuple(getattr(record, "baseline_entities", ()) or ())))


def _selection_entity_key(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _budget_audit(value: object | None) -> BudgetAudit:
    return BudgetAudit(
        agent_iterations=int(getattr(value, "agent_iterations", 0) or 0),
        tool_calls=int(getattr(value, "tool_calls", 0) or 0),
        retrieval_queries=int(getattr(value, "retrieval_queries", 0) or 0),
        llm_calls=int(getattr(value, "llm_calls", 0) or 0),
        invalid_actions=int(getattr(value, "invalid_actions", 0) or 0),
    )


def _requirement_audits(
    plan: object | None, states: tuple[object, ...], reasons: Mapping[str, str] | None = None,
    evidence: tuple[object, ...] = (),
) -> tuple[RequirementAudit, ...]:
    if plan is None:
        return ()
    states_by_id = {
        str(getattr(item, "requirement_id", "")): item
        for item in states
        if getattr(item, "requirement_id", None)
    }
    return tuple(
        RequirementAudit(
            requirement_id=str(getattr(requirement, "requirement_id", "")),
            description=str(getattr(requirement, "description", "")),
            required=bool(getattr(requirement, "required", True)),
            status=_enum_text(
                getattr(
                    states_by_id.get(str(getattr(requirement, "requirement_id", ""))),
                    "status",
                    "pending",
                )
            ),
            evidence_ids=_strings(
                getattr(
                    states_by_id.get(str(getattr(requirement, "requirement_id", ""))),
                    "evidence_ids",
                    (),
                )
            ),
            reason=(reasons or {}).get(str(getattr(requirement, "requirement_id", ""))),
            critical_anchors=derive_requirement_coverage(requirement, evidence).critical_anchors,
            covered_critical_anchors=derive_requirement_coverage(requirement, evidence).covered_anchors,
            missing_critical_anchors=derive_requirement_coverage(requirement, evidence).missing_anchors,
        )
        for requirement in tuple(getattr(plan, "requirements", ()) or ())
    )


def _llm_call(item: object) -> LLMCallAudit:
    return LLMCallAudit(
        node=str(getattr(item, "node", "unknown")),
        attempt=int(getattr(item, "attempt", 1) or 1),
        success=bool(getattr(item, "success", False)),
        finish_reason=_optional_string(getattr(item, "finish_reason", None)),
        prompt_tokens=_optional_int(getattr(item, "prompt_tokens", None)),
        completion_tokens=_optional_int(getattr(item, "completion_tokens", None)),
        error_stage=_optional_string(getattr(item, "error_stage", None)),
        recovery_action=_optional_string(getattr(item, "recovery_action", None)),
        validation_summary=_optional_string(getattr(item, "validation_summary", None)),
        projector_input_size=getattr(item, "projector_input_size", None),
        requested_output_tokens=getattr(item, "requested_output_tokens", None),
        projector_call_mode=getattr(item, "projector_call_mode", None),
        projector_repair=bool(getattr(item, "projector_repair", False)),
        projector_fallback_reason=getattr(item, "projector_fallback_reason", None),
        decision_iteration=getattr(item, "decision_iteration", None),
        decision_view_metrics=deepcopy(getattr(item, "decision_view_metrics", None)),
        projector_pre_parse_summary=deepcopy(getattr(item, "projector_pre_parse_summary", None)),
        projector_post_conversion_summary=deepcopy(getattr(item, "projector_post_conversion_summary", None)),
    )


def projector_cost_metrics(calls, evidence=()) -> dict[str, object]:
    """Costs include failed/repair calls; batch size counts initial packs only."""
    calls = [item for item in calls if item.node == "project_evidence"]
    batches = [item for item in calls if item.projector_call_mode == "batch" and not item.projector_repair]
    initial = [item for item in calls if item.projector_call_mode in {"batch", "single"} and not item.projector_repair]
    def tokens(name):
        if any(getattr(item, name) is None for item in calls):
            return None
        return sum(getattr(item, name) for item in calls)
    prompt, completion = tokens("prompt_tokens"), tokens("completion_tokens")
    sizes = [item.projector_input_size or 0 for item in batches]
    count = sum(item.projector_input_size or 0 for item in initial)
    return {
        "projector_evidence_count": count,
        "projector_llm_calls_total": len(calls),
        "projector_batch_calls": sum(item.projector_call_mode == "batch" for item in calls),
        "projector_single_calls": sum(item.projector_call_mode in {"single", "fallback_single"} for item in calls),
        "projector_repair_calls": sum(item.projector_repair for item in calls),
        "projector_fallback_single_calls": sum(item.projector_call_mode == "fallback_single" for item in calls),
        "projector_batch_count": len(batches),
        "avg_projector_batch_size": sum(sizes) / len(sizes) if sizes else 0,
        "max_projector_batch_size": max(sizes, default=0),
        "projector_prompt_tokens": prompt,
        "projector_completion_tokens": completion,
        "projector_total_tokens": prompt + completion if prompt is not None and completion is not None else None,
        "avg_evidence_per_projector_call": count / len(calls) if calls else 0,
        "projector_failed_evidence_count": sum(getattr(item, "analysis_status", None) == "failed" for item in evidence),
    }


def _validation_error(item: object) -> ValidationErrorAudit:
    node = str(getattr(item, "node", "unknown"))
    stage = str(getattr(item, "error_stage", "unknown"))
    code = str(getattr(item, "error_code", "UnknownValidationError"))
    return ValidationErrorAudit(
        node=node,
        attempt=int(getattr(item, "attempt", 1) or 1),
        stage=stage,
        code=code,
        field_path=str(getattr(item, "field_path", "root")),
        validator_name=str(getattr(item, "validator_name", "unknown")),
        safe_reason=str(getattr(item, "safe_reason", "Validation failed.")),
        expected_constraint=str(getattr(item, "expected_constraint", "unknown")),
        actual_summary=str(getattr(item, "actual_summary", "unknown")),
        recovery_action=_optional_string(getattr(item, "recovery_action", None)),
        rule_code=(
            code if node in {"build_plan", "generate_answer"}
            and stage == "domain_validation" else None
        ),
        requirement_id=_optional_string(getattr(item, "requirement_id", None)),
    )


def _canonicalization_payload(item: object | None) -> dict[str, object] | None:
    if item is None or not bool(getattr(item, "changed", False)):
        return None
    operations = [
        {
            "field_path": str(getattr(operation, "field_path", "")),
            "operation": str(getattr(operation, "operation", "")),
            "count": int(getattr(operation, "count", 0) or 0),
        }
        for operation in tuple(getattr(item, "operations", ()) or ())
    ]
    return {
        "changed": True,
        "changed_fields": list(getattr(item, "changed_fields", ()) or ()),
        "operations": operations,
    }


_PAPER_TUPLE_FIELDS = {
    "queries_tried",
    "evidence_ids",
    "requested_targets",
    "covered_targets",
    "remaining_targets",
    "requested_requirement_ids",
    "covered_requirement_ids",
    "remaining_requirement_ids",
    "requested_information",
    "addressed_information",
    "remaining_information",
    "sections_seen",
}


def _step_from_payload(item: dict[str, object]) -> ResearchStepAudit:
    values = dict(item)
    values["attempts"] = tuple(
        _attempt_from_payload(attempt)
        for attempt in _object_list(values.get("attempts"))
    )
    values["filter_reasons"] = tuple(
        FilterReasonAudit(**reason)
        for reason in _object_list(values.get("filter_reasons"))
    )
    values["error_codes"] = _strings(values.get("error_codes"))
    raw_arguments = values.get("action_arguments", {})
    if not isinstance(raw_arguments, dict):
        raise ValueError("Expected action_arguments to be a JSON object.")
    values["action_arguments"] = dict(raw_arguments)
    canonicalization = values.get("action_canonicalization")
    if canonicalization is not None and not isinstance(canonicalization, dict):
        raise ValueError("Expected action_canonicalization to be a JSON object.")
    values["action_canonicalization"] = (
        dict(canonicalization) if isinstance(canonicalization, dict) else None
    )
    for name in ("requirement_progress_before", "requirement_progress_after"):
        values[name] = tuple(tuple(pair) for pair in values.get(name, ()))
    for name in ("newly_supported_requirement_ids", "newly_exposed_section_ids"):
        values[name] = _strings(values.get(name, ()))
    for key in ("canonical_effective_requests", "request_identities", "request_executions", "requirement_target_bindings"):
        values[key] = tuple(values.get(key, ()))
    return ResearchStepAudit(**values)


def _rejected_action_from_payload(item: dict[str, object]) -> RejectedActionAudit:
    values = dict(item)
    values["requirement_ids"] = _strings(values.get("requirement_ids"))
    raw_arguments = values.get("action_arguments", {})
    if not isinstance(raw_arguments, dict):
        raise ValueError("Expected rejected action_arguments to be a JSON object.")
    values["action_arguments"] = dict(raw_arguments)
    values["blockers"] = _blocker_objects(values.get("blockers", ()))
    canonicalization = values.get("action_canonicalization")
    if canonicalization is not None and not isinstance(canonicalization, dict):
        raise ValueError("Expected action_canonicalization to be a JSON object.")
    values["action_canonicalization"] = (
        dict(canonicalization) if isinstance(canonicalization, dict) else None
    )
    return RejectedActionAudit(**values)


def _evidence_from_payload(item: dict[str, object]) -> EvidenceDigestAudit:
    values = _tuple_fields(
        item,
        {
            "covered_targets",
            "covered_requirement_ids",
            "method_entities",
            "method_families",
            "baseline_entities",
            "addressed_information",
            "projection_rejections",
            "supports_requirement_ids",
            "proposed_supports_requirement_ids",
            "verified_supports_requirement_ids",
            "selected_entity_or_fact_gain",
            "evaluated_requirement_ids",
        },
    )
    values["claims"] = tuple(
        ClaimAudit(**claim) for claim in _object_list(values.get("claims"))
    )
    projection = values.get("structured_projection", {})
    if not isinstance(projection, dict):
        raise ValueError("Expected structured_projection to be a JSON object.")
    values["structured_projection"] = dict(projection)
    multi = values.get("structured_projections", ())
    if not isinstance(multi, (list, tuple)) or any(not isinstance(p, dict) for p in multi):
        raise ValueError("Expected structured_projections to be JSON objects.")
    values["structured_projections"] = tuple(dict(p) for p in multi)
    return EvidenceDigestAudit(**values)


def _projection_type(projection: object | None) -> str | None:
    if projection is None:
        return None
    return _enum_text(getattr(projection, "projection_type", "unknown"))


def _projection_payload(projection: object | None) -> dict[str, object]:
    if projection is None:
        return {}
    payload: dict[str, object] = {
        "supported_requirement_ids": list(
            getattr(projection, "supported_requirement_ids", ())
        )
    }
    if hasattr(projection, "proposed_requirement_ids"):
        payload["proposed_requirement_ids"] = list(projection.proposed_requirement_ids)
    if hasattr(projection, "support_anchors"):
        payload["support_anchors"] = [
            {
                "requirement_id": item.requirement_id,
                "source_excerpt": item.source_excerpt,
                "support_scope": item.support_scope,
                "unresolved_aspects": list(item.unresolved_aspects),
            }
            for item in projection.support_anchors
        ]
    if hasattr(projection, "entities"):
        payload["entities"] = [
            {
                "canonical_name": str(getattr(item, "canonical_name", "")),
                "entity_type": str(getattr(item, "entity_type", "")),
                "role": _enum_text(getattr(item, "role", "unknown")),
                "ownership": _enum_text(getattr(item, "ownership", "unknown")),
            }
            for item in getattr(projection, "entities", ())
        ]
    if hasattr(projection, "facts"):
        payload["facts"] = [
            {
                "fact_type": fact.fact_type,
                "value": fact.value,
                "ownership": _enum_text(fact.ownership),
                "supported_requirement_ids": list(fact.supported_requirement_ids),
                "support_anchors": [
                    {"requirement_id": anchor.requirement_id,
                     "source_excerpt": anchor.source_excerpt,
                     "support_scope": anchor.support_scope,
                     "unresolved_aspects": list(anchor.unresolved_aspects)}
                    for anchor in fact.support_anchors
                ],
            }
            for fact in projection.facts
        ]
    for name in ("comparison_dimensions", "steps", "inputs", "outputs"):
        if hasattr(projection, name):
            payload[name] = list(getattr(projection, name))
    for name in ("fact_type", "value", "mechanism"):
        if hasattr(projection, name):
            payload[name] = str(getattr(projection, name))
    if hasattr(projection, "ownership"):
        payload["ownership"] = _enum_text(getattr(projection, "ownership"))
    return payload


def _requirement_from_payload(item: dict[str, object]) -> RequirementAudit:
    values = dict(item)
    for name in ("evidence_ids", "critical_anchors", "covered_critical_anchors", "missing_critical_anchors"):
        values[name] = _strings(values.get(name))
    return RequirementAudit(**values)


def _attempt_from_payload(item: dict[str, object]) -> SearchAttemptAudit:
    values = dict(item)
    values["lexical_terms"] = _strings(values.get("lexical_terms"))
    values["requirement_ids"] = _strings(values.get("requirement_ids"))
    return SearchAttemptAudit(**values)


def _tuple_fields(
    item: dict[str, object], field_names: set[str]
) -> dict[str, object]:
    values = dict(item)
    for name in field_names:
        values[name] = _strings(values.get(name))
    return values


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _object_list(value: object) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Expected a JSON object list.")
    return tuple(value)


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("Expected a string sequence.")
    return tuple(str(item) for item in value)


def _blocker_objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        dict(item) if isinstance(item, dict) else {"type": str(item)}
        for item in value
    )


def _enum_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(sorted(_enum_text(item) for item in value))


def _enum_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected a boolean or null.")
    return value
