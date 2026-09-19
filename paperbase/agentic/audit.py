"""Agentic Research 的安全、版本化用户审计快照。

这里的对象是运行时领域状态到产品展示层之间的稳定边界。快照只保存用户理解
研究过程所需的事实，不序列化 Prompt、隐藏推理、原始 LangGraph State、向量分数
或 chunk 内部标识。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Mapping

from .research_profile import build_research_profiles


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
    error_code: str | None = None


@dataclass(frozen=True)
class ResearchStepAudit:
    iteration: int
    action_name: str
    tool_status: str
    added_count: int
    reused_count: int
    attempts: tuple[SearchAttemptAudit, ...] = ()
    filter_total: int = 0
    filter_accepted: int = 0
    filter_rejected: int = 0
    filter_deprioritized: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    cache_errors: int = 0
    error_codes: tuple[str, ...] = ()


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
    addressed_information: tuple[str, ...] = ()
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


@dataclass(frozen=True)
class BudgetAudit:
    agent_iterations: int = 0
    tool_calls: int = 0
    retrieval_queries: int = 0
    llm_calls: int = 0
    invalid_actions: int = 0


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
    error_codes: tuple[str, ...] = ()
    clarification: str | None = None

    def to_dict(self) -> dict[str, object]:
        """转换为仅含 JSON 基础类型的字典。"""
        return asdict(self)

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
                error_codes=_strings(payload.get("error_codes")),
                clarification=_optional_string(payload.get("clarification")),
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
    profiles_by_id: dict[str, object] = {}
    if plan is not None:
        try:
            profiles_by_id = {
                item.paper_id: item
                for item in build_research_profiles(
                    plan=plan,
                    evidence_records=records,
                    observations=observations,
                )
            }
        except (AttributeError, TypeError, ValueError):
            profiles_by_id = {}

    coverage = getattr(result, "coverage", None)
    assessments = tuple(getattr(coverage, "assessments", ()) or ())
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
    if answer is not None:
        diagnostic_records += tuple(getattr(answer, "llm_diagnostics", ()) or ())
    llm_calls = tuple(_llm_call(item) for item in diagnostic_records)

    query_context = getattr(result, "query_context", None)
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
            _evidence_audit(item, selected_ids, cited_ids, titles)
            for item in records
        ),
        budget=_budget_audit(getattr(result, "budget_usage", None)),
        llm_calls=llm_calls,
        error_codes=_strings(getattr(result, "error_codes", ())),
        clarification=_optional_string(getattr(result, "clarification", None)),
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
        requested_targets=_enum_strings(getattr(profile, "requested_targets", ())),
        covered_targets=_enum_strings(getattr(profile, "covered_targets", ())),
        remaining_targets=_enum_strings(getattr(profile, "remaining_targets", ())),
        requested_information=_strings(getattr(profile, "requested_information", ())),
        addressed_information=_strings(getattr(profile, "addressed_information", ())),
        remaining_information=_strings(getattr(profile, "remaining_information", ())),
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
            error_code=_optional_string(getattr(item, "error_code", None)),
        )
        for item in tuple(getattr(observation, "search_attempts", ()) or ())
    )
    return ResearchStepAudit(
        iteration=int(getattr(observation, "iteration", 0) or 0),
        action_name=str(getattr(observation, "action_name", "unknown")),
        tool_status=_enum_text(getattr(observation, "tool_status", "unknown")),
        added_count=len(tuple(getattr(observation, "added_evidence_ids", ()) or ())),
        reused_count=len(tuple(getattr(observation, "reused_evidence_ids", ()) or ())),
        attempts=attempts,
        filter_total=int(getattr(report, "total_candidates", 0) or 0),
        filter_accepted=int(getattr(report, "accepted_candidates", 0) or 0),
        filter_rejected=int(getattr(report, "rejected_candidates", 0) or 0),
        filter_deprioritized=int(getattr(report, "deprioritized_candidates", 0) or 0),
        cache_hits=int(getattr(cache, "hits", 0) or 0),
        cache_misses=int(getattr(cache, "misses", 0) or 0),
        cache_writes=int(getattr(cache, "writes", 0) or 0),
        cache_errors=int(getattr(cache, "errors", 0) or 0),
        error_codes=_strings(getattr(observation, "error_codes", ())),
    )


def _evidence_audit(
    record: object,
    selected_ids: set[str],
    cited_ids: set[str],
    titles: Mapping[str, str],
) -> EvidenceDigestAudit:
    evidence_id = str(getattr(record, "evidence_id", ""))
    source = getattr(record, "evidence", None)
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
        addressed_information=_strings(getattr(record, "addressed_information", ())),
        claims=claims,
    )


def _budget_audit(value: object | None) -> BudgetAudit:
    return BudgetAudit(
        agent_iterations=int(getattr(value, "agent_iterations", 0) or 0),
        tool_calls=int(getattr(value, "tool_calls", 0) or 0),
        retrieval_queries=int(getattr(value, "retrieval_queries", 0) or 0),
        llm_calls=int(getattr(value, "llm_calls", 0) or 0),
        invalid_actions=int(getattr(value, "invalid_actions", 0) or 0),
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
    )


_PAPER_TUPLE_FIELDS = {
    "queries_tried",
    "evidence_ids",
    "requested_targets",
    "covered_targets",
    "remaining_targets",
    "requested_information",
    "addressed_information",
    "remaining_information",
    "sections_seen",
}


def _step_from_payload(item: dict[str, object]) -> ResearchStepAudit:
    values = dict(item)
    values["attempts"] = tuple(
        SearchAttemptAudit(**attempt)
        for attempt in _object_list(values.get("attempts"))
    )
    values["error_codes"] = _strings(values.get("error_codes"))
    return ResearchStepAudit(**values)


def _evidence_from_payload(item: dict[str, object]) -> EvidenceDigestAudit:
    values = _tuple_fields(
        item,
        {"covered_targets", "addressed_information"},
    )
    values["claims"] = tuple(
        ClaimAudit(**claim) for claim in _object_list(values.get("claims"))
    )
    return EvidenceDigestAudit(**values)


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
