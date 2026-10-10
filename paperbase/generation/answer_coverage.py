"""Per-Requirement answer coverage reported by the single Answer LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

ANSWER_SEMANTIC_SCHEMA_VERSION = "answer-semantic-status-object-v3"


class AnswerDomainValidationError(ValueError):
    """Safe failure of an Answer structure/citation/coverage constraint."""

    def __init__(self, code: str, field_path: str, safe_reason: str,
                 actual_summary: str, expected_constraint: str) -> None:
        super().__init__(safe_reason)
        self.code = code
        self.field_path = field_path
        self.safe_reason = safe_reason
        self.actual_summary = actual_summary
        self.expected_constraint = expected_constraint


@dataclass(frozen=True)
class AnswerRequirementCoverage:
    requirement_id: str
    status: Literal["complete", "partial", "unsupported"]
    evidence_ids: tuple[str, ...] = ()


def compile_answer_requirement_coverage(
    *, required_requirement_ids: Sequence[str],
    allowed_support_by_requirement: Mapping[str, Sequence[str]],
    semantic_status_by_requirement: Mapping[str, str],
    inline_citations: Sequence[str],
) -> tuple[AnswerRequirementCoverage, ...]:
    """Compile formal coverage from trusted support and validated inline IDs.

    No semantic completeness inference. Requirement order follows the Plan;
    used support follows first citation occurrence, never set iteration order.
    """
    required = tuple(required_requirement_ids)
    judged = {rid for rid in required if allowed_support_by_requirement.get(rid)}
    if (len(required) != len(set(required)) or set(semantic_status_by_requirement) != judged
            or any(not isinstance(status, str) or status not in {"complete", "partial", "unsupported"}
                   for status in semantic_status_by_requirement.values())):
        raise AnswerDomainValidationError(
            "ANSWER_REQUIREMENT_STATUS_INVALID", "requirement_status",
            "requirement_status 必须恰好包含有 selected verified support 的 required IDs，值只能 complete/partial/unsupported。",
            "semantic_status_key_or_value_domain_mismatch",
            "Exactly the selected-support required keys with complete/partial/unsupported values.",
        )
    citations = tuple(dict.fromkeys(inline_citations))
    accepted_ids = {eid for rid in judged if semantic_status_by_requirement[rid] != "unsupported"
                    for eid in allowed_support_by_requirement[rid]}
    rejected_ids = {eid for rid in judged if semantic_status_by_requirement[rid] == "unsupported"
                    for eid in allowed_support_by_requirement[rid]} - accepted_ids
    if rejected_ids.intersection(citations):
        raise AnswerDomainValidationError(
            "ANSWER_CITATION_INVALID", "inline_citations",
            "已判 unsupported 的专属证据不得继续作为回答支持引用。",
            "rejected_answer_support_cited",
            "Rejected support may only be cited when also allowed for an accepted Requirement.",
        )
    coverage = []
    for rid in required:
        allowed = set(allowed_support_by_requirement.get(rid, ()))
        status = semantic_status_by_requirement[rid] if allowed else "unsupported"
        used = tuple(eid for eid in citations if eid in allowed) if status != "unsupported" else ()
        if status != "unsupported" and not used:
            raise AnswerDomainValidationError(
                "ANSWER_CITATION_INVALID", "inline_citations",
                "complete/partial Requirement 必须在正文实际引用至少一条允许的支持 Evidence。",
                "covered_requirement_without_cited_support",
                "Each supported Requirement must use allowed support in body citations; never auto-add citations.",
            )
        coverage.append(AnswerRequirementCoverage(rid, status, used))
    return tuple(coverage)


def derive_overall_answer_status(
    coverage: tuple[AnswerRequirementCoverage, ...], citations: tuple[str, ...]
) -> Literal["complete", "partial", "insufficient"]:
    """Derive overall answer status solely from Answer coverage and citations."""
    statuses = {item.status for item in coverage}
    if not citations or not statuses or statuses == {"unsupported"}:
        return "insufficient"
    return "complete" if statuses == {"complete"} else "partial"
