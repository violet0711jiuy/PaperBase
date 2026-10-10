"""Per-Requirement answer coverage reported by the single Answer LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
