"""Source-only identifier hints for Decision retrieval focus and audit.

Lexical anchor coverage never determines support, selection, or answer
completeness. The ``complete`` field below means only that the explicit
identifier hints were seen in raw supporting Evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .contracts import EvidenceRecord, ResearchRequirement
from .grounding import verified_support_ids


_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9_])")
_SLASH_VARIABLES = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Z]\s*/\s*)+[A-Z](?![A-Za-z0-9_])")


def critical_anchors(description: str) -> tuple[str, ...]:
    """Extract explicit technical identifiers without guessing prose concepts."""
    slash_positions: set[int] = set()
    for match in _SLASH_VARIABLES.finditer(description):
        slash_positions.update(range(match.start(), match.end()))
    found: dict[str, str] = {}
    for match in _IDENTIFIER.finditer(description):
        token = match.group()
        technical = (
            "_" in token
            or ("-" in token and (any(char.isupper() or char.isdigit() for char in token)))
            or (len(token) >= 2 and token.isupper())
            or (token[0].islower() and any(char.isupper() for char in token[1:]))
            or (any(char.islower() for char in token) and sum(char.isupper() for char in token) >= 2)
            or (len(token) == 1 and match.start() in slash_positions)
        )
        if technical:
            found.setdefault(token.casefold(), token)
    return tuple(found.values())


def source_has_anchor(source: str, anchor: str) -> bool:
    """Match one identifier with real token edges and conservative PDF spacing."""
    source = unicodedata.normalize("NFKC", source).casefold()
    pieces = re.split(r"([-_])", unicodedata.normalize("NFKC", anchor).casefold())
    pattern = "".join(
        r"(?:\s*-\s*|\s+)" if part == "-" else
        r"\s*_\s*" if part == "_" else re.escape(part)
        for part in pieces
    )
    return bool(re.search(r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])", source))


@dataclass(frozen=True)
class RequirementCoverage:
    requirement_id: str
    critical_anchors: tuple[str, ...]
    covered_anchors: tuple[str, ...]
    missing_anchors: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    complete: bool


def derive_requirement_coverage(
    requirement: ResearchRequirement, evidence: tuple[EvidenceRecord, ...]
) -> RequirementCoverage:
    anchors = critical_anchors(requirement.description)
    supporting = tuple(
        record for record in evidence
        if requirement.requirement_id in verified_support_ids(record, (requirement,))
    )
    covered = tuple(
        anchor for anchor in anchors
        if any(source_has_anchor(record.evidence.content, anchor) for record in supporting)
    )
    missing = tuple(anchor for anchor in anchors if anchor not in covered)
    return RequirementCoverage(
        requirement.requirement_id, anchors, covered, missing,
        tuple(record.evidence_id for record in supporting),
        bool(supporting) and not missing,
    )
