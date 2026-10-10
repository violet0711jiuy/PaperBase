"""Evidence integrity checks for Projector support assertions.

The Projector owns Evidence-level semantic support. This module verifies IDs
and source provenance; it does not infer entailment from lexical overlap.
"""

from __future__ import annotations

from collections.abc import Iterable
import unicodedata

from .projection_reads import iter_projections

from .contracts import (
    EnumerationProjection,
    EvidenceAnalysisStatus,
    EvidenceOwnership,
    EvidenceRecord,
    FactProjection,
    ProjectedEntityRole,
    RequirementSupportAnchor,
    ResearchRequirement,
)


def normalized_source(value: str) -> str:
    """Normalize PDF whitespace while preserving the source's characters."""
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def source_contains(content: str, excerpt: str) -> bool:
    anchor = normalized_source(excerpt)
    return bool(anchor) and anchor in normalized_source(content)


def local_source_window(content: str, excerpt: str, *, limit: int = 480) -> str:
    """Expand a real excerpt only within its original Evidence text."""
    needle = normalized_source(excerpt)
    if len(needle) < 8:
        return excerpt
    chars: list[str] = []
    offsets: list[int] = []
    for offset, character in enumerate(content):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if not normalized.isspace():
                chars.append(normalized)
                offsets.append(offset)
    position = "".join(chars).find(needle)
    if position < 0:
        return excerpt
    left = offsets[position]
    right = offsets[position + len(needle) - 1] + 1
    if right - left >= limit:
        return content[left:right]
    padding = (limit - (right - left)) // 2
    left = max(0, left - padding)
    right = min(len(content), max(right, left + limit))
    return content[left:right]


def support_rejection(
    *,
    requirement: ResearchRequirement,
    record: EvidenceRecord,
    anchor: RequirementSupportAnchor,
    fact_ownership: EvidenceOwnership | None = None,
) -> str | None:
    """Reject only invalid support provenance or an internally unknown owner."""
    if anchor.requirement_id != requirement.requirement_id:
        return "requirement_id_mismatch"
    if not source_contains(record.evidence.content, anchor.source_excerpt):
        return "source_anchor_not_found"
    if fact_ownership is EvidenceOwnership.UNKNOWN:
        return "fact_ownership_unknown"
    return None


def entity_attribution_valid(role: ProjectedEntityRole, ownership: EvidenceOwnership) -> bool:
    """Reject contradictory or unknown entity provenance without judging relevance."""
    return (
        role in {ProjectedEntityRole.PROPOSED, ProjectedEntityRole.ADOPTED}
        and ownership is EvidenceOwnership.THIS_PAPER
        or role in {ProjectedEntityRole.BASELINE, ProjectedEntityRole.PRIOR_WORK}
        and ownership is EvidenceOwnership.PRIOR_WORK
    )


def verified_support_ids(
    record: EvidenceRecord,
    requirements: Iterable[ResearchRequirement] = (),
) -> tuple[str, ...]:
    """Projector support whose stored source anchors still pass integrity checks."""
    if getattr(record, "analysis_status", None) is not EvidenceAnalysisStatus.QUALIFIED:
        return ()
    requirements = tuple(requirements)
    return tuple(dict.fromkeys(rid for projection in iter_projections(record)
        for rid in verified_projection_support_ids(record, projection, requirements)))


def verified_projection_support_ids(record, projection, requirements=()):
    """The original per-projection Integrity checks, without cross-sibling trust."""
    allowed = {item.requirement_id for item in requirements}
    if isinstance(projection, EnumerationProjection):
        if not any(
            entity_attribution_valid(entity.role, entity.ownership)
            and len(name := normalized_source(entity.canonical_name)) >= 3
            and name in normalized_source(record.evidence.content)
            for entity in projection.entities
        ):
            return ()
        return tuple(
            requirement_id for requirement_id in projection.supported_requirement_ids
            if not allowed or requirement_id in allowed
            if any(
                anchor.requirement_id == requirement_id
                and source_contains(record.evidence.content, anchor.source_excerpt)
                for anchor in projection.support_anchors
            )
        )
    parts = projection.facts if isinstance(projection, FactProjection) else (projection,)
    verified: list[str] = []
    for part in parts:
        proposed = set(part.proposed_requirement_ids)
        for requirement_id in part.supported_requirement_ids:
            if requirement_id not in proposed or (allowed and requirement_id not in allowed):
                continue
            if getattr(part, "ownership", None) is EvidenceOwnership.UNKNOWN:
                continue
            if any(
                anchor.requirement_id == requirement_id
                and source_contains(record.evidence.content, anchor.source_excerpt)
                for anchor in part.support_anchors
            ):
                verified.append(requirement_id)
    return tuple(dict.fromkeys(verified))


def projection_facts(projection: object) -> tuple[str, ...]:
    """Structured facts for retrieval and audit context, never a support gate."""
    if isinstance(projection, FactProjection):
        return tuple(
            value for fact in projection.facts
            for value in (fact.fact_type, fact.value)
        )
    values: list[str] = []
    for field in ("comparison_dimensions", "steps", "inputs", "outputs"):
        values.extend(str(item) for item in getattr(projection, field, ()))
    for field in ("fact_type", "value", "mechanism"):
        value = getattr(projection, field, None)
        if value:
            values.append(str(value))
    return tuple(values)
