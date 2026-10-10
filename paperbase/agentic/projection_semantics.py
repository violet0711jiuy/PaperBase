"""Group-local Projector semantic identity, separate from search canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import unicodedata

from .contracts import EvidenceProjectionType, ResearchPlan, ResearchRequirement


PROJECTION_SEMANTIC_REQUEST_VERSION = "projection-semantic-v1"


@dataclass(frozen=True)
class ProjectionRequirementSemantic:
    requirement_id: str
    description: str
    evidence_targets: tuple[str, ...]
    required: bool
    projection_type: str | None


@dataclass(frozen=True)
class ProjectionSemanticRequest:
    question: str
    plan_goal: str
    projection_type: EvidenceProjectionType
    requirements: tuple[ProjectionRequirementSemantic, ...]


def normalize_projection_semantic_text(text: str) -> str:
    """Retain case, punctuation, numbers and symbols; normalize layout only."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def scoped_projection_requirements(
    plan: ResearchPlan, requirement_ids: tuple[str, ...],
) -> tuple[ResearchRequirement, ...]:
    """The one order authority shared by prompt input and semantic snapshot."""
    ids = tuple(requirement_ids)
    known = {r.requirement_id for r in plan.requirements}
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(known):
        raise ValueError("Projection semantic scope must be nonempty, unique and belong to Plan.")
    selected = set(ids)
    return tuple(r for r in plan.requirements if r.requirement_id in selected)


def build_projection_semantic_request(
    *, question: str, plan: ResearchPlan, projection_type: EvidenceProjectionType,
    requirement_ids: tuple[str, ...],
) -> ProjectionSemanticRequest:
    return ProjectionSemanticRequest(
        normalize_projection_semantic_text(question),
        normalize_projection_semantic_text(plan.goal),
        EvidenceProjectionType(projection_type),
        tuple(ProjectionRequirementSemantic(
            r.requirement_id, normalize_projection_semantic_text(r.description),
            tuple(sorted(target.value for target in r.evidence_targets)), r.required,
            r.projection_type.value if r.projection_type is not None else None,
        ) for r in scoped_projection_requirements(plan, requirement_ids)),
    )


def projection_semantic_snapshot(request: ProjectionSemanticRequest) -> dict[str, object]:
    """JSON-safe audit value; never written into every analysis payload."""
    return {
        "version": PROJECTION_SEMANTIC_REQUEST_VERSION,
        "question": normalize_projection_semantic_text(request.question),
        "plan_goal": normalize_projection_semantic_text(request.plan_goal),
        "projection_type": EvidenceProjectionType(request.projection_type).value,
        "requirements": [{
            "requirement_id": r.requirement_id,
            "description": normalize_projection_semantic_text(r.description),
            "evidence_targets": sorted(set(r.evidence_targets)),
            "required": r.required,
            "projection_type": r.projection_type,
        } for r in request.requirements],
    }


def projection_semantic_fingerprint(request: ProjectionSemanticRequest) -> str:
    encoded = json.dumps(projection_semantic_snapshot(request), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def projection_semantic_cache_context(request: ProjectionSemanticRequest) -> dict[str, object]:
    """Compiled opaque key material. SQLite does not interpret the Plan."""
    return {
        "projection_type": request.projection_type.value,
        "projection_requirement_ids": [r.requirement_id for r in request.requirements],
        "semantic_request_version": PROJECTION_SEMANTIC_REQUEST_VERSION,
        "semantic_fingerprint": projection_semantic_fingerprint(request),
    }
