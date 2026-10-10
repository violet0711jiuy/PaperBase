"""Requirement-driven final Evidence selection.

The selector is deliberately deterministic: projections have already made the
semantic judgement, so this module only preserves the smallest useful,
provenance-bearing set for the answer generator.  It never makes an LLM call.
"""

from __future__ import annotations

import re
from paperbase.retrieval.evidence_scores import retrieval_rank_priority

from .contracts import (
    EvidenceAnalysisStatus,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    PaperAssessmentStatus,
    QueryContext,
    ResearchGoal,
    RequirementState,
    ResearchPlan,
)
from .grounding import verified_support_ids
from .projection_reads import iter_projections


class FinalEvidenceSelector:
    """Select evidence by Requirement coverage and projected information gain.

    Paper membership is provenance, not a semantic budget: a unique Requirement
    or entity must never be discarded merely because it belongs to an already
    represented paper.
    """

    def __init__(self, *, max_units: int = 8) -> None:
        if max_units < 1:
            raise ValueError("Evidence Selector 上限必须大于零。")
        self._max_units = max_units

    def select(
        self,
        *,
        context: QueryContext | None = None,
        plan: ResearchPlan,
        coverage: object | None = None,
        evidence: tuple[EvidenceRecord, ...],
        requirement_states: tuple[RequirementState, ...] = (),
    ) -> tuple[EvidenceRecord, ...]:
        """Return final, source-preserving Evidence records without LLM calls.

        Integrity-valid Projector support is the candidate pool. Research
        progress states do not decide answer completeness or eligibility.
        """
        required = {item.requirement_id for item in plan.requirements if item.required}
        # The Registry is intentionally broad: it also stores orientation and
        # localization evidence.  The answer context is not.  A core final
        # unit must have survived analysis *and* explicitly support a final
        # required information obligation; entity novelty alone is never a
        # ticket into the answer prompt.
        eligible = tuple(
            record
            for record in evidence
            if self._eligible(record)
            and (
                not required
                # Requirement-free plans are historical checkpoint fixtures.
                # New Planner output always declares Requirements and follows
                # the strict core-context branch below.
                or bool(_supported_requirement_ids(record, plan.requirements).intersection(required))
            )
        )
        if not eligible:
            return ()

        if not required:
            return self._select_legacy(eligible, context=context, coverage=coverage)
        candidates_by_requirement = {
            requirement_id: tuple(
                record for record in eligible
                if requirement_id in _supported_requirement_ids(record, plan.requirements)
            )
            for requirement_id in required
        }
        selected: list[EvidenceRecord] = []

        # Phase 1: Requirement set cover. Unique support is selected even if it
        # exceeds the answer budget; the budget must not erase a Requirement.
        uncovered = {
            requirement_id for requirement_id, candidates in candidates_by_requirement.items()
            if candidates and not any(item in selected for item in candidates)
        }
        while uncovered:
            best = self._best(
                tuple(item for item in eligible if item not in selected), selected,
                uncovered, requirements=plan.requirements,
            )
            if best is None:
                break
            selected.append(best)
            covered = _supported_requirement_ids(best, plan.requirements)
            uncovered.difference_update(covered)
            for requirement_id in tuple(uncovered):
                if best in candidates_by_requirement[requirement_id]:
                    uncovered.remove(requirement_id)

        # Phase 2: retain distinct projected entities/facts. This keeps E10/E11
        # style independent prior-work families but discards repeated EMVE.
        while len(selected) < self._max_units:
            best = self._best(
                tuple(item for item in eligible if item not in selected), selected, set(),
                require_information_gain=True, requirements=plan.requirements,
            )
            if best is None:
                break
            selected.append(best)

        # Old checkpoints can lack RequirementState. Keep a useful context using
        # the same gain-first ordering instead of a paper/ranking quota.
        if not selected and not required:
            best = self._best(eligible, (), set())
            if best is not None:
                selected.append(best)
        return tuple(selected)

    def _select_legacy(
        self,
        eligible: tuple[EvidenceRecord, ...],
        *,
        context: QueryContext | None,
        coverage: object | None,
    ) -> tuple[EvidenceRecord, ...]:
        """Stable Phase-3/checkpoint fallback when RequirementState is absent."""
        if context is not None and context.retrieval_goal is ResearchGoal.EXHAUSTIVE_ENUMERATION:
            selected: list[EvidenceRecord] = []
            while len(selected) < self._max_units:
                best = self._best(
                    tuple(item for item in eligible if item not in selected), selected, set(),
                    require_information_gain=False,
                )
                if best is None:
                    break
                selected.append(best)
            return tuple(selected)
        ranked = sorted(
            eligible,
            key=lambda record: (*self._quality(record), retrieval_rank_priority(record.evidence), record.evidence_id),
            reverse=True,
        )
        selected: list[EvidenceRecord] = []
        positive_ids = tuple(
            item.paper_id for item in getattr(coverage, "assessments", ())
            if item.status is PaperAssessmentStatus.POSITIVE
        )
        for paper_id in positive_ids:
            candidate = next((item for item in ranked if item.evidence.paper_id == paper_id), None)
            if candidate is not None:
                selected.append(candidate)
                if len(selected) >= self._max_units:
                    return tuple(selected)
        seen_sections: set[tuple[object, ...]] = set()
        seen_sections.update(_section_identity(item) for item in selected)
        for record in ranked:
            if len(selected) >= self._max_units:
                break
            if _section_identity(record) not in seen_sections:
                selected.append(record)
                seen_sections.add(_section_identity(record))
        for record in ranked:
            if len(selected) >= self._max_units:
                break
            if record not in selected:
                selected.append(record)
        return tuple(selected)

    @classmethod
    def _best(
        cls,
        candidates: tuple[EvidenceRecord, ...],
        selected: list[EvidenceRecord] | tuple[EvidenceRecord, ...],
        uncovered: set[str],
        *,
        require_information_gain: bool = False,
        requirements: tuple[object, ...] = (),
    ) -> EvidenceRecord | None:
        scored = [
            (cls._gain(record, selected, uncovered, requirements=requirements), record)
            for record in candidates
        ]
        if require_information_gain:
            scored = [item for item in scored if item[0][1] or item[0][2]]
        if not scored:
            return None
        return max(scored, key=lambda item: (item[0], item[1].evidence_id))[1]

    @classmethod
    def _gain(
        cls, record: EvidenceRecord, selected: list[EvidenceRecord] | tuple[EvidenceRecord, ...],
        uncovered: set[str],
        *, requirements: tuple[object, ...] = (),
    ) -> tuple[float, ...]:
        selected_entities = {key for item in selected for key in _entity_keys(item)}
        selected_facts = {key for item in selected for key in _fact_keys(item)}
        selected_sections = {_section_identity(item) for item in selected}
        return (
            float(len(_supported_requirement_ids(record, requirements).intersection(uncovered))),
            float(len(_entity_keys(record) - selected_entities)),
            float(len(_fact_keys(record) - selected_facts)),
            *cls._quality(record),
            float(_section_identity(record) not in selected_sections),
            retrieval_rank_priority(record.evidence),
        )

    @staticmethod
    def _eligible(record: EvidenceRecord) -> bool:
        if record.analysis_status is not EvidenceAnalysisStatus.QUALIFIED:
            return False
        if record.projections:
            return bool(_supported_requirement_ids(record) or _entity_keys(record) or _fact_keys(record))
        return any(
            claim.relation in {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}
            and claim.ownership not in {EvidenceOwnership.GENERAL_BACKGROUND, EvidenceOwnership.UNKNOWN}
            for claim in record.claims
        )

    @staticmethod
    def _quality(record: EvidenceRecord) -> tuple[float, float]:
        direct = float(any(
            claim.relation in {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}
            and claim.ownership is not EvidenceOwnership.UNKNOWN for claim in record.claims
        ) or any(key[3] is not EvidenceOwnership.UNKNOWN for key in _entity_keys(record)))
        provenance = float(bool(record.evidence.section_id or record.evidence.section or record.evidence.source_chunk_ids))
        return (direct, provenance)


def _supported_requirement_ids(
    record: EvidenceRecord, requirements: tuple[object, ...] = ()
) -> set[str]:
    return set(verified_support_ids(record, requirements))


def _entity_keys(record: EvidenceRecord) -> set[tuple[str, str, str, EvidenceOwnership]]:
    entities = tuple(entity for projection in iter_projections(record)
                     for entity in getattr(projection, "entities", ()))
    keys = {
        (_normalize_entity(item.canonical_name), _normalize_entity(item.entity_type), item.role.value, item.ownership)
        for item in entities
    }
    keys.update((_normalize_entity(value), "method", "unknown", EvidenceOwnership.UNKNOWN) for value in record.method_entities)
    keys.update((_normalize_entity(value), "method_family", "unknown", EvidenceOwnership.UNKNOWN) for value in record.method_families)
    keys.update((_normalize_entity(value), "baseline", "unknown", EvidenceOwnership.UNKNOWN) for value in record.baseline_entities)
    return keys


def _fact_keys(record: EvidenceRecord) -> set[str]:
    projections = iter_projections(record)
    if not projections:
        if _entity_keys(record):
            return set()
        return {_normalize_entity(claim.statement) for claim in record.claims}
    return {key for projection in projections for key in _projection_fact_keys(projection)}


def _projection_fact_keys(projection) -> set[str]:
    if hasattr(projection, "facts"):
        return {
            f"{_normalize_entity(fact.fact_type)}:{_normalize_entity(fact.value)}"
            for fact in projection.facts
        } | {
            f"anchor:{' '.join(anchor.source_excerpt.casefold().split())}"
            for fact in projection.facts for anchor in fact.support_anchors
        }
    if hasattr(projection, "mechanism"):
        return {_normalize_entity(projection.mechanism),
            *(_normalize_entity(value) for value in projection.steps),
            *(f"anchor:{' '.join(anchor.source_excerpt.casefold().split())}" for anchor in projection.support_anchors)}
    return {f"anchor:{' '.join(anchor.source_excerpt.casefold().split())}"
            for anchor in getattr(projection, "support_anchors", ())}


def _section_identity(record: EvidenceRecord) -> tuple[object, ...]:
    evidence = record.evidence
    if evidence.section_id:
        return (evidence.source_type.value, "section_id", evidence.section_id)
    if evidence.section:
        return (evidence.source_type.value, "section", " ".join(evidence.section.casefold().split()))
    return (evidence.source_type.value, "chunks", tuple(sorted(evidence.source_chunk_ids)))


def _normalize_entity(value: str) -> str:
    """Conservative, deterministic normalization for known spelling variants."""
    text = value.casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "") if text in {"lstm qr", "lube ann"} else text
