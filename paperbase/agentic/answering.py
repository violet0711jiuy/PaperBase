"""Bridge selected Agentic Evidence into the single Answer LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from paperbase.generation.answer_generator import AnswerGenerationOutcome
from paperbase.generation.answer_coverage import derive_overall_answer_status
from paperbase.generation.section_expander import EvidenceClaimContext, EvidenceUnit

from .contracts import EvidenceRecord, RequirementState, ResearchAnswer, ResearchPlan
from .grounding import verified_support_ids


@dataclass(frozen=True)
class CompletionView:
    """Selected local support map, not a Requirement completeness judgement."""

    selected_verified_support_by_requirement: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def required_requirement_ids(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self.selected_verified_support_by_requirement)

    @property
    def supported_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            requirement_id for requirement_id, evidence_ids
            in self.selected_verified_support_by_requirement if evidence_ids
        )

    def support_for(self, requirement_id: str) -> tuple[str, ...]:
        return dict(self.selected_verified_support_by_requirement).get(requirement_id, ())


def build_completion_view(
    *, plan: ResearchPlan, states: tuple[RequirementState, ...],
    selected: tuple[EvidenceRecord, ...],
) -> CompletionView:
    """Map required IDs to selected integrity-valid Projector support."""
    return CompletionView(tuple(
        (
            requirement.requirement_id,
            tuple(
                record.evidence_id for record in selected
                if requirement.requirement_id in verified_support_ids(record, plan.requirements)
            ),
        )
        for requirement in plan.requirements if requirement.required
    ))


class AgenticAnswerGenerator(Protocol):
    def generate(
        self,
        *,
        query: str,
        evidence: Sequence[EvidenceUnit],
        conversation_context: Sequence[str] | None = None,
        research_requirements: Sequence[tuple[str, str]] = (),
        support_by_requirement: dict[str, tuple[str, ...]] | None = None,
    ) -> AnswerGenerationOutcome: ...


def generate_agentic_answer(
    *,
    generator: AgenticAnswerGenerator,
    query: str,
    evidence: tuple[EvidenceRecord, ...],
    completion_view: CompletionView | None = None,
    stopped_by_budget: bool = False,
    plan: ResearchPlan | None = None,
) -> ResearchAnswer:
    """Use Answer coverage as the only semantic completeness decision."""
    requirements = tuple(
        (item.requirement_id, item.description)
        for item in plan.requirements if item.required
    ) if plan is not None else ()
    supports = dict(completion_view.selected_verified_support_by_requirement) if completion_view else {}
    outcome = generator.generate(
        query=query,
        evidence=tuple(_to_evidence_unit(record, plan) for record in evidence),
        conversation_context=None,
        research_requirements=requirements,
        support_by_requirement=supports,
    )
    overall_status = None
    if outcome.status in {"success", "insufficient_evidence"} and requirements:
        overall_status = derive_overall_answer_status(
            outcome.requirement_coverage, outcome.citations
        )
    return ResearchAnswer(
        status=("insufficient_evidence" if overall_status == "insufficient" else outcome.status),
        answer=outcome.answer,
        citations=outcome.citations,
        insufficient_evidence=(overall_status == "insufficient" if overall_status else outcome.insufficient_evidence),
        partial_answer=(overall_status == "partial" if overall_status else outcome.partial_answer),
        coverage_note=outcome.coverage_note,
        requirement_coverage=outcome.requirement_coverage,
        overall_status=overall_status,
        llm_diagnostics=outcome.llm_diagnostics,
    )


def _to_evidence_unit(record: EvidenceRecord, plan: ResearchPlan | None) -> EvidenceUnit:
    source = record.evidence
    contexts = [_answer_projection_context(projection) for projection in record.projections]
    projection_context = None
    if contexts:
        projection_context = dict(contexts[0]) if len(contexts) == 1 else {}
        projection_context["projections"] = contexts
        projection_context["supported_requirement_ids"] = list(verified_support_ids(record, plan.requirements if plan else ()))
    return EvidenceUnit(
        evidence_id=record.evidence_id,
        kind=source.source_type.value,
        paper_id=source.paper_id,
        paper_title=source.paper_title or source.paper_id,
        section=source.section or "未标注章节",
        page_start=source.page_start,
        page_end=source.page_end,
        seed_chunk_ids=source.seed_chunk_ids,
        chunk_ids=source.source_chunk_ids,
        text=source.content,
        token_count=max(1, (len(source.content) + 3) // 4),
        section_id=source.section_id,
        evidence_digest=record.digest,
        claim_annotations=tuple(
            EvidenceClaimContext(
                statement=claim.statement,
                relation=claim.relation.value,
                ownership=claim.ownership.value,
                source_excerpt=claim.source_excerpt,
            )
            for claim in record.claims
        ),
        supported_requirement_ids=verified_support_ids(record, plan.requirements if plan else ()),
        projection_context=projection_context,
    )


def _answer_projection_context(projection):
    projection_context: dict[str, object] | None = None
    if projection is not None:
        projection_context = {
            "type": projection.projection_type.value,
            "supported_requirement_ids": list(projection.supported_requirement_ids),
        }
        for field in ("mechanism", "steps", "inputs", "outputs", "comparison_dimensions"):
            value = getattr(projection, field, None)
            if value:
                projection_context[field] = list(value) if isinstance(value, tuple) else value
        if hasattr(projection, "support_anchors"):
            projection_context["support_anchors"] = [
                {"requirement_id": anchor.requirement_id,
                 "source_excerpt": anchor.source_excerpt,
                 "support_scope": anchor.support_scope,
                 "unresolved_aspects": list(anchor.unresolved_aspects)}
                for anchor in projection.support_anchors
            ]
        if hasattr(projection, "facts"):
            projection_context["facts"] = [
                {"fact_type": fact.fact_type, "value": fact.value,
                 "ownership": fact.ownership.value,
                 "supported_requirement_ids": list(fact.supported_requirement_ids)}
                for fact in projection.facts
            ]
        if hasattr(projection, "entities"):
            projection_context["entities"] = [
                {"name": entity.canonical_name, "role": entity.role.value,
                 "ownership": entity.ownership.value}
                for entity in projection.entities
            ]
    return projection_context


