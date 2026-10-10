"""Deterministic research progress tracking, independent of answer completeness."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CoverageState,
    EvidenceAnalysisStatus,
    EvidenceRecord,
    RequirementState,
    RequirementStatus,
    ResearchObservation,
    ResearchPlan,
)
from .grounding import verified_support_ids


TERMINAL_REQUIREMENT_STATUSES = frozenset(
    {
        RequirementStatus.SATISFIED,
        RequirementStatus.NO_SUPPORT_FOUND,
        RequirementStatus.UNCERTAIN,
    }
)


@dataclass(frozen=True)
class RequirementTrackingResult:
    """A checkpoint-safe state projection plus audit-safe, non-LLM reasons."""

    states: tuple[RequirementState, ...]
    reasons: tuple[tuple[str, str], ...] = ()

    def reason_for(self, requirement_id: str) -> str | None:
        return dict(self.reasons).get(requirement_id)


def derive_requirement_states(
    *,
    plan: ResearchPlan,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...],
    context: QueryContext | None = None,
    coverage: CoverageState | None = None,
    previous_states: tuple[RequirementState, ...] = (),
) -> tuple[RequirementState, ...]:
    """Compatibility façade for callers that only need persisted states."""
    return track_requirement_states(
        plan=plan,
        evidence=evidence,
        observations=observations,
        context=context,
        coverage=coverage,
        previous_states=previous_states,
    ).states


def track_requirement_states(
    *,
    plan: ResearchPlan,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...],
    context: QueryContext | None = None,
    coverage: CoverageState | None = None,
    previous_states: tuple[RequirementState, ...] = (),
) -> RequirementTrackingResult:
    """Record attempts and integrity-valid Projector support, not completeness."""
    attempted = _attempted_requirement_ids(
        observations,
        fallback_requirement_ids=tuple(item.requirement_id for item in plan.requirements),
    )
    terminal_before = {
        item.requirement_id: item
        for item in previous_states
        if item.status in TERMINAL_REQUIREMENT_STATUSES
    }
    states: list[RequirementState] = []
    reasons: list[tuple[str, str]] = []
    for requirement in plan.requirements:
        requirement_id = requirement.requirement_id
        supporting = tuple(
            record.evidence_id for record in evidence
            if requirement_id in verified_support_ids(record, plan.requirements)
        )
        if supporting:
            state = RequirementState(requirement_id, RequirementStatus.SATISFIED, supporting)
            states.append(state)
            reasons.append((requirement_id, "integrity-valid Projector support found; answer completeness is undecided."))
            continue
        if requirement_id in terminal_before and terminal_before[requirement_id].status is not RequirementStatus.SATISFIED:
            state = terminal_before[requirement_id]
            states.append(state)
            reasons.append((requirement_id, "terminal state preserved from checkpoint."))
            continue
        has_attempt = requirement_id in attempted
        has_uncertain_signal = _has_uncertain_signal(evidence, requirement_id)
        if has_attempt or has_uncertain_signal or supporting:
            status = RequirementStatus.IN_PROGRESS
            reason = f"research activity exists for {requirement_id}; finalization is deferred until finish."
        else:
            status = RequirementStatus.PENDING
            reason = f"no research action has yet targeted {requirement_id}."
        states.append(RequirementState(requirement_id, status, supporting))
        reasons.append((requirement_id, reason))
    return RequirementTrackingResult(tuple(states), tuple(reasons))


def finalize_requirement_states(
    *,
    plan: ResearchPlan,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...],
    previous_states: tuple[RequirementState, ...] = (),
    research_complete: bool = True,
) -> RequirementTrackingResult:
    """Close progress; budget exhaustion leaves unsupported searches uncertain."""
    progress = track_requirement_states(
        plan=plan, evidence=evidence, observations=observations,
        previous_states=previous_states,
    )
    states: list[RequirementState] = []
    reasons: list[tuple[str, str]] = []
    for requirement in plan.requirements:
        current = next(item for item in progress.states if item.requirement_id == requirement.requirement_id)
        if current.status in TERMINAL_REQUIREMENT_STATUSES:
            states.append(current)
            reasons.append((current.requirement_id, "terminal state preserved from checkpoint."))
            continue
        supporting = tuple(
            record.evidence_id for record in evidence
            if requirement.requirement_id in verified_support_ids(record, plan.requirements)
        )
        if supporting:
            state = RequirementState(requirement.requirement_id, RequirementStatus.SATISFIED, supporting)
            reason = f"{', '.join(supporting)} provide integrity-valid local support; answer completeness is undecided."
        elif _has_uncertain_signal(evidence, requirement.requirement_id):
            state = RequirementState(requirement.requirement_id, RequirementStatus.UNCERTAIN)
            reason = "related evidence could not be reliably confirmed."
        elif current.status is RequirementStatus.IN_PROGRESS:
            status = (
                RequirementStatus.NO_SUPPORT_FOUND if research_complete
                else RequirementStatus.UNCERTAIN
            )
            state = RequirementState(requirement.requirement_id, status)
            reason = (
                "research actions completed without supporting evidence."
                if research_complete else "research stopped before support could be established."
            )
        else:
            state = current
            reason = "requirement was never researched and cannot be finalized."
        states.append(state)
        reasons.append((requirement.requirement_id, reason))
    return RequirementTrackingResult(tuple(states), tuple(reasons))


def required_requirements_researched(
    plan: ResearchPlan, states: tuple[RequirementState, ...]
) -> bool:
    by_id = {item.requirement_id: item.status for item in states}
    return all(
        not requirement.required
        or by_id.get(requirement.requirement_id)
        in {RequirementStatus.IN_PROGRESS, *TERMINAL_REQUIREMENT_STATUSES}
        for requirement in plan.requirements
    )


def required_requirements_terminal(
    plan: ResearchPlan, states: tuple[RequirementState, ...]
) -> bool:
    by_id = {item.requirement_id: item.status for item in states}
    return all(
        not requirement.required
        or by_id.get(requirement.requirement_id) in TERMINAL_REQUIREMENT_STATUSES
        for requirement in plan.requirements
    )


def _attempted_requirement_ids(
    observations: tuple[ResearchObservation, ...],
    *,
    fallback_requirement_ids: tuple[str, ...],
) -> set[str]:
    ids: set[str] = set()
    for observation in observations:
        for attempt in observation.search_attempts:
            ids.update(attempt.requirement_ids)
        action_ids = _ids_from_action_arguments(observation.action_arguments)
        ids.update(action_ids)
        # Older Action/checkpoint payloads had Subtask IDs only.  Their
        # individual requirement mapping was never persisted, so conservatively
        # treat the action as a legacy attempt for the then-current plan.
        if not action_ids and observation.action_name != "finish":
            ids.update(fallback_requirement_ids)
    return ids


def _ids_from_action_arguments(arguments: dict[str, object]) -> set[str]:
    direct = arguments.get("requirement_ids", ())
    ids = {value for value in direct if isinstance(value, str)} if isinstance(direct, (list, tuple)) else set()
    searches = arguments.get("searches", ())
    if isinstance(searches, list):
        for search in searches:
            if isinstance(search, dict):
                values = search.get("requirement_ids", ())
                if isinstance(values, (list, tuple)):
                    ids.update(value for value in values if isinstance(value, str))
    return ids


def _has_uncertain_signal(
    evidence: tuple[EvidenceRecord, ...], requirement_id: str
) -> bool:
    return any(
        requirement_id in record.requested_requirement_ids
        and record.analysis_status
        in {EvidenceAnalysisStatus.PENDING, EvidenceAnalysisStatus.FAILED}
        for record in evidence
    )
