"""Requirement research progress independent of raw Evidence semantics.

Only a validated Grounded Answer can mark a Requirement satisfied. Before an
answer attempt, this module records whether retrieval was attempted; it never
infers support from query bindings, keywords or Evidence metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CoverageState,
    EvidenceRecord,
    QueryContext,
    RequirementState,
    RequirementStatus,
    ResearchObservation,
    ResearchPlan,
)


TERMINAL_REQUIREMENT_STATUSES = frozenset({
    RequirementStatus.SATISFIED,
    RequirementStatus.NO_SUPPORT_FOUND,
    RequirementStatus.UNCERTAIN,
})


@dataclass(frozen=True)
class RequirementTrackingResult:
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
    del evidence, context, coverage
    return track_requirement_states(
        plan=plan,
        observations=observations,
        previous_states=previous_states,
    ).states


def track_requirement_states(
    *,
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    previous_states: tuple[RequirementState, ...] = (),
    evidence: tuple[EvidenceRecord, ...] = (),
    context: QueryContext | None = None,
    coverage: CoverageState | None = None,
) -> RequirementTrackingResult:
    del evidence, context, coverage
    attempted = _attempted_requirement_ids(
        observations,
        fallback_requirement_ids=tuple(item.requirement_id for item in plan.requirements),
    )
    previous = {item.requirement_id: item for item in previous_states}
    states: list[RequirementState] = []
    reasons: list[tuple[str, str]] = []
    for requirement in plan.requirements:
        old = previous.get(requirement.requirement_id)
        if old is not None and old.status in TERMINAL_REQUIREMENT_STATUSES:
            states.append(old)
            reasons.append((requirement.requirement_id, "validated answer state preserved."))
        elif requirement.requirement_id in attempted:
            states.append(RequirementState(requirement.requirement_id, RequirementStatus.IN_PROGRESS))
            reasons.append((requirement.requirement_id, "retrieval has targeted this Requirement."))
        else:
            states.append(RequirementState(requirement.requirement_id, RequirementStatus.PENDING))
            reasons.append((requirement.requirement_id, "no retrieval action has targeted this Requirement."))
    return RequirementTrackingResult(tuple(states), tuple(reasons))


def finalize_requirement_states(
    *,
    plan: ResearchPlan,
    evidence: tuple[EvidenceRecord, ...],
    observations: tuple[ResearchObservation, ...],
    previous_states: tuple[RequirementState, ...] = (),
    research_complete: bool = True,
) -> RequirementTrackingResult:
    del evidence
    progress = track_requirement_states(
        plan=plan,
        observations=observations,
        previous_states=previous_states,
    )
    states: list[RequirementState] = []
    reasons: list[tuple[str, str]] = []
    for state in progress.states:
        if state.status in TERMINAL_REQUIREMENT_STATUSES:
            states.append(state)
            reasons.append((state.requirement_id, "validated answer state preserved."))
        elif state.status is RequirementStatus.IN_PROGRESS:
            status = RequirementStatus.NO_SUPPORT_FOUND if research_complete else RequirementStatus.UNCERTAIN
            states.append(RequirementState(state.requirement_id, status))
            reasons.append((state.requirement_id, "research ended without a validated grounded answer."))
        else:
            states.append(state)
            reasons.append((state.requirement_id, "Requirement was never researched."))
    return RequirementTrackingResult(tuple(states), tuple(reasons))


def required_requirements_researched(
    plan: ResearchPlan, states: tuple[RequirementState, ...]
) -> bool:
    by_id = {item.requirement_id: item.status for item in states}
    return all(
        not requirement.required
        or by_id.get(requirement.requirement_id) in {
            RequirementStatus.IN_PROGRESS,
            *TERMINAL_REQUIREMENT_STATUSES,
        }
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
        if not action_ids and observation.action_name not in {"try_answer", "stop_research"}:
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
