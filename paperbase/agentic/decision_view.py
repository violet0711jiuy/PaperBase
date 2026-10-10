"""Small, deterministic Decision input derived from Graph state.

Decision receives plans, answer gaps, a raw Evidence catalog and the latest
tool delta. It never receives semantic projections or duplicated full Evidence
bodies. Full raw text is reserved for the Grounded Answerer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import TYPE_CHECKING

from .initial_search_hints import build_initial_search_hints
from .requirement_targets import preferred_tools_for_targets

if TYPE_CHECKING:
    from paperbase.llm.client import ValidationDiagnostic
    from .decision import DecisionState


@dataclass(frozen=True)
class CompactDecisionLimits:
    support_entries_per_requirement: int = 6
    recent_observations: int = 2
    recent_queries: int = 3
    recent_actions: int = 3
    sections: int = 10
    support_scope_chars: int = 160
    unresolved_aspects: int = 6
    unresolved_aspect_chars: int = 240
    entity_hints: int = 12
    research_leads: int = 4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class CompactDecisionView:
    payload: dict[str, object]
    metrics: dict[str, int | float]

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.payload)

    def to_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"), default=list)


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


def _runtime_decision_view(capabilities) -> dict[str, object]:
    return {
        "available_tools": capabilities.available_tools,
        "valid_requirement_ids": capabilities.valid_requirement_ids,
        "search_paper_ids": capabilities.search_paper_ids,
        "reference_paper_ids": capabilities.reference_paper_ids,
        "overview_paper_ids": capabilities.overview_paper_ids,
        "available_section_ids_by_paper": capabilities.available_section_ids_by_paper,
        "capability_reasons": capabilities.capability_reasons,
    }


def _action_arguments(arguments: dict[str, object]) -> dict[str, object]:
    return deepcopy(arguments)


def build_recent_research_events(state: "DecisionState") -> list[dict[str, object]]:
    from .decision import _observation_requirement_ids

    events: list[dict[str, object]] = []
    for observation in state.observations:
        request = _action_arguments(observation.action_arguments)
        if observation.action_name.startswith("search_") and not request:
            request = {
                "searches": [
                    {
                        "paper_id": attempt.paper_id,
                        "query": attempt.query,
                        "lexical_terms": attempt.lexical_terms,
                        "requirement_ids": attempt.requirement_ids,
                    }
                    for attempt in observation.search_attempts
                ]
            }
        events.append({
            "iteration": observation.iteration,
            "tool": observation.action_name,
            "requirement_ids": _observation_requirement_ids(observation),
            "request": request,
            "outcome": observation.tool_status.value,
            "added_evidence": len(observation.added_evidence_ids),
            "reused_evidence": len(observation.reused_evidence_ids),
            "new_sections": observation.newly_exposed_section_ids,
            "errors": observation.error_codes,
        })
    rejections = list(state.action_rejections)
    if state.last_action_rejection and state.last_action_rejection not in rejections:
        rejections.append(state.last_action_rejection)
    for rejection in rejections:
        events.append({
            "iteration": rejection.iteration,
            "tool": rejection.action_name,
            "requirement_ids": rejection.requirement_ids,
            "request": _action_arguments(rejection.action_arguments),
            "outcome": rejection.rejection_code,
            "added_evidence": 0,
            "reused_evidence": 0,
            "new_sections": (),
            "errors": (rejection.rejection_code,),
        })
    return ordered_research_events(events)


def ordered_research_events(events):
    unique: dict[str, dict[str, object]] = {}
    for event in events:
        key = json.dumps(
            [event["iteration"], event["tool"], event["request"], event["outcome"]],
            sort_keys=True, ensure_ascii=False, default=list,
        )
        unique.setdefault(key, deepcopy(event))
    return sorted(unique.values(), key=lambda item: item["iteration"])


def compute_no_progress_signal(events, *, recent_limit: int = 3):
    """Retrieval-only progress signal; answer completeness lives elsewhere."""
    actions = [
        item for item in ordered_research_events(events)
        if item["tool"].startswith("search_")
        or item["tool"] in {"get_section_context", "get_paper_overview"}
    ]
    no_new = 0
    for item in reversed(actions):
        if item["added_evidence"] > 0:
            break
        no_new += 1
    recent = actions[-recent_limit:]
    evidence_iterations = [item["iteration"] for item in actions if item["added_evidence"] > 0]
    return {
        "consecutive_no_new_evidence_actions": no_new,
        "recent_section_reads_without_new_evidence": sum(
            item["tool"] == "get_section_context" and item["added_evidence"] == 0
            for item in recent
        ),
        "recent_duplicate_rejections": sum(
            item["outcome"] in {"DuplicateAction", "DuplicatePaperQuery"}
            for item in recent
        ),
        "last_new_evidence_iteration": evidence_iterations[-1] if evidence_iterations else None,
    }


def expand_decision_references(payload):
    return json.loads(json.dumps(payload, ensure_ascii=False, default=list))


def factor_decision_payload(payload, events, *, recent_limit: int = 3):
    result = deepcopy(payload)
    result["recent_research_events"] = ordered_research_events(events)[-recent_limit:]
    return result


def build_compact_decision_view(
    state: "DecisionState",
    errors: tuple["ValidationDiagnostic", ...] | list["ValidationDiagnostic"] = (),
    *,
    limits: CompactDecisionLimits | None = None,
) -> CompactDecisionView:
    from .decision import _action_recovery_constraint, _repair_payload, _runtime_action_capabilities

    limits = limits or CompactDecisionLimits()
    capabilities = _runtime_action_capabilities(state)
    grounded = state.grounded_answer
    answers = {item.requirement_id: item for item in grounded.requirements} if grounded else {}
    requirements = []
    for requirement in state.plan.requirements:
        answer = answers.get(requirement.requirement_id)
        requirements.append({
            "requirement_id": requirement.requirement_id,
            "description": requirement.description,
            "required": requirement.required,
            "available_retrieval_targets": sorted(
                target.value for target in requirement.evidence_targets
            ),
            "preferred_tools": preferred_tools_for_targets(
                requirement.evidence_targets, capabilities.available_tools
            ),
            "accepted_answer": ({
                "status": ("complete" if answer.complete else
                           "partial" if answer.answer is not None else "gap"),
                "cited_evidence_ids": tuple(dict.fromkeys(
                    citation.evidence_id for citation in answer.citations
                )),
                "remaining_gap": answer.remaining_gap,
            } if answer is not None else None),
        })

    ordered = sorted(
        state.evidence,
        key=lambda record: (-record.last_seen_iteration, record.evidence_id),
    )
    catalog_limit = max(1, limits.support_entries_per_requirement * 2)
    visible = ordered[:catalog_limit]
    newest_iteration = max((record.last_seen_iteration for record in ordered), default=0)
    catalog: list[dict[str, object]] = []
    latest_ids: list[str] = []
    for record in visible:
        source = record.evidence
        row = {
            "evidence_id": record.evidence_id,
            "paper_id": source.paper_id,
            "section_id": source.section_id,
            "section": _clip(source.section or "", 240),
            "source_chunk_ids": source.source_chunk_ids,
            "last_seen_iteration": record.last_seen_iteration,
            "source_preview": _clip(source.content, limits.support_scope_chars),
        }
        catalog.append(row)
        if record.last_seen_iteration == newest_iteration:
            latest_ids.append(record.evidence_id)

    coverage = state.coverage
    events = build_recent_research_events(state)
    recovery = _action_recovery_constraint(state)
    if state.last_action_rejection is not None:
        rejection = state.last_action_rejection
        events = [
            event for event in events
            if not (
                event["iteration"] == rejection.iteration
                and event["tool"] == rejection.action_name
                and event["outcome"] == rejection.rejection_code
            )
        ]
    payload: dict[str, object] = {
        "view_type": "GroundedAnswerDecisionView",
        "question": state.context.resolved_query,
        "research_goal": state.plan.goal,
        "scope": state.context.scope.value,
        "trusted_paper_ids": state.context.target_paper_ids,
        "runtime_action_capabilities": _runtime_decision_view(capabilities),
        "action_recovery": recovery,
        "requirements": requirements,
        "latest_tool_evidence_ids": latest_ids,
        "evidence_catalog": catalog,
        "evidence_catalog_omitted": max(0, len(ordered) - len(visible)),
        "recent_research_events": events[-max(
            limits.recent_actions, limits.recent_observations, limits.recent_queries
        ):],
        "retrieval_stagnation": compute_no_progress_signal(events),
        "candidate_coverage": ({
            "screening_complete": coverage.coverage_complete,
            "strategy": coverage.strategy.value,
            "remaining_paper_ids": coverage.remaining_paper_ids,
        } if coverage is not None else None),
        "iteration_budget": {
            "limit": state.max_agent_iterations,
            "used": state.budget_usage.agent_iterations,
            "remaining": (
                max(0, state.max_agent_iterations - state.budget_usage.agent_iterations)
                if state.max_agent_iterations is not None else None
            ),
        },
        "previous_validation_errors": (
            [] if recovery is not None
            else [_repair_payload(error) for error in errors[-1:]]
        ),
    }
    if not state.observations and not state.action_rejections:
        payload["initial_search_hints"] = build_initial_search_hints(state.context)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=list)
    metrics = {
        "number_of_requirements": len(requirements),
        "number_of_evidence_catalog_entries": len(catalog),
        "number_of_evidence_catalog_entries_dropped": max(0, len(ordered) - len(visible)),
        "number_of_recent_research_events": len(payload["recent_research_events"]),
        "decision_view_chars": len(serialized),
        "decision_view_bytes": len(serialized.encode("utf8")),
    }
    return CompactDecisionView(payload, metrics)
