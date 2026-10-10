"""Plan-authoritative retrieval/routing intent; never support or completion."""
from __future__ import annotations

from .contracts import AgentAction, EvidenceTarget, ResearchPlan, SearchPapersAction


def targets_for_requirement_ids(
    plan: ResearchPlan, requirement_ids: tuple[str, ...]
) -> frozenset[EvidenceTarget]:
    """Union declared Plan hints; reject unknown bindings without interpreting text."""
    by_id = {item.requirement_id: item for item in plan.requirements}
    unknown = set(requirement_ids) - by_id.keys()
    if unknown:
        raise ValueError("Requirement target binding is not in Plan: " + ", ".join(sorted(unknown)))
    return frozenset(target for rid in requirement_ids for target in by_id[rid].evidence_targets)


def requirement_target_bindings(
    plan: ResearchPlan, action: AgentAction
) -> tuple[dict[str, object], ...]:
    """JSON-safe audit rows, preserving each search's identity in a batch."""
    if isinstance(action, SearchPapersAction):
        groups = [(search.requirement_ids, {"search_index": index, "paper_id": search.paper_id})
                  for index, search in enumerate(action.searches)]
    else:
        groups = [(getattr(action, "requirement_ids", ()), {})]
    rows = []
    for ids, location in groups:
        targets_for_requirement_ids(plan, ids)  # validate even empty Tool outcomes
        for rid in dict.fromkeys(ids):
            rows.append({**location, "requirement_id": rid,
                         "evidence_targets": sorted(t.value for t in targets_for_requirement_ids(plan, (rid,)))})
    return tuple(rows)


def preferred_tools_for_targets(
    targets: frozenset[EvidenceTarget], available_tools: tuple[str, ...]
) -> tuple[str, ...]:
    """Soft routing preferences intersected with the existing Runtime domain."""
    tools = []
    for target in sorted(targets, key=lambda item: item.value):
        preferred = ("search_references",) if target is EvidenceTarget.REFERENCE else (
            ("get_paper_overview",) if target is EvidenceTarget.FIELD_OVERVIEW
            else ("search_papers", "get_section_context"))
        tools.extend(tool for tool in preferred if tool in available_tools)
    return tuple(dict.fromkeys(tools))
