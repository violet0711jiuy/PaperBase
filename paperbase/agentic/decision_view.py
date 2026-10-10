"""Deterministic, ephemeral Decision input derived from the full Graph State.

No model, cache, state updates, or semantic gap inference lives here. Runtime
legality is projected once; bounded local boundaries retain Evidence provenance.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import unicodedata
from typing import TYPE_CHECKING

from .contracts import EvidenceOwnership, FactProjection
from .grounding import source_contains, verified_support_ids
from .requirement_coverage import derive_requirement_coverage
from .requirement_targets import preferred_tools_for_targets
from .initial_search_hints import build_initial_search_hints

if TYPE_CHECKING:
    from .decision import DecisionState
    from paperbase.llm.client import ValidationDiagnostic


@dataclass(frozen=True)
class CompactDecisionLimits:
    support_entries_per_requirement: int = 6
    recent_observations: int = 2
    recent_queries: int = 3
    recent_actions: int = 3
    sections: int = 10
    support_scope_chars: int = 600
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
    """A rebuilt input value, never checkpointed as business state."""

    payload: dict[str, object]
    metrics: dict[str, int | float]

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.payload)

    def to_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"), default=list)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _unique(values):
    seen = set()
    result = []
    for value in values:
        key = _normalized(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


def _runtime_decision_view(capabilities) -> dict[str, object]:
    """Expose each executable-domain fact once.

    Compatibility mirrors and executed-action telemetry stay in Graph State and
    diagnostics.  The Decision model receives only the current legal domain;
    the JSON Schema independently enforces the same domain.
    """
    return {
        "available_tools": capabilities.available_tools,
        "valid_requirement_ids": capabilities.valid_requirement_ids,
        "search_paper_ids": capabilities.search_paper_ids,
        "reference_paper_ids": capabilities.reference_paper_ids,
        "overview_paper_ids": capabilities.overview_paper_ids,
        "available_section_ids_by_paper": capabilities.available_section_ids_by_paper,
        "tool_budget_state": {
            key: value
            for key, value in capabilities.tool_budget_state.items()
            if key != "available_tools"
        },
        "capability_reasons": capabilities.capability_reasons,
    }


def _action_arguments(arguments: dict[str, object]) -> dict[str, object]:
    # Tool request shape only. Never leak injected document/debug fields.
    keys = (
        "paper_id", "section_id", "query", "lexical_terms",
        "requirement_ids", "reason",
    )
    result = {name: deepcopy(arguments[name]) for name in keys if name in arguments}
    searches = arguments.get("searches")
    if isinstance(searches, (list, tuple)):
        result["searches"] = [
            {name: deepcopy(item[name]) for name in keys if name in item}
            for item in searches if isinstance(item, dict)
        ]
    return result


def _compact_research_leads_view(record, limits):
    """Expose semantic projection content only as retrieval-planning leads.

    Formal Requirement support is intentionally omitted here.  The Decision
    Agent must use requirements[].local_supports for support authority; these lead
    values only help formulate the next query,
    section drill-down, or tool choice.
    """
    from .decision import _projections_view

    blocked_fields = {
        "supported_requirement_ids",
        "proposed_requirement_ids",
        "support_anchors",
        "source_excerpt",
        "support_scope",
        "unresolved_aspects",
    }
    text_fields = {
        "mechanism", "value", "fact_type", "canonical_name",
        "steps", "inputs", "outputs", "comparison_dimensions",
    }
    result = []
    for projection in _projections_view(record):
        truncated = False

        def bounded(value, field=None):
            nonlocal truncated
            if isinstance(value, dict):
                return {
                    key: bounded(item, key)
                    for key, item in value.items()
                    if key not in blocked_fields
                }
            if isinstance(value, (tuple, list)):
                return [bounded(item, field) for item in value]
            if isinstance(value, str) and field in text_fields:
                clipped = _clip(value, limits.support_scope_chars)
                truncated |= clipped != value
                return clipped
            return value

        view = bounded(projection)
        if not isinstance(view, dict):
            continue
        view["text_truncated"] = truncated
        result.append(view)
    return result


def _build_unfactored_decision_view(
    state: DecisionState,
    errors: tuple[ValidationDiagnostic, ...] | list[ValidationDiagnostic] = (),
    *,
    limits: CompactDecisionLimits | None = None,
) -> CompactDecisionView:
    limits = limits or CompactDecisionLimits()
    # Local imports share the existing Runtime/Guard projection without a second
    # capability policy or a module cycle.
    from .decision import (
        _action_recovery_constraint, _repair_payload,
        _runtime_action_capabilities,
    )

    support_by_id = {item.requirement_id: [] for item in state.plan.requirements}
    from .grounding import verified_projection_support_ids
    # Existing Integrity is the only support gate. Filter each Fact part too:
    # one verified sibling must not legitimize another unknown-owner part.
    for record in sorted(state.evidence, key=lambda r: -r.last_seen_iteration):
        for projection in record.projections:
            valid = set(verified_projection_support_ids(record, projection, state.plan.requirements))
            parts = projection.facts if isinstance(projection, FactProjection) else (projection,)
            for part in parts:
                if part is None or getattr(part, "ownership", None) is EvidenceOwnership.UNKNOWN:
                    continue
                supported = set(part.supported_requirement_ids)
                if hasattr(part, "proposed_requirement_ids"):
                    supported.intersection_update(part.proposed_requirement_ids)
                for anchor in part.support_anchors:
                    rid = anchor.requirement_id
                    if rid not in valid or rid not in supported or not source_contains(record.evidence.content, anchor.source_excerpt):
                        continue
                    unresolved = _unique(anchor.unresolved_aspects)
                    scope = _clip(anchor.support_scope, limits.support_scope_chars)
                    local = [_clip(s, limits.unresolved_aspect_chars) for s in unresolved[:limits.unresolved_aspects]]
                    support_by_id[rid].append({
                        "evidence_id": record.evidence_id,
                        "paper_id": record.evidence.paper_id,
                        "support_scope": scope,
                        "unresolved_aspects": local,
                        "text_truncated": scope != anchor.support_scope or local != unresolved,
                    })

    runtime_capabilities = _runtime_action_capabilities(state)
    from .decision import _observation_requirement_ids

    searched_requirement_ids = {
        requirement_id
        for observation in state.observations
        for requirement_id in _observation_requirement_ids(observation)
    }
    searched_requirement_ids.update(
        requirement_id
        for record in state.evidence
        for requirement_id in record.requested_requirement_ids
    )

    requirements = []
    dropped_support = 0
    for requirement in state.plan.requirements:
        boundaries = support_by_id[requirement.requirement_id]
        seen = set()
        unique = []
        for entry in boundaries:
            key = (entry["evidence_id"], _normalized(entry["support_scope"]), tuple(map(_normalized, entry["unresolved_aspects"])))
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        # One Evidence appears once per Requirement.  Multiple qualified anchors
        # become local coverage/boundary lists instead of repeated records.
        grouped: dict[str, dict[str, object]] = {}
        for entry in unique:
            evidence_id = entry["evidence_id"]
            group = grouped.setdefault(evidence_id, {
                "evidence_id": evidence_id,
                "paper_id": entry["paper_id"],
                "covers": [],
                "boundaries": [],
                "text_truncated": False,
            })
            group["covers"].append(entry["support_scope"])
            group["boundaries"].extend(entry["unresolved_aspects"])
            group["text_truncated"] = bool(
                group["text_truncated"] or entry["text_truncated"]
            )
        local_supports = []
        for group in grouped.values():
            group["covers"] = _unique(group["covers"])
            group["boundaries"] = _unique(group["boundaries"])
            local_supports.append(group)
        visible = local_supports[:limits.support_entries_per_requirement]
        omitted = len(local_supports) - len(visible)
        dropped_support += omitted
        hints = derive_requirement_coverage(requirement, state.evidence)
        retrieval_state = (
            "support_available"
            if local_supports
            else "searched_no_support"
            if requirement.requirement_id in searched_requirement_ids
            else "not_searched"
        )
        requirements.append({
            "requirement_id": requirement.requirement_id,
            "description": requirement.description,
            "projection_type": requirement.projection_type.value if requirement.projection_type else None,
            "required": requirement.required,
            # Retrieval state is an observation, never a completeness verdict.
            "retrieval_state": retrieval_state,
            # These are possible evidence routes, not a second completion signal.
            # Completion authority stays in the deterministic answering/Guard
            # path; the Decision view only reports what retrieval has exposed.
            "available_retrieval_targets": sorted(
                t.value for t in requirement.evidence_targets
            ),
            "preferred_tools": preferred_tools_for_targets(
                requirement.evidence_targets, runtime_capabilities.available_tools),
            "local_supports": visible,
            "local_supports_omitted": omitted,
            "lexical_hints": {
                "critical": hints.critical_anchors,
                # Lexical surface coverage helps compose a query; it does not
                # imply a semantic Requirement gap.
                "surface_missing": hints.missing_anchors,
            },
        })

    sections = []
    section_keys = set()
    for record in sorted(state.evidence, key=lambda r: (not bool(verified_support_ids(r)), -r.last_seen_iteration)):
        evidence = record.evidence
        key = (evidence.paper_id, evidence.section_id)
        if not evidence.section_id or key in section_keys:
            continue
        section_keys.add(key)
        sections.append({"section_id": evidence.section_id, "heading": _clip(evidence.section or "", 240),
                         "paper_id": evidence.paper_id, "discovered": True})
    entities = []
    for record in reversed(state.evidence):
        for name in (*record.method_entities, *record.method_families, *record.baseline_entities):
            entry = {"evidence_id": record.evidence_id, "paper_id": record.evidence.paper_id, "name": name}
            if entry not in entities:
                entities.append(entry)
    coverage = state.coverage
    # Leads exist only for Requirements with no formal support. A record may
    # support R1 and still be a useful R2 lead, so filter by the record's request
    # binding rather than dropping every partially supporting record wholesale.
    zero_support_requirement_ids = {
        requirement.requirement_id
        for requirement in state.plan.requirements
        if not support_by_id[requirement.requirement_id]
    }
    lead_records = [
        record
        for record in sorted(state.evidence, key=lambda item: -item.last_seen_iteration)
        if record.projections
        and zero_support_requirement_ids.intersection(record.requested_requirement_ids)
    ]
    visible_leads = lead_records[:limits.research_leads]
    payload = {
        "view_type": "AtomicDecisionView",
        "question": state.context.resolved_query,
        "research_goal": state.plan.goal,
        "scope": state.context.scope.value,
        "trusted_paper_ids": state.context.target_paper_ids,
        "requirements": requirements,
        "research_leads": [{
            "evidence_id": r.evidence_id,
            "paper_id": r.evidence.paper_id,
            "requirement_ids": sorted(
                zero_support_requirement_ids.intersection(r.requested_requirement_ids)
            ),
            "section_id": r.evidence.section_id,
            "section": _clip(r.evidence.section or "", 240),
            "usage": "retrieval_planning_only",
            "semantic_content": _compact_research_leads_view(r, limits),
        } for r in visible_leads],
        "entity_hints": entities[:limits.entity_hints],
        "available_context": {
            "known_sections": sections[:limits.sections],
            "sections_omitted": max(0, len(sections) - limits.sections),
        },
        "coverage": ({
            "strategy": coverage.strategy.value, "complete": coverage.coverage_complete,
            "required_paper_ids": coverage.required_paper_ids,
            "remaining_paper_ids": coverage.remaining_paper_ids,
            "screened_paper_ids": coverage.screened_paper_ids,
            "candidate_paper_ids": coverage.candidate_paper_ids,
            "uncertain_paper_ids": coverage.uncertain_paper_ids,
            "excluded_paper_ids": coverage.excluded_paper_ids,
            "assessments": [{"paper_id": a.paper_id, "status": a.status.value} for a in coverage.assessments],
        } if coverage else None),
        "action_recovery": _action_recovery_constraint(state),
        "runtime_action_capabilities": _runtime_decision_view(runtime_capabilities),
        "iteration_budget": {"limit": state.max_agent_iterations, "used": state.budget_usage.agent_iterations,
                             "remaining": max(0, state.max_agent_iterations - state.budget_usage.agent_iterations)
                             if state.max_agent_iterations is not None else None},
        "previous_validation_errors": [_repair_payload(e) for e in errors[-1:]],
    }
    # Cold-start hints stop being useful once an actual request/result exists.
    # Omitting them later avoids presenting a stale second query plan.
    if not state.observations and not state.action_rejections:
        payload["initial_search_hints"] = build_initial_search_hints(state.context)
    metrics = {
        "number_of_requirements": len(requirements),
        "number_of_support_entries": sum(len(r["local_supports"]) for r in requirements),
        "number_of_support_entries_dropped": dropped_support,
        "number_of_research_leads": len(visible_leads),
        "number_of_research_leads_dropped": max(0, len(lead_records) - len(visible_leads)),
        "number_of_sections": min(len(sections), limits.sections),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=list)
    metrics.update(decision_view_chars=len(serialized), decision_view_bytes=len(serialized.encode("utf8")))
    return CompactDecisionView(deepcopy(payload), metrics)

# P1.3a: references below are a wire representation only. Guard and action
# Schema continue to consume the full authoritative Graph State.
def build_recent_research_events(state: DecisionState) -> list[dict[str, object]]:
    """One event per observed/rejected action, never per search attempt."""
    from .decision import _observation_requirement_ids
    events = []
    for o in state.observations:
        request = _action_arguments(o.action_arguments)
        if o.action_name.startswith('search_') and not request:
            request = {'searches': [dict(paper_id=a.paper_id, query=a.query,
                lexical_terms=a.lexical_terms, requirement_ids=a.requirement_ids)
                for a in o.search_attempts]}
            if not request['searches']:
                request['searches'] = [dict(paper_id=p, query=q,
                    requirement_ids=_observation_requirement_ids(o)) for p,q in o.queries_by_paper]
        events.append(dict(iteration=o.iteration, tool=o.action_name,
            requirement_ids=_observation_requirement_ids(o), request=request,
            added_evidence=len(o.added_evidence_ids), reused_evidence=len(o.reused_evidence_ids),
            new_verified_support=o.new_supporting_evidence_count,
            new_sections=o.newly_exposed_section_ids, new_entities=o.new_goal_items,
            newly_supported_requirement_ids=o.newly_supported_requirement_ids,
            outcome=o.tool_status.value, errors=o.error_codes, goal_progress=o.goal_progress))
    rejections = list(state.action_rejections)
    if state.last_action_rejection and state.last_action_rejection not in rejections:
        rejections.append(state.last_action_rejection)
    for r in rejections:
        events.append(dict(iteration=r.iteration, tool=r.action_name,
            requirement_ids=r.requirement_ids, request=_action_arguments(r.action_arguments),
            added_evidence=0, reused_evidence=0, new_verified_support=0,
            new_sections=(), new_entities=(), outcome=r.rejection_code))
    return ordered_research_events(events)


def ordered_research_events(events):
    """Deduplicate copies of an event, never erase an actual repeated action."""
    unique = {}
    for event in events:
        key = json.dumps([event['iteration'], event['tool'], event['request'], event['outcome']],
                         sort_keys=True, ensure_ascii=False, default=list)
        unique.setdefault(key, deepcopy(event))
    return sorted(unique.values(), key=lambda e: e['iteration'])


def compute_no_progress_signal(events, *, recent_limit=3):
    """Counts only; rejected actions have zero deltas and count in streaks.

    Support and Evidence timestamps are independent; new sections, entities,
    reused Evidence and goal/status labels do not reset the streaks.
    Recent counters use the same action window as recent_research_events.
    """
    events = ordered_research_events(events)
    actions = [e for e in events if e['tool'].startswith('search_') or
               e['tool'] in ('get_section_context', 'get_paper_overview')]
    def streak(field):
        count = 0
        for e in reversed(actions):
            if e[field] > 0:
                break
            count += 1
        return count
    recent = actions[-recent_limit:]
    support_iterations = [e['iteration'] for e in actions if e['new_verified_support'] > 0]
    evidence_iterations = [e['iteration'] for e in actions if e['added_evidence'] > 0]
    targeted_support_iterations = [
        e['iteration'] for e in actions if _has_targeted_support_progress(e)
    ]
    return dict(consecutive_no_support_actions=streak('new_verified_support'),
                consecutive_no_new_evidence_actions=streak('added_evidence'),
                recent_section_reads_without_support=sum(e['tool']=='get_section_context' and
                    e['new_verified_support']==0 for e in recent),
                recent_duplicate_rejections=sum(e['outcome'] in ('DuplicateAction', 'DuplicatePaperQuery')
                                                for e in recent),
                last_new_support_iteration=support_iterations[-1] if support_iterations else None,
                last_new_evidence_iteration=evidence_iterations[-1] if evidence_iterations else None,
                consecutive_no_targeted_support_actions=_targeted_support_streak(actions),
                last_targeted_support_iteration=(targeted_support_iterations[-1]
                                                   if targeted_support_iterations else None))


def _has_targeted_support_progress(event) -> bool:
    """Whether this Action newly supported one of its bound Requirements."""

    targeted = {
        value for value in event.get('requirement_ids', ())
        if isinstance(value, str)
    }
    newly_supported = {
        value for value in event.get('newly_supported_requirement_ids', ())
        if isinstance(value, str)
    }
    return bool(targeted.intersection(newly_supported))


def _targeted_support_streak(actions) -> int:
    count = 0
    for event in reversed(actions):
        if _has_targeted_support_progress(event):
            break
        count += 1
    return count


def factor_decision_payload(payload, events, *, recent_limit=3):
    """Build a bounded atomic history: each request sits beside its result.

    Query/request interning made the model join three collections to understand
    one tool call and duplicated the same information structurally.  Keeping a
    few complete events is both smaller and easier to reason about.
    """
    p = deepcopy(payload)
    events = ordered_research_events(events)
    visible = events[-recent_limit:]
    # Preserve the last search delta even after several section/overview reads.
    last_search = next((e for e in reversed(events) if e['tool'].startswith('search_')), None)
    if last_search is not None and last_search not in visible:
        visible = sorted([last_search, *visible], key=lambda e:e['iteration'])
    p['recent_research_events'] = deepcopy(visible)
    coverage = p.pop('coverage', None)
    p['candidate_coverage'] = ({
        'screening_complete': coverage['complete'],
        'strategy': coverage['strategy'],
        'remaining_paper_ids': coverage['remaining_paper_ids'],
        'assessments': coverage['assessments'],
    } if coverage is not None else None)
    context = p['available_context']
    # Semantics appear once here; complete legal ID sets remain in Runtime.
    context['section_metadata'] = {}
    for section in context.pop('known_sections'):
        context['section_metadata'].setdefault(section['paper_id'],{})[section['section_id']] = section['heading']
    p['view_type'] = 'AtomicDecisionView'
    # Put the two authority envelopes before soft context in the model input.
    leading = ('view_type', 'question', 'research_goal', 'scope', 'trusted_paper_ids',
               'runtime_action_capabilities', 'action_recovery')
    return {**{k:p[k] for k in leading}, **{k:v for k,v in p.items() if k not in leading}}


def expand_decision_references(payload):
    """Backward-compatible diagnostic copy; atomic views have no references."""
    if 'research_queries' not in payload and 'research_requests' not in payload:
        return json.loads(json.dumps(payload, ensure_ascii=False, default=list))

    # Retain the reader for archived P1.3 payloads.
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={'$query'}:
                return payload['research_queries'][value['$query']]
            if set(value)=={'$request'}:
                return expand(payload['research_requests'][value['$request']])
            return {k:expand(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [expand(v) for v in value]
        return value
    return expand(deepcopy(payload))


def build_compact_decision_view(state, errors=(), *, limits=None):
    limits = limits or CompactDecisionLimits()
    base = _build_unfactored_decision_view(state, errors, limits=limits)
    events = build_recent_research_events(state)
    p = factor_decision_payload(base.payload, events, recent_limit=max(
        limits.recent_actions, limits.recent_observations, limits.recent_queries))
    metrics = {k:v for k,v in base.metrics.items() if k not in (
        'number_of_recent_observations','number_of_recent_queries','number_of_recent_actions',
        'number_of_history_items_dropped')}
    metrics.update(number_of_recent_research_events=len(p['recent_research_events']),
        number_of_history_items_dropped=len(events)-len(p['recent_research_events']))
    s=json.dumps(p,ensure_ascii=False,separators=(',',':'),default=list)
    metrics.update(decision_view_chars=len(s),decision_view_bytes=len(s.encode('utf8')))
    return CompactDecisionView(p,metrics)
