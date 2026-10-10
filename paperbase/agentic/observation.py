"""把原始 ToolResult 压缩为 Decision Agent 可消费的 Observation。"""

from __future__ import annotations

from paperbase.retrieval.contracts import (
    KnowledgeBaseSearchPayload,
    ReferenceSearchPayload,
    SearchPapersPayload,
    ToolResult,
)

from .contracts import (
    AgentAction,
    ActionCanonicalization,
    EvidenceFilterReport,
    EvidenceTarget,
    PaperSearchAttempt,
    RegistryDelta,
    ResearchObservation,
    ResearchPlan,
    GetPaperOverviewAction,
    GetSectionContextAction,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)

from .requirement_targets import targets_for_requirement_ids, requirement_target_bindings


def build_observation(
    *,
    iteration: int,
    result: ToolResult,
    delta: RegistryDelta,
    action: AgentAction | None = None,
    plan: ResearchPlan | None = None,
    evidence_filter_report: EvidenceFilterReport | None = None,
    action_canonicalization: ActionCanonicalization | None = None,
) -> ResearchObservation:
    queries_by_paper: list[tuple[str, str]] = []
    error_codes: list[str] = []
    search_attempts: list[PaperSearchAttempt] = []
    if isinstance(result.payload, SearchPapersPayload):
        for index, outcome in enumerate(result.payload.paper_results):
            search = (
                action.searches[index]
                if isinstance(action, SearchPapersAction)
                else outcome.search
            )
            queries_by_paper.append((outcome.paper_id, outcome.query))
            if outcome.error_code:
                error_codes.append(outcome.error_code)
            search_attempts.append(
                PaperSearchAttempt(
                    paper_id=outcome.paper_id,
                    query=outcome.query,
                    lexical_terms=outcome.lexical_terms,
                    evidence_targets=(targets_for_requirement_ids(plan, search.requirement_ids)
                        if plan is not None else frozenset(
                            EvidenceTarget(value) for value in outcome.evidence_targets)),
                    requirement_ids=search.requirement_ids,
                    status=outcome.status.value,
                    error_code=outcome.error_code,
                )
            )
    elif isinstance(result.payload, ReferenceSearchPayload):
        queries_by_paper.append((result.payload.paper_id, result.payload.query))
    elif isinstance(result.payload, KnowledgeBaseSearchPayload):
        queries_by_paper.extend(
            (evidence.paper_id, result.payload.query)
            for evidence in result.payload.evidence_candidates
        )
    if result.error_code:
        error_codes.append(result.error_code)
    return ResearchObservation(
        iteration=iteration,
        requirement_target_bindings=(requirement_target_bindings(plan, action)
            if plan is not None and action is not None else ()),
        raw_action_arguments=result.metadata.get("raw_action_arguments", _action_arguments(action)),
        canonical_effective_requests=tuple(result.metadata.get("canonical_effective_requests", ())),
        request_identities=tuple(item["request_identity"] for item in result.metadata.get("canonical_effective_requests", ())),
        request_executions=tuple(result.metadata.get("request_executions", ())),
        request_contract_version=result.metadata.get("request_contract_version", "legacy"),
        action_name=result.tool_name.value,
        tool_status=result.status,
        added_evidence_ids=delta.added_ids,
        reused_evidence_ids=delta.reused_ids,
        queries_by_paper=tuple(dict.fromkeys(queries_by_paper)),
        error_codes=tuple(dict.fromkeys(error_codes)),
        action_arguments=_action_arguments(action),
        search_attempts=tuple(search_attempts),
        evidence_filter_report=evidence_filter_report,
        action_canonicalization=action_canonicalization,
    )


def _action_arguments(action: AgentAction | None) -> dict[str, object]:
    """Expose the chosen Tool parameters without storing internal state."""
    if action is None:
        return {}
    if isinstance(action, SearchPapersAction):
        return {
            "searches": [
                {
                    "paper_id": search.paper_id,
                    "query": search.query,
                    "lexical_terms": list(search.lexical_terms),
                    "requirement_ids": list(search.requirement_ids),
                    "evidence_targets": sorted(search.evidence_targets),
                }
                for search in action.searches
            ]
        }
    if isinstance(action, SearchKnowledgeBaseAction):
        return {
            "query": action.query,
            "lexical_terms": list(action.lexical_terms),
            "requirement_ids": list(action.requirement_ids),
        }
    if isinstance(action, SearchReferencesAction):
        return {
            "paper_id": action.paper_id,
            "query": action.query,
            "lexical_terms": list(action.lexical_terms),
            "requirement_ids": list(action.requirement_ids),
        }
    if isinstance(action, GetSectionContextAction):
        return {
            "paper_id": action.paper_id,
            "section_id": action.section_id,
            "requirement_ids": list(action.requirement_ids),
        }
    if isinstance(action, GetPaperOverviewAction):
        return {
            "paper_id": action.paper_id,
            "requirement_ids": list(action.requirement_ids),
        }
    return {}
