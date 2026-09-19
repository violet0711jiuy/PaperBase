"""把原始 ToolResult 压缩为 Decision Agent 可消费的 Observation。"""

from __future__ import annotations

from paperbase.retrieval.contracts import (
    KnowledgeBaseSearchPayload,
    ReferenceSearchPayload,
    SearchPapersPayload,
    ToolResult,
)

from .contracts import EvidenceTarget, PaperSearchAttempt, RegistryDelta, ResearchObservation


def build_observation(
    *,
    iteration: int,
    result: ToolResult,
    delta: RegistryDelta,
    remaining_budget: dict[str, int],
) -> ResearchObservation:
    paper_statuses: list[tuple[str, str]] = []
    queries_by_paper: list[tuple[str, str]] = []
    error_codes: list[str] = []
    search_attempts: list[PaperSearchAttempt] = []
    if isinstance(result.payload, SearchPapersPayload):
        for outcome in result.payload.paper_results:
            paper_statuses.append((outcome.paper_id, outcome.status.value))
            queries_by_paper.append((outcome.paper_id, outcome.query))
            if outcome.error_code:
                error_codes.append(outcome.error_code)
            search_attempts.append(
                PaperSearchAttempt(
                    paper_id=outcome.paper_id,
                    query=outcome.query,
                    lexical_terms=outcome.lexical_terms,
                    evidence_targets=frozenset(
                        EvidenceTarget(value) for value in outcome.evidence_targets
                    ),
                    missing_information=outcome.missing_information,
                    target_subtask_id=outcome.target_subtask_id,
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
        action_name=result.tool_name.value,
        tool_status=result.status,
        added_evidence_ids=delta.added_ids,
        reused_evidence_ids=delta.reused_ids,
        paper_statuses=tuple(paper_statuses),
        queries_by_paper=tuple(dict.fromkeys(queries_by_paper)),
        error_codes=tuple(dict.fromkeys(error_codes)),
        no_new_evidence=not delta.added_ids,
        remaining_budget=dict(remaining_budget),
        search_attempts=tuple(search_attempts),
    )
