"""把内部逐论文检索快照转换为稳定、无 Registry 编号的 Tool 输出。"""

from __future__ import annotations

from collections.abc import Sequence

from .adapters import PaperRetrievalSnapshot
from .contracts import (
    EvidenceSourceType,
    PaperBatchExecution,
    PaperSearchOutcome,
    PaperSearchStatus,
    QueryOrigin,
    RetrievedEvidence,
    ScoreKind,
    ScoreDirection,
    SearchPapersPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)
from .evidence_scores import chunk_score_fields


def build_search_papers_tool_result(
    executions: Sequence[PaperBatchExecution],
    *,
    query_origin: QueryOrigin,
) -> ToolResult:
    outcomes = tuple(_execution_to_outcome(execution) for execution in executions)
    return ToolResult(
        status=_aggregate_status(outcomes),
        tool_name=ToolName.SEARCH_PAPERS,
        payload=SearchPapersPayload(paper_results=outcomes),
        metadata={
            "query_origin": QueryOrigin(query_origin).value,
            "requested_paper_count": len(outcomes),
        },
    )


def _execution_to_outcome(execution: PaperBatchExecution) -> PaperSearchOutcome:
    if execution.error_code is not None:
        status = (
            PaperSearchStatus.UNAVAILABLE
            if execution.error_code in {"LookupError", "FileNotFoundError"}
            else PaperSearchStatus.ERROR
        )
        return PaperSearchOutcome(
            search=execution.search,
            status=status,
            error_code=execution.error_code,
        )
    if not isinstance(execution.result, PaperRetrievalSnapshot):
        return PaperSearchOutcome(
            search=execution.search,
            status=PaperSearchStatus.ERROR,
            error_code="InvalidAdapterResult",
        )
    evidence = _snapshot_to_evidence(execution.result)
    return PaperSearchOutcome(
        search=execution.search,
        status=PaperSearchStatus.OK if evidence else PaperSearchStatus.EMPTY,
        evidence_candidates=evidence,
    )


def _snapshot_to_evidence(
    snapshot: PaperRetrievalSnapshot,
) -> tuple[RetrievedEvidence, ...]:
    chunks_by_id = {
        chunk.chunk_id: chunk for chunk in snapshot.retrieval.chunks
    }
    if snapshot.expansion is not None:
        evidence: list[RetrievedEvidence] = []
        for unit in snapshot.expansion.evidence:
            seeds = [
                chunks_by_id[chunk_id]
                for chunk_id in unit.seed_chunk_ids
                if chunk_id in chunks_by_id
            ]
            # An expanded window is not reranked itself. Inherit the best
            # returned seed's contract, never max heterogeneous seed scores.
            best_seed = min(seeds, key=lambda chunk: chunk.rank) if seeds else None
            score_fields = (chunk_score_fields(best_seed) if best_seed is not None else
                            dict(raw_score=None, score_kind=ScoreKind.NON_RANKED,
                                 score_direction=ScoreDirection.NOT_APPLICABLE, retrieval_rank=None))
            evidence.append(
                RetrievedEvidence(
                    source_type=EvidenceSourceType(unit.kind),
                    paper_id=unit.paper_id,
                    paper_title=unit.paper_title,
                    seed_chunk_ids=unit.seed_chunk_ids,
                    source_chunk_ids=unit.chunk_ids,
                    section_id=unit.section_id,
                    section=unit.section,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    content=unit.text,
                    **score_fields,
                    retrieval_metadata={
                        "reranking_status": snapshot.retrieval.reranking_status,
                        "expanded": True,
                        "score_scope": "best_seed_chunk" if best_seed is not None else "non_ranked",
                        "score_seed_chunk_id": best_seed.chunk_id if best_seed is not None else None,
                        "rerank_score_normalized": getattr(best_seed, "rerank_score_normalized", None),
                    },
                )
            )
        return tuple(evidence)
    return tuple(
        RetrievedEvidence(
            source_type=EvidenceSourceType(chunk.section_type),
            paper_id=chunk.paper_id,
            paper_title=chunk.paper_title,
            seed_chunk_ids=(chunk.chunk_id,),
            source_chunk_ids=(chunk.chunk_id,),
            section_id=chunk.section_id,
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            content=chunk.raw_text,
            **chunk_score_fields(chunk),
            retrieval_metadata={
                "rank": chunk.rank,
                "pre_rerank_rank": chunk.pre_rerank_rank,
                "reranking_status": snapshot.retrieval.reranking_status,
                "rerank_score_normalized": getattr(chunk, "rerank_score_normalized", None),
                "expanded": False,
                "source_matches": tuple(
                    {
                        "route": match.route,
                        "query": match.query,
                        "rank": match.rank,
                        "raw_score": match.raw_score,
                        "score_kind": match.score_kind.value,
                        "score_direction": match.score_direction.value,
                        "effective_weight": match.effective_weight,
                    }
                    for match in chunk.source_matches
                ),
            },
        )
        for chunk in snapshot.retrieval.chunks
    )


def _aggregate_status(outcomes: Sequence[PaperSearchOutcome]) -> ToolStatus:
    statuses = tuple(outcome.status for outcome in outcomes)
    if statuses and all(status is PaperSearchStatus.EMPTY for status in statuses):
        return ToolStatus.EMPTY
    if statuses and all(status is PaperSearchStatus.UNAVAILABLE for status in statuses):
        return ToolStatus.UNAVAILABLE
    if statuses and all(status is PaperSearchStatus.ERROR for status in statuses):
        return ToolStatus.ERROR
    has_failure = any(
        status in {PaperSearchStatus.ERROR, PaperSearchStatus.UNAVAILABLE}
        for status in statuses
    )
    has_success = any(
        status in {PaperSearchStatus.OK, PaperSearchStatus.EMPTY}
        for status in statuses
    )
    if has_failure and has_success:
        return ToolStatus.PARTIAL
    if has_failure:
        return ToolStatus.ERROR
    return ToolStatus.OK
