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
    SearchPapersPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


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
            paper_id=execution.paper_id,
            status=status,
            query=execution.query,
            lexical_terms=execution.lexical_terms,
            error_code=execution.error_code,
        )
    if not isinstance(execution.result, PaperRetrievalSnapshot):
        return PaperSearchOutcome(
            paper_id=execution.paper_id,
            status=PaperSearchStatus.ERROR,
            query=execution.query,
            lexical_terms=execution.lexical_terms,
            error_code="InvalidAdapterResult",
        )
    evidence = _snapshot_to_evidence(execution.result)
    return PaperSearchOutcome(
        paper_id=execution.paper_id,
        status=PaperSearchStatus.OK if evidence else PaperSearchStatus.EMPTY,
        query=execution.query,
        lexical_terms=execution.lexical_terms,
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
            seed_scores = [
                chunks_by_id[chunk_id].rerank_score
                if chunks_by_id[chunk_id].rerank_score is not None
                else chunks_by_id[chunk_id].fused_score
                for chunk_id in unit.seed_chunk_ids
                if chunk_id in chunks_by_id
            ]
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
                    score=max(seed_scores) if seed_scores else None,
                    retrieval_metadata={
                        "reranking_status": snapshot.retrieval.reranking_status,
                        "expanded": True,
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
            score=chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score,
            retrieval_metadata={
                "rank": chunk.rank,
                "pre_rerank_rank": chunk.pre_rerank_rank,
                "reranking_status": snapshot.retrieval.reranking_status,
                "expanded": False,
                "source_matches": tuple(
                    {
                        "route": match.route,
                        "query": match.query,
                        "rank": match.rank,
                        "raw_score": match.raw_score,
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
