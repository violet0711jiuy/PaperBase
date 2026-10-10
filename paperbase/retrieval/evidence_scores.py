"""Score provenance at the RetrievedChunk → RetrievedEvidence boundary.

No raw-score normalization/calibration is performed here. Consumers may use
the producer's ordinal rank as a late deterministic tie-break, after semantic
support/quality, but must not compare raw scores across channels or requests.
"""
from .contracts import RetrievedEvidence, ScoreDirection, ScoreKind


def chunk_score_fields(chunk: object) -> dict[str, object]:
    """Map known chunk producer contracts, not numeric sign or guessed ranges."""
    if chunk.section_type == "bibliography":
        match = next((m for m in chunk.source_matches if m.route == "bm25_bibliography"), None)
        return dict(raw_score=match.raw_score if match is not None else None,
                    score_kind=ScoreKind(getattr(match, 'score_kind', ScoreKind.LEGACY_UNKNOWN)) if match is not None else ScoreKind.OTHER,
                    score_direction=ScoreDirection(getattr(match, 'score_direction', ScoreDirection.NOT_APPLICABLE)) if match is not None else ScoreDirection.NOT_APPLICABLE,
                    retrieval_rank=chunk.rank)
    if chunk.section_type in {"overview", "section_context"}:
        return dict(raw_score=None, score_kind=ScoreKind.NON_RANKED,
                    score_direction=ScoreDirection.NOT_APPLICABLE, retrieval_rank=None)
    return dict(raw_score=chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score,
                score_kind=ScoreKind.RERANKER if chunk.rerank_score is not None else ScoreKind.RRF,
                score_direction=ScoreDirection.HIGHER_IS_BETTER, retrieval_rank=chunk.rank)


def retrieval_rank_priority(evidence: RetrievedEvidence) -> float:
    """Higher priority for earlier local rank; missing ranks retain stable ties.

    This is an ordinal tie-break only, not a cross-query relevance estimate.
    Unknown legacy raw scores are never substituted for a missing rank.
    """
    return -evidence.retrieval_rank if evidence.retrieval_rank is not None else float('-inf')
