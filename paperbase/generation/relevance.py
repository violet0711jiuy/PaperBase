"""回答生成前的低相关性保护，避免 Top-K 的无关结果被误当作论文证据。"""

from __future__ import annotations

from paperbase.retrieval.hybrid_retriever import RetrievalResult


NO_RELEVANT_EVIDENCE_MESSAGE = "当前知识库中没有与该问题相关的论文证据，无法根据已收录论文回答。"


def has_insufficient_retrieval_relevance(
    retrieval: RetrievalResult,
    *,
    min_rerank_score: float,
) -> bool:
    """只在 Reranker 成功且全部正文命中都低分时拒答，避免把未重排结果误判为无关。"""
    if min_rerank_score <= 0 or retrieval.reranking_status != "success":
        return False
    scores = tuple(
        chunk.rerank_score
        for chunk in retrieval.chunks
        if chunk.section_type == "content" and chunk.rerank_score is not None
    )
    return bool(scores) and max(scores) < min_rerank_score
