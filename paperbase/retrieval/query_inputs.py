"""把不同来源的 Query 转成可审计的检索计划。"""

from __future__ import annotations

from collections.abc import Sequence

from .requests import compile_lexical_terms, normalize_semantic_query
from .query_rewriter import QueryRewritePlan


def build_agent_supplied_plan(
    query: str,
    *,
    lexical_terms: Sequence[str] = (),
    max_lexical_terms: int = 5,
) -> QueryRewritePlan:
    """Compatibility API; max_lexical_terms caps fallback hints, not compiled tokens."""
    normalized_query = normalize_semantic_query(query)
    # Compatibility API delegates effective normalization to the compiler.
    # Active Agentic retrievers consume a fully compiled request instead.
    effective_terms = compile_lexical_terms(normalized_query, lexical_terms, max_fallback_hints=max_lexical_terms)
    return QueryRewritePlan(
        original_query=normalized_query,
        resolved_query=normalized_query,
        lexical_keywords_en=effective_terms,
        semantic_status="unavailable",
        lexical_status="valid_fallback" if effective_terms else "empty",
        rewrite_status="not_run",
        validation_diagnostics=("agent_supplied_query_bypassed_rewriter",),
        search_bibliography=False,
    )


def plan_from_canonical(request):
    """Consume compiled inputs verbatim: no rewriter/normalization/fallback."""
    return QueryRewritePlan(
        original_query=request.semantic_query, resolved_query=request.semantic_query,
        lexical_keywords_en=request.lexical_terms, semantic_status="unavailable",
        lexical_status="valid_fallback" if request.lexical_terms else "empty",
        rewrite_status="not_run",
        validation_diagnostics=("canonical_request_bypassed_rewriter",),
        search_bibliography=False,
    )
