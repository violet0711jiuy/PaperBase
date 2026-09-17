"""把不同来源的 Query 转成可审计的检索计划。"""

from __future__ import annotations

from collections.abc import Sequence

from .lexical_terms import extract_lexical_terms, normalize_lexical_terms
from .query_rewriter import QueryRewritePlan


def build_agent_supplied_plan(
    query: str,
    *,
    lexical_terms: Sequence[str] = (),
    max_lexical_terms: int = 5,
) -> QueryRewritePlan:
    """为 Agent 已生成的 Query 建立确定性计划，绝不调用 LLM。"""
    normalized_query = " ".join(query.split())
    if not normalized_query:
        raise ValueError("Agent-supplied query must not be empty.")
    explicit_terms = normalize_lexical_terms(
        lexical_terms, max_terms=max_lexical_terms
    )
    effective_terms = explicit_terms or extract_lexical_terms(
        normalized_query, max_terms=max_lexical_terms
    )
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
