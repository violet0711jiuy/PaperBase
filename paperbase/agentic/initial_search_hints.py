"""Shared Decision-facing serialization of advisory cold-start hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import QueryContext


def build_initial_search_hints(context: QueryContext) -> dict[str, object]:
    """Expose persisted hints without compiling, translating, or rewriting them."""
    return {
        "authority": "advisory_only",
        "language": "en",
        "semantic_concept_query": context.semantic_query_en,
        "lexical_concepts": list(context.lexical_keywords_en),
    }
