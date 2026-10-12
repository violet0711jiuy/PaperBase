from __future__ import annotations

import json

from paperbase.agentic.contracts import (
    QueryScope,
    ScopeResolution,
    ScopeResolutionStatus,
    ScopeSource,
)
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.schemas import QueryUnderstandingDraft


class Client:
    def __init__(self, payload): self.payload = payload; self.calls = []
    def complete_json_diagnostic(self, **kwargs):
        from paperbase.llm.client import LLMCompletion
        self.calls.append(kwargs)
        return LLMCompletion(json.dumps(self.payload, ensure_ascii=False))


def _resolution():
    return ScopeResolution(
        ScopeResolutionStatus.RESOLVED, QueryScope.SINGLE_PAPER,
        ScopeSource.UI_SELECTION, ("P1",),
    )


def test_query_understanding_no_longer_generates_retrieval_queries() -> None:
    client = Client({
        "resolved_query": "How does ESDTW work?",
        "intent": "explain", "retrieval_goal": "find_evidence",
        "evidence_targets": ["method"],
    })
    result = QueryUnderstandingService(client).understand("How does ESDTW work?", _resolution())
    schema = QueryUnderstandingDraft.model_json_schema()
    assert "semantic_query_en" not in schema["properties"]
    assert "lexical_keywords_en" not in schema["properties"]
    assert not hasattr(result.context, "semantic_query_en")
    assert not hasattr(result.context, "lexical_keywords_en")
    assert not result.used_fallback


def test_invalid_understanding_uses_literal_query_without_retrieval_hints() -> None:
    result = QueryUnderstandingService(Client({"bad": "shape"})).understand(
        "这篇论文用了什么动态图方法？", _resolution()
    )
    assert result.used_fallback
    assert result.context.resolved_query == "这篇论文用了什么动态图方法？"
    assert not hasattr(result.context, "semantic_query_en")
    assert not hasattr(result.context, "lexical_keywords_en")


def test_legacy_retrieval_hint_fields_are_rejected() -> None:
    payload = {
        "resolved_query": "How does ESDTW work?",
        "semantic_query_en": "dynamic graph forecasting",
        "lexical_keywords_en": ["ESDTW"],
        "intent": "explain",
        "retrieval_goal": "find_evidence",
        "evidence_targets": ["method"],
    }
    try:
        QueryUnderstandingDraft.model_validate(payload)
    except Exception:
        pass
    else:
        raise AssertionError("retired retrieval hint fields must not be accepted")
