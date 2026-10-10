from __future__ import annotations

from dataclasses import replace
import json

from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceTarget,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
    ScopeResolution,
    ScopeResolutionStatus,
    ScopeSource,
)
from paperbase.agentic.decision import DecisionState
from paperbase.agentic.decision_view import build_compact_decision_view
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.search_requests import compile_action_requests
from paperbase.agentic.contracts import SearchPapersAction
from paperbase.retrieval.contracts import PaperSearchSpec


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


def test_llm_generated_lexical_hints_are_preserved_as_advice() -> None:
    client = Client({
        "resolved_query": "How does ESDTW work?",
        "semantic_query_en": "dynamic graph forecasting",
        "lexical_keywords_en": ["ESDTW", "dynamic graph", "ESDTW"],
        "intent": "explain", "retrieval_goal": "find_evidence",
        "evidence_targets": ["method"],
    })
    result = QueryUnderstandingService(client).understand("How does ESDTW work?", _resolution())
    assert result.context.lexical_keywords_en == ("ESDTW", "dynamic graph")
    assert result.context.semantic_query_en == "dynamic graph forecasting"
    assert not result.used_fallback


def test_invalid_understanding_does_not_invent_translation_or_keywords() -> None:
    result = QueryUnderstandingService(Client({"bad": "shape"})).understand(
        "这篇论文用了什么动态图方法？", _resolution()
    )
    assert result.used_fallback
    assert result.context.semantic_query_en is None
    assert result.context.lexical_keywords_en == ()


def test_hints_appear_once_only_on_cold_start_and_never_change_search_identity() -> None:
    context = QueryUnderstandingService().understand("How does ESDTW work?", _resolution()).context
    context = replace(context, semantic_query_en="dynamic graph forecasting",
                      lexical_keywords_en=("ESDTW", "dynamic graph"))
    plan = ResearchPlan("Explain", (
        ResearchRequirement("R1", "Explain ESDTW", frozenset({EvidenceTarget.METHOD})),
    ))
    view = build_compact_decision_view(DecisionState(
        context=context, plan=plan, evidence=(), observations=(), budget_usage=BudgetUsage(),
    )).to_dict()
    assert view["initial_search_hints"]["lexical_concepts"] == ["ESDTW", "dynamic graph"]
    action = SearchPapersAction((PaperSearchSpec(
        "P1", "explicit query", ("explicit",), requirement_ids=("R1",)
    ),))
    changed = replace(context, semantic_query_en="other", lexical_keywords_en=("other",))
    assert compile_action_requests(action, context=context) == compile_action_requests(action, context=changed)
