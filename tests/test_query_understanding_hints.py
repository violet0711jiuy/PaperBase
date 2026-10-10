"""QU cold-start advice is independent of executable Search authority; no provider."""
from copy import deepcopy
from dataclasses import replace
import json
import inspect
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import QueryContext, SearchPapersAction, SearchReferencesAction, SearchKnowledgeBaseAction, CorpusScopeSnapshot
from paperbase.agentic.decision import LLMDecisionAgent, _legacy_state_prompt
from paperbase.agentic.decision_view import build_compact_decision_view
from paperbase.agentic.graph import LangGraphResearchWorkflow, _SEMANTIC_AUTHORITY_VERSION
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.agentic.initial_search_hints import build_initial_search_hints
from paperbase.agentic.planner import ResearchPlanner
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.schemas import QueryUnderstandingDraft
from paperbase.agentic.search_requests import compile_action_requests, canonical_action_identity
from paperbase.agentic.scope_resolver import ScopeResolver
from paperbase.agentic.tools import ResearchToolExecutor
from paperbase.config import AgenticResearchSettings
from paperbase.prompts.agentic_query_understanding import QUERY_UNDERSTANDING_MAX_LEXICAL_HINTS, QUERY_UNDERSTANDING_SYSTEM_PROMPT
from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT, LEGACY_AGENT_DECISION_SYSTEM_PROMPT
from paperbase.retrieval.adapters import FormalKnowledgeBaseAdapter
from paperbase.retrieval.facade import RetrievalService
from paperbase.retrieval.requests import compile_lexical_terms
from test_agentic_checkpoint_semantic_authority import historical_states, install_historical_v2
from test_agentic_decision_tools import _single_context, _kb_context
from test_agentic_langgraph import _CatalogDatabase, _EvidenceAwareDecisionAgent, _Executor, _SupportingEvidenceAnalyzer
from test_agentic_p0_action_control import _QueuedClient, _decision, _search, _state
from test_retrieval_tool_results import _snapshot


def draft(terms, query="这篇论文用了什么动态图方法？"):
    return json.dumps(dict(resolved_query=query, semantic_query_en="dynamic graph methods",
        lexical_keywords_en=terms, intent="retrieve", retrieval_goal="find_evidence",
        evidence_targets=["method"]), ensure_ascii=False)


def understand(raw, query="这篇论文用了什么动态图方法？"):
    resolution = ScopeResolver(_CatalogDatabase()).resolve(query, selected_paper_ids=("A",))
    client = _QueuedClient((raw,)) if raw is not None else None
    return QueryUnderstandingService(client).understand(query, resolution), client


def hinted(context):
    return replace(context, semantic_query_en="graph neural network propagation",
        lexical_keywords_en=("graph", "propagation"))


def test_valid_eight_hints_retained_with_model_provenance_and_single_limit():
    terms = [f"k{i}" for i in range(1, 9)]
    result, client = understand(draft(terms))
    assert result.context.lexical_keywords_en == tuple(terms)
    assert result.context.semantic_query_en == "dynamic graph methods"
    assert not result.used_fallback and result.llm_calls == len(client.calls) == 1
    assert result.llm_diagnostics[0].success
    schema = client.calls[0]["json_schema"]
    assert schema["properties"]["lexical_keywords_en"]["maxItems"] == QUERY_UNDERSTANDING_MAX_LEXICAL_HINTS == 8
    assert f"最多 {QUERY_UNDERSTANDING_MAX_LEXICAL_HINTS}" in QUERY_UNDERSTANDING_SYSTEM_PROMPT
    # This separate compiler authority stays at five; it is not QU's limit.
    assert inspect.signature(compile_lexical_terms).parameters["max_fallback_hints"].default == 5


def test_search_fallback_passes_its_own_five_hint_limit_to_extractor(monkeypatch):
    from paperbase.retrieval import requests
    original = requests.extract_lexical_terms
    calls = []
    def capture(query, *, max_terms):
        calls.append(max_terms)
        return original(query, max_terms=max_terms)
    monkeypatch.setattr(requests, "extract_lexical_terms", capture)
    requests.compile_lexical_terms("ESDTW DCRNN AGCRN MTGNN STGCN LSTM GRU TCN", ())
    assert calls == [5]


@pytest.mark.parametrize("terms, expected", [
    ([" a ", "b", "a", "c", "b", " d ", "c", "e"], ("a", "b", "c", "d", "e")),
    ([], ()), (["model-only"], ("model-only",)),
])
def test_model_hints_trim_stable_dedupe_without_deterministic_fill(terms, expected):
    query = "How does ESDTW work?"
    result, _ = understand(draft(terms, query), query)
    assert not result.used_fallback
    assert result.context.lexical_keywords_en == expected


@pytest.mark.parametrize("raw", [draft([f"k{i}" for i in range(9)]), "{invalid", draft([" "])])
def test_invalid_response_uses_existing_one_attempt_empty_hint_fallback(raw):
    result, client = understand(raw)
    assert result.used_fallback and result.llm_calls == len(client.calls) == 1
    assert result.context.semantic_query_en is None
    assert result.context.lexical_keywords_en == ()
    assert not result.llm_diagnostics[0].success
    assert len(result.llm_diagnostics) == 1
    assert result.context.resolved_query == "这篇论文用了什么动态图方法？"
    assert result.context.target_paper_ids == ("A",)
    assert result.context.intent.value == "retrieve"
    assert result.context.retrieval_goal.value == "find_evidence"
    assert {t.value for t in result.context.default_evidence_targets} == {"method"}


@pytest.mark.parametrize("query", ["这篇论文用了什么动态图方法？", "How does ESDTW work?", "méthode du graphe dynamique"])
def test_unavailable_model_does_not_claim_deterministic_translation(query):
    result, _ = understand(None, query)
    assert result.used_fallback and result.llm_calls == 0 and result.llm_diagnostics == ()
    assert result.context.semantic_query_en is None and result.context.lexical_keywords_en == ()
    assert result.context.resolved_query == query


def test_compact_and_legacy_share_hint_contract_without_state_mutation():
    state = replace(_state(), context=hinted(_state().context))
    before = deepcopy(state)
    expected = build_initial_search_hints(state.context)
    compact = build_compact_decision_view(state).to_dict()
    legacy = json.loads(_legacy_state_prompt(state, []))
    assert compact["initial_search_hints"] == legacy["initial_search_hints"] == expected
    assert expected == dict(authority="advisory_only", language="en",
        semantic_concept_query="graph neural network propagation", lexical_concepts=["graph", "propagation"])
    assert "retrieval_hints" not in compact and "retrieval_hints" not in legacy
    assert state == before
    expected["lexical_concepts"].append("independent view")
    assert state.context.lexical_keywords_en == ("graph", "propagation")


@pytest.mark.parametrize("query,terms", [
    ("dynamic graph adjacency", ["dynamic", "adjacency"]),
    ("论文采用的动态图构建方法", ["动态图", "邻接矩阵"]),
    ("graph neural network propagation", ["graph", "propagation"]),
    ("unrelated alternative focus", ["alternative", "focus"]),
])
def test_decision_can_ignore_translate_or_explicitly_reuse_hints_through_real_tool(query, terms):
    state = replace(_state(), context=hinted(_single_context()))
    client = _QueuedClient((_decision(_search(query=query, lexical_terms=terms)),))
    action = LLMDecisionAgent(client, max_attempts=1).decide(state).action
    assert action.searches[0].query == query and action.searches[0].lexical_terms == tuple(terms)
    wire_view = json.loads(client.calls[0]["user_prompt"])
    assert wire_view["initial_search_hints"]["authority"] == "advisory_only"
    class Retriever:
        def retrieve(self, query, **kwargs):
            self.query, self.kwargs = query, kwargs
            return _snapshot("A", with_chunk=True).retrieval
    backend = Retriever()
    executor = ResearchToolExecutor(retrieval_service=RetrievalService(formal_adapter=FormalKnowledgeBaseAdapter(
        database=SimpleNamespace(get_document=lambda _: {}), retriever=backend)), database=object())
    requests = executor.compile_requests(action, context=state.context)
    tracker = BudgetAndGuardTracker(AgenticResearchSettings())
    assert tracker.validate_action(action, context=state.context, plan=state.plan, snapshot=None, canonical_requests=requests).accepted
    result = executor.execute(action, context=state.context, canonical_requests=requests)
    assert result.status.value == "ok"
    assert backend.kwargs["canonical_request"] is requests[0]
    assert backend.query == query
    assert backend.kwargs["lexical_terms"] == compile_lexical_terms(query, terms)
    assert requests == executor.compile_requests(action, context=replace(state.context,
        semantic_query_en="changed cold-start concept", lexical_keywords_en=("different",)))


@pytest.mark.parametrize("kind", ["paper", "reference", "kb"])
@pytest.mark.parametrize("terms", [(), ("dynamic", "adjacency")])
def test_all_compiler_identities_ignore_context_hints(kind, terms):
    from paperbase.retrieval.contracts import PaperSearchSpec
    context = _kb_context() if kind == "kb" else _single_context()
    action = (SearchPapersAction((PaperSearchSpec("A", "dynamic graph adjacency", terms),)) if kind == "paper"
        else SearchReferencesAction("A", "dynamic graph adjacency", terms) if kind == "reference"
        else SearchKnowledgeBaseAction("dynamic graph adjacency", terms))
    snapshot = CorpusScopeSnapshot("S", ("A", "B"), "2026-10-03") if kind == "kb" else None
    a = compile_action_requests(action, context=hinted(context), snapshot=snapshot)
    b = compile_action_requests(action, context=replace(context, semantic_query_en=None, lexical_keywords_en=()), snapshot=snapshot)
    assert a == b and canonical_action_identity(a) == canonical_action_identity(b)


def workflow_with_understanding(client):
    executor = _Executor()
    workflow = LangGraphResearchWorkflow(scope_resolver=ScopeResolver(_CatalogDatabase()),
        query_understanding=QueryUnderstandingService(client), planner=ResearchPlanner(),
        decision_agent=_EvidenceAwareDecisionAgent(), evidence_analyzer=_SupportingEvidenceAnalyzer(),
        tool_executor=executor, settings=AgenticResearchSettings(), checkpointer=InMemorySaver())
    return workflow, executor


def test_multi_round_graph_calls_understanding_model_once():
    client = _QueuedClient((draft(["method"], "A 和 B 的方法有什么区别？"),))
    workflow, executor = workflow_with_understanding(client)
    execution = workflow.start("A 和 B 的方法有什么区别？", selected_paper_ids=("A", "B"), thread_id="hint-once")
    assert not execution.interrupted and execution.result is not None
    assert len(executor.calls) == 2
    assert [action.searches[0].query for action in executor.calls] == ["A method", "B method after A"]
    assert len(client.calls) == 1
    diagnostics = execution.state["llm_diagnostics"]
    assert sum(row["node"] == "understand_query" for row in diagnostics) == 1
    assert execution.state["semantic_authority_version"] == _SEMANTIC_AUTHORITY_VERSION == "projector-requirement-directed-multi-projection-v4"


def test_thirteen_historical_contexts_roundtrip_inspect_without_hint_migration():
    rows = list(historical_states())
    assert len(rows) == 13
    workflow, _ = workflow_with_understanding(None)
    for index, (_, row, state) in enumerate(rows):
        original = row["full_decision_state"]["context"]
        restored = load_contract(dump_contract(state.context, QueryContext), QueryContext)
        assert restored == state.context
        assert restored.semantic_query_en == original.get("semantic_query_en")
        assert restored.lexical_keywords_en == tuple(original.get("lexical_keywords_en", ()))
        thread = f"history-hints-{index}"
        install_historical_v2(workflow, thread, state)
        inspected = load_contract(workflow.inspect(thread).state["query_context"], QueryContext)
        assert inspected == restored
        with pytest.raises(ValueError, match="Incompatible checkpoint semantic authority"):
            workflow.resume(thread)


def test_prompt_separates_initial_advice_execution_language_and_reformulation():
    for prompt in (AGENT_DECISION_SYSTEM_PROMPT, LEGACY_AGENT_DECISION_SYSTEM_PROMPT):
        assert "initial_search_hints" in prompt and "retrieval_hints" not in prompt
        assert "不是已经编译的 Search Request" in prompt
        assert "完全忽略" in prompt and "Action.query" in prompt and "Action.lexical_terms" in prompt
        assert "文档语言已知" in prompt and "未知时" in prompt
        assert "符号和数字" in prompt
    assert QueryUnderstandingDraft.model_json_schema()["properties"]["lexical_keywords_en"]["maxItems"] == 8
