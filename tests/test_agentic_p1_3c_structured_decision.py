"""Capture the final SDK request; no provider calls in these tests."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator
from langgraph.checkpoint.memory import InMemorySaver

from paperbase.agentic.contracts import ActionRejectionContext
from paperbase.agentic.decision import DecisionError, LLMDecisionAgent, _decision_json_schema
from paperbase.agentic.factory import create_agentic_research_service
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.config import AgenticResearchSettings, load_settings
from paperbase.llm.client import LLMRuntimeSettings, OpenAICompatibleChatClient
from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT
from test_agentic_p0_action_control import _decision, _search, _state
from test_agentic_llm_contracts import _Database, _Retriever, ROOT


def real_client(outputs):
    calls = []
    values = iter(outputs)
    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=next(values), reasoning_content=None),
            finish_reason='stop')], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))
    client = object.__new__(OpenAICompatibleChatClient)
    client._settings = LLMRuntimeSettings(api_key='test', base_url='https://example.test',
        model='qwen-test', timeout_seconds=30, temperature=0.2, max_tokens=8000,
        structured_output_mode='json_object')
    client._extra_body = {'enable_thinking': False}
    def sdk(max_retries=2):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               max_retries=max_retries, with_options=lambda **kwargs: sdk(**kwargs))
    client._client = sdk()
    return client, calls


def assert_strict_request(request, temperature=0):
    assert request['temperature'] == temperature
    for cap in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'):
        assert cap not in request
    assert request['response_format']['type'] == 'json_schema'
    assert request['response_format']['json_schema']['strict'] is True
    wire = request['response_format']['json_schema']['schema']
    assert wire['additionalProperties'] is False
    search = wire['$defs']['SearchSpecDraft']
    assert 'evidence_targets' not in search['properties']
    assert 'lexical_terms' in search['required']
    assert search['properties']['lexical_terms']['minItems'] == 1
    assert search['properties']['requirement_ids']['maxItems'] == 1
    assert search['additionalProperties'] is False
    assert search['properties']['paper_id']['enum'] == ['A', 'B']
    # Native strict decoding receives the contract once in response_format;
    # Decision's system prompt does not duplicate the full Schema text.
    assert request['messages'][0]['content'] == AGENT_DECISION_SYSTEM_PROMPT
    return wire


def test_prompt_does_not_duplicate_schema_example_and_wire_accepts_action():
    assert '```json' not in AGENT_DECISION_SYSTEM_PROMPT
    example = json.dumps({"decision": {"action": "search_papers", "searches": [{
        "paper_id": "A",
        "query": "How does the calibration term work?",
        "lexical_terms": ["calibration"],
        "requirement_ids": ["R1"],
    }]}})
    client, calls = real_client([example])
    result = LLMDecisionAgent(client, max_attempts=1).decide(_state())
    assert len(calls) == result.llm_calls == 1
    wire = assert_strict_request(calls[0])
    Draft202012Validator(wire).validate(json.loads(example))
    assert result.action.action_name == 'search_papers'
    search, = result.action.searches
    assert search.paper_id == 'A' and search.requirement_ids == ('R1',)
    assert search.query == 'How does the calibration term work?'
    assert search.lexical_terms == ('calibration',)
    assert search.evidence_targets == frozenset({'method'})


@pytest.mark.parametrize('temperature', [0, 1])
@pytest.mark.parametrize('thinking_enabled', [False, True])
def test_first_pass_and_structure_repair_both_send_configured_temperature_strict_schema(temperature, thinking_enabled):
    malformed = json.loads(_decision(_search()))
    del malformed['decision']['searches'][0]['lexical_terms']
    client, calls = real_client([json.dumps(malformed), _decision(_search())])
    original = client._settings
    result = LLMDecisionAgent(client, temperature=temperature, thinking_enabled=thinking_enabled).decide(_state())
    assert result.llm_calls == 2
    for request in calls:
        assert_strict_request(request, temperature)
        assert request['extra_body']['enable_thinking'] is thinking_enabled
    assert client._settings == original
    assert original.max_tokens == 8000
    assert original.temperature == 0.2 and original.structured_output_mode == 'json_object'


@pytest.mark.parametrize('tool', ['search_papers', 'search_knowledge_base', 'search_references'])
def test_lexical_repair_feedback_matches_actual_provider_schema_limit(tool):
    payload = dict(query='adaptive graph', lexical_terms=['a','b','c','d','e','f'],
                   requirement_ids=['R1'])
    if tool == 'search_papers':
        payload.update(paper_id='A')
        decision = dict(action=tool, searches=[payload])
        definition = 'SearchSpecDraft'
    else:
        decision = dict(action=tool, **payload)
        if tool == 'search_references':
            decision['paper_id'] = 'A'
        definition = 'SearchReferencesDraft' if tool == 'search_references' else 'SearchKnowledgeBaseDraft'
    client, calls = real_client([json.dumps({'decision':decision}),
                                json.dumps({'decision':dict(action='finish',reason='No useful path')})])
    result = LLMDecisionAgent(client).decide(_state())
    assert result.llm_calls == 2
    feedback = json.loads(calls[1]['messages'][1]['content'])['previous_validation_errors'][0]
    maximum = calls[1]['response_format']['json_schema']['schema']['$defs'][definition]['properties']['lexical_terms']['maxItems']
    assert maximum == 5
    assert feedback['error_code'] == 'TooManyLexicalTerms'
    assert feedback['expected_constraint'] == f'at most {maximum} lexical terms'


def test_runtime_recovery_keeps_zero_without_raw_query_exclusion():
    rejection = ActionRejectionContext('search_papers', {'searches': [_search()]},
        ('R1',), 'DuplicatePaperQuery', 'already executed', 'DuplicatePaperQuery', 2)
    state = replace(_state(), last_action_rejection=rejection, action_rejections=(rejection,))
    client, calls = real_client([_decision(_search(query='new gap'))])
    LLMDecisionAgent(client).decide(state)
    wire = assert_strict_request(calls[0])
    assert 'allOf' not in wire['$defs']['SearchSpecDraft']
    assert wire['$defs']['SearchSpecDraft']['properties']['paper_id']['enum'] == ['A', 'B']


@pytest.mark.parametrize('extra', ['$defs', 'reasoning'])
@pytest.mark.parametrize('location', ['root', 'decision'])
def test_extra_model_fields_rejected_on_wire_and_locally(extra, location):
    raw = json.loads(_decision(_search()))
    (raw if location == 'root' else raw['decision'])[extra] = {}
    client, calls = real_client([json.dumps(raw)])
    with pytest.raises(DecisionError):
        LLMDecisionAgent(client, max_attempts=1).decide(_state())
    wire = assert_strict_request(calls[0])
    assert list(Draft202012Validator(wire).iter_errors(raw))


def test_schema_does_not_change_and_guard_allows_effectively_changed_lexical_request():
    state = _state()
    before = deepcopy(_decision_json_schema(state))
    client, calls = real_client([_decision(_search()), _decision(_search(lexical_terms=['different']))])
    agent = LLMDecisionAgent(client)
    first = agent.decide(state).action
    second = agent.decide(state).action
    tracker = BudgetAndGuardTracker(AgenticResearchSettings())
    args = dict(context=state.context, plan=state.plan, snapshot=None)
    assert tracker.validate_action(first, **args).accepted
    tracker.record_tool_call(first)
    assert tracker.validate_action(second, **args).accepted
    assert tracker.validate_action(first, **args).code == 'DuplicateAction'
    assert _decision_json_schema(state) == before


@pytest.mark.parametrize('temperature', [0, 1])
@pytest.mark.parametrize('thinking_enabled', [False, True])
def test_production_factory_removes_agentic_caps_and_keeps_node_modes(temperature, thinking_enabled):
    client, _ = real_client([])
    settings = load_settings(ROOT / 'config.example.yaml')
    settings = settings.model_copy(update={'agentic_research': settings.agentic_research.model_copy(
        update={'llm_sampling_profile': 'configured', 'decision_temperature': temperature,
                'decision_thinking_enabled': thinking_enabled, 'planner_temperature': 0,
                'projector_temperature': 0, 'answer_temperature': 0,
                'schema_critical_output_mode': 'json_object'})})
    with patch('paperbase.agentic.factory.load_settings', return_value=settings):
        service = create_agentic_research_service(config_path=ROOT / 'config.example.yaml',
            client=client, database=_Database(), hybrid_retriever=_Retriever(), checkpointer=InMemorySaver())
    context = service._workflow._context
    decision = context.decision_agent._client._settings
    assert (decision.temperature, decision.structured_output_mode,
            decision.strict_schema_required_fields,
            decision.schema_in_system_prompt) == (temperature, 'json_schema', True, False)
    assert decision.max_tokens is None
    for other in (context.query_understanding._client, context.candidate_selector._client):
        assert other._settings.temperature == 0.2
        assert other._settings.structured_output_mode == 'json_object'
        assert other._settings.max_tokens is None
    for other in (context.planner._client, context.evidence_analyzer._client, context.answer_generator._client):
        assert other._settings.temperature == 0
        assert other._settings.structured_output_mode == 'json_object'
        assert other._settings.max_tokens is None
    assert client._settings.temperature == 0.2
    assert client._settings.structured_output_mode == 'json_object'
    assert client._settings.max_tokens == 8000
    assert context.decision_agent._client._extra_body['enable_thinking'] is thinking_enabled
    for other in (context.query_understanding._client, context.candidate_selector._client,
                  context.planner._client, context.evidence_analyzer._client, context.answer_generator._client):
        assert other._extra_body['enable_thinking'] is False
    assert client._extra_body['enable_thinking'] is False
