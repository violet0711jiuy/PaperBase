"""Assert final SDK kwargs for the explicit Agentic Qwen sampling recipe."""
from unittest.mock import patch

import pytest

from paperbase.agentic.factory import create_agentic_research_service, AgenticResearchConfigurationError
from paperbase.config import load_settings
from langgraph.checkpoint.memory import InMemorySaver
from test_agentic_llm_contracts import ROOT, _Database, _Retriever, _Client
from test_agentic_p1_3c_structured_decision import real_client
from test_agentic_p0_action_control import _state, _decision, _search


def build(client, thinking=True):
    settings = load_settings(ROOT / 'config.example.yaml')
    settings = settings.model_copy(update={'agentic_research': settings.agentic_research.model_copy(
        update={'llm_sampling_profile':'qwen38_recommended', 'decision_thinking_enabled':thinking,
                'schema_critical_output_mode':'json_schema'})})
    with patch('paperbase.agentic.factory.load_settings', return_value=settings):
        return create_agentic_research_service(config_path=ROOT / 'config.example.yaml', client=client,
            database=_Database(), hybrid_retriever=_Retriever(), checkpointer=InMemorySaver())


def assert_sampling(request, thinking):
    assert request['temperature'] == (1 if thinking else 0.7)
    assert request['top_p'] == (0.95 if thinking else 0.8)
    assert request['presence_penalty'] == (0 if thinking else 1.5)
    assert request['extra_body'] == {'enable_thinking':thinking, 'top_k':20,
                                   'min_p':0.0, 'repetition_penalty':1.0}
    assert 'max_tokens' not in request


@pytest.mark.parametrize('thinking', [False, True])
def test_factory_final_requests_align_all_agentic_nodes_without_mutating_base(thinking):
    client, calls = real_client(['{}'] * 7)
    original = client._settings
    context = build(client, thinking)._workflow._context
    nodes = (context.query_understanding._client, context.planner._client,
             context.decision_agent._client, context.candidate_selector._client,
             context.evidence_analyzer._client, context.answer_generator._client)
    for node in nodes:
        node.complete_json_diagnostic(system_prompt='s', user_prompt='u',
                                      json_schema={'type':'object'}, schema_name='test')
    for index, call in enumerate(calls):
        assert_sampling(call, thinking and index == 2)
        assert call['response_format']['type'] == ('json_object' if index in (0,3) else 'json_schema')
    assert client._settings == original
    assert client._extra_body == {'enable_thinking':False}
    client.complete_json_diagnostic(system_prompt='s', user_prompt='u',
                                   json_schema={'type':'object'}, schema_name='base')
    assert calls[-1]['temperature'] == 0.2
    assert calls[-1]['max_tokens'] == 8000
    assert 'top_p' not in calls[-1] and 'presence_penalty' not in calls[-1]
    assert calls[-1]['extra_body'] == {'enable_thinking':False}
    assert not getattr(client, '_sampling_parameters', {})


def test_sampling_survives_node_repair_and_structured_client_views():
    malformed = '{"decision":{"action":"search_papers","searches":[]}}'
    client, calls = real_client([malformed, _decision(_search())])
    agent = build(client)._workflow._context.decision_agent
    result = agent.decide(_state())
    assert result.llm_calls == 2
    for call in calls:
        assert_sampling(call, True)
        assert call['response_format']['json_schema']['strict'] is True


def test_profile_not_silently_ignored_on_client_without_sampling_support():
    with pytest.raises(AgenticResearchConfigurationError, match='cannot apply'):
        build(_Client())


def test_sampling_survives_timeout_and_repair_temperature_clones_and_plain_requests():
    client, calls = real_client(['text'])
    view = client.with_max_tokens(None).with_sampling_parameters(top_p=0.8, presence_penalty=1.5,
        top_k=20, min_p=0.0, repetition_penalty=1.0).with_temperature(0.7).with_timeout_seconds(60)
    view.with_repair_temperature().complete(system_prompt='s', user_prompt='u')
    assert_sampling(calls[0], False)
    assert calls[0]['timeout'] == 60
