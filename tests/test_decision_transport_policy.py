"""Real OpenAI SDK, mocked HTTP transport: zero external provider calls."""
import json
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI
from langgraph.checkpoint.memory import InMemorySaver

from paperbase.agentic.factory import create_agentic_research_service
from paperbase.agentic.decision import DecisionError
from paperbase.config import load_settings
from paperbase.llm.client import LLMRuntimeSettings,OpenAICompatibleChatClient
from test_agentic_llm_contracts import ROOT,_Database,_Retriever
from test_agentic_p0_action_control import _decision,_search,_state


def build(tmp_path,handler):
    runtime=LLMRuntimeSettings('test-only','https://example.test/v1','qwen-test',30,.2,8000)
    client=OpenAICompatibleChatClient(runtime)
    client._client.close()
    client._client=OpenAI(api_key='test-only',base_url=runtime.base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),timeout=30)
    settings=load_settings(ROOT/'config.example.yaml')
    settings=settings.model_copy(update={'agentic_research':settings.agentic_research.model_copy(update={
        'evidence_analysis_cache_path':tmp_path/'cache.sqlite3'})})
    with patch('paperbase.agentic.factory.load_settings',return_value=settings):
        service=create_agentic_research_service(config_path=ROOT/'config.example.yaml',client=client,
            database=_Database(),hybrid_retriever=_Retriever(),checkpointer=InMemorySaver())
    return client,service


def response(request,content):
    return httpx.Response(200,request=request,json={'id':'test','model':'qwen-test','object':'chat.completion','created':0,
        'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
        'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}})


def test_timeout_uses_two_node_attempts_without_sdk_retry_multiplication(tmp_path):
    seen=[]
    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout('mock timeout',request=request)
    base,service=build(tmp_path,handler)
    try:
        ctx=service._workflow._context
        with pytest.raises(DecisionError) as error:
            ctx.decision_agent.decide(_state())
        assert len(seen)==error.value.llm_calls==2
        assert [d.error_stage for d in error.value.llm_diagnostics]==['timeout','timeout']
        assert ctx.decision_agent._client._client.max_retries==0
        assert base._client.max_retries==2
        assert ctx.planner._client._client is base._client
        assert ctx.evidence_analyzer._client._client is base._client
        assert ctx.answer_generator._client._client is base._client
        assert ctx.decision_agent._client._client._client is base._client._client
        for request in seen:
            assert request.extensions['timeout']['read']==60
            payload=json.loads(request.content)
            assert payload['temperature']==1 and payload['enable_thinking'] is True
            assert payload['response_format']['json_schema']['strict'] is True
            assert 'max_tokens' not in payload
        assert base._settings.timeout_seconds==30
    finally:
        service.close();base._client.close()


@pytest.mark.parametrize('first', ['timeout','schema'])
def test_request_or_structure_recovery_still_succeeds_in_original_attempt_pool(tmp_path,first):
    seen=[]
    def handler(request):
        seen.append(request)
        if len(seen)==1:
            if first=='timeout': raise httpx.ReadTimeout('mock timeout',request=request)
            return response(request,'{"decision":{"action":"search_papers","searches":[]}}')
        return response(request,_decision(_search()))
    base,service=build(tmp_path,handler)
    try:
        result=service._workflow._context.decision_agent.decide(_state())
        assert result.llm_calls==len(seen)==2
        assert result.action.action_name=='search_papers'
        assert result.llm_diagnostics[-1].success
        assert all(r.extensions['timeout']['read']==60 for r in seen)
    finally:
        service.close();base._client.close()
