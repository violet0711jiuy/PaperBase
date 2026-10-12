"""No-network regressions for request rejection, safe audit and wire Schema."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import pytest
from jsonschema import Draft202012Validator
from paperbase.llm.client import LLMRequestError, _provider_request_error, OpenAICompatibleChatClient, ZhipuAIChatClient, LLMRuntimeSettings
from paperbase.llm.diagnostics import failure_record,recovery_action_for,request_is_nonretryable
from paperbase.agentic.grounded_answer import LLMGroundedAnswerer,GroundedAnswerAttemptsExhausted,EvidencePack,_response_schema
from paperbase.agentic.decision import LLMDecisionAgent,DecisionError
from paperbase.agentic.candidate_selection import LLMCandidateSelector
from paperbase.agentic.contracts import CorpusScopeSnapshot
from paperbase.agentic.audit import AgenticAuditSnapshot, build_agentic_audit_snapshot
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.llm.client import LLMCallRecord
from test_agentic_audit import _result
from test_agentic_grounded_answer import QueueClient,plan,evidence,complete_payload,item
from test_agentic_candidate_selection import _MetadataSource
from test_agentic_decision_view import _state


class ProviderError(Exception):
    def __init__(self,status=400,**details):
        super().__init__('Unsafe exception prose: sk-do-not-persist / private paper text')
        self.status_code=status
        self.body={'error':details}


def rejection():
    return _provider_request_error(ProviderError(code='invalid_parameter_error',message="Format error: 'response_format.json_schema.schema'; array 'uniqueItems' unsupported. sk-do-not-persist PRIVATE_PAPER"),'LLM request failed')


def test_safe_metadata_uses_existing_diagnostic_contract_without_raw_error_prose():
    error=rejection()
    assert error.error_stage=='invalid_request'
    assert error.status_code==400 and error.error_code=='invalid_parameter_error'
    assert error.field_path=='response_format.json_schema.schema'
    assert error.schema_keywords==('uniqueItems',)
    record=failure_record(node='grounded_answer',attempt=1,error=error,can_retry=True)
    diagnostic=record.validation_diagnostics[0]
    assert record.recovery_action is None and diagnostic.recovery_action is None
    assert diagnostic.error_code==error.error_code
    assert diagnostic.field_path==error.field_path
    assert 'HTTP 400' in diagnostic.actual_summary and 'uniqueItems' in diagnostic.actual_summary
    assert 'sk-do-not-persist' not in json.dumps(asdict(record))
    assert 'PRIVATE_PAPER' not in json.dumps(asdict(record))


def test_unknown_provider_values_cannot_smuggle_private_text_into_audit():
    error=_provider_request_error(ProviderError(code='sk-secret',param='PRIVATE_PAPER',message='full request and PRIVATE_PAPER and token sk-secret'), 'LLM request failed')
    assert error.error_code is None and error.field_path is None and error.schema_keywords==()
    assert 'secret' not in json.dumps(asdict(failure_record(node='x',attempt=1,error=error,can_retry=True)))


def test_safe_request_diagnostic_round_trips_through_existing_audit_and_checkpoint():
    record = failure_record(node="grounded_answer", attempt=1, error=rejection(), can_retry=True)
    records = (record,)
    record_type = tuple[LLMCallRecord, ...]
    assert load_contract(dump_contract(records, record_type), record_type) == records
    result = replace(_result(), llm_diagnostics=(record,))
    snapshot = build_agentic_audit_snapshot(result)
    assert AgenticAuditSnapshot.from_json(snapshot.to_json()) == snapshot
    assert "invalid_parameter_error" in snapshot.to_json()
    assert "uniqueItems" in snapshot.to_json()
    assert "sk-do-not-persist" not in snapshot.to_json()
    assert "PRIVATE_PAPER" not in snapshot.to_json()


@pytest.mark.parametrize('status',[400,401,403,404,405,413,415,422])
def test_nonrecoverable_status_never_retries_same_request(status):
    error=_provider_request_error(ProviderError(status),'request failed')
    assert request_is_nonretryable(error)
    assert recovery_action_for(error,can_retry=True) is None
    client=QueueClient(error,complete_payload())
    with pytest.raises(GroundedAnswerAttemptsExhausted) as caught:
        LLMGroundedAnswerer(client).answer(question='Explain',plan=plan(),evidence=(evidence(),))
    assert len(client.calls)==len(caught.value.diagnostics)==1
    assert caught.value.diagnostics[0].error_stage=='invalid_request'


@pytest.mark.parametrize('status,stage',[(408,'request_error'),(429,'rate_limit'),(500,'provider_5xx'),(503,'provider_5xx')])
def test_transient_provider_errors_keep_bounded_retry_and_do_not_invent_answer_repairs(status,stage):
    error=_provider_request_error(ProviderError(status),'request failed')
    assert error.error_stage==stage and not request_is_nonretryable(error)
    client=QueueClient(error,complete_payload())
    result=LLMGroundedAnswerer(client).answer(question='Explain',plan=plan(),evidence=(evidence(),))
    assert result.complete and len(client.calls)==2
    assert result.llm_diagnostics[0].recovery_action=='retry_request'
    assert json.loads(client.calls[1]['user_prompt'])['validation_errors']==[]


@pytest.mark.parametrize('stage',['connection_error','timeout'])
def test_connection_and_timeout_retries_remain_enabled(stage):
    client=QueueClient(LLMRequestError('request failed',error_stage=stage),complete_payload())
    assert LLMGroundedAnswerer(client).answer(question='Explain',plan=plan(),evidence=(evidence(),)).complete
    assert len(client.calls)==2


def test_request_rejection_during_coverage_repair_preserves_valid_prose_and_sibling():
    first=complete_payload();first['requirements'][1]['evidence_ids']=['E99']
    client=QueueClient(first,rejection(),{'requirements':[item('R2')]})
    result=LLMGroundedAnswerer(client,max_attempts=3).answer(question='Explain',plan=plan(),evidence=(evidence(),))
    assert len(client.calls)==2 and result.answer==first['answer']
    assert result.requirements[0].complete and not result.requirements[1].complete
    assert result.requirements[1].remaining_gap and result.llm_diagnostics[-1].recovery_action is None


def test_duplicate_coverage_ids_still_require_repair_without_wire_unique_items():
    first = complete_payload()
    first["requirements"][1]["evidence_ids"] = ["E1", "E1"]
    client = QueueClient(first, {"requirements": [item("R2")]})
    result = LLMGroundedAnswerer(client).answer(question="Explain", plan=plan(), evidence=(evidence(),))
    assert result.complete and len(client.calls) == 2
    assert result.llm_diagnostics[0].validation_diagnostics[0].error_code == "ANSWER_COVERAGE_REFERENCES_INVALID"
    assert json.loads(client.calls[1]["user_prompt"])["validated_answer"]["answer"] == first["answer"]


def test_decision_request_rejection_stops_without_second_decision_call():
    client=QueueClient(rejection(),{'decision':{'action':'stop_research','reason':'done'}})
    with pytest.raises(DecisionError) as caught:
        LLMDecisionAgent(client).decide(_state())
    assert len(client.calls)==caught.value.llm_calls==1
    assert caught.value.llm_diagnostics[0].validation_diagnostics[0].validator_name=='provider_request_validator'


def test_candidate_request_rejection_stops_the_batch_without_schema_repair():
    client=QueueClient(rejection())
    result=LLMCandidateSelector(client,_MetadataSource()).select(question='Explain',snapshot=CorpusScopeSnapshot.create(('A',)))
    assert len(client.calls)==result.llm_calls==1
    assert result.used_fallback


def test_candidate_request_rejection_does_not_resend_invalid_schema_for_later_batches():
    client=QueueClient(rejection())
    result=LLMCandidateSelector(client,_MetadataSource(),batch_size=1).select(question='Explain',snapshot=CorpusScopeSnapshot.create(('A','B','C')))
    assert len(client.calls)==result.llm_calls==1
    assert {a.paper_id for a in result.selection.assessments}=={'A','B','C'}
    assert all(a.disposition.value=='uncertain' for a in result.selection.assessments)


def test_legacy_request_stage_with_explicit_400_also_stops():
    client=QueueClient(LLMRequestError('bad request',status_code=400),complete_payload())
    with pytest.raises(GroundedAnswerAttemptsExhausted) as caught:
        LLMGroundedAnswerer(client).answer(question='Explain',plan=plan(),evidence=(evidence(),))
    assert len(client.calls)==1
    assert caught.value.diagnostics[0].error_stage=='invalid_request'
    assert caught.value.diagnostics[0].validation_diagnostics[0].error_stage=='invalid_request'


@pytest.mark.parametrize('repair,citations',[(False,()),(True,('E1',)),(True,())])
def test_whole_answer_and_coverage_repair_wire_schemas_keep_supported_constraints(repair,citations):
    schema=_response_schema([('R1','Explain')],EvidencePack((evidence(),)),repair=repair,citations=citations)
    assert 'uniqueItems' not in json.dumps(schema)
    Draft202012Validator.check_schema(schema)
    ids=schema['$defs']['RequirementCoverageDraft']['properties']['evidence_ids']
    assert ids['maxItems']==(len(citations) if repair else 1)
    if not repair:
        assert schema['properties']['citations']['items']['enum']==['E1']
    if citations:
        assert ids['items']['enum']==list(citations)


@pytest.mark.parametrize('provider,structured',[(OpenAICompatibleChatClient,True),(OpenAICompatibleChatClient,False),(ZhipuAIChatClient,True)])
def test_sdk_error_is_wrapped_safely_for_all_active_client_entry_points(provider,structured):
    client=object.__new__(provider)
    client._settings=LLMRuntimeSettings(api_key='sk-do-not-persist',base_url='https://example.test',model='test',timeout_seconds=10,temperature=0,max_tokens=None,structured_output_mode='json_schema' if provider is OpenAICompatibleChatClient else 'json_object')
    client._extra_body={};client._thinking={'type':'disabled'}
    def fail(**kwargs):raise ProviderError(code='invalid_parameter_error',param='response_format',message='PRIVATE_PAPER sk-do-not-persist')
    client._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(LLMRequestError) as caught:
        if structured:client.complete_json_diagnostic(system_prompt='system',user_prompt='user',json_schema={'type':'object'},schema_name='test')
        else:client.complete(system_prompt='system',user_prompt='user')
    assert caught.value.error_stage=='invalid_request'
    assert caught.value.error_code=='invalid_parameter_error' and caught.value.field_path=='response_format'
    assert 'PRIVATE_PAPER' not in str(caught.value) and 'sk-do-not-persist' not in str(caught.value)
