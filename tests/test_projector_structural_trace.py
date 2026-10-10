"""Projector observability A-G, privacy, ordering and additive compatibility."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from paperbase.agentic.audit import _llm_call, AgenticAuditSnapshot
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import EvidenceProjectionType as Shape
from paperbase.agentic.evidence_analysis import EvidenceAnalysisRequest, projection_requirement_scope
from paperbase.agentic.evidence_projection import LLMEvidenceProjector
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.agentic.grounding import verified_support_ids
from paperbase.agentic.projector_trace import pre_parse_summary
from paperbase.agentic.projection_semantics import build_projection_semantic_request, projection_semantic_fingerprint
from paperbase.llm.client import LLMCallRecord, LLMRequestError
from test_agentic_batch_projector import CONTEXT, PLAN, Client, record, fact, item

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_real_provider(monkeypatch):
    import paperbase.llm.client as module
    import paperbase.agentic.factory as factory
    def forbidden(*args,**kwargs):raise AssertionError('Structural trace tests cannot call provider')
    monkeypatch.setattr(module,'create_chat_completion_client',forbidden)
    monkeypatch.setattr(factory,'create_agentic_research_service',forbidden)


def run(responses,n=1,*,cache=None):
    records=tuple(replace(record(i),requested_requirement_ids=('R1',)) for i in range(1,n+1))
    plan=replace(PLAN,requirements=(replace(PLAN.requirements[0],projection_type=Shape.FACT),))
    request=EvidenceAnalysisRequest('facts',records,CONTEXT,plan,
        tuple((r.evidence_id,projection_requirement_scope(r,plan)) for r in records))
    before=deepcopy(asdict(request))
    client=Client(responses)
    result=LLMEvidenceProjector(client,cache=cache).analyze(request=request)
    assert asdict(request)==before
    return result,client,request


@pytest.mark.parametrize('n',[1,2])
def test_a_valid_assertion_visible_at_both_boundaries(n):
    response={'projections':[fact(1)]} if n==1 else {'items':[item(1),item(2)]}
    result,client,req=run([response],n)
    d=result.llm_diagnostics[0]
    pre=d.projector_pre_parse_summary;post=d.projector_post_conversion_summary
    assert pre['json_parseable'] and pre['item_count']==n
    assert pre['semantic_request_fingerprint']==projection_semantic_fingerprint(build_projection_semantic_request(
        question=req.question,plan=req.plan,projection_type=Shape.FACT,requirement_ids=('R1',)))
    p=pre['items'][0]['projections'][0]
    assert p['supported_requirement_ids']==p['support_candidate_requirement_ids']==['R1']
    assert p['support_candidate_count']==1
    assert p['support_candidates'][0]['source_span_ids']==['S1']
    assert post['conversion_succeeded']
    for i in post['items']:
        assert i['supported_requirement_ids']==i['proposed_requirement_ids']==i['support_anchor_requirement_ids']==i['verified_support_ids']==['R1']
    if n==2:assert pre['items'][0]['evidence_id'].startswith('source:') and pre['items'][0]['runtime_evidence_id']=='E1'


@pytest.mark.parametrize('n',[1,2])
def test_b_empty_assertion_already_empty_before_typed_parser(n):
    empty=dict(fact(1),supported_requirement_ids=[],support_candidates=[])
    response={'projections':[empty]} if n==1 else {'items':[item(1,[empty]),item(2)]}
    result,_,_=run([response],n)
    d=result.llm_diagnostics[0];p=d.projector_pre_parse_summary['items'][0]['projections'][0]
    assert p['value_present'] and p['supported_requirement_id_count']==p['support_candidate_count']==0
    out=d.projector_post_conversion_summary['items'][0]
    assert out['analysis_status']=='qualified' and out['projection_count']==1
    assert out['proposed_requirement_ids']==out['supported_requirement_ids']==out['verified_support_ids']==[]


@pytest.mark.parametrize('behavior',['mismatch','provenance','scope'])
def test_c_d_e_assertions_visible_before_formal_rejection(behavior):
    bad=fact(1)
    if behavior=='mismatch':bad['support_candidates']=[]
    elif behavior=='provenance':bad['support_candidates'][0]['source_span_ids']=['S999']
    else:
        bad['supported_requirement_ids']=['R99'];bad['support_candidates'][0]['requirement_id']='R99'
    result,_,_=run([{'projections':[bad]}]*2)
    assert result.analyses[0].status.value=='failed'
    for d in result.llm_diagnostics:
        pre=d.projector_pre_parse_summary
        assert pre['items'][0]['projections'][0]['supported_requirement_ids']==(['R99'] if behavior=='scope' else ['R1'])
        assert d.projector_post_conversion_summary is None
        assert d.error_stage=='domain_validation' and d.validation_diagnostics
    assert result.llm_diagnostics[1].projector_repair


def test_f_invalid_json_keeps_hash_and_length_without_regex_id_extraction():
    raw='broken { "supported_requirement_ids": ["R99"], SECRET_SOURCE_SENTINEL'
    result,_,_=run([raw,raw])
    for d in result.llm_diagnostics:
        p=d.projector_pre_parse_summary
        assert not p['json_parseable'] and p['content_chars']==len(raw)
        assert p['content_sha256']==sha256(raw.encode()).hexdigest()
        assert p['items'] is None and p['item_count'] is None
        assert 'R99' not in json.dumps(p) and 'SECRET_SOURCE_SENTINEL' not in json.dumps(p)
        assert d.projector_post_conversion_summary is None


def test_g_mixed_batch_preserves_each_pre_item_and_partial_post():
    empty=dict(fact(2),supported_requirement_ids=[],support_candidates=[])
    bad=fact(3,excerpt='INVALID_PEER_SENTINEL')
    response={'items':[item(1),item(2,[empty]),item(3,[bad])]}
    result,_,_=run([response,response,{'projections':[fact(3)]}],3)
    d=result.llm_diagnostics[0]
    pre=d.projector_pre_parse_summary
    assert [i['runtime_evidence_id'] for i in pre['items']]==['E1','E2','E3']
    assert [i['projections'][0]['supported_requirement_ids'] for i in pre['items']]==[['R1'],[],['R1']]
    post=d.projector_post_conversion_summary
    assert not post['conversion_succeeded'] and post['converted_item_count']==2
    assert [i['runtime_evidence_id'] for i in post['items']]==['E1','E2']
    assert result.llm_diagnostics[-1].projector_call_mode=='fallback_single'
    assert result.llm_diagnostics[-1].projector_pre_parse_summary['fallback_reason']
    assert all(a.status.value=='qualified' for a in result.analyses)


def test_unknown_transport_and_arbitrary_requirement_ids_never_echo_strings():
    raw={'items':[{'evidence_id':'SECRET_PROVIDER_ID_SENTINEL'*100,'projections':[
        dict(fact(1),supported_requirement_ids=['R99','SECRET_ID_SENTINEL'],
             support_candidates=[dict(requirement_id='SECRET_ID_SENTINEL',source_span_ids=['SECRET_SOURCE_SENTINEL'],
                 support_scope='SECRET_SCOPE_SENTINEL',unresolved_aspects=[])])]}]}
    # Unknown item forces envelope repair and fallback; inspect its FIRST raw summary.
    result,_,_=run([raw,raw,{'projections':[fact(1)]},{'projections':[fact(2)]}],2)
    pre=result.llm_diagnostics[0].projector_pre_parse_summary
    i=pre['items'][0]
    assert not i['known_input_id'] and i['evidence_id'] is i['runtime_evidence_id'] is None
    assert i['projections'][0]['supported_requirement_ids']==['R99']
    assert i['projections'][0]['unrecordable_supported_id_count']==1
    encoded=json.dumps(pre)
    assert all(s not in encoded for s in ('SECRET_PROVIDER_ID_SENTINEL','SECRET_ID_SENTINEL','SECRET_SOURCE_SENTINEL','SECRET_SCOPE_SENTINEL'))


def test_trace_is_created_before_single_parser_and_batch_validator(monkeypatch):
    import paperbase.agentic.evidence_projection as module
    events=[]
    pre=module.pre_parse_summary;parse=module._parse_projection_batch;validate=module._validate_transport_items
    def observe(**kwargs):events.append('pre');return pre(**kwargs)
    def typed(*args,**kwargs):events.append('typed');assert events[0]=='pre';return parse(*args,**kwargs)
    def batch(*args,**kwargs):events.append('batch');assert events[0]=='pre';return validate(*args,**kwargs)
    monkeypatch.setattr(module,'pre_parse_summary',observe)
    monkeypatch.setattr(module,'_parse_projection_batch',typed)
    monkeypatch.setattr(module,'_validate_transport_items',batch)
    run([{'projections':[fact(1)]}]);assert events[:2]==['pre','typed']
    events.clear();run([{'items':[item(1),item(2)]}],2);assert events[:2]==['pre','batch']


def test_schema_failure_and_request_failure_have_honest_absent_post():
    bad=dict(fact(1),value=None)
    result,_,_=run([{'projections':[bad]}]*2)
    assert all(d.projector_pre_parse_summary and d.projector_post_conversion_summary is None for d in result.llm_diagnostics)
    assert result.llm_diagnostics[0].error_stage=='schema_validation'
    result,_,_=run([LLMRequestError('request failed')])
    assert result.llm_diagnostics[0].projector_pre_parse_summary is None
    assert result.llm_diagnostics[0].projector_post_conversion_summary is None


def test_cache_hit_never_invents_completion_trace(tmp_path):
    cache=SQLiteEvidenceAnalysisCache(tmp_path/'cache.sqlite3')
    run([{'projections':[fact(1)]}],cache=cache)
    result,client,_=run([],cache=cache)
    assert result.llm_calls==0 and not result.llm_diagnostics and not client.calls
    assert result.cache_stats.hits==1


def test_privacy_of_new_public_llm_call_audit_trace():
    raw=fact(1)
    raw.update(value='SECRET_VALUE_SENTINEL',fact_type='SECRET_FACT_TYPE_SENTINEL')
    raw['support_candidates'][0]['support_scope']='SECRET_SCOPE_SENTINEL'
    raw['support_candidates'][0]['unresolved_aspects']=['SECRET_GAP_SENTINEL']
    result,client,_=run([{'projections':[raw]}])
    # Existing Evidence audit is deliberately unchanged; only the additive
    # LLM-call audit is the privacy boundary authorized by the user.
    payload=asdict(_llm_call(result.llm_diagnostics[0]))
    text=json.dumps(payload)
    assert all(s not in text for s in ('SECRET_VALUE_SENTINEL','SECRET_FACT_TYPE_SENTINEL','SECRET_SCOPE_SENTINEL','SECRET_GAP_SENTINEL',record(1).evidence.content))
    assert 'system_prompt' not in text and 'user_prompt' not in text
    assert payload['projector_pre_parse_summary']['items'][0]['projections'][0]['support_candidates'][0]['source_span_ids']==['S1']


def test_new_call_checkpoint_and_audit_roundtrip_old_records_remain_none():
    result,_,_=run([{'projections':[fact(1)]}])
    records=result.llm_diagnostics
    assert load_contract(dump_contract(records,tuple[LLMCallRecord,...]),tuple[LLMCallRecord,...])==records
    old=load_contract([dict(node='project_evidence',attempt=1,success=True)],tuple[LLMCallRecord,...])[0]
    assert old.projector_pre_parse_summary is old.projector_post_conversion_summary is None
    historical=AgenticAuditSnapshot.from_json((ROOT/'eval/results/e1_live_smoke_20261003/cases/fact_001.audit.json').read_text(encoding='utf-8'))
    assert all(d.projector_pre_parse_summary is d.projector_post_conversion_summary is None for d in historical.llm_calls)
    assert AgenticAuditSnapshot.from_json(historical.to_json())==historical


def test_all_text_sentinels_are_absent_from_new_public_diagnostics():
    # A malformed response may contain any extra text. Observability must not
    # echo it even when the production schema rejects the response.
    sentinels = ('SECRET_SOURCE_SENTINEL', 'SECRET_MECHANISM_SENTINEL',
                 'SECRET_VALUE_SENTINEL', 'SECRET_PROMPT_SENTINEL',
                 'SECRET_REASONING_SENTINEL', 'SECRET_EVIDENCE_SENTINEL')
    raw = dict(fact(1), value=sentinels[2], mechanism=sentinels[1],
               prompt=sentinels[3], reasoning=sentinels[4], evidence_content=sentinels[5])
    raw['support_candidates'][0]['source_span_ids'] = [sentinels[0]]
    result, _, _ = run([{'projections': [raw]}] * 2)
    for diagnostic in result.llm_diagnostics:
        payload = json.dumps(asdict(_llm_call(diagnostic)))
        assert all(sentinel not in payload for sentinel in sentinels)
        assert 'source_span_ids' in payload
        assert diagnostic.projector_post_conversion_summary is None


@pytest.mark.parametrize('behavior', ['valid', 'empty', 'rejection', 'mixed'])
def test_observability_does_not_change_requests_results_or_recovery(monkeypatch, behavior):
    import paperbase.agentic.evidence_projection as module
    if behavior == 'valid':
        responses, n = [{'projections': [fact(1)]}], 1
    elif behavior == 'empty':
        responses, n = [{'projections': [dict(fact(1), supported_requirement_ids=[], support_candidates=[])]}], 1
    elif behavior == 'rejection':
        responses, n = [{'projections': [fact(1, excerpt='invalid')]}] * 2, 1
    else:
        payload = {'items': [item(1), item(2, [fact(2, excerpt='invalid')])]}
        responses, n = [payload, payload, {'projections': [fact(2)]}], 2
    observed, observed_client, _ = run(deepcopy(responses), n)
    monkeypatch.setattr(module, 'pre_parse_summary', lambda **kwargs: None)
    monkeypatch.setattr(module, 'post_conversion_summary', lambda **kwargs: None)
    plain, plain_client, _ = run(deepcopy(responses), n)
    stripped = replace(observed, llm_diagnostics=tuple(replace(
        d, projector_pre_parse_summary=None, projector_post_conversion_summary=None
    ) for d in observed.llm_diagnostics))
    assert stripped == plain
    assert observed_client.calls == plain_client.calls
    assert observed_client.budgets == plain_client.budgets
