"""Read-only support characterization using fixed live sources and fake responses."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3

import pytest
from pydantic import ValidationError

from paperbase.agentic.contracts import (
    EvidenceRecord, EvidenceProjectionType as Shape, EvidenceAnalysisStatus as Status,
    ResearchPlan, ResearchRequirement, QueryContext, Intent, QueryScope, ScopeSource,
    ResearchGoal, EvidenceTarget, RegistryState,
)
from paperbase.agentic.evidence_analysis import EvidenceAnalysisRequest, projection_requirement_scope, merge_analysis_groups
from paperbase.agentic.evidence_projection import LLMEvidenceProjector, _parse_projection_batch, _convert, ProjectionRequest, _projection_validation_diagnostics
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.agentic.grounding import verified_support_ids, normalized_source
from paperbase.agentic.schemas import transport_projection_model
from paperbase.llm.client import LLMCompletion
from paperbase.retrieval.contracts import RetrievedEvidence

ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/'eval/results/e1_live_smoke_20261003'
SAMPLES={'method_001':'E3','fact_001':'E1'}


def sample(case,eid=None):
    """Known fields from artifacts; unknown original goal/question are explicit fixture values."""
    result=json.loads((LIVE/'cases'/f'{case}.result.json').read_text(encoding='utf-8'))
    audit=json.loads((LIVE/'cases'/f'{case}.audit.json').read_text(encoding='utf-8'))
    eid=eid or SAMPLES[case]
    tr=next(e for e in result['case_trace']['evidence'] if e['evidence_id']==eid)
    ev=next(e for e in audit['evidence'] if e['evidence_id']==eid)
    targets={b['requirement_id']:b['evidence_targets'] for step in audit['steps'] for b in step['requirement_target_bindings']}
    requirements=tuple(ResearchRequirement(r['requirement_id'],r['description'],frozenset(targets[r['requirement_id']]),
        r['required'],r['projection_type']) for r in result['case_trace']['requirements'])
    plan=ResearchPlan('RCA fixture goal; original live Plan.goal was not persisted',requirements)
    with sqlite3.connect((ROOT/'storage/paperbase.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[db.execute('SELECT * FROM chunks WHERE chunk_id=?',(c,)).fetchone() for c in tr['source_chunk_ids']]
    content='\n\n'.join(r['raw_text'].strip() for r in rows).strip()
    evidence=RetrievedEvidence('content',tr['paper_id'],tuple(tr['seed_chunk_ids']),tuple(tr['source_chunk_ids']),content,
        ev['paper_title'],rows[0]['section_id'],ev['section'],ev['page_start'],ev['page_end'])
    # Step 1 has one executed paper request; every newly registered E1..En has its scope.
    bound=tuple(audit['steps'][0]['action_arguments']['searches'][0]['requirement_ids'])
    rec=EvidenceRecord(eid,evidence,(),1,1,requested_requirement_ids=bound)
    ctx=QueryContext('RCA fixture question','RCA fixture question',Intent.SYNTHESIZE,QueryScope.SINGLE_PAPER,
        (tr['paper_id'],),ScopeSource.UI_SELECTION,ResearchGoal.FIND_EVIDENCE,frozenset({EvidenceTarget.METHOD}))
    return rec,plan,ctx


def request(rec,plan,ctx,peer=False):
    records=(rec,)
    if peer:
        p=replace(rec,evidence_id='PEER',fingerprint=None,assessment=None,
            evidence=replace(rec.evidence,source_chunk_ids=('synthetic-peer',),seed_chunk_ids=('synthetic-peer',),
                content='Synthetic peer. '+rec.evidence.content))
        records+=(p,)
    return EvidenceAnalysisRequest(ctx.resolved_query,records,ctx,plan,
        tuple((r.evidence_id,projection_requirement_scope(r,plan)) for r in records))


class FakeClient:
    def __init__(self,behavior='valid'):
        self.behavior=behavior
        self.calls=[]
    def complete_json_diagnostic(self,**kwargs):
        self.calls.append(kwargs)
        payload=json.loads(kwargs['user_prompt'])
        is_batch=kwargs['schema_name']=='paperbase_evidence_transport_batch'
        if self.behavior=='fallback_valid' and is_batch:
            return LLMCompletion('{malformed')
        if self.behavior=='partial_retention' and is_batch and len(self.calls)==2:
            return LLMCompletion('{malformed')
        parts=[]
        for i,e in enumerate(payload['evidence']):
            shape=payload['projection_type'];ids=payload['projection_requirement_ids']
            source_text='\n'.join(span['text'] for span in e['source_spans'])
            part=dict(projection_type=shape,status='qualified',supported_requirement_ids=list(ids),
                support_candidates=[dict(requirement_id=rid,source_span_ids=[e['source_spans'][0]['span_id']],
                    support_scope='Deterministic characterization of a source-local assertion',unresolved_aspects=[]) for rid in ids])
            if shape=='fact':
                threshold='0.8' in source_text and 'quantile' in source_text.lower()
                part.update(fact_type='quantile_violation_threshold' if threshold else 'architecture_components',
                    value='0.8 σ' if threshold else 'Three parallel dense layers use softplus, ReLU, and exponential activation functions.',ownership='this_paper')
            elif shape=='mechanism':part.update(mechanism='Separate branches estimate the mean and standard deviation; the multi-activation ensemble captures diverse uncertainty patterns.',
                steps=['Estimate the mean','Estimate the standard deviation with parallel activation functions'],inputs=['Dense layer output'],outputs=['Mean','Standard deviation'])
            elif shape=='comparison':part.update(comparison_dimensions=['ReLU captures sudden uncertainty changes; Softplus models gradual uncertainty evolution'])
            else:part.update(entities=[dict(canonical_name='softplus',entity_type='method',role='adopted',ownership='this_paper')])
            if self.behavior=='empty_support':part.update(supported_requirement_ids=[],support_candidates=[])
            if self.behavior=='out_of_scope':
                part['supported_requirement_ids']=['R_UNKNOWN'];part['support_candidates'][0]['requirement_id']='R_UNKNOWN'
            if self.behavior=='bad_anchor':part['support_candidates'][0]['source_span_ids']=['S999']
            if self.behavior=='missing_anchor':part['support_candidates']=[]
            if self.behavior=='unknown_owner' and shape=='fact':part['ownership']='unknown'
            if self.behavior=='partial_retention' and is_batch and i==1:part['value']=None
            if not is_batch:part['evidence_id']='WRONG_EVIDENCE' if self.behavior=='unknown_evidence' else e['evidence_id']
            parts.append(dict(evidence_id='source:unknown' if self.behavior=='unknown_evidence' else e['evidence_id'],projections=[part]))
        response=dict(items=parts) if is_batch else dict(projections=[p for item in parts for p in item['projections']])
        return LLMCompletion(json.dumps(response),prompt_tokens=1,completion_tokens=1)


@pytest.fixture(autouse=True)
def forbid_real_provider(monkeypatch):
    import paperbase.llm.client as module
    import paperbase.agentic.factory as factory
    def forbidden(*args,**kwargs):raise AssertionError('RCA cannot construct/call a real provider or Agent')
    monkeypatch.setattr(module,'create_chat_completion_client',forbidden)
    monkeypatch.setattr(factory,'create_agentic_research_service',forbidden)


def run(case,behavior='valid',peer=False,*,only_fact=True):
    rec,plan,ctx=sample(case)
    if only_fact:rec=replace(rec,requested_requirement_ids=('R1',))
    req=request(rec,plan,ctx,peer)
    before=deepcopy(asdict(req));client=FakeClient(behavior)
    result=LLMEvidenceProjector(client).analyze(request=req)
    registry=EvidenceRegistry(RegistryState(req.evidence,content_counter=len(req.evidence)))
    registry.apply_analyses(result.analyses,plan=plan)
    assert asdict(req)==before
    return registry.get(rec.evidence_id),result,client,req


@pytest.mark.parametrize('case',SAMPLES)
@pytest.mark.parametrize('path',['single','batch','fallback','partial_retention'])
def test_valid_support_survives_all_stages(case,path):
    behavior={'fallback':'fallback_valid','partial_retention':'partial_retention'}.get(path,'valid')
    current,result,client,req=run(case,behavior,peer=path!='single')
    assert current.analysis_status is Status.QUALIFIED
    assert current.projection.facts[0].value==('0.8 σ' if case=='fact_001' else 'Three parallel dense layers use softplus, ReLU, and exponential activation functions.')
    assert current.projection.proposed_requirement_ids==current.projection.supported_requirement_ids==('R1',)
    assert len(current.projection.support_anchors)==1
    assert normalized_source(current.projection.support_anchors[0].source_excerpt) in normalized_source(current.evidence.content)
    assert current.assessment.evaluated_requirement_ids==('R1',)
    assert verified_support_ids(current,req.plan.requirements)==('R1',)
    if path=='fallback':
        assert result.llm_calls==4
        assert any(d.projector_call_mode=='fallback_single' for d in result.llm_diagnostics)
    if path=='partial_retention':
        assert result.llm_calls==3
        assert all(e['evidence_id']=='PEER' for e in json.loads(client.calls[-1]['user_prompt'])['evidence'])


@pytest.mark.parametrize('case',SAMPLES)
@pytest.mark.parametrize('peer',[False,True])
def test_qualified_facts_without_support_are_legal_not_inferred(case,peer):
    current,result,client,req=run(case,'empty_support',peer)
    assert current.analysis_status is Status.QUALIFIED
    assert current.projection.facts
    assert current.projection.supported_requirement_ids==current.projection.proposed_requirement_ids==()
    assert current.projection.support_anchors==()
    assert current.assessment.evaluated_requirement_ids==('R1',)
    assert current.assessment.projection_rejections==()
    assert verified_support_ids(current,req.plan.requirements)==()
    assert result.llm_calls==1


@pytest.mark.parametrize('case',SAMPLES)
@pytest.mark.parametrize('peer',[False,True])
@pytest.mark.parametrize('behavior',['out_of_scope','bad_anchor','unknown_evidence','unknown_owner','missing_anchor'])
def test_invalid_support_is_rejected_and_recovery_cannot_bless_it(case,peer,behavior):
    current,result,client,req=run(case,behavior,peer)
    assert current.analysis_status is Status.FAILED
    assert not current.projections and current.assessment.evaluated_requirement_ids==()
    assert verified_support_ids(current,req.plan.requirements)==()
    assert result.llm_calls==(6 if peer else 2)
    assert any(d.projector_repair for d in result.llm_diagnostics)
    if peer:assert any(d.projector_call_mode=='fallback_single' for d in result.llm_diagnostics)


def test_live_method_binding_produces_fact_r1_and_mechanism_r2_no_scope_loss():
    current,result,client,req=run('method_001',peer=True,only_fact=False)
    assert req.evidence[0].requested_requirement_ids==('R1','R2')
    payloads=[json.loads(c['user_prompt']) for c in client.calls]
    assert [(p['projection_type'],p['projection_requirement_ids']) for p in payloads]==[('fact',['R1']),('mechanism',['R2'])]
    assert current.assessment.evaluated_requirement_ids==('R1','R2')
    assert verified_support_ids(current,req.plan.requirements)==('R1','R2')


def test_failed_incremental_group_preserves_old_fact_support_without_attesting_scope():
    old,_,_,req=run('method_001')
    expanded=replace(old,requested_requirement_ids=('R1','R2'))
    client=FakeClient('bad_anchor')
    result=LLMEvidenceProjector(client).analyze(request=request(expanded,req.plan,req.context))
    assert result.analyses[0]==old.assessment
    assert json.loads(client.calls[0]['user_prompt'])['projection_requirement_ids']==['R2']
    registry=EvidenceRegistry(RegistryState((expanded,),content_counter=1))
    registry.apply_analyses(result.analyses,plan=req.plan)
    assert verified_support_ids(registry.get(old.evidence_id),req.plan.requirements)==('R1',)


def test_legacy_generic_integrity_error_does_not_invent_specific_rejection():
    error=ValueError('Projection group integrity failed: asserted support has invalid provenance.')
    rec,_,_=sample('method_001')
    diagnostics=_projection_validation_diagnostics(error,1,True,(rec,))
    assert diagnostics[0].error_code=='support_provenance_invalid'
    assert diagnostics[0].error_stage=='domain_validation'
    assert diagnostics[0].recovery_action=='repair_domain'


def test_legacy_untyped_filter_can_retain_facts_but_leaves_proposed_ids_and_rejections():
    rec,plan,ctx=sample('fact_001')
    plan=replace(plan,requirements=(replace(plan.requirements[0],projection_type=None),))
    result=LLMEvidenceProjector(FakeClient('bad_anchor')).analyze(request=request(rec,plan,ctx))
    current=replace(rec,assessment=result.analyses[0])
    assert current.analysis_status is Status.QUALIFIED
    assert current.projection.facts and current.projection.supported_requirement_ids==()
    assert current.projection.proposed_requirement_ids==('R1',)
    assert any('source_span_not_found' in r for r in current.assessment.projection_rejections)
    assert result.llm_calls==1  # Explicit legacy behavior; typed v4 live samples cannot take it.


@pytest.mark.parametrize('case',SAMPLES)
@pytest.mark.parametrize('support',['valid','empty_support'])
def test_provider_batch_schema_and_local_single_schema_accept_expected_shapes(case,support):
    current,result,client,req=run(case,support,True)
    payload=json.loads(client.calls[0]['user_prompt'])
    fake=FakeClient(support)
    completion=fake.complete_json_diagnostic(**client.calls[0])
    transport_projection_model('fact').model_validate_json(completion.content)
    raw=json.loads(completion.content)['items'][0]['projections'][0]
    raw['evidence_id']=req.evidence[0].evidence_id
    _parse_projection_batch(Shape.FACT,json.dumps(dict(projections=[raw])))


@pytest.mark.parametrize('shape',['fact','mechanism','comparison','enumeration'])
def test_explicit_schema_support_fields_roundtrip_consistent_across_shapes(shape):
    rec,plan,ctx=sample('method_001')
    plan=replace(plan,requirements=(replace(plan.requirements[0],projection_type=Shape(shape)),))
    rec=replace(rec,requested_requirement_ids=('R1',))
    client=FakeClient()
    result=LLMEvidenceProjector(client).analyze(request=request(rec,plan,ctx))
    current=replace(rec,assessment=result.analyses[0])
    assert verified_support_ids(current,plan.requirements)==('R1',)
    assert current.assessment.evaluated_requirement_ids==('R1',)
