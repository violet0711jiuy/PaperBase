"""E1 synthetic traces A-H, current public DTO capture, and historical compatibility."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path

import pytest

from paperbase.evaluation.contracts import (CaseTrace, EvidenceTrace, RetrievalCandidate, ResearchStep,
    RequirementTrace, AnswerTrace, DatasetIssue, EVALUATION_SCHEMA_VERSION, json_safe)
from paperbase.evaluation.dataset import validate_golden_dataset
from paperbase.evaluation.diagnostic import evaluate_agentic_case
from paperbase.evaluation.reporting import aggregate_evaluations, metrics_by_primary_type
from paperbase.evaluation.trace import trace_from_artifacts, trace_from_result, result_artifact
from paperbase.evaluation.run_agentic_e1 import replay, safe_model_summary, SMOKE_CASE_IDS

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    import paperbase.llm.client as client
    import paperbase.agentic.factory as factory
    def forbidden(*args, **kwargs):
        raise AssertionError('E1 tests must not construct/call a real provider')
    monkeypatch.setattr(client, 'create_chat_completion_client', forbidden)
    monkeypatch.setattr(factory, 'create_agentic_research_service', forbidden)
    monkeypatch.setattr(client, 'complete_json_with_diagnostics', forbidden)


def golden(*, gold=('g1',), answerable=True, kind='method', case_id='method_001'):
    return dict(id=case_id,question='Explain the model',primary_type=kind,tags=['single_hop','easy'],paper_id='P',
        answerable=answerable,reference_answer='Reference for future E2' if answerable else None,
        required_facts=['Required fact'] if answerable else [],relevant_evidence=[dict(paper_id='P',
        section='References' if kind=='bibliography' else 'Methods',page_start=1,page_end=1,chunk_ids=list(gold))] if answerable else [],
        expected_bibliography_intent=kind=='bibliography')


def trace(*, retrieved=('g1',), registered=('g1',), verified=True, selected=True,
          cited=True, insufficient=False, stop='agent_finished', answerable=True):
    evidence=() if not registered else (EvidenceTrace('E1','P',tuple(registered),qualified=True,
        projected=True,verified_requirement_ids=('R1',) if verified else (),projection_types=('mechanism',),
        selected=selected,cited=cited),)
    return CaseTrace('completed',stop,evidence,(ResearchStep(1,'search_papers',True,True,1,
        tuple(RetrievalCandidate('P',(c,),retrieval_rank=i) for i,c in enumerate(retrieved,1)),1 if verified and registered else 0),),
        (RequirementTrace('R1','Explain the model',True,'mechanism','in_progress'),),
        ('E1',) if selected and registered else (),tuple(registered) if selected else (),
        AnswerTrace(not insufficient,insufficient,False,('E1',) if cited and registered else (), 'success'),
        counters=(('tool_calls',1),('retrieval_queries',1),('agent_iterations',2),('llm_calls',3)),
        guard_rejection_codes=(),validation_error_count=0,llm_nodes=(('decide_action',True),),decision_calls=1)


@pytest.mark.parametrize('case,arguments,stage,label',[
    ('A',dict(retrieved=(),registered=(),verified=False,selected=False,cited=False),'retrieval','retrieval_miss'),
    ('B',dict(verified=False,selected=False,cited=False),'projection','gold_registered_not_verified'),
    ('C',dict(selected=False,cited=False),'selection','gold_verified_not_selected'),
    ('D',dict(insufficient=True,cited=False),'answer','false_abstention'),
    ('H',dict(verified=False,selected=False,cited=False,stop='budget_exhausted'),'projection','gold_registered_not_verified'),
])
def test_funnel_failure_matrix_a_b_c_d_h(case,arguments,stage,label):
    evaluation=evaluate_agentic_case(golden(),trace(**arguments))
    assert evaluation.primary_failure_stage==stage
    assert label in evaluation.failure_labels
    if case=='H':
        assert 'budget_exhausted' in evaluation.failure_labels
    if case=='B':
        assert 'retrieval_miss' not in evaluation.failure_labels


def test_registry_loss_has_its_own_earliest_stage():
    row=evaluate_agentic_case(golden(),trace(registered=(),selected=False,cited=False))
    assert row.primary_failure_stage=='registry'
    assert row.failure_labels==('gold_retrieved_not_registered',)


@pytest.mark.parametrize('insufficient,stage,label',[(True,None,None),(False,'answer','unsafe_non_abstention')])
def test_unanswerable_e_f(insufficient,stage,label):
    row=evaluate_agentic_case(golden(answerable=False,kind='unanswerable'),trace(retrieved=(),registered=(),
        selected=False,cited=False,insufficient=insufficient))
    assert row.primary_failure_stage==stage
    assert row.answer.correct_abstention==insufficient
    assert row.answer.unsafe_non_abstention==(not insufficient)
    assert row.failure_labels==((label,) if label else ())
    assert row.retrieval.cumulative_gold_retrieval_recall is None


def test_partial_gold_g_and_legacy_top5_are_separate_from_cumulative():
    t=replace(trace(),legacy_ranked_chunk_ids=('other','g2'))
    row=evaluate_agentic_case(golden(gold=('g1','g2')),t)
    assert row.retrieval.hit_at_5==1 and row.retrieval.recall_at_5==0.5
    assert row.retrieval.initial_search_gold_recall==row.retrieval.cumulative_gold_retrieval_recall==0.5
    assert row.primary_failure_stage=='retrieval' and 'retrieval_partial' in row.failure_labels


def test_success_not_forced_into_failure_or_semantic_correctness():
    row=evaluate_agentic_case(golden(),trace())
    assert row.primary_failure_stage is None and row.failure_labels==()
    assert row.evidence_funnel.gold_cited_chunk_ids==('g1',)
    assert row.answer.citation_validity==1
    assert 'correctness' not in row.to_dict()
    changed={**golden(),'reference_answer':'Opposite words','required_facts':['unmapped alternative text']}
    assert evaluate_agentic_case(changed,trace())==row


def test_multichunk_expanded_seed_bibliography_provenance_and_reused_units():
    t=trace(registered=('expanded_gold','g1','other'),retrieved=('g1',))
    ev=replace(t.evidence[0],evidence_id='REF1',seed_chunk_ids=('seed_gold',),projection_types=('enumeration','fact'))
    steps=(replace(t.steps[0],candidates=(RetrievalCandidate('P',('expanded_gold','g1','other'),('seed_gold',)),)),
        replace(t.steps[0],iteration=2,new_verified_support=0,candidates=(RetrievalCandidate('P',('expanded_gold','g1','other'),('seed_gold',)),)))
    t=replace(t,evidence=(ev,),steps=steps,selected_evidence_ids=('REF1',),selected_source_chunk_ids=None,
        answer=replace(t.answer,citations=('REF1',)))
    row=evaluate_agentic_case(golden(gold=('expanded_gold','seed_gold','g1'),kind='bibliography'),t)
    for stage in ('retrieved','registered','projected','verified','selected','cited'):
        assert set(getattr(row.evidence_funnel,f'gold_{stage}_chunk_ids'))=={'expanded_gold','seed_gold','g1'}
    assert row.research.verified_supported_requirement_count==1
    assert row.efficiency.executed_actions==2 and row.efficiency.zero_support_actions==1
    assert row.answer.citation_count==1


def test_initial_is_first_really_executed_search_context_only_affects_cumulative():
    base=trace(retrieved=(),registered=('g1',))
    context=ResearchStep(1,'get_section_context',True,True,0,(RetrievalCandidate('P',('g1',)),),0)
    skipped=ResearchStep(2,'search_references',True,False,0,(),0)
    actual=ResearchStep(3,'search_papers',True,True,1,(RetrievalCandidate('P',('other',)),),1)
    row=evaluate_agentic_case(golden(),replace(base,steps=(context,skipped,actual)))
    assert row.retrieval.initial_search_hit is False
    assert row.retrieval.cumulative_gold_retrieval_recall==1
    assert row.retrieval.queries_until_first_gold is None
    assert 'queries_until_first_gold_attribution' in row.observability_gaps


def test_queries_until_first_gold_stagnation_counts_reset_and_control_secondary():
    t=trace(stop='budget_exhausted',verified=False,selected=False,cited=False)
    steps=tuple(ResearchStep(i,'search_papers',True,True,1,(RetrievalCandidate('P',('g1' if i==3 else 'x',)),),n)
        for i,n in enumerate((0,0,1,0,0,0),1))
    row=evaluate_agentic_case(golden(),replace(t,steps=steps,guard_rejection_codes=('DuplicateAction','DuplicatePaperQuery')))
    assert row.retrieval.queries_until_first_gold==3
    assert row.efficiency.consecutive_no_progress_peak==3
    assert row.efficiency.zero_support_actions==5
    assert row.efficiency.actions_until_first_verified_support==3
    assert row.primary_failure_stage=='projection'
    assert {'duplicate_loop','agent_search_stagnation','budget_exhausted'}.issubset(row.failure_labels)


def test_batched_first_gold_query_is_unknown_not_guessed():
    t=trace()
    row=evaluate_agentic_case(golden(),replace(t,steps=(replace(t.steps[0],executed_queries=2),)))
    assert row.retrieval.queries_until_first_gold is None
    assert 'queries_until_first_gold_attribution' in row.observability_gaps


def test_unknown_prior_search_does_not_steal_initial_from_known_later_action():
    t=trace()
    unknown=replace(t.steps[0],retrieval_executed=None,candidates=None)
    row=evaluate_agentic_case(golden(),replace(t,steps=(unknown,replace(t.steps[0],iteration=2))))
    assert row.retrieval.initial_search_hit is None
    assert row.retrieval.queries_until_first_gold is None


def test_legacy_fields_never_backfill_verified_or_chunk_identity():
    result=dict(run_status='partial',stop_reason='budget_exhausted',answer=dict(insufficient_evidence=True,citations=[]),
        selected_evidence_ids=[],selected_source_chunk_ids=[])
    audit=dict(evidence=[dict(evidence_id='E1',analysis_status='qualified',supports_requirement_ids=['R1'],selected=False,cited=False)],
        steps=[dict(iteration=1,action_name='search_papers',added_count=1)],requirements=[dict(requirement_id='R1',required=True,description='goal')])
    before=deepcopy((result,audit))
    row=evaluate_agentic_case(golden(),trace_from_artifacts(result,audit))
    assert row.evidence_funnel.gold_registered_chunk_ids is None
    assert row.evidence_funnel.gold_verified_chunk_ids is None
    assert 'retrieval_miss' not in row.failure_labels
    assert 'verified_support_authority' in row.observability_gaps
    assert (result,audit)==before


def test_invalid_citation_requires_registry_and_selector_membership():
    t=trace()
    row=evaluate_agentic_case(golden(),replace(t,answer=replace(t.answer,citations=('E1','E99'))))
    assert row.answer.citation_validity==0.5 and row.answer.invalid_citation_ids==('E99',)
    assert row.primary_failure_stage=='answer'
    row=evaluate_agentic_case(golden(),replace(t,selected_evidence_ids=(),selected_source_chunk_ids=()))
    assert row.answer.citation_validity==0 and 'invalid_citation' in row.failure_labels


def test_verified_gold_unit_dropped_cannot_be_hidden_by_unverified_selected_unit():
    t=trace()
    unverified=replace(t.evidence[0],evidence_id='E2',verified_requirement_ids=(),selected=True)
    verified=replace(t.evidence[0],selected=False)
    t=replace(t,evidence=(verified,unverified),selected_evidence_ids=('E2',),answer=replace(t.answer,citations=('E2',)))
    row=evaluate_agentic_case(golden(),t)
    assert row.evidence_funnel.gold_selected_chunk_ids==('g1',)
    assert row.evidence_funnel.gold_verified_selected_chunk_ids==()
    assert row.primary_failure_stage=='selection'
    assert aggregate_evaluations([row])['evidence_funnel']['verified_to_selected']['value']==0


def test_micro_funnel_denominators_and_unknown_metrics_not_zero():
    a=evaluate_agentic_case(golden(gold=('g1','g2')),trace())
    b=evaluate_agentic_case(golden(),trace())
    unknown=evaluate_agentic_case(golden(),CaseTrace(None,None,None,None,None,None,None,None))
    dataset_error=evaluate_agentic_case(golden(),trace(),dataset_issues=(DatasetIssue('method_001','missing_chunk','g1'),))
    m=aggregate_evaluations([a,b,unknown,dataset_error])
    assert m['dataset_error_cases']==1 and m['agent_case_count']==3
    assert m['evidence_funnel']['gold_retrieved_rate']['value']==2/3
    assert m['evidence_funnel']['gold_retrieved_rate']['observed_cases']==2
    assert m['evidence_funnel']['registered_to_verified']['value']==1
    assert 'dataset_error' not in m['failure_label_counts']
    assert metrics_by_primary_type([a,b])['method']['case_count']==2


def database_catalog(rows):
    return {'P'}, {c:dict(chunk_id=c,paper_id='P',section=e['section'],section_type='bibliography' if r['primary_type']=='bibliography' else 'content',
        page_start=1,page_end=1) for r in rows for e in r['relevant_evidence'] for c in e['chunk_ids']}


@pytest.mark.parametrize('fault',['duplicate','paper','missing_chunk','bibliography','unanswerable','too_few_gold','invalid_id_type','unsafe_id'])
def test_golden_validation_errors_are_dataset_not_agent_failures(fault):
    rows=[golden()];papers,chunks=database_catalog(rows)
    if fault=='duplicate': rows.append(deepcopy(rows[0]))
    elif fault=='paper': papers=set()
    elif fault=='missing_chunk': chunks={}
    elif fault=='bibliography': rows[0]=golden(kind='bibliography');chunks['g1']['section_type']='content'
    elif fault=='unanswerable': rows[0]['answerable']=False
    elif fault=='too_few_gold': rows[0]['relevant_evidence']=[]
    elif fault=='invalid_id_type': rows[0]['id']=['invalid']
    elif fault=='unsafe_id': rows[0]['id']='../escape'
    validation=validate_golden_dataset(rows,paper_ids=papers,chunks=chunks)
    assert not validation.valid
    row=evaluate_agentic_case(rows[0],trace(),dataset_issues=validation.issues)
    assert row.primary_failure_stage=='dataset' and row.failure_labels==('dataset_error',)


def test_frozen_forty_schema_validate_readonly_and_smoke_all_types():
    from paperbase.evaluation.dataset import load_validated_golden
    path=ROOT/'eval/datasets/golden_dataset_v1_2.jsonl';before=path.read_bytes()
    rows,validation=load_validated_golden(path,ROOT/'storage/paperbase.sqlite3')
    assert validation.valid and validation.case_count==40
    assert 'expected_bibliography_intent' in validation.actual_fields
    assert 'expected_bibliography' not in validation.actual_fields
    smoke=[r for r in rows if r['id'] in SMOKE_CASE_IDS]
    assert len(smoke)==8 and len({r['primary_type'] for r in smoke})==7
    assert path.read_bytes()==before


def test_manifest_model_summary_allows_no_secret_prompt_or_reasoning():
    raw=dict(model='qwen',provider='modelscope',temperature=0,api_key='SECRET',secret='SECRET',
        raw_hidden_reasoning='PRIVATE',prompt='PRIVATE',nested=dict(api_key='SECRET'))
    summary=safe_model_summary(raw)
    assert summary==dict(model='qwen',provider='modelscope',temperature=0)
    assert 'SECRET' not in json.dumps(summary) and 'PRIVATE' not in json.dumps(summary)
    raw['base_url']='https://user:SECRET@api.example.test/v1?api_key=SECRET'
    summary=safe_model_summary(raw)
    assert summary['endpoint_host']=='api.example.test' and 'SECRET' not in json.dumps(summary)


def public_result():
    from paperbase.agentic.contracts import (ResearchRunResult, ResearchRunStatus, StopReason, BudgetUsage,
        ResearchObservation, EvidenceFilterReport, EvidenceFilterDecision, EvidenceAnalysis,
        EvidenceAnalysisStatus, MechanismProjection, FactProjection, FactItem, RequirementSupportAnchor,
        EvidenceOwnership, EvidenceProjectionType, ResearchAnswer)
    from paperbase.retrieval.contracts import ToolStatus
    from test_requirement_multi_projection import source, typed_plan
    from test_agentic_batch_projector import CONTEXT
    plan=typed_plan(EvidenceProjectionType.MECHANISM,EvidenceProjectionType.FACT)
    r=source();excerpt=r.evidence.content
    mechanism=MechanismProjection(r.evidence_id,'Alpha encoder',supported_requirement_ids=('R1',),proposed_requirement_ids=('R1',),
        support_anchors=(RequirementSupportAnchor('R1',excerpt,'encoder'),))
    fact=FactProjection(r.evidence_id,(FactItem('model','Alpha',EvidenceOwnership.THIS_PAPER,
        supported_requirement_ids=('R2',),proposed_requirement_ids=('R2',),support_anchors=(RequirementSupportAnchor('R2',excerpt,'Alpha'),)),))
    r=replace(r,assessment=EvidenceAnalysis(r.evidence_id,EvidenceAnalysisStatus.QUALIFIED,projections=(mechanism,fact),evaluated_requirement_ids=('R1','R2')))
    obs=ResearchObservation(1,'search_papers',ToolStatus.OK,('E1',),(),
        evidence_filter_report=EvidenceFilterReport(1,1,0,0,(EvidenceFilterDecision(r.fingerprint,r.evidence.paper_id,r.evidence.source_chunk_ids,'accepted',()),)),
        request_executions=({'executed':True},),new_supporting_evidence_count=1)
    result=ResearchRunResult(ResearchRunStatus.COMPLETED,StopReason.AGENT_FINISHED,CONTEXT,plan,(r,),(obs,),(),BudgetUsage(2,1,1,3),
        selected_evidence=(r,),answer=ResearchAnswer('success','Alpha is the model.',('E1',)))
    return result


def test_current_public_result_capture_multi_projection_provenance_and_roundtrip():
    result=public_result()
    before=deepcopy(result)
    trace=trace_from_result(result)
    assert trace.evidence[0].verified_requirement_ids==('R1','R2')
    assert trace.evidence[0].projection_types==('mechanism','fact')
    payload=json_safe(result_artifact(result))
    assert trace_from_artifacts(payload,{})==trace
    assert result==before
    row=evaluate_agentic_case(golden(gold=result.evidence[0].evidence.source_chunk_ids),trace)
    assert row.retrieval.cumulative_gold_retrieval_recall==1
    assert row.research.verified_supported_requirement_count==2
    assert row.answer.gold_citation_coverage==1


def test_public_admission_rejection_is_retrieved_not_registered():
    from paperbase.agentic.contracts import ResearchAnswer
    result=public_result()
    report=result.observations[0].evidence_filter_report
    decision=replace(report.decisions[0],decision='rejected',reasons=('below_rerank_threshold',))
    report=replace(report,accepted_candidates=0,rejected_candidates=1,decisions=(decision,))
    observation=replace(result.observations[0],added_evidence_ids=(),new_supporting_evidence_count=0,evidence_filter_report=report)
    g=golden(gold=result.evidence[0].evidence.source_chunk_ids)
    result=replace(result,evidence=(),selected_evidence=(),observations=(observation,),
        answer=ResearchAnswer('fallback','Insufficient evidence',(),insufficient_evidence=True))
    row=evaluate_agentic_case(g,trace_from_result(result))
    assert row.evidence_funnel.gold_retrieved_chunk_ids==tuple(g['relevant_evidence'][0]['chunk_ids'])
    assert row.evidence_funnel.gold_registered_chunk_ids==()
    assert row.primary_failure_stage=='registry'
    assert 'retrieval_miss' not in row.failure_labels


def test_decision_invocations_are_deduplicated_across_repair_attempts():
    from paperbase.llm.client import LLMCallRecord
    result=replace(public_result(),llm_diagnostics=(
        LLMCallRecord('decide_action',1,False,decision_iteration=1),
        LLMCallRecord('decide_action',2,True,decision_iteration=1),
        LLMCallRecord('decide_action',1,True,decision_iteration=2)))
    row=evaluate_agentic_case(golden(),trace_from_result(result))
    assert row.efficiency.decision_calls==2
    assert row.efficiency.decision_llm_calls==3
    legacy=evaluate_agentic_case(golden(),trace_from_artifacts({},dict(llm_calls=[
        dict(node='decide_action',success=True,attempt=1),dict(node='decide_action',success=True,attempt=2)])))
    assert legacy.efficiency.decision_calls is None and legacy.efficiency.decision_llm_calls==2


def test_only_context_has_no_initial_search_and_does_not_fake_top5():
    t=trace()
    context=replace(t.steps[0],tool='get_section_context',executed_queries=0)
    row=evaluate_agentic_case(golden(),replace(t,steps=(context,)))
    assert row.retrieval.initial_search_hit is None
    assert row.retrieval.hit_at_5 is None
    assert row.retrieval.cumulative_gold_retrieval_recall==1


def test_unexecuted_guard_action_does_not_count_as_retrieved():
    t=trace(registered=(),selected=False,cited=False)
    rejected=replace(t.steps[0],executed=False,retrieval_executed=False)
    row=evaluate_agentic_case(golden(),replace(t,steps=(rejected,)))
    assert row.retrieval.initial_search_hit is None
    assert row.evidence_funnel.gold_retrieved_chunk_ids==()
    assert row.efficiency.executed_actions==0


def test_missing_answer_partial_and_planner_fallback_are_distinct_labels():
    t=trace()
    no_answer=replace(t,answer=replace(t.answer,has_answer=False,citations=()))
    assert evaluate_agentic_case(golden(),no_answer).primary_failure_stage=='answer'
    partial=replace(t,answer=replace(t.answer,partial_answer=True))
    assert 'partial_answer' in evaluate_agentic_case(golden(),partial).failure_labels
    fallback=replace(t,planning_mode='literal_fallback',llm_nodes=(('build_plan',False),))
    assert 'planner_validation_fallback' in evaluate_agentic_case(golden(),fallback).failure_labels
    repaired=replace(t,planning_mode='model',llm_nodes=(('build_plan',False),('build_plan',True)))
    assert 'planner_validation_fallback' not in evaluate_agentic_case(golden(),repaired).failure_labels


@pytest.mark.parametrize('setup_fail',[False,True])
def test_live_entry_contract_uses_fake_service_and_writes_manifest_before_case(monkeypatch,tmp_path,setup_fail):
    from types import SimpleNamespace
    from paperbase.evaluation import run_agentic_e1 as runner
    from paperbase.evaluation.contracts import GoldenValidation
    import paperbase.agentic.factory as factory
    import paperbase.config as config
    import paperbase.llm.client as client
    out=tmp_path/'mock-live'
    monkeypatch.setattr(runner,'load_validated_golden',lambda *a:([golden()],GoldenValidation(1,(),(),True)))
    settings=SimpleNamespace(config_path=tmp_path/'config.yaml',agentic_research=SimpleNamespace(model_dump=lambda **k:dict(max_agent_iterations=8)))
    monkeypatch.setattr(config,'load_settings',lambda *a:settings)
    monkeypatch.setattr(client,'load_llm_runtime_settings',lambda *a:SimpleNamespace(model='fake',provider='mock',temperature=0,api_key='SECRET'))
    class FakeService:
        calls=0
        closed=False
        def research(self,*args,**kwargs):
            assert (out/'run_manifest.json').is_file()
            self.calls+=1
            return public_result()
        def close(self): self.closed=True
    service=FakeService()
    def create(**kwargs):
        if setup_fail:
            raise RuntimeError('SECRET provider setup detail')
        return service
    monkeypatch.setattr(factory,'create_agentic_research_service',create)
    records=runner.live(dataset=ROOT/'eval/datasets/golden_dataset_v1_2.jsonl',output=out,database=tmp_path/'unused')
    assert len(records)==1 and service.calls==(0 if setup_fail else 1) and service.closed==(not setup_fail)
    if setup_fail:
        assert records[0].primary_failure_stage=='scope'
    for suffix in ('result','audit','evaluation'):
        assert (out/f'cases/method_001.{suffix}.json').is_file()
    manifest=(out/'run_manifest.json').read_text(encoding='utf-8')
    assert 'SECRET' not in manifest and 'projector-requirement-directed-multi-projection-v4' in manifest
    assert 'SECRET' not in (out/'cases/method_001.result.json').read_text(encoding='utf-8')


def test_live_invalid_golden_stops_before_provider_construction(monkeypatch,tmp_path):
    from paperbase.evaluation import run_agentic_e1 as runner
    from paperbase.evaluation.contracts import GoldenValidation
    validation=GoldenValidation(1,(DatasetIssue('method_001','missing_chunk','g1'),),(),True)
    monkeypatch.setattr(runner,'load_validated_golden',lambda *a:([golden()],validation))
    records=runner.live(dataset=ROOT/'eval/datasets/golden_dataset_v1_2.jsonl',output=tmp_path/'blocked',database=tmp_path/'unused')
    assert records[0].primary_failure_stage=='dataset'


def test_incomplete_tool_observations_never_prove_a_retrieval_miss():
    result=replace(public_result(),evidence=(),selected_evidence=(),observations=())
    assert result.budget_usage.tool_calls==1
    row=evaluate_agentic_case(golden(),trace_from_result(result))
    assert row.evidence_funnel.gold_retrieved_chunk_ids is None
    assert row.efficiency.executed_actions is None
    assert 'incomplete_tool_observations' in row.observability_gaps
    assert 'retrieval_miss' not in row.failure_labels


def test_scope_failure_precedes_absent_plan_in_legacy_audit():
    row=evaluate_agentic_case(golden(),trace_from_artifacts(
        dict(run_status='needs_clarification',stop_reason='scope_unresolved'),dict(requirements=[],evidence=[])))
    assert row.primary_failure_stage=='scope'


def test_fatal_run_stop_without_observable_layers_is_control_not_success():
    row=evaluate_agentic_case(golden(),CaseTrace('failed','fatal_error',None,None,None,None,None,None))
    assert row.primary_failure_stage=='control' and 'fatal_run_stop' in row.failure_labels


def test_historical_replay_smoke_writes_only_new_evaluation_without_provider(tmp_path):
    source=ROOT/'storage/evaluation/agentic_eval_glm45_p0_final_guard_20260923_40case'
    paths=[p for p in (source/'cases').glob('*.json')]
    before={p:p.read_bytes() for p in paths}
    records,summary=replay(dataset=ROOT/'eval/datasets/golden_dataset_v1_2.jsonl',source=source,
        output=tmp_path/'e1',database=ROOT/'storage/paperbase.sqlite3',smoke=True)
    assert len(records)==8 and summary['case_count']==8
    assert all(p.read_bytes()==raw for p,raw in before.items())
    manifest=json.loads((tmp_path/'e1/run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['evaluation_schema_version']==EVALUATION_SCHEMA_VERSION
    assert manifest['production_semantic_authority_version'] is None
    assert len(list((tmp_path/'e1/cases').glob('*.evaluation.json')))==8
    assert (tmp_path/'e1/evaluation_report.md').is_file()
    with pytest.raises(ValueError,match='outside'):
        replay(dataset=ROOT/'eval/datasets/golden_dataset_v1_2.jsonl',source=source,output=source/'e1',database=ROOT/'storage/paperbase.sqlite3')
