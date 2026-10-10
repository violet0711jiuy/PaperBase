"""Batch transport recovery and per-Evidence semantic boundary regression."""
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from paperbase.agentic.contracts import (
    EvidenceAnalysisStatus, EvidenceProjectionType, EvidenceRecord, EvidenceTarget, Intent,
    QueryContext, QueryScope, ResearchGoal, ResearchPlan, ResearchRequirement, ScopeSource,
)
from paperbase.agentic.evidence_analysis import EvidenceAnalysisRequest
from paperbase.agentic.evidence_projection import (
    BatchProjectorInput, LLMEvidenceProjector, ProjectionRequest, _validate_transport_items,
)
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.agentic.audit import projector_cost_metrics
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.schemas import transport_projection_model
from paperbase.llm.client import LLMCompletion, LLMCallRecord, LLMRequestError
from paperbase.retrieval.contracts import RetrievedEvidence, EvidenceSourceType

CONTEXT = QueryContext(
    original_query='facts', resolved_query='facts', intent=Intent.SYNTHESIZE,
    scope=QueryScope.SINGLE_PAPER, target_paper_ids=('P',), scope_source=ScopeSource.UI_SELECTION,
    retrieval_goal=ResearchGoal.FIND_EVIDENCE,
    default_evidence_targets=frozenset({EvidenceTarget.PRIOR_WORK}),
)
PLAN = ResearchPlan(goal='facts', requirements=(ResearchRequirement(
    'R1', 'facts', required=True, evidence_targets=frozenset({EvidenceTarget.PRIOR_WORK})),))
REQUEST = ProjectionRequest('facts', CONTEXT, PLAN, ('R1',))


def record(number):
    return EvidenceRecord(f'E{number}', RetrievedEvidence(
        source_type=EvidenceSourceType.BIBLIOGRAPHY, paper_id='P', paper_title='Paper',
        section_id='refs', section='References', seed_chunk_ids=(f'c{number}',),
        source_chunk_ids=(f'c{number}',), content=f'Unique reference fact {number}.',
    ), supporting_queries=('facts',), first_seen_iteration=1, last_seen_iteration=1)


def fact(number, *, excerpt=None):
    span_ids = ['S1'] if excerpt is None or excerpt == record(number).evidence.content else ['S999']
    return dict(projection_type='fact', evidence_id=f'E{number}', status='qualified',
        fact_type='reference', value=f'Unique reference fact {number}.', ownership='prior_work',
        supported_requirement_ids=['R1'], support_candidates=[dict(requirement_id='R1',
        source_span_ids=span_ids, support_scope=f'fact {number}',
        unresolved_aspects=['Another requested aspect remains local.'])])


def item(number, projections=None):
    parts = [fact(number)] if projections is None else projections
    return dict(evidence_id='source:' + record(number).fingerprint, projections=[
        dict(part, evidence_id='source:' + record(number).fingerprint)
        if part.get('evidence_id') == f'E{number}' else part for part in parts])


def batch(*numbers):
    return {'items': [item(number) for number in numbers]}


def wire(payload):
    return {'items': [dict(evidence_id=e['evidence_id'], projections=[
        {k: v for k, v in p.items() if k != 'evidence_id'} for p in e['projections']
    ]) for e in payload['items']]}


def _analysis_request(*, question, evidence, context, plan, projection_requirement_ids):
    """Bind pre-scope test fixtures explicitly to the scope this test exercises."""
    records = tuple(replace(record, requested_requirement_ids=tuple(projection_requirement_ids))
                    for record in evidence)
    return EvidenceAnalysisRequest(question, records, context, plan,
        tuple((record.evidence_id, tuple(projection_requirement_ids)) for record in records))


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.budgets = []
    def with_max_tokens(self, budget):
        self.budgets.append(budget)
        return self
    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMCompletion(response if isinstance(response, str) else json.dumps(response),
                             prompt_tokens=100, completion_tokens=20)


def analyze(client, records=None, **kwargs):
    return LLMEvidenceProjector(client, **kwargs).analyze(request=_analysis_request(question='facts', evidence=tuple(records or [record(1), record(2), record(3)]), context=CONTEXT, plan=PLAN, projection_requirement_ids=('R1',)))


@pytest.mark.parametrize('order', [(1, 2, 3), (3, 1, 2)])
def test_normal_and_out_of_order_use_one_call_and_map_by_id(order):
    client = Client([batch(*order)])
    result = analyze(client)
    assert result.llm_calls == 1
    assert [a.evidence_id for a in result.analyses] == ['E1', 'E2', 'E3']
    assert [a.projection.facts[0].value for a in result.analyses] == [fact(n)['value'] for n in (1, 2, 3)]
    payload = json.loads(client.calls[0]['user_prompt'])
    assert len(payload['projection_requirements']) == 1
    assert all('projection_requirements' not in e for e in payload['evidence'])
    assert payload['evidence'][0]['kind'] == 'bibliography'
    assert client.budgets == [None]
    assert result.analyses[0].projection.support_anchors[0].unresolved_aspects
    metrics = projector_cost_metrics(result.llm_diagnostics)
    assert metrics['projector_evidence_count'] == 3
    assert metrics['projector_total_tokens'] == 120
    assert metrics['avg_evidence_per_projector_call'] == 3


@pytest.mark.parametrize('bad,code', [
    (batch(1, 3), 'missing_id'),
    (batch(1, 2, 2, 3), 'duplicate_id'),
    (batch(1, 2, 3, 99), 'unknown_id'),
])
def test_id_errors_are_structural_and_repaired(bad, code):
    client = Client([bad, batch(3, 1, 2)])
    result = analyze(client)
    assert result.llm_calls == 2
    assert all(a.status is EvidenceAnalysisStatus.QUALIFIED for a in result.analyses)
    feedback = json.loads(client.calls[1]['user_prompt'])['output_validation_feedback'][0]
    assert code in [v['code'] for v in feedback['violations']]
    assert all({'evidence_id', 'field_path', 'code', 'expected_constraint'} <= set(v) for v in feedback['violations'])
    assert result.llm_diagnostics[0].error_stage == 'domain_validation'
    assert result.llm_diagnostics[1].projector_repair


def test_partial_failure_preserves_valid_items_even_if_repair_is_unparseable():
    invalid = fact(2)
    invalid['value'] = None
    client = Client([{'items': [item(1), item(2, [invalid]), item(3)]}, '{bad json', {'projections': [fact(2)]}])
    result = analyze(client)
    assert result.llm_calls == 3
    assert all(a.status is EvidenceAnalysisStatus.QUALIFIED for a in result.analyses)
    assert [e['evidence_id'] for e in json.loads(client.calls[2]['user_prompt'])['evidence']] == ['E2']
    feedback = json.loads(client.calls[1]['user_prompt'])['output_validation_feedback'][0]['violations']
    assert any(v['evidence_id'] == 'source:' + record(2).fingerprint and v['field_path'].endswith('.value') for v in feedback)
    metrics = projector_cost_metrics(result.llm_diagnostics)
    assert metrics['projector_fallback_single_calls'] == 1
    assert metrics['projector_repair_calls'] == 1
    assert metrics['projector_total_tokens'] == 360


def test_complete_envelope_failure_falls_back_to_bounded_single_calls():
    client = Client(['{bad', '{bad'] + [{'projections': [fact(n)]} for n in (1, 2, 3)])
    result = analyze(client, max_attempts=5)
    assert result.llm_calls == 5
    assert projector_cost_metrics(result.llm_diagnostics)['projector_fallback_single_calls'] == 3
    assert all(a.projection for a in result.analyses)


def test_exhausted_single_does_not_discard_successful_peers():
    client = Client([batch(1, 3), batch(1, 3), '{bad', '{bad'])
    result = analyze(client)
    assert [a.status for a in result.analyses] == [EvidenceAnalysisStatus.QUALIFIED, EvidenceAnalysisStatus.FAILED, EvidenceAnalysisStatus.QUALIFIED]
    assert result.llm_calls == 4


def test_unknown_source_span_is_rejected_by_existing_integrity():
    client = Client([{'items': [item(1, [fact(1, excerpt=record(2).evidence.content)]), item(2)]}])
    result = analyze(client, [record(1), record(2)])
    assert result.analyses[0].projection.supported_requirement_ids == ()
    assert any('source_span_not_found' in r for r in result.analyses[0].projection_rejections)
    assert result.analyses[1].projection.supported_requirement_ids == ('R1',)
    assert result.llm_calls == 1  # Integrity rejection is not a structural repair.


def test_fact_multi_projection_and_zero_projection_items():
    second = dict(fact(1), value='Another fact from the same reference')
    response = {'items': [item(1, [fact(1), second]), item(2, [])]}
    transport_projection_model('fact').model_validate_json(json.dumps(wire(response)))
    result = analyze(Client([response]), [record(1), record(2)])
    assert len(result.analyses[0].projection.facts) == 2
    assert result.analyses[1].status is EvidenceAnalysisStatus.IRRELEVANT


def test_cache_filters_batch_and_survives_membership_and_id_changes(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / 'cache.sqlite3')
    analyze(Client([{'projections': [fact(1)]}]), [record(1)], cache=cache)
    client = Client([batch(3, 2)])
    result = analyze(client, cache=cache)
    assert result.cache_stats.hits == 1
    assert result.cache_stats.writes == 2
    assert [e['evidence_id'] for e in json.loads(client.calls[0]['user_prompt'])['evidence']] == ['source:' + record(n).fingerprint for n in (2, 3)]
    renamed = replace(record(2), evidence_id='E20', assessment=None)
    hit = analyze(Client([]), [renamed], cache=cache)
    assert hit.llm_calls == 0
    assert hit.analyses[0].projection.evidence_id == 'E20'


def test_irrelevant_cache_result_is_written_and_reused(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / 'cache.sqlite3')
    result = analyze(Client([{'items': [item(1, []), item(2, [])]}]), [record(1), record(2)], cache=cache)
    assert result.cache_stats.writes == 2
    assert analyze(Client([]), [record(2)], cache=cache).cache_stats.hits == 1


def test_dedup_precedes_cache_and_packing():
    records = [record(1), replace(record(1), evidence_id='E10', assessment=None), record(2)]
    client = Client([batch(1, 2)])
    result = analyze(client, records)
    assert len(json.loads(client.calls[0]['user_prompt'])['evidence']) == 2
    assert [a.evidence_id for a in result.analyses] == ['E1', 'E10', 'E2']
    assert result.analyses[1].projection.evidence_id == 'E10'


def test_packing_processes_short_tail_without_waiting():
    client = Client([batch(1, 2), batch(3, 4), {'projections': [fact(5)]}])
    result = analyze(client, [record(n) for n in range(1, 6)], batch_size=2)
    assert result.llm_calls == 3
    assert [len(json.loads(c['user_prompt'])['evidence']) for c in client.calls] == [2, 2, 1]
    assert projector_cost_metrics(result.llm_diagnostics)['avg_projector_batch_size'] == 2


def test_legacy_output_budget_is_ignored_and_diagnostic_survives_checkpoint():
    projector = LLMEvidenceProjector(Client([]), batch_base_tokens=100, batch_per_evidence_tokens=1000, batch_max_output_tokens=3000)
    assert [projector.output_budget(n) for n in (1, 2, 4, 32)] == [None, None, None, None]
    result = analyze(Client([batch(1, 2)]), [record(1), record(2)])
    serialized = dump_contract(result.llm_diagnostics, tuple[LLMCallRecord, ...])
    assert load_contract(serialized, tuple[LLMCallRecord, ...]) == result.llm_diagnostics


def test_inner_id_mismatch_is_not_mapped_by_position():
    response = {'items': [item(1, [fact(2)]), item(2)]}
    valid, errors = _validate_transport_items((record(1), record(2)), REQUEST, EvidenceProjectionType.FACT, json.dumps(response))
    assert [v.evidence_id for v in valid] == ['E2']
    assert errors[0]['code'] == 'projection_id_mismatch'


@pytest.mark.parametrize('kind', ['enumeration', 'comparison', 'mechanism'])
def test_other_projection_types_reuse_single_production_schema(kind):
    fields = {'enumeration': {'entities': []}, 'comparison': {'comparison_dimensions': ['dimension']},
              'mechanism': {'mechanism': 'method', 'steps': [], 'inputs': [], 'outputs': []}}[kind]
    parts = [dict(projection_type=kind, evidence_id=f'E{n}', status='qualified', supported_requirement_ids=[], support_candidates=[], **fields) for n in (1, 2)]
    response = {'items': [item(n, [parts[n - 1]]) for n in (1, 2)]}
    transport_projection_model(kind).model_validate_json(json.dumps(wire(response)))
    request = replace(REQUEST, context=replace(CONTEXT,
        retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION if kind == 'enumeration' else ResearchGoal.COMPARISON_SUPPORT if kind == 'comparison' else ResearchGoal.FIND_EVIDENCE,
        resolved_query='how' if kind == 'mechanism' else 'facts'))
    result = LLMEvidenceProjector(Client([response])).analyze(request=_analysis_request(question='facts', evidence=(record(1), record(2)), context=request.context, plan=PLAN, projection_requirement_ids=('R1',)))
    assert result.llm_calls == 1
    assert all(a.projection.projection_type.value == kind for a in result.analyses)


def test_provider_failure_does_not_erase_evidence_and_records_reason():
    client = Client([LLMRequestError('provider', error_stage='connection_error'), {'projections': [fact(1)]}, {'projections': [fact(2)]}])
    result = analyze(client, [record(1), record(2)])
    assert result.llm_calls == 3
    assert result.llm_diagnostics[0].projector_fallback_reason == 'connection_error'
    assert all(a.projection for a in result.analyses)


def test_transport_binds_outer_ids_without_redundant_payload_ids():
    response = wire(batch(3, 1, 2))
    transport_projection_model('fact').model_validate_json(json.dumps(response))
    result = analyze(Client([response]))
    assert result.llm_calls == 1
    assert [a.projection.evidence_id for a in result.analyses] == ['E1', 'E2', 'E3']
    schema = transport_projection_model('fact').model_json_schema()
    for definition in schema['$defs'].values():
        properties = definition.get('properties', {})
        if 'projection_type' in properties:
            assert 'evidence_id' not in properties


def test_transport_fact_status_fields_still_enforce_production_constraints():
    bad = dict(fact(1), status='irrelevant', fact_type='reference', value=None)
    with pytest.raises(ValidationError):
        transport_projection_model('fact').model_validate_json(json.dumps(wire({'items': [item(1, [bad])]})))


def test_batch_source_namespace_is_stable_and_disjoint_from_requirement_ids():
    sources = tuple(replace(record(n), evidence_id=f'REF{n}', assessment=None) for n in (1, 2, 3))
    first = BatchProjectorInput(REQUEST, EvidenceProjectionType.FACT, sources[:2]).to_payload()
    reordered = BatchProjectorInput(REQUEST, EvidenceProjectionType.FACT, tuple(reversed(sources))).to_payload()
    assert [e['evidence_id'] for e in first['evidence']] == ['source:' + record(n).fingerprint for n in (1, 2)]
    assert {e['evidence_id'] for e in first['evidence']} <= {e['evidence_id'] for e in reordered['evidence']}
    assert first['projection_requirement_ids'] == ['R1']


def test_two_requirement_facts_bind_to_same_reference_source_without_id_collision():
    source = replace(record(1), evidence_id='REF1', assessment=None, fingerprint='', evidence=replace(record(1).evidence,
        content='Unique reference fact 1. Unique reference fact 2.'))
    unrelated = replace(record(3), evidence_id='REF2', assessment=None)
    plan = replace(PLAN, requirements=PLAN.requirements + (replace(PLAN.requirements[0], requirement_id='R2'),))
    first = {k: v for k, v in fact(1).items() if k != 'evidence_id'}
    second = {k: v for k, v in fact(2).items() if k != 'evidence_id'}
    second['supported_requirement_ids'] = ['R2']
    second['support_candidates'][0]['requirement_id'] = 'R2'
    response = {'items': [dict(evidence_id='source:' + unrelated.fingerprint, projections=[]),
                          dict(evidence_id='source:' + source.fingerprint, projections=[first, second])]}
    result = LLMEvidenceProjector(Client([response])).analyze(request=_analysis_request(question='facts', evidence=(source, unrelated), context=CONTEXT, plan=plan, projection_requirement_ids=('R1', 'R2')))
    assert result.llm_calls == 1
    assert result.analyses[0].evidence_id == result.analyses[0].projection.evidence_id == 'REF1'
    assert result.analyses[0].projection.supported_requirement_ids == ('R1', 'R2')
    assert len(result.analyses[0].projection.facts) == 2
    assert result.analyses[1].evidence_id == 'REF2'
    assert result.analyses[1].status is EvidenceAnalysisStatus.IRRELEVANT


def test_single_schema_feedback_identifies_source_path_type_and_constraint():
    bad = fact(1); bad['support_candidates'][0]['unresolved_aspects'] = 'wrong type'
    c = Client([{'projections':[bad]}, {'projections':[fact(1)]}])
    result = analyze(c, [record(1)], max_attempts=5)
    assert result.llm_calls == 2
    feedback = json.loads(c.calls[1]['user_prompt'])['output_validation_feedback'][0]
    violation = next(v for v in feedback['violations'] if v['code']=='list_type')
    assert violation['evidence_id']=='E1'
    assert 'unresolved_aspects' in violation['field_path']
    assert violation['expected_constraint']
    assert result.llm_diagnostics[0].validation_diagnostics


def test_provider_length_splits_batch_and_never_retries_single_indefinitely():
    class LengthClient(Client):
        def complete_json_diagnostic(self, **kwargs):
            r = super().complete_json_diagnostic(**kwargs)
            return replace(r, finish_reason='length') if len(self.calls) in (1,2) else r
    c=LengthClient(['{bad','{bad',{'projections':[fact(2)]}])
    result=analyze(c,[record(1),record(2)],max_attempts=5)
    assert result.llm_calls==3
    assert result.analyses[0].status is EvidenceAnalysisStatus.FAILED
    assert result.analyses[1].status is EvidenceAnalysisStatus.QUALIFIED
    assert [x.error_stage for x in result.llm_diagnostics[:2]]==['truncation','truncation']
    assert not any(x.projector_repair for x in result.llm_diagnostics)
    assert all(x.requested_output_tokens is None for x in result.llm_diagnostics)


def test_mechanism_response_schema_encodes_existing_status_branches():
    from paperbase.agentic.schemas import MechanismProjectionBatchDraft
    base={'projection_type':'mechanism','evidence_id':'E1','status':'qualified','mechanism':None,
          'steps':[],'inputs':[],'outputs':[],'supported_requirement_ids':[],'support_candidates':[]}
    with pytest.raises(ValidationError):MechanismProjectionBatchDraft.model_validate({'projections':[base]})
    base['mechanism']='A mechanism.'
    assert MechanismProjectionBatchDraft.model_validate({'projections':[base]})
    base.update(status='irrelevant',mechanism=None)
    assert MechanismProjectionBatchDraft.model_validate({'projections':[base]})
    base['steps']=['A non-empty step.']
    with pytest.raises(ValidationError):MechanismProjectionBatchDraft.model_validate({'projections':[base]})
