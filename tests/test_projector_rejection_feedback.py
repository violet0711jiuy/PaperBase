"""Precise repair feedback without changing Projector support acceptance."""
from dataclasses import asdict
import json

import pytest

from paperbase.agentic.audit import _llm_call
from paperbase.agentic.evidence_projection import _projection_repair_feedback
from test_agentic_batch_projector import fact, item
from test_projector_structural_trace import run, no_real_provider


def rejected_fact(code):
    draft = fact(1)
    if code == 'missing_source_anchor':
        draft['support_candidates'] = []
    elif code == 'source_span_not_found':
        draft['support_candidates'][0]['source_span_ids'] = ['S999']
    elif code == 'fact_ownership_unknown':
        draft['ownership'] = 'unknown'
    elif code == 'not_proposed':
        draft['supported_requirement_ids'] = []
    return draft


@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('code', [
    'missing_source_anchor', 'source_span_not_found', 'fact_ownership_unknown', 'not_proposed',
])
def test_specific_integrity_reason_reaches_existing_repair_and_audit(code, n):
    bad = rejected_fact(code)
    responses = [{'projections': [bad]}, {'projections': [fact(1)]}] if n == 1 else [
        {'items': [item(1, [bad]), item(2)]}, {'items': [item(1), item(2)]},
    ]
    result, client, _ = run(responses, n)
    assert result.llm_calls == len(client.calls) == 2
    assert all(a.status.value == 'qualified' for a in result.analyses)
    diagnostic, = result.llm_diagnostics[0].validation_diagnostics
    assert diagnostic.error_code == code
    assert diagnostic.validator_name == 'ProjectorIntegrity'
    assert diagnostic.requirement_id == 'R1'
    assert diagnostic.error_stage == 'domain_validation'
    assert diagnostic.recovery_action == 'repair_domain'
    assert json.loads(diagnostic.actual_summary)['evidence_id'] == 'E1'
    feedback = json.loads(client.calls[1]['user_prompt'])['output_validation_feedback'][0]
    violation, = feedback['violations']
    assert violation['code'] == code and violation['requirement_id'] == 'R1'
    assert violation['field_path'].startswith('items.0.projections.' if n == 2 else 'projections.')
    assert violation['evidence_id'].startswith('source:') if n == 2 else violation['evidence_id'] == 'E1'
    assert 'PRIVATE_INVALID_ANCHOR' not in json.dumps(asdict(_llm_call(result.llm_diagnostics[0])))
    # Repair retains the original full source and the same provider schema.
    first, second = [json.loads(c['user_prompt']) for c in client.calls]
    second.pop('output_validation_feedback')
    second.pop('repair_context')
    assert first == second
    assert client.calls[0]['json_schema'] == client.calls[1]['json_schema']
    if n == 2:
        assert result.llm_diagnostics[0].projector_post_conversion_summary['converted_item_count'] == 1


@pytest.mark.parametrize('error_type,code', [
    (ValueError, 'projection_domain_validation'), (TypeError, 'projection_schema_validation'),
])
def test_unrecognized_error_is_not_a_status_branch_or_raw_exception(error_type, code):
    feedback = _projection_repair_feedback(error_type('PRIVATE_EXCEPTION_CONTENT'), evidence_id='E1')
    assert feedback['violations'][0]['code'] == code
    assert 'PRIVATE_EXCEPTION_CONTENT' not in json.dumps(feedback)


def test_unknown_candidate_id_is_not_echoed_in_integrity_feedback():
    bad = dict(fact(1), supported_requirement_ids=[])
    bad['support_candidates'][0]['requirement_id'] = 'PRIVATE_MODEL_REQUIREMENT_ID'
    result, client, _ = run([{'projections': [bad]}, {'projections': [fact(1)]}])
    diagnostic, = result.llm_diagnostics[0].validation_diagnostics
    assert diagnostic.error_code == 'not_proposed' and diagnostic.requirement_id is None
    feedback = json.loads(client.calls[1]['user_prompt'])['output_validation_feedback']
    assert 'PRIVATE_MODEL_REQUIREMENT_ID' not in json.dumps(feedback)
    assert 'PRIVATE_MODEL_REQUIREMENT_ID' not in json.dumps(asdict(_llm_call(result.llm_diagnostics[0])))


def test_empty_assertion_is_not_rejected_or_retried_or_invented():
    empty = dict(fact(1), supported_requirement_ids=[], support_candidates=[])
    result, client, _ = run([{'projections': [empty]}])
    assert result.llm_calls == len(client.calls) == 1
    assert result.analyses[0].status.value == 'qualified'
    assert result.analyses[0].projection.supported_requirement_ids == ()
    assert result.analyses[0].projection.facts[0].value == empty['value']
    assert not result.llm_diagnostics[0].validation_diagnostics


@pytest.mark.parametrize('n', [1, 2])
def test_unknown_span_feedback_requires_current_evidence_ids_in_single_and_batch(n):
    bad = fact(1, excerpt='Unique...fact 1.')
    responses = [{'projections': [bad]}, {'projections': [fact(1)]}] if n == 1 else [
        {'items': [item(1, [bad]), item(2)]}, {'items': [item(1), item(2)]},
    ]
    result, client, _ = run(responses, n)
    assert result.llm_calls == len(client.calls) == 2
    assert all(a.status.value == 'qualified' for a in result.analyses)
    feedback = json.loads(client.calls[1]['user_prompt'])['output_validation_feedback'][0]
    violation, = feedback['violations']
    assert violation['code'] == 'source_span_not_found'
    assert 'only source span IDs listed on this Evidence' in violation['expected_constraint']
    for call in client.calls:
        assert '最小有界原文区间' in call['system_prompt']
        assert '不要复制 source text' in call['system_prompt']
    first, second = [json.loads(c['user_prompt']) for c in client.calls]
    second.pop('output_validation_feedback')
    second.pop('repair_context')
    assert first == second
    assert client.calls[0]['json_schema'] == client.calls[1]['json_schema']


def test_literal_source_ellipsis_is_still_accepted_without_repair():
    from dataclasses import replace
    from paperbase.agentic.contracts import EvidenceProjectionType
    from paperbase.agentic.evidence_analysis import EvidenceAnalysisRequest
    from paperbase.agentic.evidence_projection import LLMEvidenceProjector
    from paperbase.agentic.grounding import verified_support_ids
    from test_agentic_batch_projector import record, PLAN, CONTEXT, Client
    text = 'Unique...reference fact 1.'
    current = record(1)
    current = replace(current, evidence=replace(current.evidence, content=text), fingerprint=None,
                      requested_requirement_ids=('R1',))
    plan = replace(PLAN, requirements=(replace(PLAN.requirements[0], projection_type=EvidenceProjectionType.FACT),))
    req = EvidenceAnalysisRequest('facts', (current,), CONTEXT, plan, (('E1', ('R1',)),))
    client = Client([{'projections': [fact(1)]}])
    result = LLMEvidenceProjector(client).analyze(request=req)
    current = replace(current, assessment=result.analyses[0])
    assert verified_support_ids(current, plan.requirements) == ('R1',)
    assert result.llm_calls == len(client.calls) == 1
    assert not result.llm_diagnostics[0].projector_repair


def test_source_grounded_general_background_fact_can_establish_declared_support():
    from dataclasses import replace
    from paperbase.agentic.grounding import verified_support_ids
    background = dict(fact(1),ownership='general_background')
    result, client, request = run([{'projections':[background]}])
    current = replace(request.evidence[0],assessment=result.analyses[0])
    assert verified_support_ids(current,request.plan.requirements) == ('R1',)
    assert len(client.calls) == 1


@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('length', [484, 1600])
def test_long_support_scope_is_preserved_without_repair(n, length):
    from dataclasses import replace
    from paperbase.agentic.grounding import verified_support_ids
    draft = fact(1)
    scope = 's' * length
    draft['support_candidates'][0]['support_scope'] = scope
    response = {'projections': [draft]} if n == 1 else {'items': [item(1, [draft]), item(2)]}
    result, client, request = run([response], n)
    current = replace(request.evidence[0], assessment=result.analyses[0])
    assert verified_support_ids(current, request.plan.requirements) == ('R1',)
    assert current.projection.support_anchors[0].support_scope == scope
    assert result.llm_calls == len(client.calls) == 1
    assert not any(d.projector_repair for d in result.llm_diagnostics)
    from paperbase.llm.client import _strict_response_schema
    wire = _strict_response_schema(client.calls[0]['json_schema'])
    definitions = [value for value in wire['$defs'].values() if 'support_scope' in value.get('properties', {})]
    assert definitions
    assert all('maxLength' not in value['properties']['support_scope'] for value in definitions)
    assert all(value['properties']['support_scope']['minLength'] == 1 for value in definitions)


@pytest.mark.parametrize('value', ['', '   ', None, 123])
def test_support_scope_still_requires_nonempty_strict_text(value):
    from pydantic import ValidationError
    from paperbase.agentic.schemas import RequirementSupportCandidateDraft
    candidate = dict(fact(1)['support_candidates'][0], support_scope=value)
    with pytest.raises(ValidationError):
        RequirementSupportCandidateDraft.model_validate(candidate)


def test_long_support_scope_does_not_bypass_source_integrity():
    bad = rejected_fact('source_span_not_found')
    bad['support_candidates'][0]['support_scope'] = 's' * 1600
    result, client, _ = run([{'projections': [bad]}] * 2)
    assert result.analyses[0].status.value == 'failed'
    assert result.llm_calls == len(client.calls) == 2
    assert all(d.validation_diagnostics[0].error_code == 'source_span_not_found'
               for d in result.llm_diagnostics)


def test_schema_failure_stays_distinct_from_provenance_failure():
    malformed = fact(1)
    malformed.pop('value')
    result, client, _ = run([{'projections': [malformed]}, {'projections': [fact(1)]}])
    first = result.llm_diagnostics[0]
    assert first.error_stage == 'schema_validation'
    assert all(d.validator_name == 'ProjectorStructure' for d in first.validation_diagnostics)
    assert any(d.error_code == 'missing' and d.field_path.endswith('.value') for d in first.validation_diagnostics)
    assert result.llm_calls == len(client.calls) == 2


@pytest.mark.parametrize('code', ['source_span_not_found', 'missing_source_anchor'])
def test_failed_repair_keeps_original_two_attempt_limit(code):
    bad = {'projections': [rejected_fact(code)]}
    result, client, _ = run([bad, bad])
    assert result.llm_calls == len(client.calls) == 2
    assert result.analyses[0].status.value == 'failed'
    assert result.llm_diagnostics[-1].validation_diagnostics[0].error_code == code
    assert result.llm_diagnostics[-1].validation_diagnostics[0].recovery_action == 'fallback'
