"""Saved experiment_001 + synthetic drafts: zero provider, no production fixes."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('answer_coverage_offline_replay',
    ROOT / 'output/answer_coverage_rca_20261006/replay.py')
rca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rca)

# These tests reproduce the archived pre-redesign contract, not a production
# compatibility parser. New Agentic provider fixtures live in semantic tests.
import sys
before = ROOT / 'output/answer_contract_simplification_20261006/before_source'
legacy_spec = importlib.util.spec_from_file_location(
    'paperbase.generation._historical_answer_rca', before / 'paperbase/generation/answer_generator.py')
legacy = importlib.util.module_from_spec(legacy_spec)
sys.modules[legacy_spec.name] = legacy
legacy_spec.loader.exec_module(legacy)
prompt_spec = importlib.util.spec_from_file_location(
    '_historical_answer_prompt_rca', before / 'paperbase/prompts/answer_generation.py')
legacy_prompt = importlib.util.module_from_spec(prompt_spec)
prompt_spec.loader.exec_module(legacy_prompt)
legacy.build_answer_generation_user_prompt = legacy_prompt.build_answer_generation_user_prompt
legacy.ANSWER_GENERATION_SYSTEM_PROMPT = legacy_prompt.ANSWER_GENERATION_SYSTEM_PROMPT
legacy.AGENTIC_REQUIREMENT_COVERAGE_INSTRUCTION = legacy_prompt.AGENTIC_REQUIREMENT_COVERAGE_INSTRUCTION
for name in ('AnswerGenerationDraft', 'AnswerDomainValidationError', 'GroundedAnswerGenerator',
             '_compose_answer', '_normalize_and_validate_answer', '_answer_validation_diagnostics',
             '_answer_response_schema', '_citation_repair_prompt', '_validate_citation_repair'):
    setattr(rca, name, getattr(legacy, name))


@pytest.fixture(autouse=True)
def forbid_real_provider_and_agent(monkeypatch):
    import paperbase.llm.client as client
    import paperbase.generation.answer_generator as answer
    import paperbase.agentic.factory as factory
    def forbidden(*args, **kwargs):
        raise AssertionError('Answer RCA must not construct/call a provider or Agent')
    monkeypatch.setattr(client, 'create_chat_completion_client', forbidden)
    monkeypatch.setattr(answer, 'create_chat_completion_client', forbidden)
    monkeypatch.setattr(factory, 'create_agentic_research_service', forbidden)
    monkeypatch.setattr(client.OpenAICompatibleChatClient, 'complete_json_diagnostic', forbidden)
    monkeypatch.setattr(client.ZhipuAIChatClient, 'complete_json_diagnostic', forbidden)


def test_saved_authority_matches_formal_replay():
    plan, states, selected, view, requirements, mapping, units = rca.sample()
    assert [r.requirement_id for r in plan.requirements] == ['R1', 'R2']
    assert all(r.required for r in plan.requirements)
    assert [r.evidence_id for r in selected] == ['E3', 'E1', 'E2']
    assert mapping == {'R1': ('E1', 'E2'), 'R2': ('E3',)}
    assert view.required_requirement_ids == view.supported_requirement_ids == ('R1', 'R2')
    assert {r.requirement_id: r.evidence_ids for r in states} == mapping
    assert [u.evidence_id for u in units] == ['E3', 'E1', 'E2']
    assert [u.supported_requirement_ids for u in units] == [('R2',), ('R1',), ('R1',)]


def test_case_1_canonical_valid_answer_passes_generator_and_bridge():
    answer, client = rca.fake_generate([rca.canonical_draft()])
    assert len(client.calls) == 1 and answer.status == 'success'
    assert answer.overall_status == 'complete'
    assert not answer.insufficient_evidence and not answer.partial_answer
    assert {r.requirement_id for r in answer.requirement_coverage} == {'R1', 'R2'}
    assert answer.citations == ('E1', 'E3')


@pytest.mark.parametrize('mode', ['missing_R2', 'all_missing', 'duplicate_R1', 'unknown_R99', 'omitted_field'])
def test_case_2_exact_row_domain_diagnostic_matches_saved_both_attempts(mode):
    draft = rca.canonical_draft()
    if mode == 'missing_R2':
        draft['requirement_coverage'] = draft['requirement_coverage'][:1]
    elif mode == 'all_missing':
        draft['requirement_coverage'] = []
    elif mode == 'duplicate_R1':
        draft['requirement_coverage'] = [draft['requirement_coverage'][0]] * 2
    elif mode == 'unknown_R99':
        draft['requirement_coverage'][1]['requirement_id'] = 'R99'
    else:
        del draft['requirement_coverage']
    with pytest.raises(rca.AnswerDomainValidationError) as caught:
        rca.normalize(draft)
    error = caught.value
    assert error.code == 'ANSWER_REQUIREMENT_COVERAGE_INVALID'
    assert error.field_path == 'requirement_coverage'
    assert error.actual_summary == 'missing_duplicate_or_unknown_requirement_id'
    historical = rca.read(rca.LIVE / 'cases/experiment_001.audit.json')
    observed = [e for e in historical['validation_errors'] if e['node'] == 'generate_answer']
    assert [e['attempt'] for e in observed] == [1, 2]
    assert all(e['code'] == error.code and e['expected_constraint'] == error.expected_constraint
               and e['actual_summary'] == error.actual_summary for e in observed)


def test_case_3_optional_requirements_are_not_sent_by_actual_bridge():
    plan, _, _, _, _, _, _ = rca.sample()
    optional = rca.ResearchRequirement('R3', 'COUNTERFACTUAL optional requirement',
        evidence_targets=plan.requirements[0].evidence_targets, required=False, projection_type='fact')
    plan = replace(plan, requirements=plan.requirements + (optional,))
    answer, client = rca.fake_generate([rca.canonical_draft()], plan=plan)
    assert answer.status == 'success'
    body = client.calls[0]['user_prompt']
    requirements = json.loads(body.split('<research_requirements>\n', 1)[1].split('\n</research_requirements>', 1)[0])
    assert [r['requirement_id'] for r in requirements] == ['R1', 'R2']
    assert 'R3' not in client.calls[0]['json_schema']['properties']['requirement_coverage']['items']['anyOf'][0]['properties']['requirement_id']['const']


def test_case_4_unsupported_requirement_cannot_claim_covered():
    _, _, _, _, _, mapping, _ = rca.sample()
    with pytest.raises(rca.AnswerDomainValidationError) as caught:
        rca.normalize(rca.canonical_draft(), mapping=dict(mapping, R2=()))
    assert caught.value.code == 'ANSWER_CITATION_INVALID'
    assert caught.value.actual_summary == 'invalid_support_evidence_id'


@pytest.mark.parametrize('wrong_id', ['E3', 'E999', 'R1', 'REF1', 'paper_5b6a1007fa7514bf_chunk_0067'])
def test_case_5_correct_requirement_with_wrong_evidence_id_fails(wrong_id):
    draft = rca.canonical_draft()
    draft['requirement_coverage'][0]['evidence_ids'] = [wrong_id]
    with pytest.raises(rca.AnswerDomainValidationError) as caught:
        rca.normalize(draft)
    assert caught.value.code == 'ANSWER_CITATION_INVALID'
    assert caught.value.field_path == 'requirement_coverage.evidence_ids'


def test_case_6_shared_evidence_allowed_if_explicit_support_map_allows_it():
    # Actual sample has no intersection. This positive counterfactual tests
    # cardinality only; it does not assert E1 truly supports historical R2.
    _, _, _, _, _, mapping, _ = rca.sample()
    assert not set(mapping['R1']).intersection(mapping['R2'])
    draft = rca.canonical_draft()
    draft['requirement_coverage'][1]['evidence_ids'] = ['E1']
    result = rca.normalize(draft, mapping=dict(mapping, R2=('E1',)))
    assert all(r.evidence_ids == ['E1'] for r in result.requirement_coverage)


def test_case_7_partial_when_one_required_requirement_has_no_verified_support():
    plan, states, selected, _, _, _, _ = rca.sample()
    missing = tuple(replace(r, assessment=rca.EvidenceAnalysis(r.evidence_id, 'irrelevant'))
                    if r.evidence_id == 'E3' else r for r in selected)
    view = rca.build_completion_view(plan=plan, states=states, selected=missing)
    assert view.support_for('R2') == ()
    draft = rca.canonical_draft()
    draft['requirement_coverage'][1].update(status='unsupported', evidence_ids=[])
    draft.update(direct_answer='A source-local fact. [E1]', evidence_explanation='A source-local fact. [E1]', citations=['E1'])
    answer, _ = rca.fake_generate([draft], selected=missing)
    assert answer.overall_status == 'partial'
    assert answer.partial_answer and not answer.insufficient_evidence


def test_all_unsupported_is_insufficient_and_flags_do_not_override_coverage():
    draft = rca.canonical_draft()
    draft.update(insufficient_evidence=True, partial_answer=True)
    result = rca.normalize(draft)
    assert not result.insufficient_evidence and not result.partial_answer
    for row in draft['requirement_coverage']:
        row.update(status='unsupported', evidence_ids=[])
    draft.update(direct_answer='Insufficient evidence.', evidence_explanation='Insufficient evidence.', citations=[])
    result = rca.normalize(draft, mapping={'R1': (), 'R2': ()})
    assert result.insufficient_evidence and not result.partial_answer


def test_research_state_satisfied_is_not_an_answer_completeness_gate():
    plan, states, selected, view, _, _, _ = rca.sample()
    different = tuple(replace(s, status='uncertain', evidence_ids=()) for s in states)
    assert rca.build_completion_view(plan=plan, states=different, selected=selected) == view
    partial = rca.canonical_draft()
    partial['requirement_coverage'][0]['status'] = 'partial'
    answer, _ = rca.fake_generate([partial])
    assert all(s.status.value == 'satisfied' for s in states)
    assert answer.overall_status == 'partial'


def test_repair_can_fill_missing_row_without_changing_fact_text():
    valid = rca.canonical_draft()
    missing = dict(valid, requirement_coverage=valid['requirement_coverage'][:1])
    answer, client = rca.fake_generate([missing, valid])
    assert answer.status == 'success' and len(client.calls) == 2
    assert client.calls[0]['json_schema'] == client.calls[1]['json_schema']
    payload = json.loads(client.calls[1]['user_prompt'].split('<answer_citation_repair>\n', 1)[1].split('\n</answer_citation_repair>', 1)[0])
    assert payload['validation_errors'][0]['code'] == 'ANSWER_REQUIREMENT_COVERAGE_INVALID'
    assert payload['allowed_coverage_evidence_ids'] == {'R1': ['E1', 'E2'], 'R2': ['E3']}
    assert payload['allowed_evidence_ids'] == ['E1', 'E2', 'E3']
    assert payload['original_answer_json']['requirement_coverage'] == missing['requirement_coverage']
    assert len(answer.requirement_coverage) == 2


def test_repeated_same_row_error_falls_back_without_inventing_coverage():
    valid = rca.canonical_draft()
    missing = dict(valid, requirement_coverage=valid['requirement_coverage'][:1])
    answer, client = rca.fake_generate([missing, missing])
    assert answer.status == 'fallback' and len(client.calls) == 2
    assert answer.requirement_coverage == () and answer.citations == ()
    assert [d.recovery_action for d in answer.llm_diagnostics] == ['repair_citation', 'fallback']
    assert [d.validation_diagnostics[0].error_code for d in answer.llm_diagnostics] == ['ANSWER_REQUIREMENT_COVERAGE_INVALID'] * 2


def test_repaired_row_domain_then_wrong_support_has_distinct_diagnostic():
    valid = rca.canonical_draft()
    missing = dict(valid, requirement_coverage=valid['requirement_coverage'][:1])
    wrong = deepcopy(valid)
    wrong['requirement_coverage'][1]['evidence_ids'] = ['E2']
    answer, _ = rca.fake_generate([missing, wrong])
    assert [d.validation_diagnostics[0].error_code for d in answer.llm_diagnostics] == ['ANSWER_REQUIREMENT_COVERAGE_INVALID', 'ANSWER_CITATION_INVALID']


def test_citation_coverage_desynchronization_cannot_produce_historical_rule():
    draft = rca.canonical_draft()
    draft.update(direct_answer='Source fact. [E1]', evidence_explanation='Source fact. [E1]', citations=['E1'])
    with pytest.raises(rca.AnswerDomainValidationError) as caught:
        rca.normalize(draft)
    assert caught.value.code == 'ANSWER_CITATION_INVALID'
    assert caught.value.actual_summary == 'covered_requirement_without_cited_support'


def test_compose_preserves_coverage_and_does_not_renumber_ids():
    parsed = rca.AnswerGenerationDraft.model_validate(rca.canonical_draft())
    composed = rca._compose_answer(parsed)
    assert composed.requirement_coverage == parsed.requirement_coverage
    assert composed.citations == parsed.citations


def test_schema_can_admit_duplicate_ids_but_formal_validator_rejects_them():
    from jsonschema import Draft202012Validator
    _, _, _, _, reqs, mapping, units = rca.sample()
    schema = rca._strict_response_schema(rca._answer_response_schema(reqs, mapping, [u.evidence_id for u in units]))
    valid = rca.canonical_draft()
    Draft202012Validator(schema).validate(valid)
    duplicate = deepcopy(valid)
    duplicate['requirement_coverage'] = [duplicate['requirement_coverage'][0]] * 2
    Draft202012Validator(schema).validate(duplicate)
    with pytest.raises(rca.AnswerDomainValidationError, match='恰好'):
        rca.normalize(duplicate)


def test_history_only_preserves_fallback_not_rejected_drafts():
    saved = rca.read(rca.LIVE / 'cases/experiment_001.result.json')
    audit = rca.read(rca.LIVE / 'cases/experiment_001.audit.json')
    assert 'state' not in saved and 'draft' not in saved
    assert saved['answer']['status'] == 'fallback'
    assert saved['answer']['requirement_coverage'] == audit['answer_requirement_coverage'] == []
    assert [d['recovery_action'] for d in audit['validation_errors'] if d['node'] == 'generate_answer'] == ['repair_citation', 'fallback']
