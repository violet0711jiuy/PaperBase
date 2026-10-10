"""New Answer provider contract, compiler purity, grounding and final DTO."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from paperbase.generation.answer_coverage import (
    compile_answer_requirement_coverage as compile_coverage, derive_overall_answer_status,
    AnswerDomainValidationError,
)
from paperbase.generation.answer_generator import (
    AnswerSemanticDraft, GroundedAnswerGenerator, _answer_response_schema, _compile_semantic_answer,
    _validate_citation_repair,
)
from paperbase.generation.section_expander import EvidenceUnit
from paperbase.agentic.contracts import ResearchAnswer
from paperbase.agentic.checkpoint_serde import load_contract, dump_contract
from paperbase.config import AnswerGenerationSettings
from paperbase.llm.client import LLMCompletion


def test_classic_heading_change_cannot_contaminate_agentic_system_prompt():
    from paperbase.prompts import answer_generation as prompts
    source = Path(prompts.__file__).read_text(encoding='utf-8')
    changed = source.replace('# 回答完整性', '# Classic completion heading')
    assert changed != source
    namespace = {}
    exec(compile(changed, str(prompts.__file__), 'exec'), namespace)
    assert namespace['ANSWER_SEMANTIC_SYSTEM_PROMPT'] == prompts.ANSWER_SEMANTIC_SYSTEM_PROMPT
    assert namespace['ANSWER_GENERATION_SYSTEM_PROMPT'] != prompts.ANSWER_GENERATION_SYSTEM_PROMPT


class Queue:
    def __init__(self, responses):
        self.responses = deepcopy(list(responses))
        self.calls = []

    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        return LLMCompletion(json.dumps(self.responses.pop(0)), finish_reason='stop')


def draft(statuses=None, direct='Fact. [E1]', explanation='Fact. [E2]', reading=None, note=None):
    return dict(direct_answer=direct, evidence_explanation=explanation, reading_interpretation=reading,
                requirement_status={'R1': 'complete', 'R2': 'partial'} if statuses is None else statuses,
                coverage_note=note)


def units(ids=('E1', 'E2', 'REF1')):
    return tuple(EvidenceUnit(eid, 'bibliography' if eid.startswith('REF') else 'content',
        'P', 'Paper', 's', 1, 1, ('c',), ('c',), 'Source fact.', 3) for eid in ids)


def generate(responses, mapping=None):
    mapping = {'R1': ('E1',), 'R2': ('E2',)} if mapping is None else mapping
    client = Queue(responses)
    out = GroundedAnswerGenerator(settings=AnswerGenerationSettings(), client=client).generate(
        query='Facts?', evidence=units(),
        research_requirements=tuple((rid, 'A fact') for rid in mapping),
        support_by_requirement=mapping,
    )
    return out, client


@pytest.fixture(autouse=True)
def no_real_provider(monkeypatch):
    import paperbase.llm.client as llm
    import paperbase.agentic.factory as factory
    def forbidden(*args, **kwargs):
        raise AssertionError('Semantic contract tests cannot call real provider')
    monkeypatch.setattr(llm, 'create_chat_completion_client', forbidden)
    monkeypatch.setattr(factory, 'create_agentic_research_service', forbidden)


@pytest.mark.parametrize('statuses,mapping,cites,expected,overall', [
    ({'R1':'complete','R2':'partial'}, {'R1':('E1',),'R2':('E2',)}, ('E1','E2'),
     [('R1','complete',('E1',)),('R2','partial',('E2',))], 'partial'),
    ({'R1':'complete'}, {'R1':('E1',),'R2':()}, ('E1',),
     [('R1','complete',('E1',)),('R2','unsupported',())], 'partial'),
    ({}, {'R1':(),'R2':()}, (),
     [('R1','unsupported',()),('R2','unsupported',())], 'insufficient'),
    ({'R1':'complete','R2':'partial'}, {'R1':('E1',),'R2':('E1',)}, ('E1',),
     [('R1','complete',('E1',)),('R2','partial',('E1',))], 'partial'),
    ({'R1':'complete'}, {'R1':('E1','E2')}, ('E2',),
     [('R1','complete',('E2',))], 'complete'),
    ({'R1':'complete','R2':'complete'}, {'R1':('E1',),'R2':('E2',)}, ('E1','E2'),
     [('R1','complete',('E1',)),('R2','complete',('E2',))], 'complete'),
    ({'R1':'partial','R2':'complete'}, {'R1':('E1',),'R2':('E2',)}, ('E1','E2'),
     [('R1','partial',('E1',)),('R2','complete',('E2',))], 'partial'),
])
def test_compiler_a_b_c_d_e_k_l_m_n(statuses, mapping, cites, expected, overall):
    before = deepcopy((statuses, mapping, cites))
    coverage = compile_coverage(required_requirement_ids=tuple(mapping),
        allowed_support_by_requirement=mapping, semantic_status_by_requirement=statuses, inline_citations=cites)
    assert [(r.requirement_id,r.status,r.evidence_ids) for r in coverage] == expected
    assert derive_overall_answer_status(coverage, cites) == overall
    assert (statuses, mapping, cites) == before
    assert coverage == compile_coverage(required_requirement_ids=tuple(mapping),
        allowed_support_by_requirement=mapping, semantic_status_by_requirement=statuses, inline_citations=cites)


@pytest.mark.parametrize('statuses', [
    {'R1':'complete','R99':'partial'}, {'R1':'complete'}, {'R1':'complete','R2':'partial','R3':'complete'},
    {'R1':'invalid','R2':'partial'},
])
def test_h_i_j_schema_and_defensive_compiler_exact_supported_keys(statuses):
    schema = _answer_response_schema((('R1','a'),('R2','b'),('R3','missing')),
                                    {'R1':('E1',),'R2':('E2',),'R3':()})
    assert list(Draft202012Validator(schema).iter_errors(draft(statuses)))
    with pytest.raises(AnswerDomainValidationError) as error:
        compile_coverage(required_requirement_ids=('R1','R2','R3'),
            allowed_support_by_requirement={'R1':('E1',),'R2':('E2',),'R3':()},
            semantic_status_by_requirement=statuses, inline_citations=('E1','E2'))
    assert error.value.code == 'ANSWER_REQUIREMENT_STATUS_INVALID'


def test_f_no_auto_citation_or_coverage_fill():
    with pytest.raises(AnswerDomainValidationError) as error:
        compile_coverage(required_requirement_ids=('R1',), allowed_support_by_requirement={'R1':('E1','E2')},
            semantic_status_by_requirement={'R1':'complete'}, inline_citations=())
    assert error.value.code == 'ANSWER_CITATION_INVALID'
    out, q = generate([draft(direct='Uncited fact.', explanation='Uncited fact.')]*2)
    assert out.status == 'fallback' and len(q.calls) == 2
    assert out.citations == out.requirement_coverage == ()


@pytest.mark.parametrize('citation', ['E999','R1','REQ:R1'])
def test_g_unknown_citation_and_requirement_namespace_still_rejected(citation):
    value = draft(direct='Fact. ['+citation+']')
    out, q = generate([value]*2)
    assert out.status == 'fallback'
    assert out.llm_diagnostics[0].validation_diagnostics[0].error_code == 'ANSWER_UNKNOWN_CITATION'
    assert len(q.calls) == 2


def test_c_all_unsupported_uses_zero_provider_calls_and_no_model_upgrade():
    out, q = generate([], mapping={'R1':(), 'R2':()})
    assert q.calls == []
    assert out.insufficient_evidence and not out.partial_answer
    assert [r.status for r in out.requirement_coverage] == ['unsupported','unsupported']


def test_provider_removes_bookkeeping_final_dto_retains_it():
    out, q = generate([draft()])
    assert out.status == 'success' and out.partial_answer
    schema = q.calls[0]['json_schema']
    assert set(schema['properties']) == {'direct_answer','evidence_explanation','reading_interpretation','requirement_status','coverage_note'}
    assert q.calls[0]['schema_name'] == 'paperbase_grounded_answer_semantic_v3'
    assert schema['properties']['requirement_status']['required'] == ['R1','R2']
    assert schema['properties']['requirement_status']['additionalProperties'] is False
    for extra in ('citations','requirement_coverage','insufficient_evidence','partial_answer','overall_status'):
        with pytest.raises(ValidationError):
            AnswerSemanticDraft.model_validate(dict(draft(), **{extra: []}))
    answer = ResearchAnswer(out.status,out.answer,out.citations,out.insufficient_evidence,out.partial_answer,
        out.coverage_note,out.requirement_coverage,
        derive_overall_answer_status(out.requirement_coverage,out.citations))
    payload = dump_contract(answer, ResearchAnswer)
    assert {'citations','requirement_coverage','insufficient_evidence','partial_answer','overall_status'} <= set(payload)
    assert load_contract(payload, ResearchAnswer) == answer


def test_parser_uses_three_body_fields_in_first_occurrence_order_not_note():
    value = draft({'R1':'complete'}, direct='First. [E2][E1]', explanation='Second. [E2]',
                  reading='Meaning. [REF1]', note='Optional note [E1]')
    result = _compile_semantic_answer(AnswerSemanticDraft.model_validate(value),
        allowed_evidence_ids={'E1','E2','REF1'}, research_requirements=(('R1','fact'),),
        support_by_requirement={'R1':('E1','E2','REF1')})
    assert result.citations == ['E2','E1','REF1']
    assert result.requirement_coverage[0].evidence_ids == ('E2','E1','REF1')
    only_note = draft({'R1':'complete'}, direct='Fact.', explanation='Fact.',note='Missing detail [E1]')
    with pytest.raises(AnswerDomainValidationError):
        _compile_semantic_answer(AnswerSemanticDraft.model_validate(only_note),
            allowed_evidence_ids={'E1'}, research_requirements=(('R1','fact'),),
            support_by_requirement={'R1':('E1',)})


def test_missing_status_key_repair_keeps_body_and_schema():
    bad = draft({'R1':'complete'})
    out, q = generate([bad, draft()])
    assert out.status == 'success' and len(q.calls) == 2
    assert q.calls[0]['json_schema'] == q.calls[1]['json_schema']
    assert out.llm_diagnostics[0].validation_diagnostics[0].field_path == 'requirement_status'


def test_schema_field_failure_repair_preserves_fact_text():
    bad = draft()
    del bad['coverage_note']
    out, q = generate([bad,draft()])
    assert out.status == 'success' and len(q.calls) == 2
    assert out.llm_diagnostics[0].error_stage == 'schema_validation'


@pytest.mark.parametrize('change', ['facts','upgrade'])
def test_repair_cannot_add_or_rephrase_facts_or_upgrade_partial(change):
    before = draft({'R1':'partial','R2':'partial'}, direct='Fact. [E999]')
    after = draft({'R1':'partial','R2':'partial'})
    if change == 'facts':
        after['direct_answer'] = 'New fact. [E1]'
    else:
        after['requirement_status']['R1'] = 'complete'
    out, q = generate([before,after])
    assert out.status == 'fallback' and len(q.calls) == 2
    assert out.llm_diagnostics[1].validation_diagnostics[0].error_code == (
        'ANSWER_REPAIR_CONTENT_CHANGED' if change == 'facts' else 'ANSWER_REPAIR_COVERAGE_UPGRADED')


def test_repair_allows_safe_downgrade_and_correcting_inline_citation():
    before = draft(direct='Fact. [E999]')
    after = draft({'R1':'partial','R2':'partial'})
    out, q = generate([before,after])
    assert out.status == 'success' and out.partial_answer and len(q.calls) == 2


def test_historical_final_answers_still_parse_without_provider_parser():
    root = Path(__file__).resolve().parents[1]
    for path in (root/'eval/results/e1_live_smoke_20261003/cases').glob('*.result.json'):
        raw = json.loads(path.read_text(encoding='utf-8'))['answer']
        answer = load_contract(raw, ResearchAnswer)
        assert load_contract(dump_contract(answer, ResearchAnswer), ResearchAnswer) == answer


def test_equivalent_valid_draft_preserves_final_dto_against_archived_generator():
    # Reuse the test-only pre-migration RCA snapshot, never a production parser.
    from test_answer_coverage_rca import legacy
    statuses = {'R1':'complete','R2':'partial'}
    value = draft(statuses)
    old = legacy.AnswerGenerationDraft.model_validate(dict(
        direct_answer=value['direct_answer'], evidence_explanation=value['evidence_explanation'],
        reading_interpretation=None, citations=['E1','E2'], insufficient_evidence=False,
        partial_answer=True, coverage_note=None,
        requirement_coverage=[dict(requirement_id=rid,status=status,evidence_ids=[eid])
            for rid,status,eid in [('R1','complete','E1'),('R2','partial','E2')]],
    ))
    old_result = legacy._normalize_and_validate_answer(legacy._compose_answer(old),
        allowed_evidence_ids={'E1','E2'}, research_requirements=(('R1','first'),('R2','second')),
        support_by_requirement={'R1':('E1',),'R2':('E2',)})
    old_outcome = legacy._to_outcome(old_result,status='success')
    new_outcome, _ = generate([value])
    assert asdict(old_outcome) | {'llm_diagnostics':()} == asdict(new_outcome) | {'llm_diagnostics':()}


def test_final_native_sdk_request_has_only_semantic_fields_and_strict_keys():
    from types import SimpleNamespace
    from paperbase.llm.client import OpenAICompatibleChatClient, LLMRuntimeSettings
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(draft()), reasoning_content=None), finish_reason='stop')], usage=None)
    # Inject SDK transport without constructing an SDK client/network connection.
    client = object.__new__(OpenAICompatibleChatClient)
    client._settings = LLMRuntimeSettings('test-only', 'https://example.invalid', 'test-model', 30, .2, 1600,
                                         structured_output_mode='json_object')
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._extra_body = {}
    result = GroundedAnswerGenerator(settings=AnswerGenerationSettings(),client=client).generate(
        query='Facts?',evidence=units(),research_requirements=(('R1','first'),('R2','second')),
        support_by_requirement={'R1':('E1',),'R2':('E2',)})
    assert result.status=='success' and len(requests)==1
    request=requests[0]
    assert request['temperature']==0
    fmt=request['response_format']
    assert fmt['type']=='json_schema' and fmt['json_schema']['strict'] is True
    schema=fmt['json_schema']['schema']
    assert set(schema['required'])==set(AnswerSemanticDraft.model_fields)
    assert schema['properties']['requirement_status']['required']==['R1','R2']
    assert schema['properties']['requirement_status']['additionalProperties'] is False
    assert not {'citations','requirement_coverage','insufficient_evidence','partial_answer'}.intersection(schema['properties'])


def test_supported_mapping_can_be_semantically_declined_without_mutating_upstream():
    mapping = {'R1': ('E1',), 'R2': ('E2',)}
    before = deepcopy(mapping)
    value = draft({'R1':'unsupported','R2':'unsupported'},
                  direct='当前证据不足以确定目标测量结果。',
                  explanation='当前片段只有相关背景，无法回答核心问题。', note='缺少目标测量。')
    schema = _answer_response_schema((('R1','a'),('R2','b')),mapping)
    assert not list(Draft202012Validator(schema).iter_errors(value))
    out, q = generate([value], mapping=mapping)
    assert len(q.calls) == 1
    assert out.status == 'insufficient_evidence' and out.insufficient_evidence
    assert not out.partial_answer and not out.citations
    assert [(r.status,r.evidence_ids) for r in out.requirement_coverage] == [('unsupported',()),('unsupported',())]
    assert mapping == before
    assert all(d.success for d in out.llm_diagnostics)


def test_mixed_answer_rejection_keeps_other_supported_requirement():
    value=draft({'R1':'unsupported','R2':'complete'},direct='Supported fact. [E2]',
                explanation='R1 lacks the requested information; R2 is established. [E2]')
    out,q=generate([value])
    assert len(q.calls)==1 and out.status=='success' and out.partial_answer
    assert [(r.status,r.evidence_ids) for r in out.requirement_coverage] == [('unsupported',()),('complete',('E2',))]
    assert out.citations==('E2',)


def test_shared_evidence_remains_usable_for_an_accepted_requirement():
    rows=compile_coverage(required_requirement_ids=('R1','R2'),
        allowed_support_by_requirement={'R1':('E1',),'R2':('E1',)},
        semantic_status_by_requirement={'R1':'unsupported','R2':'partial'},inline_citations=('E1',))
    assert rows[0].evidence_ids==() and rows[1].evidence_ids==('E1',)
    assert derive_overall_answer_status(rows,('E1',))=='partial'


def test_declined_exclusive_support_cannot_be_cited_as_answer_support():
    with pytest.raises(AnswerDomainValidationError) as error:
        compile_coverage(required_requirement_ids=('R1','R2'),
            allowed_support_by_requirement={'R1':('E1',),'R2':('E2',)},
            semantic_status_by_requirement={'R1':'unsupported','R2':'partial'},inline_citations=('E1','E2'))
    assert error.value.code=='ANSWER_CITATION_INVALID'


def test_abstention_cannot_hide_a_citation_in_appended_coverage_note():
    value=draft({'R1':'unsupported'},direct='当前证据不足。',explanation='无法判断目标事实。',note='只有相关背景。 [E1]')
    with pytest.raises(AnswerDomainValidationError) as error:
        _compile_semantic_answer(AnswerSemanticDraft.model_validate(value),allowed_evidence_ids={'E1'},
            research_requirements=(('R1','目标事实'),),support_by_requirement={'R1':('E1',)})
    assert error.value.code=='ANSWER_CITATION_INVALID'


def test_repair_can_decline_support_but_cannot_restore_it():
    before=AnswerSemanticDraft.model_validate(draft({'R1':'partial'},direct='Fact. [E1]',explanation='Fact. [E1]'))
    declined=AnswerSemanticDraft.model_validate(draft({'R1':'unsupported'},direct='Fact.',explanation='Fact.'))
    _validate_citation_repair(before,declined)
    with pytest.raises(AnswerDomainValidationError) as error:
        _validate_citation_repair(declined,before)
    assert error.value.code=='ANSWER_REPAIR_COVERAGE_UPGRADED'


def test_saved_unanswerable_context_has_a_valid_abstention_exit(tmp_path):
    """Saved question/Requirements/source with a scripted refusal, not a model result."""
    from paperbase.llm.answer_output_capture import capture_answer_outputs
    from test_agentic_p1_3c_structured_decision import real_client
    root=Path(__file__).resolve().parents[1]
    context=json.loads((root/'tests/fixtures/unanswerable_answer_context.json').read_text(encoding='utf-8'))
    entries=context['requirements']
    mapping={r['requirement_id']:tuple(r['allowed_supporting_evidence_ids']) for r in entries}
    value=draft({rid:'unsupported' for rid in mapping},direct='当前证据不足以判断是否报告了实际能耗或GPU功耗。',
                explanation='当前片段提供训练时间和复杂度，没有目标测量；不能推断整篇论文未报告。')
    client,calls=real_client([json.dumps(value,ensure_ascii=False)])
    capture=tmp_path/'answer.jsonl'
    with capture_answer_outputs(capture,case_id='unanswerable_001'):
        out=GroundedAnswerGenerator(settings=AnswerGenerationSettings(),client=client).generate(
            query=context['question'],evidence=(replace(units(('E1',))[0],text=context['source_text'],
                token_count=(len(context['source_text'])+3)//4,projection_context=context['structured_projection']),),
            research_requirements=tuple((r['requirement_id'],r['description']) for r in entries),support_by_requirement=mapping)
    assert out.status=='insufficient_evidence' and len(calls)==1
    saved=json.loads(capture.read_text(encoding='utf-8'))
    assert saved['schema_name']=='paperbase_grounded_answer_semantic_v3'
    assert json.loads(saved['raw_content'])==value
