"""P1.2a structural/provenance boundaries, independent of model semantics."""
import json
from dataclasses import replace
import pytest
from jsonschema import Draft202012Validator
from paperbase.config import AnswerGenerationSettings
from paperbase.generation.answer_generator import (
    AnswerSemanticDraft, AnswerGenerationResult,
    AnswerDomainValidationError, GroundedAnswerGenerator,
    _compile_semantic_answer, _normalize_and_validate_answer, _validate_citation_repair,
    _answer_response_schema,
)
from paperbase.generation.section_expander import EvidenceUnit
from paperbase.prompts.answer_generation import build_answer_generation_user_prompt
from paperbase.prompts.agentic_planner import RESEARCH_PLANNER_SYSTEM_PROMPT
from paperbase.agentic.planner import ResearchPlanner
from paperbase.llm.client import LLMCompletion

def units(ids=('E1','E2','E3')):
    return tuple(EvidenceUnit(evidence_id=i,kind='content',paper_id='P',paper_title='P',
        section='s',page_start=1,page_end=1,seed_chunk_ids=('c',),chunk_ids=('c',),
        text='A source fact.',token_count=4) for i in ids)

def draft(status='complete',body='A source fact. [E1]'):
    return AnswerSemanticDraft(direct_answer=body,evidence_explanation='A source fact. [E1]',
        reading_interpretation=None,requirement_status={'R1':status},coverage_note=None)

def validate(d,allowed={'E1','E2','E3'},supports={'R1':('E1','E2')}):
    return _compile_semantic_answer(d,allowed_evidence_ids=allowed,
        research_requirements=(('R1','A fact'),),support_by_requirement=supports)

@pytest.mark.parametrize('body',['A source fact. [E99]','A source fact. [REQ:R1]','Uncited fact.'])
def test_domain_rejects_uncited_unknown_namespace(body):
    d = draft(body=body)
    if body == 'Uncited fact.':
        d.evidence_explanation = body
    with pytest.raises(AnswerDomainValidationError):validate(d)

def test_only_actually_cited_support_is_compiled():
    assert validate(draft()).requirement_coverage[0].evidence_ids==('E1',)
    d=draft();d.evidence_explanation+=' [E2]'
    assert validate(d).requirement_coverage[0].evidence_ids==('E1','E2')

def test_prompt_partitions_support_and_preserves_reference_namespace():
    p=build_answer_generation_user_prompt(query='q',evidence=units(('REF1',)),
        research_requirements=(('R1','fact'),('R2','fact2')),
        support_by_requirement={'R1':('REF1','E99'),'R2':('E2',)})
    supported=json.loads(p.split('<requirements_with_verified_support>\n')[1].split('\n</requirements_with_verified_support>')[0])
    unsupported=json.loads(p.split('<requirements_without_verified_support>\n')[1].split('\n</requirements_without_verified_support>')[0])
    assert supported[0]['allowed_supporting_evidence_ids']==['REF1']
    assert unsupported[0]['requirement_id']=='R2'
    assert supported[0]['requirement_label']=='REQ:R1'
    assert '"evidence_id": "REF1", "in_text_citation": "[REF1]"' in p
    d=draft(body='A source fact. [REF1]');d.evidence_explanation='A source fact. [REF1]'
    assert validate(d,allowed={'REF1'},supports={'R1':('REF1',)}).citations==['REF1']
    with pytest.raises(AnswerDomainValidationError):validate(d,allowed={'E1'},supports={'R1':('E1',)})

class Queue:
    def __init__(self,values):self.values=list(values);self.calls=[];self.caps=[];self.temps=[]
    def with_max_tokens(self,n):self.caps.append(n);return self
    def with_temperature(self,n):self.temps.append(n);return self
    def complete_json_diagnostic(self,**kw):
        self.calls.append(kw);return LLMCompletion(json.dumps(self.values.pop(0)),finish_reason='stop')

def generate(client):
    return GroundedAnswerGenerator(settings=AnswerGenerationSettings(),client=client).generate(
        query='q',evidence=units(),research_requirements=(('R1','A fact'),),support_by_requirement={'R1':('E1','E2')})

def test_one_citation_repair_keeps_context_and_revalidates():
    bad=draft(body='A source fact. [E99]');good=draft()
    c=Queue([bad.model_dump(),good.model_dump()]);out=generate(c)
    assert out.status=='success' and len(c.calls)==2
    assert c.caps==[None] and c.temps==[0]
    assert out.llm_diagnostics[0].recovery_action=='repair_citation'
    feedback=json.loads(c.calls[1]['user_prompt'].split('<answer_citation_repair>\n')[1].split('\n</answer_citation_repair>')[0])
    assert feedback['allowed_coverage_evidence_ids']=={'R1':['E1','E2']}
    assert feedback['original_answer_json']==bad.model_dump()
    assert '<evidence_context>' in c.calls[1]['user_prompt']

def test_second_citation_failure_falls_back_no_loop():
    bad=draft(body='A source fact. [E99]').model_dump();c=Queue([bad,bad]);out=generate(c)
    assert out.status=='fallback' and len(c.calls)==2
    assert out.llm_diagnostics[-1].recovery_action=='fallback'

def test_repair_cannot_invent_facts_or_remove_negation_or_upgrade():
    before=draft(body='It does not improve performance. [E1]')
    for body in ('It improves performance. [E1]','It does improve performance. [E1]',
                 'It does not improve performance. [E1] New fact. [E1]'):
        with pytest.raises(AnswerDomainValidationError):_validate_citation_repair(before,draft(body=body))
    partial=draft(status='partial')
    with pytest.raises(AnswerDomainValidationError):_validate_citation_repair(partial,draft())

def test_repair_can_delete_whole_unsupported_sentence_and_fix_only_citations():
    before=draft(body='A source fact. [E99] Unsupported claim.')
    after=draft(body='A source fact. [E1]',status='partial')
    _validate_citation_repair(before,after)
    c=Queue([before.model_dump(),after.model_dump()]);assert generate(c).status=='success'


def test_repair_treats_ref_tokens_as_citations_without_loosening_body_guard():
    before = draft(body='Reference metadata. [REF99]')
    before.evidence_explanation = 'Reference metadata. [REF99]'
    after = draft(body='Reference metadata. [REF1]')
    after.evidence_explanation = 'Reference metadata. [REF1]'
    _validate_citation_repair(before, after)
    after.direct_answer = 'Changed reference claim. [REF1]'
    with pytest.raises(AnswerDomainValidationError, match='不得增加或改写'):
        _validate_citation_repair(before, after)

def test_planner_preserves_unknown_acronym_and_uses_independent_view():
    from paperbase.agentic.contracts import QueryContext,Intent,QueryScope,ScopeSource,ResearchGoal,EvidenceTarget
    question='Explain XZY-QV.'
    context=QueryContext(original_query=question,resolved_query=question,intent=Intent.SYNTHESIZE,
        scope=QueryScope.SINGLE_PAPER,target_paper_ids=('P',),scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.FIND_EVIDENCE,default_evidence_targets=frozenset({EvidenceTarget.METHOD}))
    c=Queue([{'goal':question,'requirements':[{'description':question,'required':True,'evidence_targets':['method'], "projection_type": "fact"}]}])
    out=ResearchPlanner(c).plan(context)
    assert out.plan.requirements[0].description==question
    assert '不猜测 acronym expansion' in RESEARCH_PLANNER_SYSTEM_PROMPT
    assert c.temps==[0] and c.caps==[None]

def test_dynamic_schema_encodes_only_supported_status_keys():
    schema=_answer_response_schema((('R1','fact'),('R2','no support')),{'R1':('E1','E2'),'R2':()},('E1','REF1'))
    assert 'citations' not in schema['properties']
    status=schema['properties']['requirement_status']
    assert status['required']==['R1'] and set(status['properties'])=={'R1'}
    assert status['additionalProperties'] is False
    with pytest.raises(AnswerDomainValidationError):validate(draft(body='Fact. [E99]'))


def test_dynamic_citation_schema_accepts_only_raw_allowed_evidence_ids():
    schema = _answer_response_schema((), {}, ('REF1',))
    validator = Draft202012Validator(schema)
    base = {
        'direct_answer': 'Reference fact. [REF1]',
        'evidence_explanation': 'Reference metadata. [REF1]',
        'reading_interpretation': None,
        'citations': ['REF1'],
        'insufficient_evidence': False,
        'partial_answer': False,
        'coverage_note': None,
    }
    assert not list(validator.iter_errors(base))
    for invalid in ('[REF1]', 'R1', 'E99', 'REF99'):
        assert list(validator.iter_errors({**base, 'citations': [invalid]}))


def test_mixed_evidence_citations_work_and_native_numbers_are_not_evidence_ids():
    item = AnswerGenerationResult(
        answer='Body fact [E1]. Bibliography contains [2] A and [3] B. [REF1]',
        citations=['E1', 'REF1'],
        insufficient_evidence=False,
    )
    result = _normalize_and_validate_answer(
        item,
        allowed_evidence_ids={'E1', 'REF1'},
    )
    assert result.citations == ['E1', 'REF1']
    assert '[2]' in result.answer and '[3]' in result.answer


def test_requirement_id_is_never_accepted_as_body_citation():
    item = AnswerGenerationResult(
        answer='Requirement is not evidence. [R1]',
        citations=[],
        insufficient_evidence=True,
    )
    with pytest.raises(AnswerDomainValidationError, match='Requirement ID'):
        _normalize_and_validate_answer(item, allowed_evidence_ids={'REF1'})
