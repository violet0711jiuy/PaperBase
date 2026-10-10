"""Shared Guard preflight / bounded Decision recovery, no provider calls."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    AgentAction, ActionRejectionContext, FinishAction, GuardState, RegistryState,
    QueryContext, ResearchPlan, RequirementState, SearchPapersAction,
)
from paperbase.agentic.decision import (
    LLMDecisionAgent, DecisionError, _decision_json_schema, _to_action,
)
from paperbase.agentic.graph import _assess_candidate_with_guard, _decide_action, _tracker, _validate_action
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.agentic.schemas import AGENT_DECISION_ADAPTER
from paperbase.agentic.search_requests import compile_action_requests
from paperbase.agentic.action_control import ActionCanonicalizer
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import PaperSearchSpec
from test_agentic_decision_view import state, record

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT/'output/answer_contract_simplification_20261006/e1_answer_semantic_20261006/cases'

class Client:
    def __init__(self, *payloads): self.payloads=list(payloads); self.calls=[]
    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        assert self.payloads, 'unexpected additional Decision call'
        return json.dumps(self.payloads.pop(0))

def wire(query='already searched', lexical=('adaptive graph',), requirements=('R1',)):
    return {'decision': {'action':'search_papers','searches':[dict(paper_id='P',query=query,
        lexical_terms=list(lexical),requirement_ids=list(requirements))]}}

def action(payload):
    canonical, _ = ActionCanonicalizer().canonicalize(payload)
    return _to_action(AGENT_DECISION_ADAPTER.validate_json(json.dumps(canonical)).decision)

class Executor:
    def compile_requests(self, action, **kwargs):
        # The executor's effective backend parameters must participate in identity.
        return compile_action_requests(action,context=kwargs.get('context'),snapshot=kwargs.get('corpus_snapshot'),
                                       parameters={'P': {'actual_backend_parameter': 73}})

def fixture(first=None, *, max_attempts=2):
    s=state(evidence=(record(),)); settings=AgenticResearchSettings(node_max_attempts=max_attempts)
    runtime=SimpleNamespace(context=SimpleNamespace(settings=settings,tool_executor=Executor()))
    tracker=BudgetAndGuardTracker(settings)
    if first is not None:
        first_action=action(first)
        requests=runtime.context.tool_executor.compile_requests(first_action,context=s.context)
        assert tracker.validate_action(first_action,context=s.context,plan=s.plan,snapshot=None,
                                       canonical_requests=requests).accepted
        tracker.record_decision(1);tracker.record_tool_call(first_action,canonical_requests=requests)
    g=dict(query_context=dump_contract(s.context,QueryContext),plan=dump_contract(s.plan,ResearchPlan),
        registry_state=dump_contract(RegistryState(records=s.evidence),RegistryState),
        guard_state=dump_contract(tracker.snapshot(),GuardState),
        requirement_states=dump_contract(s.requirement_states,tuple[RequirementState,...]))
    s=replace(s,budget_usage=tracker.usage(),guard_preflight=lambda a: _assess_candidate_with_guard(
        g,runtime,a,_tracker(g,settings)))
    return s,g,runtime

def run(first, *candidates, max_attempts=2):
    s,g,r=fixture(first,max_attempts=max_attempts)
    client=Client(*candidates)
    outcome=LLMDecisionAgent(client,max_attempts=max_attempts).decide(s)
    return outcome,client,s,g,r

def test_a_first_legal_action_no_recovery_and_authoritative_guard_still_runs():
    out,c,s,g,r=run(None,wire())
    assert len(c.calls)==1 and not out.recovery_rejections
    g['action']=dump_contract(out.action,AgentAction)
    assert _validate_action(g,r)['next_node']=='execute_tool'

@pytest.mark.parametrize('replacement',[
    wire('different canonical request'),
    {'decision':dict(action='get_section_context',paper_id='P',section_id='section-E1',requirement_ids=['R1'])},
    {'decision':dict(action='finish',reason='No useful additional research')},
])
def test_b_c_e_f_duplicate_recovers_to_independently_chosen_legal_candidate(replacement):
    a=wire();out,c,s,g,r=run(a,a,replacement)
    assert len(c.calls)==2 and out.llm_calls==2
    assert len(out.recovery_rejections)==1 and out.recovery_rejections[0].rejection_code=='DuplicateAction'
    feedback=json.loads(c.calls[1]['user_prompt'])
    assert feedback['action_recovery']['blocked_action']['action_arguments']['searches'][0]['query']=='already searched'
    assert feedback['action_recovery']['rejection_code']=='DuplicateAction'
    assert feedback['runtime_action_capabilities']['available_tools']
    error=out.llm_diagnostics[0].validation_diagnostics[0]
    details=json.loads(error.actual_summary)
    assert details['rejected_action_identity'].startswith('search-action-v1:')
    assert details['canonical_effective_requests'][0]['request_identity'].startswith('search-v1:')
    # Check actual provider input, not just diagnostics or intermediate State.
    visible=feedback['previous_validation_errors'][0]['rejected_effective_request']
    assert visible['rejected_action_identity']==details['rejected_action_identity']
    assert visible['canonical_effective_requests']==details['canonical_effective_requests']
    assert visible['affected_requirement_ids']==list(out.recovery_rejections[0].requirement_ids)
    assert out.llm_diagnostics[0].error_stage=='guard_preflight_rejection'
    assert out.llm_diagnostics[1].decision_iteration==out.llm_diagnostics[0].decision_iteration
    g['action']=dump_contract(out.action,AgentAction)
    assert _validate_action(g,r)['next_node']=='execute_tool'  # Finish closure remains separate.

def test_d_raw_different_but_same_effective_request_still_rejected():
    original=wire('already searched',('adaptive graph','retrieval'))
    changed=wire('already   searched',('GRAPH RETRIEVAL','adaptive'),('R2',))
    s,g,r=fixture(original)
    assert asdict(action(original))!=asdict(action(changed))
    assert s.guard_preflight(action(original))[1][0].identity==s.guard_preflight(action(changed))[1][0].identity
    c=Client(changed,wire('new effective request'))
    out=LLMDecisionAgent(c).decide(s)
    assert out.recovery_rejections[0].rejection_code=='DuplicateAction' and len(c.calls)==2

def test_g_recovery_is_one_extra_candidate_even_with_larger_validation_pool():
    a=wire();s,g,r=fixture(a,max_attempts=4);before=deepcopy(g)
    c=Client(a,a,a,a)
    with pytest.raises(DecisionError) as captured: LLMDecisionAgent(c,max_attempts=4).decide(s)
    assert captured.value.llm_calls==2 and len(c.calls)==2
    assert len(captured.value.recovery_rejections)==2 and g==before

def test_preflight_preserves_all_runtime_ledgers_and_actual_compiler_parameters():
    a=wire();s,g,r=fixture(a); before=deepcopy(g)
    assessment,requests=s.guard_preflight(action(a))
    assert assessment.code=='DuplicateAction' and g==before
    wrong_identity=compile_action_requests(action(a),context=s.context)[0].identity
    assert wrong_identity!=requests[0].identity
    assert load_contract(g['guard_state'],GuardState).usage.invalid_actions==0

def test_graph_success_counts_one_iteration_two_calls_zero_invalid_and_keeps_history():
    a=wire();s,g,r=fixture(a);old=load_contract(g['guard_state'],GuardState)
    c=Client(a,wire('legal new request'));r.context.decision_agent=LLMDecisionAgent(c)
    update=_decide_action(g,r);g.update(update)
    now=load_contract(g['guard_state'],GuardState)
    assert now.usage.agent_iterations==old.usage.agent_iterations+1
    assert now.usage.llm_calls==old.usage.llm_calls+2 and now.usage.invalid_actions==0
    history=load_contract(g['action_rejections'],tuple[ActionRejectionContext,...])
    assert len(history)==1 and g['last_action_rejection'] is not None
    assert _validate_action(g,r)['next_node']=='execute_tool'

def test_graph_failed_recovery_counts_only_one_failed_iteration_and_preserves_both_rejections():
    a=wire();s,g,r=fixture(a);r.context.decision_agent=LLMDecisionAgent(Client(a,a))
    update=_decide_action(g,r);g.update(update)
    assert load_contract(g['guard_state'],GuardState).usage.invalid_actions==1
    assert len(load_contract(g['action_rejections'],tuple[ActionRejectionContext,...]))==2
    assert update['next_node']=='decide_action'
    r.context.decision_agent=LLMDecisionAgent(Client(a,a));g.update(_decide_action(g,r))
    assert load_contract(g['guard_state'],GuardState).usage.invalid_actions==2
    assert g['stop_reason']=='invalid_action_limit'

def test_recovery_schema_failure_does_not_trigger_unbounded_extra_repair():
    a=wire();s,g,r=fixture(a,max_attempts=4);c=Client(a,{'decision':{'action':'finish'}},a)
    with pytest.raises(DecisionError) as e: LLMDecisionAgent(c,max_attempts=4).decide(s)
    assert e.value.llm_calls==2 and len(e.value.recovery_rejections)==1

def test_k_semantic_near_duplicate_distinct_identity_is_not_rejected():
    original=wire('how is C_i defined');near=wire('definition and role of calibration term C_i')
    out,c,s,g,r=run(original,near)
    assert len(c.calls)==1 and not out.recovery_rejections

def test_l_requirement_only_changes_do_not_bypass_search_identity():
    a=wire();s,g,r=fixture(a)
    assert s.guard_preflight(action(wire(requirements=('R2',))))[0].code=='DuplicateAction'

def test_guard_feedback_serializer_does_not_forward_unrelated_diagnostic_or_provider_data():
    from paperbase.agentic.decision import _repair_payload
    from paperbase.llm.client import ValidationDiagnostic
    summary=dict(rejected_tool='search_references',rejected_action_identity='search-action-v1:test',
                 canonical_effective_requests=[dict(kind='search_references',request_identity='search-v1:test',
                 semantic_query=None,lexical_terms=['svr'],top_k=5,unrelated_internal_value='not-for-prompt')],
                 unrelated_provider_value='not-for-prompt')
    diagnostic=ValidationDiagnostic(node='decide_action',attempt=1,error_stage='guard_preflight_rejection',
        error_code='DuplicateAction',field_path='decision',validator_name='runtime_guard_preflight',
        safe_reason='effective request already executed',expected_constraint='choose a legal candidate',
        actual_summary=json.dumps(summary),recovery_action='replan_action')
    visible=_repair_payload(diagnostic)['rejected_effective_request']
    assert visible['canonical_effective_requests'][0]['semantic_query'] is None
    assert 'not-for-prompt' not in json.dumps(visible)
    assert 'rejected_effective_request' not in _repair_payload(replace(diagnostic,validator_name='pydantic.missing'))

def test_schema_local_and_guard_layers_remain_distinct():
    a=wire();s,g,r=fixture(a)
    assert not list(Draft202012Validator(_decision_json_schema(s)).iter_errors(a))
    assert s.guard_preflight(action(a))[0].code=='DuplicateAction'
    unavailable={'decision':dict(action='get_section_context',paper_id='P',section_id='not-known',requirement_ids=['R1'])}
    assert list(Draft202012Validator(_decision_json_schema(s)).iter_errors(unavailable))


def test_reference_duplicate_recovery_uses_lexical_only_canonical_contract():
    refs = {'decision': dict(action='search_references', paper_id='P',
        query='reference query', lexical_terms=['adaptive'], requirement_ids=['R1'])}
    finish = {'decision': dict(action='finish', reason='Current scope researched')}
    out, client, state, _, runtime = run(refs, refs, finish)
    assert out.llm_calls == 2
    rejected = out.llm_diagnostics[0]
    assert rejected.error_stage == 'guard_preflight_rejection'
    assert rejected.validation_diagnostics[0].error_code == 'DuplicateAction'
    feedback = json.loads(client.calls[1]['user_prompt'])
    assert feedback['action_recovery']['blocked_action']['action_name'] == 'search_references'
    summary = feedback['previous_validation_errors'][0]['rejected_effective_request']
    request = runtime.context.tool_executor.compile_requests(action(refs), context=state.context)[0]
    assert summary['canonical_effective_requests'] == [dict(
        request_identity=request.identity, kind='search_references', paper_ids=['P'],
        semantic_query=None, lexical_terms=['adaptive'], lexical_query='"adaptive"', top_k=5)]
    assert out.recovery_rejections[0].rejection_code == 'DuplicateAction'

def test_overview_exact_repeat_is_guard_blocked_and_can_recover():
    overview={'decision':dict(action='get_paper_overview',paper_id='P',requirement_ids=['R1'])}
    out,c,s,g,r=run(overview,overview,{'decision':dict(action='finish',reason='Current scope researched')})
    assert not list(Draft202012Validator(_decision_json_schema(s)).iter_errors(overview))
    assert out.recovery_rejections[0].rejection_code=='DuplicateAction' and len(c.calls)==2

def test_partial_batch_duplicate_paper_query_uses_original_guard():
    from paperbase.agentic.contracts import QueryScope
    s,g,r=fixture();context=replace(s.context,scope=QueryScope.SELECTED_PAPERS,target_paper_ids=('P','Q'))
    g['query_context']=dump_contract(context,QueryContext)
    batch=SearchPapersAction((PaperSearchSpec('P','already searched',('adaptive graph',),requirement_ids=('R1',)),
                              PaperSearchSpec('Q','other paper',('graph',),requirement_ids=('R1',))))
    requests=r.context.tool_executor.compile_requests(batch,context=context)
    tracker=BudgetAndGuardTracker(r.context.settings);tracker.record_decision(1)
    tracker.record_tool_call(batch,canonical_requests=requests)
    g['guard_state']=dump_contract(tracker.snapshot(),GuardState)
    s=replace(s,context=context,budget_usage=tracker.usage())
    c=Client(wire(),wire('new request'))
    out=LLMDecisionAgent(c).decide(s)
    assert out.recovery_rejections[0].rejection_code=='DuplicatePaperQuery'

@pytest.mark.parametrize('cid',['method_002','result_001'])
@pytest.mark.parametrize('replacement',['search','section','finish'])
def test_latest_fixed_audit_request_replay(cid,replacement):
    # Actual saved requests/identities; full original Decision State was not saved.
    a=json.loads((LATEST/f'{cid}.audit.json').read_text(encoding='utf-8'))
    rejected=a['rejected_actions'][-1]
    payload={'decision':dict(deepcopy(rejected['action_arguments']),action='search_papers')}
    legacy_action=SearchPapersAction(tuple(PaperSearchSpec(
        paper_id=item['paper_id'],query=item['query'],
        lexical_terms=tuple(item.get('lexical_terms',())),
        requirement_ids=tuple(item['requirement_ids']),
    ) for item in payload['decision']['searches']))
    # Older audits may contain an empty lexical list. Preserve that historical
    # request for identity replay, but make the simulated new Decision output
    # obey the current LLM contract instead of synthesizing terms in production.
    for item in payload['decision']['searches']:
        if not item.get('lexical_terms'):
            item['lexical_terms']=['calibration term','C_i','prediction interval coverage']
    paper=payload['decision']['searches'][0]['paper_id']
    s,g,r=fixture()
    reqids=tuple(q['requirement_id'] for q in a['requirements'])
    from paperbase.agentic.contracts import ResearchRequirement, RequirementStatus
    targets={binding['requirement_id']:frozenset(binding['evidence_targets'])
             for step in a['steps'] for binding in step['requirement_target_bindings']}
    plan=replace(s.plan,requirements=tuple(ResearchRequirement(q['requirement_id'],q['description'],targets[q['requirement_id']],required=q['required']) for q in a['requirements']))
    context=replace(s.context,target_paper_ids=(paper,))
    evidence=tuple(replace(e,evidence=replace(e.evidence,paper_id=paper)) for e in s.evidence)
    reqstates=tuple(RequirementState(rid,RequirementStatus.IN_PROGRESS) for rid in reqids)
    g.update(query_context=dump_contract(context,QueryContext),plan=dump_contract(plan,ResearchPlan),
        registry_state=dump_contract(RegistryState(records=evidence),RegistryState),
        requirement_states=dump_contract(reqstates,tuple[RequirementState,...]))
    parameters={paper:a['steps'][0]['canonical_effective_requests'][0]['retrieval_parameters']}
    def compile_actual(act,**kwargs): return compile_action_requests(act,context=context,parameters=parameters)
    r.context.tool_executor.compile_requests=compile_actual
    legacy_requests=compile_actual(legacy_action)
    prior=next(step for step in a['steps'] if step['action_arguments']==rejected['action_arguments'])
    assert legacy_requests[0].identity==prior['request_identities'][0]
    current_action=action(payload)
    requests=compile_actual(current_action)
    tracker=BudgetAndGuardTracker(r.context.settings);tracker.record_decision(1)
    tracker.record_tool_call(current_action,canonical_requests=requests)
    g['guard_state']=dump_contract(tracker.snapshot(),GuardState)
    s=replace(s,context=context,plan=plan,evidence=evidence,requirement_states=reqstates,budget_usage=tracker.usage())
    s=replace(s,guard_preflight=lambda act:_assess_candidate_with_guard(g,r,act,_tracker(g,r.context.settings)))
    different=deepcopy(payload)
    if replacement=='search': different['decision']['searches'][0]['query']+=' alternative research focus'
    elif replacement=='section': different={'decision':dict(action='get_section_context',paper_id=paper,section_id='section-E1',requirement_ids=[reqids[0]])}
    else: different={'decision':dict(action='finish',reason='Current research path has low value')}
    client=Client(payload,different);out=LLMDecisionAgent(client).decide(s)
    assert out.llm_calls==2 and out.recovery_rejections[0].rejection_code=='DuplicateAction'
    assert out.action.action_name==different['decision']['action']
