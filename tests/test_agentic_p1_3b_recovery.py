"""P1.3b lifecycle tests use real Graph nodes and original legality Guard."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from types import SimpleNamespace

import pytest
from paperbase.agentic.checkpoint_serde import dump_contract,load_contract
from paperbase.agentic.contracts import (ActionRejectionContext,AgentAction,GuardState,RegistryState,
    QueryContext,ResearchPlan,ResearchObservation,GetSectionContextAction,SearchPapersAction,
    CoverageState,CoverageStrategy)
from paperbase.agentic.decision import _action_recovery_constraint,_runtime_action_capabilities,LLMDecisionAgent
from paperbase.agentic.decision_view import build_compact_decision_view,expand_decision_references
from paperbase.agentic.graph import _validate_action,_execute_tool
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import PaperSearchSpec,ToolStatus,ToolResult,ToolName,SectionContextPayload
from test_agentic_decision_view import state,record,observation,Client

canonical=lambda x:json.loads(json.dumps(x,default=list))

def rejected_state():
    o=observation(1)
    r=ActionRejectionContext('search_papers',o.action_arguments,('R2',),'DuplicateAction','executed','DuplicateAction',2)
    return replace(state(evidence=(record(),),observations=(o,),iteration=2),
                   last_action_rejection=r,action_rejections=(r,))


def test_active_constraints_are_inline_and_history_is_atomic():
    s=rejected_state();p=build_compact_decision_view(s).payload
    assert p['action_recovery']['blocked_action']['action_arguments']==s.last_action_rejection.action_arguments
    assert '$request' not in json.dumps(p['action_recovery']) and '$query' not in json.dumps(p['action_recovery'])
    assert '$request' not in json.dumps(p['recent_research_events'])
    assert 'recent_unavailable_actions' not in p['runtime_action_capabilities']
    assert not {'research_requests','research_queries'} & p.keys()
    expanded=expand_decision_references(p)
    assert expanded['action_recovery']==canonical(_action_recovery_constraint(s))


def test_history_does_not_reactivate_recovery_or_repair_temperature():
    s=replace(rejected_state(),last_action_rejection=None)
    assert _action_recovery_constraint(s) is None
    assert build_compact_decision_view(s).payload['action_recovery'] is None
    assert build_compact_decision_view(s).payload['recent_research_events'][-1]['outcome']=='DuplicateAction'
    class RepairClient(Client):
        def with_repair_temperature(self):raise AssertionError('historical rejection triggered pending recovery')
    LLMDecisionAgent(RepairClient()).decide(s)


def lifecycle_fixture(status=ToolStatus.OK,raises=False):
    s=state(evidence=(record(),))
    settings=AgenticResearchSettings(node_max_attempts=4)
    first=SearchPapersAction((PaperSearchSpec('P','focused gap 1',('term1',),requirement_ids=('R2',)),))
    tracker=BudgetAndGuardTracker(settings)
    assert tracker.validate_action(first,context=s.context,plan=s.plan,snapshot=None).accepted
    tracker.record_decision(1);tracker.record_tool_call(first)
    graph=dict(query_context=dump_contract(s.context,QueryContext),plan=dump_contract(s.plan,ResearchPlan),
               registry_state=dump_contract(RegistryState(records=s.evidence),RegistryState),
               guard_state=dump_contract(tracker.snapshot(),GuardState),action=dump_contract(first,AgentAction))
    class Executor:
        def execute(self,action,**kwargs):
            if raises:raise RuntimeError('tool failed')
            return ToolResult(status,ToolName.GET_SECTION_CONTEXT,SectionContextPayload('P','section-E1'))
    runtime=SimpleNamespace(context=SimpleNamespace(settings=settings,tool_executor=Executor()))
    return s,graph,runtime


def decision_state_from_graph(s,g):
    return replace(s,last_action_rejection=load_contract(g['last_action_rejection'],ActionRejectionContext)
                   if g.get('last_action_rejection') is not None else None,
        action_rejections=load_contract(g['action_rejections'],tuple[ActionRejectionContext,...]))


def test_rejection_reject_again_then_success_consumes_only_current_pointer():
    s,g,runtime=lifecycle_fixture()
    g.update(_validate_action(g,runtime))
    pending=decision_state_from_graph(s,g)
    assert _action_recovery_constraint(pending)['rejection_code']=='DuplicateAction'
    # Another illegal action updates the pointer and appends history.
    tracker=BudgetAndGuardTracker(runtime.context.settings,state=load_contract(g['guard_state'],GuardState))
    tracker.record_decision(1);g['guard_state']=dump_contract(tracker.snapshot(),GuardState)
    g.update(_validate_action(g,runtime))
    pending=decision_state_from_graph(s,g)
    assert pending.last_action_rejection.iteration==2 and len(pending.action_rejections)==2
    history=deepcopy(g['action_rejections'])
    g['action']=dump_contract(GetSectionContextAction('P','section-E1',('R2',)),AgentAction)
    accepted=_validate_action(g,runtime)
    assert accepted['next_node']=='execute_tool' and 'last_action_rejection' not in accepted
    g.update(accepted)
    assert _action_recovery_constraint(decision_state_from_graph(s,g)) is not None
    executed=_execute_tool(g,runtime)
    assert executed['last_action_rejection'] is None
    g.update(executed)
    assert g['action_rejections']==history
    next_state=decision_state_from_graph(s,g)
    p=build_compact_decision_view(next_state).payload
    assert p['action_recovery'] is None
    assert sum(e['outcome']=='DuplicateAction' for e in p['recent_research_events'])==2


@pytest.mark.parametrize('status',[ToolStatus.EMPTY,ToolStatus.PARTIAL])
def test_successful_empty_or_partial_execution_also_consumes_recovery(status):
    s,g,runtime=lifecycle_fixture(status=status);g.update(_validate_action(g,runtime))
    g['action']=dump_contract(GetSectionContextAction('P','section-E1',('R2',)),AgentAction)
    g.update(_validate_action(g,runtime));g.update(_execute_tool(g,runtime))
    assert g['last_action_rejection'] is None and len(g['action_rejections'])==1


@pytest.mark.parametrize('status,raises',[(ToolStatus.ERROR,False),(ToolStatus.UNAVAILABLE,False),(ToolStatus.OK,True)])
def test_failed_execution_does_not_consume_pending_recovery(status,raises):
    s,g,runtime=lifecycle_fixture(status=status,raises=raises);g.update(_validate_action(g,runtime))
    pending=deepcopy(g['last_action_rejection'])
    g['action']=dump_contract(GetSectionContextAction('P','section-E1',('R2',)),AgentAction)
    g.update(_validate_action(g,runtime));g.update(_execute_tool(g,runtime))
    assert g['last_action_rejection']==pending


def test_candidate_screening_serialization_is_separate_from_requirement_retrieval_state():
    s=replace(state(),coverage=CoverageState(('P',),(),CoverageStrategy.EXHAUSTIVE))
    before=deepcopy(s);p=build_compact_decision_view(s).payload
    assert 'coverage' not in p and 'complete' not in p['candidate_coverage']
    assert p['candidate_coverage']['screening_complete']==s.coverage.coverage_complete
    # A persisted RequirementStatus does not get echoed as a Decision hint;
    # this view derives only observable retrieval history and local supports.
    assert p['requirements'][1]['retrieval_state']=='not_searched'
    assert 'status' not in p['requirements'][1] and 'complete' not in p['requirements'][1]
    assert s==before


def test_accepted_finish_consumes_recovery_without_erasing_history():
    from paperbase.agentic.contracts import FinishAction,RequirementState
    from paperbase.agentic.graph import _assess_action
    s,g,runtime=lifecycle_fixture();g.update(_validate_action(g,runtime))
    history=deepcopy(g['action_rejections'])
    g['registry_state']=dump_contract(RegistryState(records=(*s.evidence,record('E2',rid='R2'))),RegistryState)
    g['requirement_states']=dump_contract(s.requirement_states,tuple[RequirementState,...])
    g['action']=dump_contract(FinishAction('researched current scope'),AgentAction)
    update=_assess_action(g,runtime)
    assert update['stop_reason']=='agent_finished' and update['last_action_rejection'] is None
    g.update(update)
    assert g['action_rejections']==history
