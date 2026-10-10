from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from paperbase.agentic.contracts import (AgentAction, GuardState, QueryContext, RegistryState,
    ResearchPlan, SearchReferencesAction, SearchPapersAction, SearchKnowledgeBaseAction, CorpusScopeSnapshot)
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.graph import _validate_action
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import PaperSearchSpec, ToolStatus
from paperbase.retrieval.requests import validate_effective_request
from paperbase.agentic.search_requests import compile_action_requests
from test_canonical_search_requests import real_reference_tool
from test_agentic_query_dedup import _plan
from test_agentic_decision_tools import _single_context, _kb_context


def test_reference_guard_and_direct_executor_do_not_enter_backend_or_spend_query(real_reference_tool):
    executor, context, chunks, db = real_reference_tool
    db.search_compiled_lexical = Mock(wraps=db.search_compiled_lexical)
    db.get_document = Mock(wraps=db.get_document)
    action=SearchReferencesAction(chunks[0].paper_id,'SVR',('the','and'),('R1',))
    requests=executor.compile_requests(action,context=context)
    assert requests[0].lexical_terms==() and requests[0].lexical_query==''
    tracker=BudgetAndGuardTracker(AgenticResearchSettings())
    args=dict(context=context,plan=_plan(),snapshot=None,canonical_requests=requests)
    decision=tracker.validate_action(action,**args)
    assert not decision.accepted and decision.code=='NoEffectiveLexicalQuery'
    assert not tracker.snapshot().executed_action_fingerprints
    result=executor.execute(action,context=context,canonical_requests=requests)
    assert result.status is ToolStatus.ERROR and result.error_code==decision.code
    receipt=result.metadata['request_executions'][0]
    assert not receipt['executed'] and not receipt['successful']
    assert result.metadata['request_rejection_detail']==decision.detail
    db.get_document.assert_not_called(); db.search_compiled_lexical.assert_not_called()
    tracker.record_tool_call(action,canonical_requests=requests,result=result)
    assert tracker.retrieval_queries==0 and tracker.queries_per_paper.get(chunks[0].paper_id,0)==0
    assert not tracker.snapshot().executed_paper_query_fingerprints
    fixed=replace(action,lexical_terms=('SVR',))
    fixed_requests=executor.compile_requests(fixed,context=context)
    assert tracker.validate_action(fixed,**{**args,'canonical_requests':fixed_requests}).accepted
    result=executor.execute(fixed,context=context,canonical_requests=fixed_requests)
    tracker.record_tool_call(fixed,canonical_requests=fixed_requests,result=result)
    assert result.status is ToolStatus.OK and tracker.retrieval_queries==1
    assert tracker.queries_per_paper[chunks[0].paper_id]==1
    assert tracker.snapshot().executed_paper_query_fingerprints==(fixed_requests[0].identity,)


def test_graph_rejects_before_execution_and_provides_replan_context(real_reference_tool):
    executor,context,chunks,db=real_reference_tool
    executor.execute=Mock(wraps=executor.execute)
    db.search_compiled_lexical=Mock(wraps=db.search_compiled_lexical)
    action=SearchReferencesAction(chunks[0].paper_id,'SVR',('the','and'),('R1',))
    state=dict(action=dump_contract(action,AgentAction),query_context=dump_contract(context,QueryContext),
        registry_state=dump_contract(RegistryState(),RegistryState),plan=dump_contract(_plan(),ResearchPlan),
        guard_state=dump_contract(GuardState(),GuardState))
    runtime=SimpleNamespace(context=SimpleNamespace(settings=AgenticResearchSettings(),tool_executor=executor))
    update=_validate_action(state,runtime)
    assert update['next_node']=='decide_action'
    assert update['last_action_rejection']['rejection_code']=='NoEffectiveLexicalQuery'
    assert 'Revise lexical_terms' in update['last_action_rejection']['rejection_reason']
    restored=load_contract(update['guard_state'],GuardState)
    assert restored.usage.retrieval_queries==0 and not restored.executed_action_fingerprints
    executor.execute.assert_not_called();db.search_compiled_lexical.assert_not_called()


def test_reference_empty_wire_hints_still_use_explicit_compiler_fallback(real_reference_tool):
    executor,context,chunks,db=real_reference_tool
    action=SearchReferencesAction(chunks[0].paper_id,'SVR wind',(),('R1',))
    request=executor.compile_requests(action,context=context)[0]
    assert 'svr' in request.lexical_terms and validate_effective_request(request) is None
    result=executor.execute(action,context=context)
    assert result.status is ToolStatus.OK and result.metadata['request_executions'][0]['executed']


def test_empty_lexical_remains_executable_for_paper_and_kb():
    paper=SearchPapersAction((PaperSearchSpec('A','SVR',('the','and'),requirement_ids=('R1',)),))
    kb=SearchKnowledgeBaseAction('SVR',('the','and'),('R1',))
    snapshot=CorpusScopeSnapshot('corpus',('A',),'2026-10-02')
    for action,context,scope in ((paper,_single_context(),None),(kb,_kb_context(),snapshot)):
        request=compile_action_requests(action,context=context,snapshot=scope)[0]
        assert request.lexical_terms==() and request.semantic_query=='SVR'
        assert validate_effective_request(request) is None
        assert BudgetAndGuardTracker(AgenticResearchSettings()).validate_action(
            action,context=context,plan=_plan(),snapshot=scope,canonical_requests=(request,)).accepted
