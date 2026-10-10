"""Effective request identity regressions; real FTS5 and no LLM calls."""
from dataclasses import asdict, replace
from types import SimpleNamespace
from copy import deepcopy

import pytest
from pydantic import TypeAdapter

from paperbase.agentic.contracts import (GuardState, RegistryDelta, ResearchObservation,
    SearchPapersAction, SearchReferencesAction, SearchKnowledgeBaseAction, CorpusScopeSnapshot)
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.agentic.search_requests import compile_action_requests
from paperbase.agentic.tools import ResearchToolExecutor
from paperbase.agentic.observation import build_observation
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.config import AgenticResearchSettings
from paperbase.database import MetadataDatabase
from paperbase.retrieval.requests import (CanonicalSearchRequest, compile_paper_request,
    compile_reference_request, compile_kb_request, request_trace)
from paperbase.retrieval.contracts import (PaperSearchSpec, ToolResult, ToolName,
    ReferenceSearchPayload, ToolStatus, QueryOrigin, RetrievalSource)
from paperbase.retrieval.facade import RetrievalService
from paperbase.retrieval.adapters import FormalKnowledgeBaseAdapter
from test_agentic_query_dedup import _context, _plan
from test_agentic_decision_tools import _single_context, _kb_context
from test_database import _sample_result
from test_retrieval_tool_results import _snapshot
from paperbase.agentic.audit import _step_audit


def paper(query='focus', terms=('encoder',), **kwargs):
    return compile_paper_request(paper_id='A', query=query, lexical_terms=terms,
                                top_k=kwargs.pop('top_k',5), source=kwargs.pop('source','formal_knowledge_base'), **kwargs)


def reference(query='ignored', terms=('wind speed',), **kwargs):
    return compile_reference_request(paper_id='A', query=query, lexical_terms=terms,
                                    top_k=kwargs.pop('top_k',5), **kwargs)


def test_paper_lexical_change_changes_identity_and_normalized_hints_do_not():
    assert paper(terms=('encoder',)).identity != paper(terms=('decoder',)).identity
    assert paper(terms=('Graph,  WAVENET!', 'the', 'graph')).identity == paper(terms=('wavenet','graph')).identity
    assert paper(' focus  ').identity == paper('focus').identity
    # Dense case and punctuation are preserved: no assumption about the embedder.
    assert paper('Focus').identity != paper('focus').identity
    assert paper('focus?').identity != paper('focus').identity


def test_reference_identity_only_contains_effective_lexical_request():
    first=reference('query one', ('WIND, speed!',))
    assert first.identity == reference('different query',('speed','wind')).identity
    assert first.identity != reference('query one',('SVR',)).identity
    assert 'semantic_query' not in asdict(first) and 'query' not in asdict(first)
    assert reference('SVR wind',()).identity != reference('LSTM wind',()).identity
    # Explicit all-stopword hints do not silently inject fallback query tokens.
    assert reference('SVR',('the','and')).lexical_terms == ()


def test_top_k_scope_source_and_live_parameters_are_identity_fields():
    assert paper(top_k=1).identity != paper(top_k=5).identity
    assert paper(source='workspace',workspace_id='W').identity != paper().identity
    assert paper(retrieval_parameters={'rrf_k':60}).identity != paper(retrieval_parameters={'rrf_k':20}).identity
    assert compile_kb_request(paper_ids=('B','A','A'),query='focus',lexical_terms=('x',),top_k=5).identity == compile_kb_request(paper_ids=('A','B'),query='focus',lexical_terms=('x',),top_k=5).identity
    assert compile_kb_request(paper_ids=('A',),query='focus',lexical_terms=('x',),top_k=5).identity != compile_kb_request(paper_ids=('B',),query='focus',lexical_terms=('x',),top_k=5).identity


def test_guard_validation_is_attempt_only_then_successful_execution_blocks():
    tracker=BudgetAndGuardTracker(AgenticResearchSettings())
    action=SearchPapersAction((PaperSearchSpec('A','focus',('encoder',),requirement_ids=('R1',)),))
    args=dict(context=_context(),plan=_plan(),snapshot=None)
    assert tracker.validate_action(action,**args).accepted
    assert tracker.validate_action(action,**args).accepted
    assert tracker.snapshot().executed_action_fingerprints == ()
    assert all(not receipt['executed'] for receipt in tracker.snapshot().request_executions)
    tracker.record_tool_call(action)
    assert tracker.validate_action(action,**args).code == 'DuplicateAction'
    changed=replace(action,searches=(replace(action.searches[0],lexical_terms=('decoder',)),))
    assert tracker.validate_action(changed,**args).accepted
    restored=BudgetAndGuardTracker(AgenticResearchSettings(),state=tracker.snapshot())
    assert restored.validate_action(action,**args).code == 'DuplicateAction'


def test_legacy_executed_search_keys_fail_open_with_reason_and_usage_retained():
    old=GuardState(executed_action_fingerprints=('[ ["A","focus",["x"]] ]',),
                   executed_paper_query_fingerprints=('["A","focus"]',))
    restored=BudgetAndGuardTracker(AgenticResearchSettings(),state=old)
    action=SearchPapersAction((PaperSearchSpec('A','focus',('x',),requirement_ids=('R1',)),))
    assert restored.validate_action(action,context=_context(),plan=_plan(),snapshot=None).accepted
    assert restored.snapshot().legacy_request_notes and restored.usage()==old.usage


@pytest.fixture
def real_reference_tool(tmp_path):
    imported=_sample_result(tmp_path)
    texts=('SVR 2015 wind forecasting speed prediction.', 'LSTM 2020 air temperature regression.')
    chunks=tuple(replace(chunk,section_type='bibliography',content_kind='body',front_matter_type=None,
                         section='References',raw_text=text) for chunk,text in zip(imported.chunks,texts,strict=True))
    db=MetadataDatabase(tmp_path/'references.sqlite3')
    db.import_chunking_result(replace(imported,chunks=chunks))
    executor=ResearchToolExecutor(retrieval_service=RetrievalService(),database=db)
    context=replace(_single_context(),target_paper_ids=(chunks[0].paper_id,))
    return executor,context,chunks,db


def test_real_fts5_2_by_2_rca_and_non_phrase_hints(real_reference_tool):
    executor,context,chunks,db=real_reference_tool
    matrix=[]
    for query in ('SVR 2015 wind','LSTM 2020 air'):
        for lexical in (('2015 SVR wind speed prediction',),('2020 LSTM air temperature',)):
            action=SearchReferencesAction(chunks[0].paper_id,query,lexical)
            result=executor.execute(action,context=context)
            assert result.status is ToolStatus.OK
            effective=result.metadata['canonical_effective_requests'][0]
            assert '"2015 SVR wind speed prediction"' not in effective['lexical_query']
            candidates=tuple(e.seed_chunk_ids for e in result.payload.evidence_candidates)
            # Actual FTS receipt is sufficient to replay exactly, including rank.
            rows=db.search_compiled_lexical(effective['lexical_query'],paper_ids=effective['paper_ids'],top_k=effective['top_k'],bibliography=True)
            assert candidates==tuple((row['chunk_id'],) for row in rows)
            delta=RegistryDelta(added_ids=(),reused_ids=())
            observation=build_observation(iteration=1,result=result,delta=delta,action=action)
            restored=load_contract(dump_contract((observation,),tuple[ResearchObservation,...]),tuple[ResearchObservation,...])[0]
            assert restored.request_contract_version=='search-v1'
            assert restored.raw_action_arguments['query']==query
            assert restored.canonical_effective_requests[0]==effective
            assert restored.request_identities==(effective['request_identity'],)
            assert restored.request_executions[0]['executed'] and restored.request_executions[0]['successful']
            step=_step_audit(restored,{})
            assert step.canonical_effective_requests==restored.canonical_effective_requests
            assert step.request_identities==restored.request_identities
            matrix.append((effective['request_identity'],candidates))
    assert matrix[0]==matrix[2] and matrix[1]==matrix[3]
    assert matrix[0] != matrix[1]
    assert matrix[0][1]==((chunks[0].chunk_id,),)
    assert matrix[1][1]==((chunks[1].chunk_id,),)


@pytest.mark.parametrize('entered', [False,True])
def test_failed_execution_receipt_is_explicit_and_retry_not_poisoned(real_reference_tool,entered):
    executor,context,chunks,db=real_reference_tool
    action=SearchReferencesAction(chunks[0].paper_id,'ignored',('SVR',),('R1',))
    if entered:
        def fail(*args,**kwargs): raise RuntimeError('fts transient failure')
        db.search_compiled_lexical=fail
    else:
        db.get_document=lambda _:None
    requests=executor.compile_requests(action,context=context)
    tracker=BudgetAndGuardTracker(AgenticResearchSettings())
    args=dict(context=context,plan=_plan(),snapshot=None,canonical_requests=requests)
    assert tracker.validate_action(action,**args).accepted
    result=executor.execute(action,context=context,canonical_requests=requests)
    receipt=result.metadata['request_executions'][0]
    assert receipt['executed'] is entered and receipt['failed'] and not receipt['successful']
    tracker.record_tool_call(action,canonical_requests=requests,result=result)
    assert tracker.validate_action(action,**args).accepted
    assert not tracker.snapshot().executed_action_fingerprints
    assert tracker.usage().retrieval_queries == int(entered)


def test_formal_adapter_guard_and_retriever_share_compiled_object():
    class Retriever:
        def retrieve(self,query,**kwargs):
            self.query=query; self.kwargs=kwargs
            return _snapshot('A',with_chunk=True).retrieval
    retriever=Retriever()
    adapter=FormalKnowledgeBaseAdapter(database=SimpleNamespace(get_document=lambda _:{}),retriever=retriever)
    executor=ResearchToolExecutor(retrieval_service=RetrievalService(formal_adapter=adapter),database=object(),top_k_per_paper=3)
    action=SearchPapersAction((PaperSearchSpec('A','Focused   query',('Wind Speed!',),requirement_ids=('R1',)),))
    requests=executor.compile_requests(action,context=_single_context())
    tracker=BudgetAndGuardTracker(AgenticResearchSettings(top_k_per_paper=3))
    assert tracker.validate_action(action,context=_single_context(),plan=_plan(),snapshot=None,canonical_requests=requests).accepted
    result=executor.execute(action,context=_single_context(),canonical_requests=requests)
    assert retriever.kwargs['canonical_request'] is requests[0]
    assert retriever.query == requests[0].semantic_query
    assert retriever.kwargs['lexical_terms']==requests[0].lexical_terms
    assert result.metadata['canonical_effective_requests']==request_trace(requests)
    tracker.record_tool_call(action,canonical_requests=requests,result=result)
    assert tracker.snapshot().executed_paper_query_fingerprints==(requests[0].identity,)


def test_requests_roundtrip_and_no_llm_no_mutation():
    action=SearchPapersAction((PaperSearchSpec('A','Focus',('wind speed',),requirement_ids=('R1',)),))
    before=deepcopy(action)
    requests=compile_action_requests(action,context=_single_context())
    restored=load_contract(dump_contract(requests,tuple[CanonicalSearchRequest,...]),tuple[CanonicalSearchRequest,...])
    assert restored==requests and restored[0].identity==requests[0].identity
    assert action==before
    legacy=TypeAdapter(ResearchObservation).validate_python(dict(iteration=1,action_name='search_references',tool_status='empty',added_evidence_ids=[],reused_evidence_ids=[]))
    assert legacy.request_contract_version=='legacy' and legacy.request_identities==()


def test_reference_guard_ignores_raw_query_but_allows_changed_fts():
    tracker=BudgetAndGuardTracker(AgenticResearchSettings())
    first=SearchReferencesAction('A','first query',('wind speed',),('R1',))
    args=dict(context=_single_context(),plan=_plan(),snapshot=None)
    assert tracker.validate_action(first,**args).accepted
    tracker.record_tool_call(first)
    same=replace(first,query='entirely different',lexical_terms=('SPEED!','wind'))
    assert tracker.validate_action(same,**args).code=='DuplicateAction'
    assert tracker.validate_action(replace(first,lexical_terms=('SVR',)),**args).accepted


def test_kb_scope_and_actual_input_receipt():
    from test_agentic_decision_tools import _GlobalRetriever
    backend=_GlobalRetriever()
    snapshot=CorpusScopeSnapshot('snapshot',('B','A'),'2026-10-02T00:00:00Z')
    executor=ResearchToolExecutor(retrieval_service=RetrievalService(),database=object(),global_retriever=backend,top_k_per_paper=2)
    action=SearchKnowledgeBaseAction('Dense   focus',('Wind SPEED!',))
    requests=executor.compile_requests(action,context=_kb_context(),corpus_snapshot=snapshot)
    result=executor.execute(action,context=_kb_context(),corpus_snapshot=snapshot,canonical_requests=requests)
    assert result.status is ToolStatus.OK
    assert backend.kwargs['paper_ids']==('A','B')
    assert backend.kwargs['lexical_terms']==('speed','wind')
    assert backend.kwargs['result_limit']==2
    assert backend.kwargs['canonical_request'] is requests[0]
    assert result.metadata['canonical_effective_requests']==request_trace(requests)
    assert result.metadata['request_executions'][0]['executed']


def test_partial_batch_only_successful_request_is_deduplicated():
    class Adapter:
        def search(self,search,**kwargs):
            if search.paper_id=='B': raise RuntimeError('backend failed')
            return _snapshot('A',with_chunk=True)
    executor=ResearchToolExecutor(retrieval_service=RetrievalService(formal_adapter=Adapter()),database=object())
    action=SearchPapersAction(tuple(PaperSearchSpec(p,'focus',('x',),requirement_ids=('R1',)) for p in ('A','B')))
    requests=executor.compile_requests(action,context=_context())
    tracker=BudgetAndGuardTracker(AgenticResearchSettings())
    args=dict(context=_context(),plan=_plan(),snapshot=None)
    assert tracker.validate_action(action,canonical_requests=requests,**args).accepted
    result=executor.execute(action,context=_context(),canonical_requests=requests)
    assert result.status is ToolStatus.PARTIAL
    assert [r['successful'] for r in result.metadata['request_executions']]==[True,False]
    tracker.record_tool_call(action,canonical_requests=requests,result=result)
    assert tracker.validate_action(SearchPapersAction((action.searches[0],)),canonical_requests=requests[:1],**args).code=='DuplicatePaperQuery'
    assert tracker.validate_action(SearchPapersAction((action.searches[1],)),canonical_requests=requests[1:],**args).accepted


def test_compiled_workspace_tokens_preserve_chinese_index_semantics():
    request=paper(terms=('风速 预测',),source='workspace',workspace_id='W')
    assert '风' in request.lexical_terms and '风速' in request.lexical_terms
    assert request.lexical_query==' '.join(request.lexical_terms)


def test_empty_scoped_fts_does_not_turn_into_global_search(real_reference_tool):
    executor,context,chunks,db=real_reference_tool
    assert db.search_compiled_lexical('"svr"',paper_ids=(),top_k=3,bibliography=True)==()


def test_request_contract_rejects_uncompiled_fields():
    with pytest.raises(ValueError): replace(reference(),lexical_query='"wind speed"')
    with pytest.raises(ValueError): replace(paper(),top_k=0)


def test_legacy_pending_action_at_execute_edge_is_compiled(real_reference_tool):
    from paperbase.agentic.graph import _execute_tool
    from paperbase.agentic.contracts import AgentAction, QueryContext, RegistryState
    executor,context,chunks,db=real_reference_tool
    action=SearchReferencesAction(chunks[0].paper_id,'raw query',('SVR',),('R1',))
    state=dict(action=dump_contract(action,AgentAction),query_context=dump_contract(context,QueryContext),
               registry_state=dump_contract(RegistryState(),RegistryState),
               guard_state=dump_contract(GuardState(),GuardState))
    runtime=SimpleNamespace(context=SimpleNamespace(settings=AgenticResearchSettings(),tool_executor=executor))
    update=_execute_tool(state,runtime)
    result=load_contract(update['tool_result'],ToolResult)
    assert result.status is ToolStatus.OK
    assert result.metadata['canonical_effective_requests'][0]['lexical_query']=='"svr"'
    restored=load_contract(update['guard_state'],GuardState)
    assert restored.executed_paper_query_fingerprints==(result.metadata['canonical_effective_requests'][0]['request_identity'],)
