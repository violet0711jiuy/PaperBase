"""Score provenance and cross-channel regressions, without LLM/model calls."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from paperbase.database import MetadataDatabase

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (EvidenceRelation, ResearchPlan, RegistryState,
    ResearchObservation, SearchReferencesAction)
from paperbase.agentic.decision import _visible_evidence
from paperbase.agentic.evidence_admission import EvidenceAdmissionFilter
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.agentic.observation import build_observation
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.agentic.tools import ResearchToolExecutor, _retrieved_chunk_to_evidence
from paperbase.retrieval.contracts import (RetrievedEvidence, EvidenceSourceType, ScoreKind,
    ScoreDirection, PaperBatchExecution, PaperSearchSpec, QueryOrigin)
from paperbase.retrieval.evidence_fingerprint import build_evidence_fingerprint
from paperbase.retrieval.tool_results import build_search_papers_tool_result
from test_agentic_phase3 import _record
from test_agentic_p0_action_control import _state
from test_agentic_decision_tools import _RetrievalService, _single_context
from test_retrieval_tool_results import _snapshot
from test_database import _sample_result


def reference_records():
    best = _record('REF1', 'A', relation=EvidenceRelation.SUPPORTS)
    best = replace(best, fingerprint='', evidence=replace(best.evidence, source_type=EvidenceSourceType.BIBLIOGRAPHY,
        raw_score=-20, score_kind=ScoreKind.BM25_FTS5, score_direction=ScoreDirection.LOWER_IS_BETTER,
        retrieval_rank=1))
    worse = replace(_record('REF2', 'A', relation=EvidenceRelation.SUPPORTS), fingerprint='', evidence=replace(best.evidence,
        raw_score=-0.01, retrieval_rank=2, seed_chunk_ids=('worse',), source_chunk_ids=('worse',)))
    return best, worse


def test_selector_and_decision_prefer_bm25_rank_not_larger_negative_value():
    best, worse = reference_records()
    plan = ResearchPlan('reference support')
    assert FinalEvidenceSelector(max_units=1).select(plan=plan, evidence=(worse,best)) == (best,)
    state = replace(_state(), evidence=(worse,best))
    assert _visible_evidence(state,limit=1) == (best,)


@pytest.mark.parametrize('raw_rrf,raw_reranker', [(0.001,1000), (9999,-1000)])
def test_heterogeneous_raw_scores_have_no_effect_on_selector_or_decision(raw_rrf,raw_reranker):
    base = _record('E1','A',relation=EvidenceRelation.SUPPORTS)
    ranked_first = replace(base,evidence=replace(base.evidence,raw_score=raw_rrf,
        score_kind=ScoreKind.RRF,retrieval_rank=1))
    ranked_second = replace(_record('E2','A',relation=EvidenceRelation.SUPPORTS),fingerprint='',evidence=replace(base.evidence,raw_score=raw_reranker,
        score_kind=ScoreKind.RERANKER,retrieval_rank=2,source_chunk_ids=('second',),seed_chunk_ids=('second',)))
    candidates=(ranked_second,ranked_first)
    assert FinalEvidenceSelector(max_units=1).select(plan=ResearchPlan('facts'), evidence=candidates) == (ranked_first,)
    assert _visible_evidence(replace(_state(),evidence=candidates),limit=1) == (ranked_first,)


def test_unknown_legacy_scores_do_not_create_cross_source_relevance():
    best,worse=reference_records()
    def legacy(record,value):
        e=RetrievedEvidence(record.evidence.source_type,'A',record.evidence.seed_chunk_ids,
            record.evidence.source_chunk_ids,record.evidence.content,score=value)
        return replace(record,evidence=e)
    # Missing ranks use stable evidence-ID ties; varying legacy values cannot change selection.
    first=(legacy(best,-999),legacy(worse,999))
    second=(legacy(best,999),legacy(worse,-999))
    select=lambda es:FinalEvidenceSelector(max_units=1).select(plan=ResearchPlan('facts'),evidence=es)[0].evidence_id
    assert select(first)==select(second)
    visible=lambda es:_visible_evidence(replace(_state(),evidence=es),limit=1)[0].evidence_id
    assert visible(first)==visible(second)


def test_bibliography_tool_uses_real_fts5_order_and_survives_admission_registry_path(tmp_path):
    imported=_sample_result(tmp_path)
    chunks=tuple(replace(chunk, section_type='bibliography',content_kind='body',front_matter_type=None,
        section='References',raw_text=text) for chunk,text in zip(imported.chunks,(
            'Graph WaveNet Graph WaveNet Graph WaveNet.', 'Graph WaveNet. ' + 'Other background. '*40),strict=True))
    database=MetadataDatabase(tmp_path/'scores.sqlite3')
    database.import_chunking_result(replace(imported,chunks=chunks))
    paper_id=chunks[0].paper_id
    context=replace(_single_context(),target_paper_ids=(paper_id,))
    executor=ResearchToolExecutor(retrieval_service=_RetrievalService(),database=database)
    action=SearchReferencesAction(paper_id,'citation',('Graph WaveNet',))
    result=executor.execute(action,context=context)
    evidence=result.payload.evidence_candidates
    assert len(evidence)==2
    assert evidence[0].raw_score < evidence[1].raw_score < 0
    assert evidence[0].seed_chunk_ids==(chunks[0].chunk_id,)
    assert [e.retrieval_rank for e in evidence]==[1,2]
    admitted=EvidenceAdmissionFilter().apply(result)
    assert admitted.report.accepted_candidates==2
    registry=EvidenceRegistry()
    delta=registry.register_tool_result(admitted.tool_result,action=action,iteration=1)
    assert len(delta.added_ids)==2
    observation=build_observation(iteration=1,result=admitted.tool_result,delta=delta,action=action,
                                  evidence_filter_report=admitted.report)
    restored=load_contract(dump_contract((observation,),tuple[ResearchObservation,...]),tuple[ResearchObservation,...])[0]
    first=restored.evidence_filter_report.decisions[0]
    assert (first.raw_score,first.score_kind,first.score_direction,first.retrieval_rank)==(
        evidence[0].raw_score,ScoreKind.BM25_FTS5,ScoreDirection.LOWER_IS_BETTER,1)


def test_workspace_bibliography_preserves_positive_bm25_direction():
    from test_ask_paper import _retriever, _snapshot as workspace_snapshot
    result=_retriever(workspace_snapshot(Path('/temporary/score_contract'))).retrieve('How does the method work?',result_limit=3)
    chunk=next(c for c in result.chunks if c.section_type=='bibliography')
    evidence=_retrieved_chunk_to_evidence(chunk)
    assert evidence.raw_score > 0
    assert evidence.score_kind is ScoreKind.BM25
    assert evidence.score_direction is ScoreDirection.HIGHER_IS_BETTER
    assert evidence.retrieval_rank==chunk.rank
    assert evidence.raw_score==chunk.source_matches[0].raw_score


def test_snapshot_expansion_inherits_best_seed_rank_without_max_mixed_scores():
    snapshot=_snapshot('A',with_chunk=True,expanded=True)
    first=replace(snapshot.retrieval.chunks[0],rerank_score=None,fused_score=0.01,rank=1)
    second=replace(first,chunk_id='A-chunk-2',rerank_score=100,rank=2)
    unit=replace(snapshot.expansion.content_evidence[0],seed_chunk_ids=(second.chunk_id,first.chunk_id))
    snapshot=replace(snapshot,retrieval=replace(snapshot.retrieval,chunks=(first,second)),
                     expansion=replace(snapshot.expansion,content_evidence=(unit,)))
    result=build_search_papers_tool_result((PaperBatchExecution(PaperSearchSpec('A','q'),snapshot),),
                                          query_origin=QueryOrigin.AGENT_SUPPLIED)
    e=result.payload.paper_results[0].evidence_candidates[0]
    assert (e.raw_score,e.score_kind,e.retrieval_rank)==(0.01,ScoreKind.RRF,1)
    assert e.retrieval_metadata['score_scope']=='best_seed_chunk'
    assert e.retrieval_metadata['score_seed_chunk_id']==first.chunk_id


def test_hybrid_bibliography_fused_zero_is_not_mislabeled_rrf_or_reranker():
    from paperbase.retrieval.hybrid_retriever import RetrievalSourceMatch
    chunk=_snapshot('A',with_chunk=True).retrieval.chunks[0]
    chunk=replace(chunk,section_type='bibliography',rerank_score=None,fused_score=0,
        source_matches=(RetrievalSourceMatch('bm25_bibliography','citation',1,-8,0,
            ScoreKind.BM25_FTS5,ScoreDirection.LOWER_IS_BETTER),))
    e=_retrieved_chunk_to_evidence(chunk)
    assert (e.raw_score,e.score_kind,e.score_direction)==(-8,ScoreKind.BM25_FTS5,ScoreDirection.LOWER_IS_BETTER)
    snapshot=_snapshot('A',with_chunk=True)
    snapshot=replace(snapshot,retrieval=replace(snapshot.retrieval,chunks=(chunk,)))
    result=build_search_papers_tool_result((PaperBatchExecution(PaperSearchSpec('A','q'),snapshot),),query_origin=QueryOrigin.AGENT_SUPPLIED)
    assert result.payload.paper_results[0].evidence_candidates[0].score_kind is ScoreKind.BM25_FTS5


@pytest.mark.parametrize('source',[EvidenceSourceType.SECTION_CONTEXT,EvidenceSourceType.OVERVIEW])
def test_non_ranked_evidence_enters_admission_registry_selector_and_decision(source):
    from test_agentic_evidence_admission import _result
    record=_record('E1','A',relation=EvidenceRelation.SUPPORTS)
    non_ranked=replace(record.evidence,source_type=source,raw_score=None,score=None,
        score_kind=ScoreKind.NON_RANKED,score_direction=ScoreDirection.NOT_APPLICABLE,retrieval_rank=None)
    record=replace(record,evidence=non_ranked,fingerprint='')
    assert EvidenceAdmissionFilter().apply(_result((non_ranked,))).report.accepted_candidates==1
    assert FinalEvidenceSelector(max_units=1).select(plan=ResearchPlan('facts'),evidence=(record,))==(record,)
    assert _visible_evidence(replace(_state(),evidence=(record,)),limit=1)==(record,)
    restored=TypeAdapter(RetrievedEvidence).validate_json(TypeAdapter(RetrievedEvidence).dump_json(non_ranked))
    assert restored==non_ranked


def test_new_score_checkpoint_roundtrip_and_explicit_legacy_unknown_migration():
    best,_=reference_records()
    state=RegistryState(records=(best,))
    dto=dump_contract(state,RegistryState)
    assert load_contract(dto,RegistryState)==state
    old=deepcopy(dto)
    evidence=old['records'][0]['evidence']
    for name in ('raw_score','score_kind','score_direction','retrieval_rank'):evidence.pop(name)
    evidence['score']=-7.2
    evidence['retrieval_metadata']={'rank':2,'reranking_status':'success'}
    restored=load_contract(old,RegistryState).records[0].evidence
    assert restored.score==-7.2 and restored.raw_score==-7.2
    assert restored.score_kind is ScoreKind.LEGACY_UNKNOWN
    assert restored.score_direction is ScoreDirection.NOT_APPLICABLE
    assert restored.retrieval_rank==2
    # Fingerprint/cache identity uses content/provenance, not retrieval score.
    assert build_evidence_fingerprint(best.evidence)==build_evidence_fingerprint(replace(best.evidence,raw_score=-100,retrieval_rank=9))


@pytest.mark.parametrize('kwargs',[
    dict(raw_score=1,score_kind=ScoreKind.BM25_FTS5,score_direction=ScoreDirection.HIGHER_IS_BETTER),
    dict(raw_score=0,score_kind=ScoreKind.NON_RANKED),
    dict(retrieval_rank=0),dict(raw_score=float('nan')),dict(raw_score=float('inf')),
])
def test_invalid_score_contracts_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        RetrievedEvidence(EvidenceSourceType.CONTENT,'A',('c',),('c',),'source',**kwargs)
