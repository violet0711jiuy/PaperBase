from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from paperbase.agentic.contracts import (
    BudgetUsage,
    CorpusScopeSnapshot,
    EvidenceRecord,
    EvidenceTarget,
    GetSectionContextAction,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
    ScopeSource,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)
from paperbase.agentic.decision import DecisionState, LLMDecisionAgent
from paperbase.agentic.tools import ResearchToolExecutor
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchSpec,
    QueryOrigin,
    RetrievedEvidence,
    SearchPapersPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


def _single_context():
    return QueryContext(
        original_query="method",
        resolved_query="method",
        intent=Intent.RETRIEVE,
        scope=QueryScope.SINGLE_PAPER,
        target_paper_ids=("A",),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _kb_context():
    return QueryContext(
        original_query="all methods",
        resolved_query="all methods",
        intent=Intent.SYNTHESIZE,
        scope=QueryScope.KNOWLEDGE_BASE,
        target_paper_ids=(),
        scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
        retrieval_goal=ResearchGoal.SYNTHESIS_SUPPORT,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _requirement_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="method",
        requirements=(
            ResearchRequirement(
                "R1",
                "识别论文采用的方法",
                required=True,
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
            ),
        ),
    )


class _QueuedClient:
    def __init__(self, values):
        self.values = list(values)

    def complete_json(self, **kwargs):
        return self.values.pop(0)


class _RetrievalService:
    def __init__(self):
        self.call = None

    def search_papers(self, request, **kwargs):
        self.call = (request, kwargs)
        return ToolResult(
            status=ToolStatus.EMPTY,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(()),
        )


class _Database:
    def get_document(self, paper_id):
        return {"paper_id": paper_id}

    def search_compiled_lexical(self, terms, *, paper_ids, top_k, bibliography=False):
        paper_id = paper_ids[0]
        return (
            {
                "paper_id": paper_id,
                "paper_title": "Paper A",
                "chunk_id": "ref_1",
                "section_id": "refs",
                "section": "References",
                "page_start": 10,
                "page_end": 10,
                "raw_text": "A cited work.",
                "bm25_score": -1.5,
            },
        )

    def get_section(self, section_id):
        return {"paper_id": "A", "section_title": "Method"}

    def list_content_chunks_in_section_id(self, *, paper_id, section_id):
        return (
            {
                "paper_title": "Paper A",
                "chunk_id": "c1",
                "page_start": 2,
                "page_end": 2,
                "raw_text": "first",
            },
            {
                "paper_title": "Paper A",
                "chunk_id": "c2",
                "page_start": 3,
                "page_end": 3,
                "raw_text": "second",
            },
        )


class _GlobalRetriever:
    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, **kwargs):
        self.kwargs = kwargs
        chunk = SimpleNamespace(
            section_type="content",
            paper_id="A",
            paper_title="Paper A",
            chunk_id="c1",
            section_id="method",
            section="Method",
            page_start=2,
            page_end=2,
            raw_text="method evidence",
            rerank_score=0.8,
            fused_score=0.1,
            rank=1,
        )
        return SimpleNamespace(chunks=(chunk,))


class _FailingDatabase(_Database):
    def search_compiled_lexical(self, terms, *, paper_ids, top_k, bibliography=False):
        paper_id = paper_ids[0]
        raise RuntimeError("fts unavailable")


class AgenticDecisionAndToolTests(unittest.TestCase):
    def test_decision_repair_rejects_duplicate_paper_in_one_batch(self) -> None:
        def search(query: str) -> dict[str, object]:
            return {
                "paper_id": "A",
                "query": query,
                "lexical_terms": ["method"],
                "evidence_targets": ["method"],
                "requirement_ids": ["R1"],
            }

        invalid = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "searches": [search("method"), search("method details")],
                },
            }
        )
        valid = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "searches": [search("method")],
                },
            }
        )
        plan = _requirement_plan()

        decision = LLMDecisionAgent(_QueuedClient((invalid, valid))).decide(
            DecisionState(_single_context(), plan, (), (), BudgetUsage())
        )

        self.assertEqual(decision.llm_calls, 2)
        self.assertEqual(len(decision.action.searches), 1)

    def test_decision_repair_rejects_agent_supplied_top_k(self) -> None:
        invalid = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "searches": [
                        {
                            "paper_id": "A",
                            "query": "method",
                            "lexical_terms": ["method"],
                            "evidence_targets": ["method"],
                            "requirement_ids": ["R1"],
                        }
                    ],
                    "top_k": 100,
                }
            }
        )
        valid = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "searches": [
                        {
                            "paper_id": "A",
                            "query": "method",
                            "lexical_terms": ["method"],
                            "evidence_targets": ["method"],
                            "requirement_ids": ["R1"],
                        }
                    ],
                }
            }
        )
        context = _single_context()
        plan = _requirement_plan()
        decision = LLMDecisionAgent(_QueuedClient((invalid, valid))).decide(
            DecisionState(context, plan, (), (), BudgetUsage())
        )
        self.assertEqual(decision.llm_calls, 2)
        self.assertIsInstance(decision.action, SearchPapersAction)

    def test_decision_repair_rejects_unknown_section_reference(self) -> None:
        unknown_section = json.dumps(
            {
                "decision": {
                    "action": "get_section_context",
                    "paper_id": "A",
                    "section_id": "invented",
                    "requirement_ids": ["R1"],
                }
            }
        )
        valid_search = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "searches": [
                        {
                            "paper_id": "A",
                            "query": "method",
                            "lexical_terms": ["method"],
                            "evidence_targets": ["method"],
                            "requirement_ids": ["R1"],
                        }
                    ],
                }
            }
        )
        plan = _requirement_plan()
        decision = LLMDecisionAgent(
            _QueuedClient((unknown_section, valid_search))
        ).decide(DecisionState(_single_context(), plan, (), (), BudgetUsage()))
        self.assertEqual(decision.llm_calls, 2)
        self.assertIsInstance(decision.action, SearchPapersAction)

    def test_decision_accepts_section_id_exposed_by_evidence(self) -> None:
        known_section = json.dumps(
            {
                "decision": {
                    "action": "get_section_context",
                    "paper_id": "A",
                    "section_id": "method",
                    "requirement_ids": ["R1"],
                }
            }
        )
        plan = _requirement_plan()
        record = EvidenceRecord(
            "E1",
            RetrievedEvidence(
                source_type=EvidenceSourceType.CONTENT,
                paper_id="A",
                section_id="method",
                seed_chunk_ids=("c1",),
                source_chunk_ids=("c1",),
                content="method evidence",
            ),
            ("method",),
            1,
            1,
        )
        decision = LLMDecisionAgent(_QueuedClient((known_section,))).decide(
            DecisionState(_single_context(), plan, (record,), (), BudgetUsage())
        )
        self.assertEqual(decision.llm_calls, 1)
        self.assertIsInstance(decision.action, GetSectionContextAction)

    def test_search_papers_injects_runtime_top_k_and_agent_origin(self) -> None:
        retrieval = _RetrievalService()
        executor = ResearchToolExecutor(
            retrieval_service=retrieval,
            database=_Database(),
            top_k_per_paper=3,
        )
        executor.execute(
            SearchPapersAction((PaperSearchSpec("A", "method"),)),
            context=_single_context(),
        )
        request, _ = retrieval.call
        self.assertEqual(request.top_k_per_paper, 3)
        self.assertEqual(request.query_origin, QueryOrigin.AGENT_SUPPLIED)

    def test_reference_and_section_tools_return_typed_evidence(self) -> None:
        executor = ResearchToolExecutor(
            retrieval_service=_RetrievalService(), database=_Database()
        )
        references = executor.execute(
            SearchReferencesAction("A", "Transformer citation", ("Transformer",)),
            context=_single_context(),
        )
        self.assertEqual(references.tool_name, ToolName.SEARCH_REFERENCES)
        self.assertEqual(
            references.payload.evidence_candidates[0].source_type,
            EvidenceSourceType.BIBLIOGRAPHY,
        )
        section = executor.execute(
            GetSectionContextAction("A", "method"), context=_single_context()
        )
        self.assertEqual(section.tool_name, ToolName.GET_SECTION_CONTEXT)
        self.assertEqual(section.payload.evidence_candidates[0].source_chunk_ids, ("c1", "c2"))
        from paperbase.retrieval.contracts import ScoreKind, ScoreDirection
        reference = references.payload.evidence_candidates[0]
        self.assertIs(reference.score_kind, ScoreKind.BM25_FTS5)
        self.assertIs(reference.score_direction, ScoreDirection.LOWER_IS_BETTER)
        self.assertEqual(reference.retrieval_rank, 1)
        self.assertIsNone(reference.score)
        non_ranked = section.payload.evidence_candidates[0]
        self.assertIs(non_ranked.score_kind, ScoreKind.NON_RANKED)
        self.assertIsNone(non_ranked.raw_score)
        self.assertIsNone(non_ranked.retrieval_rank)

    def test_global_search_is_frozen_to_corpus_snapshot(self) -> None:
        retriever = _GlobalRetriever()
        executor = ResearchToolExecutor(
            retrieval_service=_RetrievalService(),
            database=_Database(),
            global_retriever=retriever,
        )
        snapshot = CorpusScopeSnapshot.create(("A", "B"))
        result = executor.execute(
            SearchKnowledgeBaseAction("graph methods", ("graph",)),
            context=_kb_context(),
            corpus_snapshot=snapshot,
        )
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(retriever.kwargs["paper_ids"], ("A", "B"))
        self.assertEqual(result.metadata["corpus_snapshot_id"], snapshot.snapshot_id)

    def test_reference_backend_failure_becomes_tool_error(self) -> None:
        executor = ResearchToolExecutor(
            retrieval_service=_RetrievalService(), database=_FailingDatabase()
        )
        result = executor.execute(
            SearchReferencesAction("A", "Transformer", ("Transformer",)),
            context=_single_context(),
        )
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
