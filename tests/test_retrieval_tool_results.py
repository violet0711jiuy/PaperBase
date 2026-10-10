"""RetrievedEvidence、逐论文状态和 ToolResult 聚合语义测试。"""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from paperbase.generation.section_expander import EvidenceUnit, ExpansionResult
from paperbase.retrieval import (
    PaperBatchExecution,
    PaperSearchSpec,
    PaperRetrievalSnapshot,
    QueryOrigin,
    RetrievedChunk,
    ToolName,
    ToolResult,
    ToolStatus,
)
from paperbase.retrieval.hybrid_retriever import RetrievalResult, RetrievalSourceMatch
from paperbase.retrieval.query_rewriter import QueryRewritePlan
from paperbase.retrieval.tool_results import build_search_papers_tool_result


def _retrieval(paper_id: str, *, with_chunk: bool) -> RetrievalResult:
    chunks = ()
    if with_chunk:
        chunks = (
            RetrievedChunk(
                rank=1,
                chunk_id=f"{paper_id}-chunk-1",
                vector_id=1,
                paper_id=paper_id,
                paper_title=f"Title {paper_id}",
                section="3 Method",
                content_kind="body",
                front_matter_type=None,
                section_type="content",
                page_start=3,
                page_end=3,
                raw_text="First line.\n\nSecond line.",
                pre_rerank_rank=1,
                rerank_score=0.91,
                fused_score=0.02,
                source_matches=(
                    RetrievalSourceMatch(
                        route="dense_resolved",
                        query="method",
                        rank=1,
                        raw_score=0.8,
                        effective_weight=1.0,
                    ),
                ),
                section_id=f"{paper_id}-section-method",
            ),
        )
    return RetrievalResult(
        query="method",
        rewrite_plan=QueryRewritePlan(original_query="method", resolved_query="method"),
        reranking_status="success",
        chunks=chunks,
    )


def _snapshot(paper_id: str, *, with_chunk: bool, expanded: bool = False) -> PaperRetrievalSnapshot:
    retrieval = _retrieval(paper_id, with_chunk=with_chunk)
    expansion = None
    if expanded and with_chunk:
        chunk = retrieval.chunks[0]
        expansion = ExpansionResult(
            content_evidence=(
                EvidenceUnit(
                    evidence_id="E1",
                    kind="content",
                    paper_id=paper_id,
                    paper_title=chunk.paper_title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    seed_chunk_ids=(chunk.chunk_id,),
                    chunk_ids=(chunk.chunk_id, f"{paper_id}-chunk-2"),
                    text="First line.\n\nExpanded neighbor.",
                    token_count=8,
                    section_id=chunk.section_id,
                ),
            ),
            bibliography_evidence=(),
        )
    return PaperRetrievalSnapshot(
        paper_id=paper_id,
        retrieval=retrieval,
        expansion=expansion,
    )


def _success(paper_id: str, *, with_chunk: bool, expanded: bool = False) -> PaperBatchExecution:
    return PaperBatchExecution(
        search=PaperSearchSpec(
            paper_id=paper_id, query=f"{paper_id} method", lexical_terms=(paper_id,),
            evidence_targets=frozenset({"method"}), requirement_ids=("R1",),
        ),
        result=_snapshot(paper_id, with_chunk=with_chunk, expanded=expanded),
    )


def _failure(paper_id: str, error_code: str) -> PaperBatchExecution:
    return PaperBatchExecution(
        search=PaperSearchSpec(paper_id=paper_id, query="method"),
        error_code=error_code,
        error_message="simulated",
    )


class RetrievalToolResultTests(unittest.TestCase):
    def test_expanded_evidence_preserves_full_provenance_without_registry_id(self) -> None:
        result = build_search_papers_tool_result(
            (_success("paper-a", with_chunk=True, expanded=True),),
            query_origin=QueryOrigin.AGENT_SUPPLIED,
        )

        evidence = result.payload.paper_results[0].evidence_candidates[0]
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(evidence.paper_id, "paper-a")
        self.assertEqual(evidence.seed_chunk_ids, ("paper-a-chunk-1",))
        self.assertEqual(
            evidence.source_chunk_ids,
            ("paper-a-chunk-1", "paper-a-chunk-2"),
        )
        self.assertEqual(evidence.section_id, "paper-a-section-method")
        self.assertEqual(evidence.content, "First line.\n\nExpanded neighbor.")
        self.assertFalse(hasattr(evidence, "evidence_id"))
        self.assertEqual(result.metadata["query_origin"], "agent_supplied")

    def test_status_aggregation_distinguishes_empty_partial_error_and_unavailable(self) -> None:
        cases = (
            (( _success("a", with_chunk=False), _success("b", with_chunk=True)), ToolStatus.OK),
            (( _failure("a", "RuntimeError"), _success("b", with_chunk=True)), ToolStatus.PARTIAL),
            (( _success("a", with_chunk=False), _success("b", with_chunk=False)), ToolStatus.EMPTY),
            (( _failure("a", "RuntimeError"), _failure("b", "ValueError")), ToolStatus.ERROR),
            (( _failure("a", "LookupError"), _failure("b", "FileNotFoundError")), ToolStatus.UNAVAILABLE),
        )
        for executions, expected in cases:
            with self.subTest(expected=expected):
                result = build_search_papers_tool_result(
                    executions,
                    query_origin=QueryOrigin.AGENT_SUPPLIED,
                )
                self.assertEqual(result.status, expected)

    def test_unexpanded_chunk_keeps_section_id_and_retrieval_routes(self) -> None:
        result = build_search_papers_tool_result(
            (_success("paper-a", with_chunk=True),),
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
        )

        evidence = result.payload.paper_results[0].evidence_candidates[0]
        self.assertEqual(evidence.section_id, "paper-a-section-method")
        self.assertEqual(evidence.retrieval_metadata["rank"], 1)
        self.assertEqual(
            evidence.retrieval_metadata["source_matches"][0]["route"],  # type: ignore[index]
            "dense_resolved",
        )

    def test_search_intent_survives_success_and_error_outcomes(self) -> None:
        executions = (
            _success("paper-a", with_chunk=True),
            PaperBatchExecution(
                    search=PaperSearchSpec(
                        paper_id="paper-b", query="paper-b limitation",
                        lexical_terms=("limitation",),
                        evidence_targets=frozenset({"claim"}), requirement_ids=("R2",),
                    ),
                    error_code="RuntimeError",
            ),
        )
        result = build_search_papers_tool_result(
            executions,
            query_origin=QueryOrigin.AGENT_SUPPLIED,
        )

        success, failure = result.payload.paper_results
        self.assertEqual(success.evidence_targets, frozenset({"method"}))
        self.assertEqual(success.requirement_ids, ("R1",))
        self.assertEqual(failure.evidence_targets, frozenset({"claim"}))
        self.assertEqual(failure.requirement_ids, ("R2",))
        self.assertIs(success.search, executions[0].search)
        self.assertIs(failure.search, executions[1].search)

    def test_tool_name_must_match_payload_kind(self) -> None:
        @dataclass(frozen=True)
        class _WrongPayload:
            kind: str = "get_paper_overview"

        with self.assertRaisesRegex(ValueError, "does not match payload kind"):
            ToolResult(
                status=ToolStatus.OK,
                tool_name=ToolName.SEARCH_PAPERS,
                payload=_WrongPayload(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
