"""Agentic RetrievalService 的请求契约与显式来源分发测试。"""

from __future__ import annotations

import unittest

from paperbase.retrieval import (
    PaperSearchSpec,
    QueryOrigin,
    RetrievalService,
    RetrievalSource,
    SearchPapersRequest,
)


class _RecordingAdapter:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[PaperSearchSpec, QueryOrigin, int | None]] = []

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> dict[str, str]:
        self.calls.append((search, query_origin, result_limit))
        return {"adapter": self.label, "paper_id": search.paper_id}


class _HeterogeneousAdapter(_RecordingAdapter):
    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> dict[str, str]:
        self.calls.append((search, query_origin, result_limit))
        if search.paper_id == "paper-broken":
            raise RuntimeError("simulated isolated failure")
        return {
            "adapter": self.label,
            "paper_id": search.paper_id,
            "query": search.query,
        }


class RetrievalContractsTests(unittest.TestCase):
    def test_search_spec_normalizes_and_deduplicates_text_fields(self) -> None:
        spec = PaperSearchSpec(
            paper_id=" paper-a ",
            query="  compare   method ",
            lexical_terms=(" GNN ", "GNN", "message passing"),
            evidence_targets=frozenset({" method "}),
        )

        self.assertEqual(spec.paper_id, "paper-a")
        self.assertEqual(spec.query, "compare method")
        self.assertEqual(spec.lexical_terms, ("GNN", "message passing"))
        self.assertEqual(spec.evidence_targets, frozenset({"method"}))

    def test_batch_request_rejects_duplicate_paper_ids_and_invalid_limit(self) -> None:
        first = PaperSearchSpec(paper_id="paper-a", query="method")
        second = PaperSearchSpec(paper_id="paper-a", query="results")
        with self.assertRaisesRegex(ValueError, "duplicate paper_id"):
            SearchPapersRequest(
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                searches=(first, second),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            SearchPapersRequest(
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                searches=(first,),
                top_k_per_paper=0,
            )


class RetrievalServiceTests(unittest.TestCase):
    def test_heterogeneous_batch_uses_each_query_and_per_paper_limit(self) -> None:
        adapter = _HeterogeneousAdapter("formal")
        service = RetrievalService(formal_adapter=adapter)
        request = SearchPapersRequest(
            query_origin=QueryOrigin.AGENT_SUPPLIED,
            searches=(
                PaperSearchSpec(
                    paper_id="paper-a",
                    query="A encoder mechanism",
                    lexical_terms=("A-Encoder",),
                ),
                PaperSearchSpec(
                    paper_id="paper-b",
                    query="B temporal decoder",
                    lexical_terms=("B-Decoder",),
                ),
            ),
            top_k_per_paper=3,
        )

        executions = service.search_papers(
            request,
            source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
        )

        self.assertEqual(
            [call[0].query for call in adapter.calls],
            ["A encoder mechanism", "B temporal decoder"],
        )
        self.assertEqual([call[2] for call in adapter.calls], [3, 3])
        self.assertEqual(
            [execution.result["query"] for execution in executions],  # type: ignore[index]
            ["A encoder mechanism", "B temporal decoder"],
        )
        self.assertTrue(all(execution.succeeded for execution in executions))

    def test_batch_preserves_successes_when_one_paper_fails(self) -> None:
        adapter = _HeterogeneousAdapter("formal")
        service = RetrievalService(formal_adapter=adapter)
        request = SearchPapersRequest(
            query_origin=QueryOrigin.AGENT_SUPPLIED,
            searches=(
                PaperSearchSpec(paper_id="paper-a", query="A method"),
                PaperSearchSpec(paper_id="paper-broken", query="Broken method"),
                PaperSearchSpec(paper_id="paper-c", query="C method"),
            ),
        )

        executions = service.search_papers(
            request,
            source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
        )

        self.assertEqual(len(adapter.calls), 3)
        self.assertTrue(executions[0].succeeded)
        self.assertFalse(executions[1].succeeded)
        self.assertEqual(executions[1].error_code, "RuntimeError")
        self.assertTrue(executions[2].succeeded)
        self.assertEqual(executions[2].result["paper_id"], "paper-c")  # type: ignore[index]

    def test_explicit_source_dispatch_keeps_formal_and_workspace_separate(self) -> None:
        formal = _RecordingAdapter("formal")
        workspace = _RecordingAdapter("workspace")
        service = RetrievalService(formal_adapter=formal)
        service.register_workspace("ws-a", workspace)
        spec = PaperSearchSpec(paper_id="paper-a", query="method")

        formal_result = service.search_paper(
            spec,
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
            source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
        )
        workspace_result = service.search_paper(
            spec,
            query_origin=QueryOrigin.AGENT_SUPPLIED,
            source=RetrievalSource.WORKSPACE,
            workspace_id="ws-a",
            result_limit=3,
        )

        self.assertEqual(formal_result["adapter"], "formal")
        self.assertEqual(workspace_result["adapter"], "workspace")
        self.assertEqual(formal.calls[0][1], QueryOrigin.INITIAL_UNDERSTANDING)
        self.assertEqual(workspace.calls[0][1], QueryOrigin.AGENT_SUPPLIED)
        self.assertEqual(workspace.calls[0][2], 3)

    def test_workspace_source_requires_registered_workspace(self) -> None:
        service = RetrievalService()
        spec = PaperSearchSpec(paper_id="paper-a", query="method")
        with self.assertRaisesRegex(ValueError, "workspace_id is required"):
            service.search_paper(
                spec,
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                source=RetrievalSource.WORKSPACE,
            )
        with self.assertRaisesRegex(LookupError, "not registered"):
            service.search_paper(
                spec,
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                source=RetrievalSource.WORKSPACE,
                workspace_id="missing",
            )


if __name__ == "__main__":
    unittest.main()
