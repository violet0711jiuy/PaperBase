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
