"""Workspace Adapter 的论文隔离与 Facade 接入测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

from paperbase.retrieval import (
    PaperSearchSpec,
    QueryOrigin,
    RetrievalService,
    RetrievalSource,
    WorkspacePaperAdapter,
)
from paperbase.staging.sections import WorkspaceSectionSnapshot


class _RecordingWorkspaceRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"query": query, **kwargs})
        return {"query": query, "scope": kwargs["trusted_scope"]}


class WorkspaceRetrievalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = WorkspaceSectionSnapshot(
            workspace_id="workspace-a",
            root_dir=Path("/temporary/workspace-a"),
            paper_id="paper-a",
            paper_title="Paper A",
            sections=(),
            chunks=(),
        )
        self.retriever = _RecordingWorkspaceRetriever()
        self.adapter = WorkspacePaperAdapter(
            snapshot=self.snapshot,
            retriever=self.retriever,  # type: ignore[arg-type]
        )

    def test_adapter_passes_program_owned_scope_to_workspace_retriever(self) -> None:
        result = self.adapter.search(
            PaperSearchSpec(paper_id="paper-a", query="encoder method"),
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
            result_limit=4,
        )

        scope = result["scope"]
        self.assertEqual(getattr(scope, "paper_id"), "paper-a")
        self.assertEqual(getattr(scope, "paper_title"), "Paper A")
        self.assertEqual(self.retriever.calls[0]["result_limit"], 4)

    def test_adapter_rejects_a_different_paper_id(self) -> None:
        with self.assertRaisesRegex(LookupError, "paper_id mismatch"):
            self.adapter.search(
                PaperSearchSpec(paper_id="paper-b", query="method"),
                query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
            )
        self.assertEqual(self.retriever.calls, [])

    def test_facade_routes_registered_workspace_without_formal_fallback(self) -> None:
        service = RetrievalService()
        service.register_workspace("workspace-a", self.adapter)

        result = service.search_paper(
            PaperSearchSpec(paper_id="paper-a", query="results"),
            query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
            source=RetrievalSource.WORKSPACE,
            workspace_id="workspace-a",
        )

        self.assertEqual(result["query"], "results")
        with self.assertRaisesRegex(LookupError, "Formal knowledge-base adapter"):
            service.search_paper(
                PaperSearchSpec(paper_id="paper-a", query="results"),
                query_origin=QueryOrigin.INITIAL_UNDERSTANDING,
                source=RetrievalSource.FORMAL_KNOWLEDGE_BASE,
            )


if __name__ == "__main__":
    unittest.main()
