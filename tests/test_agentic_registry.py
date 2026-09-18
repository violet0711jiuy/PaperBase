from __future__ import annotations

import unittest

from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchOutcome,
    PaperSearchStatus,
    ReferenceSearchPayload,
    RetrievedEvidence,
    SearchPapersPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


def _evidence(source_type=EvidenceSourceType.CONTENT):
    return RetrievedEvidence(
        source_type=source_type,
        paper_id="paper_a",
        seed_chunk_ids=("chunk_1",),
        source_chunk_ids=("chunk_1", "chunk_2"),
        content="evidence text",
    )


class EvidenceRegistryTests(unittest.TestCase):
    def test_exact_duplicate_reuses_id_and_merges_queries(self) -> None:
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(
                (
                    PaperSearchOutcome(
                        paper_id="paper_a",
                        status=PaperSearchStatus.OK,
                        query="first query",
                        lexical_terms=(),
                        evidence_candidates=(_evidence(),),
                    ),
                )
            ),
        )
        registry = EvidenceRegistry()
        first = registry.register_tool_result(
            result, iteration=1, target_subtask_ids=("S1",)
        )
        second = registry.register_tool_result(
            result,
            iteration=2,
            supporting_queries=("second query",),
            target_subtask_ids=("S2",),
        )
        self.assertEqual(first.added_ids, ("E1",))
        self.assertEqual(second.reused_ids, ("E1",))
        self.assertEqual(registry.records[0].last_seen_iteration, 2)
        self.assertIn("second query", registry.records[0].supporting_queries)
        self.assertEqual(registry.records[0].originating_tools, ("search_papers",))
        self.assertEqual(registry.records[0].target_subtask_ids, ("S1", "S2"))

    def test_bibliography_uses_independent_r_numbering(self) -> None:
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_REFERENCES,
            payload=ReferenceSearchPayload(
                paper_id="paper_a",
                query="citation",
                evidence_candidates=(_evidence(EvidenceSourceType.BIBLIOGRAPHY),),
            ),
        )
        delta = EvidenceRegistry().register_tool_result(result, iteration=1)
        self.assertEqual(delta.added_ids, ("R1",))


if __name__ == "__main__":
    unittest.main()
