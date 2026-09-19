from __future__ import annotations

import unittest

from paperbase.agentic.contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRelation,
    EvidenceTarget,
)
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

    def test_registry_keeps_search_targets_gaps_and_stable_fingerprint(self) -> None:
        outcome = PaperSearchOutcome(
            paper_id="paper_a",
            status=PaperSearchStatus.OK,
            query="定位核心方法",
            lexical_terms=("encoder",),
            evidence_targets=frozenset({"method"}),
            missing_information=("编码器如何工作",),
            target_subtask_id="S-method",
            evidence_candidates=(_evidence(),),
        )
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload((outcome,)),
        )

        first_registry = EvidenceRegistry()
        first_registry.register_tool_result(result, iteration=1)
        record = first_registry.records[0]
        self.assertEqual(
            {target.value for target in record.requested_evidence_targets},
            {"method"},
        )
        self.assertEqual(record.requested_missing_information, ("编码器如何工作",))
        self.assertEqual(record.target_subtask_ids, ("S-method",))

        second_evidence = RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="paper_a",
            seed_chunk_ids=("chunk_1",),
            source_chunk_ids=("chunk_2", "chunk_1"),
            content="  EVIDENCE   text ",
            score=0.01,
        )
        second_result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(
                (
                    PaperSearchOutcome(
                        paper_id="paper_a",
                        status=PaperSearchStatus.OK,
                        query="另一轮检索",
                        lexical_terms=(),
                        evidence_candidates=(second_evidence,),
                    ),
                )
            ),
        )
        second_registry = EvidenceRegistry()
        second_registry.register_tool_result(second_result, iteration=9)
        self.assertEqual(second_registry.records[0].fingerprint, record.fingerprint)

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

    def test_analysis_coverage_is_persisted_on_record(self) -> None:
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(
                (
                    PaperSearchOutcome(
                        paper_id="paper_a",
                        status=PaperSearchStatus.OK,
                        query="方法",
                        lexical_terms=(),
                        evidence_targets=frozenset({"method"}),
                        missing_information=("核心机制",),
                        evidence_candidates=(_evidence(),),
                    ),
                )
            ),
        )
        registry = EvidenceRegistry()
        registry.register_tool_result(result, iteration=1)
        registry.apply_analyses(
            (
                EvidenceAnalysis(
                    evidence_id="E1",
                    status=EvidenceAnalysisStatus.QUALIFIED,
                    digest="方法摘要",
                    covered_targets=frozenset({EvidenceTarget.METHOD}),
                    addressed_information=("核心机制",),
                    claims=(
                        EvidenceClaim(
                            "E1:C1",
                            "该方法使用编码器。",
                            EvidenceRelation.SUPPORTS,
                            EvidenceOwnership.THIS_PAPER,
                            "原文直接陈述。",
                            "evidence text",
                        ),
                    ),
                ),
            )
        )

        record = registry.records[0]
        self.assertEqual(record.covered_targets, frozenset({EvidenceTarget.METHOD}))
        self.assertEqual(record.addressed_information, ("核心机制",))

    def test_highly_overlapping_windows_reuse_stable_evidence_id_across_rounds(self) -> None:
        def result_for(evidence, query):
            return ToolResult(
                status=ToolStatus.OK,
                tool_name=ToolName.SEARCH_PAPERS,
                payload=SearchPapersPayload(
                    (
                        PaperSearchOutcome(
                            paper_id="paper_a",
                            status=PaperSearchStatus.OK,
                            query=query,
                            lexical_terms=(),
                            evidence_candidates=(evidence,),
                        ),
                    )
                ),
            )

        first = RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="paper_a",
            seed_chunk_ids=("c1",),
            source_chunk_ids=("c1", "c2", "c3"),
            section_id="method",
            content="method window one",
        )
        expanded = RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="paper_a",
            seed_chunk_ids=("c1",),
            source_chunk_ids=("c1", "c2", "c3", "c4"),
            section_id="method",
            content="method window one plus neighbor",
        )
        registry = EvidenceRegistry()

        added = registry.register_tool_result(
            result_for(first, "method"), iteration=1
        )
        reused = registry.register_tool_result(
            result_for(expanded, "method details"), iteration=2
        )

        self.assertEqual(added.added_ids, ("E1",))
        self.assertEqual(reused.reused_ids, ("E1",))
        self.assertEqual(len(registry.records), 1)
        self.assertEqual(
            registry.records[0].supporting_queries,
            ("method", "method details"),
        )


if __name__ == "__main__":
    unittest.main()
