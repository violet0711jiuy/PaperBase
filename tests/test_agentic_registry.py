from __future__ import annotations

import unittest

from paperbase.agentic.contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRelation,
    EvidenceTarget,
    GetPaperOverviewAction,
    GetSectionContextAction,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    KnowledgeBaseSearchPayload,
    PaperOverviewPayload,
    PaperSearchOutcome,
    PaperSearchSpec,
    PaperSearchStatus,
    ReferenceSearchPayload,
    RetrievedEvidence,
    SearchPapersPayload,
    SectionContextPayload,
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


def _action_for_result(result):
    if isinstance(result.payload, SearchPapersPayload):
        return SearchPapersAction(tuple(item.search for item in result.payload.paper_results))
    return SearchReferencesAction(result.payload.paper_id, result.payload.query, requirement_ids=("R-ref",))


class EvidenceRegistryTests(unittest.TestCase):
    def test_exact_duplicate_reuses_id_and_merges_queries(self) -> None:
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(
                (
                    PaperSearchOutcome(
                        search=PaperSearchSpec(paper_id="paper_a", query="first query", requirement_ids=("R1",)),
                        status=PaperSearchStatus.OK,
                        evidence_candidates=(_evidence(),),
                    ),
                )
            ),
        )
        registry = EvidenceRegistry()
        first = registry.register_tool_result(result, action=_action_for_result(result), iteration=1)
        second_result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload((PaperSearchOutcome(
                search=PaperSearchSpec(paper_id="paper_a", query="second query", requirement_ids=("R2",)),
                status=PaperSearchStatus.OK,
                evidence_candidates=(_evidence(),),
            ),)),
        )
        second = registry.register_tool_result(
            second_result,
            action=_action_for_result(second_result),
            iteration=2,
        )
        self.assertEqual(first.added_ids, ("E1",))
        self.assertEqual(second.reused_ids, ("E1",))
        self.assertEqual(registry.records[0].last_seen_iteration, 2)
        self.assertIn("second query", registry.records[0].supporting_queries)
        self.assertEqual(registry.records[0].originating_tools, ("search_papers",))
        self.assertEqual(registry.records[0].requested_requirement_ids, ("R1", "R2"))

    def test_registry_keeps_search_targets_gaps_and_stable_fingerprint(self) -> None:
        outcome = PaperSearchOutcome(
            search=PaperSearchSpec(
                paper_id="paper_a", query="定位核心方法", lexical_terms=("encoder",),
                evidence_targets=frozenset({"method"}), requirement_ids=("R-method",),
            ),
            status=PaperSearchStatus.OK,
            evidence_candidates=(_evidence(),),
        )
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload((outcome,)),
        )

        first_registry = EvidenceRegistry()
        first_registry.register_tool_result(result, action=_action_for_result(result), iteration=1)
        record = first_registry.records[0]
        self.assertEqual(
            {target.value for target in record.requested_evidence_targets},
            {"method"},
        )
        self.assertEqual(record.requested_requirement_ids, ("R-method",))

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
                        search=PaperSearchSpec(paper_id="paper_a", query="另一轮检索"),
                        status=PaperSearchStatus.OK,
                        evidence_candidates=(second_evidence,),
                    ),
                )
            ),
        )
        second_registry = EvidenceRegistry()
        second_registry.register_tool_result(second_result, action=_action_for_result(second_result), iteration=9)
        self.assertEqual(second_registry.records[0].fingerprint, record.fingerprint)

    def test_bibliography_uses_independent_ref_numbering(self) -> None:
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_REFERENCES,
            payload=ReferenceSearchPayload(
                paper_id="paper_a",
                query="citation",
                evidence_candidates=(_evidence(EvidenceSourceType.BIBLIOGRAPHY),),
            ),
        )
        registry = EvidenceRegistry()
        delta = registry.register_tool_result(result, action=_action_for_result(result), iteration=1)
        self.assertEqual(delta.added_ids, ("REF1",))
        self.assertEqual(registry.records[0].requested_requirement_ids, ("R-ref",))

    def test_content_reference_and_requirement_namespaces_do_not_collide(self) -> None:
        registry = EvidenceRegistry()
        content = tuple(
            RetrievedEvidence(
                source_type=EvidenceSourceType.CONTENT,
                paper_id="paper_a",
                seed_chunk_ids=(f"body_{index}",),
                source_chunk_ids=(f"body_{index}",),
                content=f"body {index}",
            )
            for index in (1, 2)
        )
        references = tuple(
            RetrievedEvidence(
                source_type=EvidenceSourceType.BIBLIOGRAPHY,
                paper_id="paper_a",
                seed_chunk_ids=(f"reference_{index}",),
                source_chunk_ids=(f"reference_{index}",),
                content=f"reference {index}",
            )
            for index in (1, 2)
        )
        content_result = ToolResult(
            ToolStatus.OK,
            ToolName.SEARCH_PAPERS,
            SearchPapersPayload((PaperSearchOutcome(
                PaperSearchSpec("paper_a", "body", requirement_ids=("R1", "R2")),
                PaperSearchStatus.OK,
                content,
            ),)),
        )
        reference_result = ToolResult(
            ToolStatus.OK,
            ToolName.SEARCH_REFERENCES,
            ReferenceSearchPayload("paper_a", "references", references),
        )

        content_delta = registry.register_tool_result(
            content_result,
            action=SearchPapersAction((PaperSearchSpec(
                "paper_a", "body", requirement_ids=("R1", "R2")
            ),)),
            iteration=1,
        )
        reference_delta = registry.register_tool_result(
            reference_result,
            action=SearchReferencesAction(
                "paper_a", "references", requirement_ids=("R1", "R2")
            ),
            iteration=2,
        )

        self.assertEqual(content_delta.added_ids, ("E1", "E2"))
        self.assertEqual(reference_delta.added_ids, ("REF1", "REF2"))
        self.assertEqual(
            {record.evidence_id: record.requested_requirement_ids for record in registry.records},
            {
                "E1": ("R1", "R2"),
                "E2": ("R1", "R2"),
                "REF1": ("R1", "R2"),
                "REF2": ("R1", "R2"),
            },
        )

    def test_single_action_bindings_reach_evidence_for_each_tool(self) -> None:
        cases = (
            (
                SearchKnowledgeBaseAction("method", requirement_ids=("R-kb",)),
                ToolName.SEARCH_KNOWLEDGE_BASE,
                KnowledgeBaseSearchPayload("method", (_evidence(),)),
                "R-kb",
            ),
            (
                SearchReferencesAction("paper_a", "citation", requirement_ids=("R-ref",)),
                ToolName.SEARCH_REFERENCES,
                ReferenceSearchPayload("paper_a", "citation", (_evidence(EvidenceSourceType.BIBLIOGRAPHY),)),
                "R-ref",
            ),
            (
                GetSectionContextAction("paper_a", "method", requirement_ids=("R-section",)),
                ToolName.GET_SECTION_CONTEXT,
                SectionContextPayload("paper_a", "method", (_evidence(EvidenceSourceType.SECTION_CONTEXT),)),
                "R-section",
            ),
            (
                GetPaperOverviewAction("paper_a", requirement_ids=("R-overview",)),
                ToolName.GET_PAPER_OVERVIEW,
                PaperOverviewPayload("paper_a", (_evidence(EvidenceSourceType.OVERVIEW),)),
                "R-overview",
            ),
        )
        for action, tool_name, payload, requirement_id in cases:
            with self.subTest(tool=tool_name):
                registry = EvidenceRegistry()
                result = ToolResult(ToolStatus.OK, tool_name, payload)
                registry.register_tool_result(result, action=action, iteration=1)
                self.assertEqual(registry.records[0].requested_requirement_ids, (requirement_id,))

    def test_multi_search_uses_each_action_search_binding(self) -> None:
        first = PaperSearchSpec("paper_a", "first", requirement_ids=("R1",))
        second = PaperSearchSpec("paper_b", "second", requirement_ids=("R2",))
        other_evidence = RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="paper_b", seed_chunk_ids=("chunk_b",),
            source_chunk_ids=("chunk_b",), content="other evidence",
        )
        result = ToolResult(
            ToolStatus.OK, ToolName.SEARCH_PAPERS,
            SearchPapersPayload((
                PaperSearchOutcome(PaperSearchSpec("paper_a", "first"), PaperSearchStatus.OK, (_evidence(),)),
                PaperSearchOutcome(PaperSearchSpec("paper_b", "second"), PaperSearchStatus.OK, (other_evidence,)),
            )),
        )
        registry = EvidenceRegistry()
        registry.register_tool_result(
            result, action=SearchPapersAction((first, second)), iteration=1,
        )
        self.assertEqual(
            {item.evidence.paper_id: item.requested_requirement_ids for item in registry.records},
            {"paper_a": ("R1",), "paper_b": ("R2",)},
        )

    def test_empty_result_records_no_evidence(self) -> None:
        action = SearchReferencesAction("paper_a", "missing", requirement_ids=("R1",))
        result = ToolResult(ToolStatus.EMPTY, ToolName.SEARCH_REFERENCES, ReferenceSearchPayload("paper_a", "missing"))
        registry = EvidenceRegistry()
        delta = registry.register_tool_result(result, action=action, iteration=1)
        self.assertEqual(delta.added_ids, ())
        self.assertEqual(registry.records, ())

    def test_analysis_coverage_is_persisted_on_record(self) -> None:
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(
                (
                    PaperSearchOutcome(
                        search=PaperSearchSpec(paper_id="paper_a", query="方法", evidence_targets=frozenset({"method"})),
                        status=PaperSearchStatus.OK,
                        evidence_candidates=(_evidence(),),
                    ),
                )
            ),
        )
        registry = EvidenceRegistry()
        registry.register_tool_result(result, action=_action_for_result(result), iteration=1)
        registry.apply_analyses(
            (
                EvidenceAnalysis(
                    evidence_id="E1",
                    status=EvidenceAnalysisStatus.QUALIFIED,
                    digest="方法摘要",
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
        self.assertEqual(record.assessment.status, EvidenceAnalysisStatus.QUALIFIED)
        self.assertEqual(record.covered_targets, frozenset())

    def test_highly_overlapping_windows_reuse_stable_evidence_id_across_rounds(self) -> None:
        def result_for(evidence, query):
            return ToolResult(
                status=ToolStatus.OK,
                tool_name=ToolName.SEARCH_PAPERS,
                payload=SearchPapersPayload(
                    (
                        PaperSearchOutcome(
                            search=PaperSearchSpec(paper_id="paper_a", query=query),
                            status=PaperSearchStatus.OK,
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

        first_result = result_for(first, "method")
        second_result = result_for(expanded, "method details")
        added = registry.register_tool_result(
            first_result, action=_action_for_result(first_result), iteration=1
        )
        reused = registry.register_tool_result(
            second_result, action=_action_for_result(second_result), iteration=2
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
