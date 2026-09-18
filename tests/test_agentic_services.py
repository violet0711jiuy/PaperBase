from __future__ import annotations

import unittest

from paperbase.agentic.contracts import (
    EvidenceTarget,
    Intent,
    QueryScope,
    ScopeResolutionStatus,
    ScopeSource,
)
from paperbase.agentic.planner import ResearchPlanner
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.scope_resolver import ScopeResolver, TrustedWorkspaceScope


class _CatalogDatabase:
    def __init__(self) -> None:
        self.rows = (
            {
                "paper_id": "paper_a",
                "paper_title": "Attention Is All You Need",
                "source_filename": "attention.pdf",
            },
            {
                "paper_id": "paper_b",
                "paper_title": "BERT Pre-training of Deep Bidirectional Transformers",
                "source_filename": "bert.pdf",
            },
        )

    def list_documents(self):
        return self.rows

    def get_document(self, paper_id: str):
        return next((row for row in self.rows if row["paper_id"] == paper_id), None)


class _InvalidJSONClient:
    def complete_json(self, **kwargs):
        return "{not-json"


class AgenticServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = ScopeResolver(_CatalogDatabase())

    def test_trusted_workspace_wins_over_query_title(self) -> None:
        result = self.resolver.resolve(
            "BERT Pre-training of Deep Bidirectional Transformers 使用了什么方法？",
            trusted_workspace=TrustedWorkspaceScope("workspace_only", "Temporary Paper"),
        )
        self.assertEqual(result.status, ScopeResolutionStatus.RESOLVED)
        self.assertEqual(result.target_paper_ids, ("workspace_only",))
        self.assertEqual(result.scope_source, ScopeSource.TRUSTED_WORKSPACE)

    def test_duplicate_title_requires_clarification(self) -> None:
        database = _CatalogDatabase()
        database.rows = (
            database.rows[0],
            {
                "paper_id": "paper_a_copy",
                "paper_title": "Attention Is All You Need",
                "source_filename": "attention-copy.pdf",
            },
        )
        result = ScopeResolver(database).resolve("请解释 Attention Is All You Need 的方法")
        self.assertEqual(result.status, ScopeResolutionStatus.AMBIGUOUS)
        self.assertEqual(result.candidate_paper_ids, ("paper_a", "paper_a_copy"))

    def test_two_titles_resolve_to_selected_papers(self) -> None:
        result = self.resolver.resolve(
            "Attention Is All You Need 和 BERT Pre-training of Deep Bidirectional Transformers 的方法有什么区别？"
        )
        self.assertEqual(result.scope, QueryScope.SELECTED_PAPERS)
        self.assertEqual(result.target_paper_ids, ("paper_a", "paper_b"))

    def test_knowledge_base_scope_freezes_membership(self) -> None:
        result = self.resolver.resolve("知识库里有哪些方法？", knowledge_base_all=True)
        self.assertEqual(result.scope, QueryScope.KNOWLEDGE_BASE)
        self.assertEqual(result.target_paper_ids, ())
        self.assertEqual(result.corpus_snapshot.paper_ids, ("paper_a", "paper_b"))

    def test_comparison_understanding_and_fallback_plan_are_per_paper(self) -> None:
        resolution = self.resolver.resolve(
            "Attention Is All You Need 和 BERT Pre-training of Deep Bidirectional Transformers 的方法有什么区别？"
        )
        understood = QueryUnderstandingService().understand(
            "Attention Is All You Need 和 BERT Pre-training of Deep Bidirectional Transformers 的方法有什么区别？",
            resolution,
        )
        self.assertEqual(understood.context.intent, Intent.COMPARE)
        self.assertEqual(
            understood.context.default_evidence_targets,
            frozenset({EvidenceTarget.METHOD}),
        )
        planned = ResearchPlanner().plan(understood.context)
        self.assertEqual(planned.plan.subtasks[0].target_paper_ids, ("paper_a",))
        self.assertEqual(planned.plan.subtasks[1].target_paper_ids, ("paper_b",))
        self.assertEqual(planned.plan.subtasks[2].target_paper_ids, ("paper_a", "paper_b"))

    def test_invalid_llm_output_falls_back_without_changing_scope(self) -> None:
        resolution = self.resolver.resolve("paper_a 的方法是什么？")
        result = QueryUnderstandingService(_InvalidJSONClient()).understand(
            "paper_a 的方法是什么？", resolution
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.context.target_paper_ids, ("paper_a",))
        planned = ResearchPlanner(_InvalidJSONClient()).plan(result.context)
        self.assertTrue(planned.used_fallback)


if __name__ == "__main__":
    unittest.main()
