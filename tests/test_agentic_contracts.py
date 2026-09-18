from __future__ import annotations

import unittest

from paperbase.agentic.contracts import (
    CorpusScopeSnapshot,
    EvidenceTarget,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ScopeSource,
)
from paperbase.config import AgenticResearchSettings


class AgenticContractTests(unittest.TestCase):
    def test_query_context_enforces_scope_cardinality(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            QueryContext(
                original_query="q",
                resolved_query="q",
                intent=Intent.RETRIEVE,
                scope=QueryScope.SINGLE_PAPER,
                target_paper_ids=("A", "B"),
                scope_source=ScopeSource.UI_SELECTION,
                retrieval_goal=ResearchGoal.FIND_EVIDENCE,
                default_evidence_targets=frozenset({EvidenceTarget.GENERAL}),
            )

    def test_program_owned_flags_cannot_disagree_with_semantics(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires_complete_coverage"):
            QueryContext(
                original_query="all",
                resolved_query="all",
                intent=Intent.RETRIEVE,
                scope=QueryScope.SINGLE_PAPER,
                target_paper_ids=("A",),
                scope_source=ScopeSource.UI_SELECTION,
                retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION,
                default_evidence_targets=frozenset({EvidenceTarget.GENERAL}),
                requires_complete_coverage=False,
            )

    def test_corpus_snapshot_is_membership_stable(self) -> None:
        first = CorpusScopeSnapshot.create(("B", "A"))
        second = CorpusScopeSnapshot.create(("A", "B"))
        changed = CorpusScopeSnapshot.create(("A", "C"))
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(first.paper_ids, second.paper_ids)
        self.assertNotEqual(first.snapshot_id, changed.snapshot_id)

    def test_agentic_settings_are_strict_and_bounded(self) -> None:
        settings = AgenticResearchSettings(max_agent_iterations=3)
        self.assertEqual(settings.max_agent_iterations, 3)
        with self.assertRaises(ValueError):
            AgenticResearchSettings(max_tool_calls=0)


if __name__ == "__main__":
    unittest.main()
