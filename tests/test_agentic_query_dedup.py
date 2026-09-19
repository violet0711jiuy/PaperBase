from __future__ import annotations

import unittest

from paperbase.agentic.contracts import (
    EvidenceTarget,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchSubtask,
    ScopeSource,
    SearchPapersAction,
)
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import PaperSearchSpec


def _context() -> QueryContext:
    return QueryContext(
        original_query="比较 A 和 B",
        resolved_query="比较 A 和 B",
        intent=Intent.COMPARE,
        scope=QueryScope.SELECTED_PAPERS,
        target_paper_ids=("A", "B"),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.COMPARISON_SUPPORT,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        "比较方法",
        (
            ResearchSubtask(
                "S1",
                "比较方法",
                ("A", "B"),
                frozenset({EvidenceTarget.METHOD}),
                "获得两篇方法证据",
            ),
        ),
    )


class PerPaperQueryDedupTests(unittest.TestCase):
    def test_same_paper_query_is_rejected_across_different_batches(self) -> None:
        tracker = BudgetAndGuardTracker(AgenticResearchSettings())
        first = SearchPapersAction(
            (
                PaperSearchSpec("A", "Temporal   Encoder", target_subtask_id="S1"),
                PaperSearchSpec("B", "Graph decoder", target_subtask_id="S1"),
            ),
            "S1",
        )
        repeated_in_new_batch = SearchPapersAction(
            (
                PaperSearchSpec("A", " temporal encoder ", target_subtask_id="S1"),
            ),
            "S1",
        )

        accepted = tracker.validate_action(
            first,
            context=_context(),
            plan=_plan(),
            snapshot=None,
        )
        rejected = tracker.validate_action(
            repeated_in_new_batch,
            context=_context(),
            plan=_plan(),
            snapshot=None,
        )

        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.code, "DuplicatePaperQuery")
        self.assertEqual(
            len(tracker.snapshot().executed_paper_query_fingerprints), 2
        )

    def test_query_history_survives_guard_checkpoint_restore(self) -> None:
        first_tracker = BudgetAndGuardTracker(AgenticResearchSettings())
        action = SearchPapersAction(
            (
                PaperSearchSpec("A", "method"),
                PaperSearchSpec("B", "decoder"),
            )
        )
        self.assertTrue(
            first_tracker.validate_action(
                action,
                context=_context(),
                plan=_plan(),
                snapshot=None,
            ).accepted
        )

        restored = BudgetAndGuardTracker(
            AgenticResearchSettings(), state=first_tracker.snapshot()
        )
        decision = restored.validate_action(
            SearchPapersAction((PaperSearchSpec("A", "METHOD"),)),
            context=_context(),
            plan=_plan(),
            snapshot=None,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "DuplicatePaperQuery")


if __name__ == "__main__":
    unittest.main()
