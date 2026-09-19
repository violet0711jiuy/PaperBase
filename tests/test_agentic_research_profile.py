from __future__ import annotations

import json
import unittest

from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    EvidenceTarget,
    Intent,
    PaperSearchAttempt,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
    ResearchSubtask,
    ScopeSource,
)
from paperbase.agentic.decision import DecisionState, _state_prompt
from paperbase.agentic.research_profile import build_research_profiles
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    RetrievedEvidence,
    ToolStatus,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        "比较方法",
        (
            ResearchSubtask(
                "S1",
                "比较 A 与 B 的方法",
                ("A", "B"),
                frozenset({EvidenceTarget.METHOD}),
                "两篇均有方法证据",
            ),
        ),
    )


def _context() -> QueryContext:
    return QueryContext(
        original_query="A 和 B 的方法有什么区别？",
        resolved_query="A 和 B 的方法有什么区别？",
        intent=Intent.COMPARE,
        scope=QueryScope.SELECTED_PAPERS,
        target_paper_ids=("A", "B"),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.COMPARISON_SUPPORT,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _analyzed_a() -> EvidenceRecord:
    excerpt = "A uses a temporal encoder."
    return EvidenceRecord(
        evidence_id="E1",
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="A",
            seed_chunk_ids=("a1",),
            source_chunk_ids=("a1",),
            section_id="method-a",
            section="Method",
            content=excerpt,
        ),
        supporting_queries=("A temporal encoder",),
        first_seen_iteration=1,
        last_seen_iteration=1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requested_missing_information=("A 的核心机制",),
        analysis_status=EvidenceAnalysisStatus.QUALIFIED,
        digest="A 使用时间编码器。",
        covered_targets=frozenset({EvidenceTarget.METHOD}),
        addressed_information=("A 的核心机制",),
        claims=(
            EvidenceClaim(
                "E1:C1",
                "A 使用时间编码器。",
                EvidenceRelation.SUPPORTS,
                EvidenceOwnership.THIS_PAPER,
                "原文直接陈述。",
                excerpt,
            ),
        ),
    )


class ResearchProfileTests(unittest.TestCase):
    def test_profiles_derive_remaining_gaps_without_persisting_second_state(self) -> None:
        observation = ResearchObservation(
            iteration=1,
            action_name="search_papers",
            tool_status=ToolStatus.OK,
            added_evidence_ids=("E1",),
            reused_evidence_ids=(),
            search_attempts=(
                PaperSearchAttempt(
                    paper_id="A",
                    query="A temporal encoder",
                    evidence_targets=frozenset({EvidenceTarget.METHOD}),
                    missing_information=("A 的核心机制",),
                    status="ok",
                ),
            ),
        )

        profiles = build_research_profiles(
            paper_ids=("A", "B"),
            plan=_plan(),
            evidence=(_analyzed_a(),),
            observations=(observation,),
            default_targets=frozenset({EvidenceTarget.METHOD}),
        )

        self.assertEqual(profiles[0].remaining_targets, frozenset())
        self.assertEqual(profiles[0].remaining_information, ())
        self.assertEqual(
            profiles[0].qualified_claims, ("A 使用时间编码器。",)
        )
        self.assertEqual(
            profiles[1].remaining_targets, frozenset({EvidenceTarget.METHOD})
        )

    def test_decision_prompt_exposes_cross_paper_comparison_dimension(self) -> None:
        payload = json.loads(
            _state_prompt(
                DecisionState(
                    _context(), _plan(), (_analyzed_a(),), (), BudgetUsage()
                ),
                [],
            )
        )

        profiles = {item["paper_id"]: item for item in payload["research_profiles"]}
        self.assertIn("A 使用时间编码器。", profiles["A"]["qualified_claims"])
        self.assertEqual(profiles["B"]["remaining_targets"], ["method"])


if __name__ == "__main__":
    unittest.main()
