from __future__ import annotations

import json
import unittest
from dataclasses import fields

from paperbase.agentic.contracts import (
    BudgetUsage,
    CoverageState,
    CoverageStrategy,
    ComparisonProjection,
    EvidenceAnalysis,
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
    ResearchRequirement,
    RequirementSupportAnchor,
    ScopeSource,
)
from paperbase.agentic.decision import DecisionState, _state_prompt
from paperbase.agentic.research_profile import PaperResearchProfile, build_research_profiles
from paperbase.agentic.answering import CompletionView
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    RetrievedEvidence,
    ToolStatus,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="比较方法",
        requirements=(
            ResearchRequirement(
                "R1", "比较 A 与 B 的方法",
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
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
        assessment=EvidenceAnalysis(
            evidence_id="E1",
            status=EvidenceAnalysisStatus.QUALIFIED,
            projection=ComparisonProjection("E1", ("A 使用时间编码器。",)),
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

        # Qualified content without integrity-valid Requirement support is not
        # target completion. Both paper profiles still retain the Plan hint.
        self.assertEqual(profiles[0].remaining_targets, frozenset({EvidenceTarget.METHOD}))
        self.assertEqual(
            profiles[0].qualified_claims, ("A 使用时间编码器。",)
        )
        self.assertEqual(
            profiles[1].remaining_targets, frozenset({EvidenceTarget.METHOD})
        )
        self.assertEqual(profiles[0].supported_requirement_ids, ())

    def test_progress_views_do_not_store_derived_completion_fields(self) -> None:
        profile_fields = {item.name for item in fields(PaperResearchProfile)}
        self.assertFalse({
            "remaining_targets", "requested_requirement_ids",
            "covered_requirement_ids", "remaining_requirement_ids",
            "remaining_optional_requirement_ids",
        } & profile_fields)
        completion_fields = {item.name for item in fields(CompletionView)}
        self.assertEqual(
            completion_fields, {"selected_verified_support_by_requirement"}
        )
        coverage_fields = {item.name for item in fields(CoverageState)}
        self.assertNotIn("screened_paper_ids", coverage_fields)
        self.assertNotIn("discovery_completed", coverage_fields)
        coverage = CoverageState(
            scope_paper_ids=("A", "B"), assessments=(),
            strategy=CoverageStrategy.EXHAUSTIVE,
            candidate_paper_ids=("A",), excluded_paper_ids=("B",),
        )
        self.assertEqual(coverage.screened_paper_ids, ("A", "B"))
        self.assertFalse(coverage.coverage_complete)

    def test_compact_decision_exposes_grounded_cross_paper_support_boundary(self) -> None:
        from dataclasses import replace
        record = _analyzed_a()
        record = replace(record, assessment=replace(record.assessment, projection=None, projections=(ComparisonProjection(
            "E1", ("A 使用时间编码器。",), supported_requirement_ids=("R1",),
            proposed_requirement_ids=("R1",), support_anchors=(RequirementSupportAnchor(
                "R1", record.evidence.content, "A 使用时间编码器。", ("B 的对应结构尚未说明",)),),
        ),)))
        payload = json.loads(
            _state_prompt(
                DecisionState(
                    _context(), _plan(), (record,), (), BudgetUsage()
                ),
                [],
            )
        )

        support = payload["requirements"][0]["local_supports"][0]
        self.assertEqual(support["covers"], ["A 使用时间编码器。"])
        self.assertEqual(support["paper_id"], "A")
        self.assertEqual(support["boundaries"], ["B 的对应结构尚未说明"])
        # R1 already has verified local support in the State. The B-specific
        # local boundary remains visible without a duplicate paper summary.
        self.assertNotIn("paper_state", payload)

    def test_decision_prompt_exposes_deterministic_search_delta_and_budget(self) -> None:
        observation = ResearchObservation(
            iteration=2,
            action_name="search_papers",
            tool_status=ToolStatus.OK,
            added_evidence_ids=("E2",),
            reused_evidence_ids=("E1",),
            search_attempts=(
                PaperSearchAttempt("A", "alternative method wording", requirement_ids=("R1",)),
            ),
            new_supporting_evidence_count=1,
            newly_supported_requirement_ids=("R1",),
            newly_exposed_section_ids=("methods",),
        )
        payload = json.loads(
            _state_prompt(
                DecisionState(
                    _context(), _plan(), (), (observation,),
                    BudgetUsage(queries_per_paper=(("A", 2),)),
                    per_paper_query_limit=3,
                ),
                [],
            )
        )

        from paperbase.agentic.decision_view import expand_decision_references
        payload = expand_decision_references(payload)
        event = payload["recent_research_events"][-1]
        self.assertEqual(event["request"]["searches"][0]["paper_id"], "A")
        self.assertEqual(event["new_verified_support"], 1)
        self.assertEqual(event["new_sections"], ["methods"])
        budget = {item["paper_id"]: item for item in payload["runtime_action_capabilities"]["tool_budget_state"]["per_paper"]}
        self.assertEqual(budget["A"]["remaining"], 1)


if __name__ == "__main__":
    unittest.main()
