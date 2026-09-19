from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paperbase.agentic.contracts import (
    BudgetUsage,
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    CoverageStrategy,
    CorpusScopeSnapshot,
    EvidenceOwnership,
    EvidenceQualification,
    EvidenceRecord,
    EvidenceRelation,
    EvidenceTarget,
    GetPaperOverviewAction,
    Intent,
    PaperAssessmentStatus,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
    ResearchSubtask,
    ScopeSource,
)
from paperbase.agentic.coverage import CoverageManager, CoveragePolicy
from paperbase.agentic.decision import DecisionState, _visible_evidence
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.agentic.tools import ResearchToolExecutor
from paperbase.config import PaperOverviewSettings
from paperbase.overview.repository import FormalOverviewRepository
from paperbase.overview.service import OverviewField, PaperOverview
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    RetrievedEvidence,
    ToolStatus,
)


def _context(*, exhaustive: bool = False) -> QueryContext:
    return QueryContext(
        original_query="比较 A 和 B",
        resolved_query="比较 A 和 B",
        intent=Intent.COMPARE,
        scope=QueryScope.SELECTED_PAPERS,
        target_paper_ids=("A", "B"),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=(
            ResearchGoal.EXHAUSTIVE_ENUMERATION
            if exhaustive
            else ResearchGoal.COMPARISON_SUPPORT
        ),
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requires_complete_coverage=exhaustive,
    )


def _record(
    evidence_id: str,
    paper_id: str,
    *,
    relation: EvidenceRelation | None = None,
    ownership: EvidenceOwnership = EvidenceOwnership.THIS_PAPER,
    score: float = 0.5,
    section_id: str | None = None,
) -> EvidenceRecord:
    qualification = (
        EvidenceQualification(evidence_id, relation, ownership, "测试标注")
        if relation is not None
        else None
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id=paper_id,
            section_id=section_id,
            section=section_id,
            seed_chunk_ids=(f"{paper_id}-{evidence_id}",),
            source_chunk_ids=(f"{paper_id}-{evidence_id}",),
            content=f"{paper_id} evidence",
            score=score,
        ),
        supporting_queries=("method",),
        first_seen_iteration=1,
        last_seen_iteration=1,
        target_subtask_ids=(f"S{paper_id}",),
        qualification=qualification,
    )


def _observation(*queries: tuple[str, str]) -> ResearchObservation:
    return ResearchObservation(
        iteration=1,
        action_name="search_papers",
        tool_status=ToolStatus.OK,
        added_evidence_ids=(),
        reused_evidence_ids=(),
        queries_by_paper=queries,
    )


class _FormalDatabase:
    def get_document(self, paper_id: str):
        return {"paper_id": paper_id, "paper_title": "Paper A"}

    def list_chunks(self, paper_id: str):
        return (
            {
                "chunk_id": "c1",
                "chunk_index": 0,
                "paper_id": paper_id,
                "raw_text": "Our method introduces a trainable framework and algorithm.",
                "section": "3 Method",
                "section_type": "content",
                "content_kind": "body",
                "front_matter_type": None,
                "raw_token_count": 12,
            },
        )


class _OverviewClient:
    def complete_json(self, **kwargs) -> str:
        field = {"content": "由 c1 支持。", "source_chunk_ids": ["c1"]}
        return json.dumps(
            {
                "paper_title": "Paper A",
                "research_problem": field,
                "main_method": field,
                "contributions": field,
                "datasets": field,
                "experimental_setup": field,
                "main_results": field,
                "limitations": field,
            },
            ensure_ascii=False,
        )


class AgenticPhase3Tests(unittest.TestCase):
    def test_qualification_rejects_unknown_evidence(self) -> None:
        registry = EvidenceRegistry()
        with self.assertRaisesRegex(ValueError, "未知 Evidence"):
            registry.apply_qualifications(
                (
                    EvidenceQualification(
                        "E404",
                        EvidenceRelation.SUPPORTS,
                        EvidenceOwnership.THIS_PAPER,
                        "不存在",
                    ),
                )
            )

    def test_candidate_coverage_distinguishes_positive_and_no_support(self) -> None:
        coverage = CoverageManager().evaluate(
            context=_context(),
            corpus_snapshot=None,
            evidence=(
                _record("E1", "A", relation=EvidenceRelation.SUPPORTS),
            ),
            observations=(_observation(("A", "method A"), ("B", "method B")),),
        )
        self.assertEqual(coverage.strategy, CoverageStrategy.CANDIDATE_VERIFICATION)
        self.assertTrue(coverage.coverage_complete)
        self.assertEqual(
            [item.status for item in coverage.assessments],
            [PaperAssessmentStatus.POSITIVE, PaperAssessmentStatus.NO_SUPPORT_FOUND],
        )
        self.assertIn("不代表事实不存在", coverage.assessments[1].rationale)

    def test_exhaustive_coverage_requires_alternate_query(self) -> None:
        manager = CoverageManager(CoveragePolicy(exhaustive_min_queries_per_paper=2))
        first = manager.evaluate(
            context=_context(exhaustive=True),
            corpus_snapshot=None,
            evidence=(),
            observations=(_observation(("A", "direct"), ("B", "direct")),),
        )
        self.assertFalse(first.coverage_complete)
        second = manager.evaluate(
            context=_context(exhaustive=True),
            corpus_snapshot=None,
            evidence=(),
            observations=(
                _observation(("A", "direct"), ("B", "direct")),
                _observation(("A", "alternate"), ("B", "alternate")),
            ),
        )
        self.assertTrue(second.coverage_complete)

    def test_candidate_selection_does_not_count_as_body_verification(self) -> None:
        context = QueryContext(
            original_query="哪些论文支持该方法？",
            resolved_query="哪些论文支持该方法？",
            intent=Intent.RETRIEVE,
            scope=QueryScope.KNOWLEDGE_BASE,
            target_paper_ids=(),
            scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
            retrieval_goal=ResearchGoal.FIND_EVIDENCE,
            default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        )
        discovered = replace(
            _record("E1", "A"), originating_tools=("search_knowledge_base",)
        )
        snapshot = CorpusScopeSnapshot.create(("A", "B"))
        selection = CandidateSelection(
            scope_paper_ids=("A", "B"),
            assessments=(
                CandidateAssessment("A", CandidateDisposition.CANDIDATE, "元数据直接相关"),
                CandidateAssessment("B", CandidateDisposition.EXCLUDED, "元数据明确无关"),
            ),
        )
        manager = CoverageManager()
        first = manager.evaluate(
            context=context,
            corpus_snapshot=snapshot,
            evidence=(discovered,),
            observations=(
                ResearchObservation(
                    iteration=1,
                    action_name="search_knowledge_base",
                    tool_status=ToolStatus.OK,
                    added_evidence_ids=("E1",),
                    reused_evidence_ids=(),
                    queries_by_paper=(("A", "global query"),),
                ),
            ),
            candidate_selection=selection,
        )
        self.assertFalse(first.coverage_complete)
        self.assertEqual(first.remaining_paper_ids, ("A",))
        verified = manager.evaluate(
            context=context,
            corpus_snapshot=snapshot,
            evidence=(discovered,),
            observations=(
                ResearchObservation(
                    iteration=1,
                    action_name="search_knowledge_base",
                    tool_status=ToolStatus.OK,
                    added_evidence_ids=("E1",),
                    reused_evidence_ids=(),
                    queries_by_paper=(("A", "global query"),),
                ),
                _observation(("A", "paper-specific query")),
            ),
            candidate_selection=selection,
        )
        self.assertTrue(verified.coverage_complete)
        self.assertEqual(
            verified.assessments[0].status, PaperAssessmentStatus.UNCERTAIN
        )

    def test_uncertain_paper_uses_one_shallow_query_in_exhaustive_scope(self) -> None:
        context = QueryContext(
            original_query="完整列出风速预测方法",
            resolved_query="完整列出风速预测方法",
            intent=Intent.SYNTHESIZE,
            scope=QueryScope.KNOWLEDGE_BASE,
            target_paper_ids=(),
            scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
            retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION,
            default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
            requires_complete_coverage=True,
        )
        snapshot = CorpusScopeSnapshot.create(("A", "B"))
        selection = CandidateSelection(
            scope_paper_ids=("A", "B"),
            assessments=(
                CandidateAssessment("A", CandidateDisposition.UNCERTAIN, "元数据不足"),
                CandidateAssessment("B", CandidateDisposition.EXCLUDED, "主题无关"),
            ),
        )
        coverage = CoverageManager(
            CoveragePolicy(exhaustive_min_queries_per_paper=2)
        ).evaluate(
            context=context,
            corpus_snapshot=snapshot,
            candidate_selection=selection,
            evidence=(),
            observations=(_observation(("A", "shallow verification")),),
        )
        self.assertTrue(coverage.coverage_complete)
        self.assertEqual(coverage.screened_paper_ids, ("A", "B"))
        self.assertEqual(coverage.verified_paper_ids, ("A",))
        self.assertEqual(coverage.excluded_paper_ids, ("B",))

    def test_selector_balances_positive_papers_and_filters_background(self) -> None:
        coverage = CoverageManager().evaluate(
            context=_context(),
            corpus_snapshot=None,
            evidence=(
                _record("E1", "A", relation=EvidenceRelation.SUPPORTS, score=0.9),
                _record("E2", "A", relation=EvidenceRelation.SUPPORTS, score=0.8),
                _record("E3", "B", relation=EvidenceRelation.SUPPORTS, score=0.1),
                _record(
                    "E4",
                    "B",
                    relation=EvidenceRelation.BACKGROUND_ONLY,
                    ownership=EvidenceOwnership.GENERAL_BACKGROUND,
                    score=1.0,
                ),
            ),
            observations=(_observation(("A", "method A"), ("B", "method B")),),
        )
        plan = ResearchPlan(
            "compare",
            (
                ResearchSubtask("SA", "A", ("A",), frozenset({EvidenceTarget.METHOD}), "A"),
                ResearchSubtask("SB", "B", ("B",), frozenset({EvidenceTarget.METHOD}), "B"),
            ),
        )
        selected = FinalEvidenceSelector(max_units=2).select(
            plan=plan, coverage=coverage, evidence=tuple(
                [
                    _record("E1", "A", relation=EvidenceRelation.SUPPORTS, score=0.9),
                    _record("E2", "A", relation=EvidenceRelation.SUPPORTS, score=0.8),
                    _record("E3", "B", relation=EvidenceRelation.SUPPORTS, score=0.1),
                    _record(
                        "E4",
                        "B",
                        relation=EvidenceRelation.BACKGROUND_ONLY,
                        ownership=EvidenceOwnership.GENERAL_BACKGROUND,
                        score=1.0,
                    ),
                ]
            )
        )
        self.assertEqual({item.evidence.paper_id for item in selected}, {"A", "B"})
        self.assertNotIn("E4", {item.evidence_id for item in selected})

    def test_selector_prefers_different_sections_before_same_section_fill(self) -> None:
        method_high = _record(
            "E1", "A", relation=EvidenceRelation.SUPPORTS,
            score=0.9, section_id="method",
        )
        method_medium = _record(
            "E2", "A", relation=EvidenceRelation.SUPPORTS,
            score=0.8, section_id="method",
        )
        results_low = _record(
            "E3", "A", relation=EvidenceRelation.SUPPORTS,
            score=0.3, section_id="results",
        )
        coverage = CoverageManager().evaluate(
            context=_context(),
            corpus_snapshot=None,
            evidence=(method_high, method_medium, results_low),
            observations=(_observation(("A", "method"), ("B", "method")),),
        )
        plan = ResearchPlan(
            "collect complementary evidence",
            (
                ResearchSubtask(
                    "SA", "A", ("A",), frozenset({EvidenceTarget.METHOD}), "A"
                ),
            ),
        )

        selected = FinalEvidenceSelector(max_units=2, max_per_paper=2).select(
            plan=plan,
            coverage=coverage,
            evidence=(method_high, method_medium, results_low),
        )

        self.assertEqual([item.evidence_id for item in selected], ["E1", "E3"])

    def test_decision_visibility_uses_same_section_diversity_rule(self) -> None:
        method_high = _record(
            "E1", "A", relation=EvidenceRelation.SUPPORTS,
            score=0.9, section_id="method",
        )
        method_medium = _record(
            "E2", "A", relation=EvidenceRelation.SUPPORTS,
            score=0.8, section_id="method",
        )
        results_low = _record(
            "E3", "A", relation=EvidenceRelation.SUPPORTS,
            score=0.3, section_id="results",
        )
        plan = ResearchPlan(
            "inspect sections",
            (
                ResearchSubtask(
                    "SA", "A", ("A",), frozenset({EvidenceTarget.METHOD}), "A"
                ),
            ),
        )

        visible = _visible_evidence(
            DecisionState(
                _context(),
                plan,
                (method_high, method_medium, results_low),
                (),
                BudgetUsage(),
            ),
            limit=2,
        )

        self.assertEqual([item.evidence_id for item in visible], ["E1", "E3"])

    def test_formal_overview_generation_is_explicit_and_tool_is_read_only(self) -> None:
        with TemporaryDirectory() as directory:
            repository = FormalOverviewRepository(Path(directory))
            self.assertIsNone(repository.get("A"))
            artifact = repository.generate(
                "A",
                database=_FormalDatabase(),
                settings=PaperOverviewSettings(),
                client=_OverviewClient(),
            )
            self.assertTrue(artifact.overview_path.is_file())
            executor = ResearchToolExecutor(
                retrieval_service=object(),
                database=object(),
                formal_overview_repository=repository,
            )
            result = executor.execute(
                GetPaperOverviewAction("A"), context=QueryContext(
                    original_query="A",
                    resolved_query="A",
                    intent=Intent.EXPLAIN,
                    scope=QueryScope.SINGLE_PAPER,
                    target_paper_ids=("A",),
                    scope_source=ScopeSource.UI_SELECTION,
                    retrieval_goal=ResearchGoal.FIND_EVIDENCE,
                    default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
                )
            )
            self.assertEqual(result.status, ToolStatus.OK)
            self.assertEqual(len(result.payload.evidence_candidates), 7)

    def test_formal_overview_tool_does_not_generate_missing_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            repository = FormalOverviewRepository(Path(directory))
            executor = ResearchToolExecutor(
                retrieval_service=object(),
                database=object(),
                formal_overview_repository=repository,
            )
            result = executor.execute(
                GetPaperOverviewAction("A"), context=QueryContext(
                    original_query="A",
                    resolved_query="A",
                    intent=Intent.EXPLAIN,
                    scope=QueryScope.SINGLE_PAPER,
                    target_paper_ids=("A",),
                    scope_source=ScopeSource.UI_SELECTION,
                    retrieval_goal=ResearchGoal.FIND_EVIDENCE,
                    default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
                )
            )
            self.assertEqual(result.status, ToolStatus.UNAVAILABLE)
            self.assertFalse((Path(directory) / "A").exists())


if __name__ == "__main__":
    unittest.main()
