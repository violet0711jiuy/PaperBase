from __future__ import annotations

import json
import unittest

from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    EvidenceTarget,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ScopeSource,
)
from paperbase.agentic.decision import DecisionState, _state_prompt, _visible_evidence
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


def _record(content: str = "Existing studies use ARIMA. We propose EMVE.") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="E1",
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="A",
            section_id="method",
            section="Method",
            seed_chunk_ids=("c1",),
            source_chunk_ids=("c1",),
            content=content,
        ),
        supporting_queries=("wind speed method",),
        first_seen_iteration=1,
        last_seen_iteration=1,
        requested_requirement_ids=("R1",),
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


class EvidenceAnalysisTests(unittest.TestCase):
    def test_decision_visibility_prefers_relevance_and_paper_diversity(self) -> None:
        context = QueryContext(
            original_query="wind speed forecasting methods",
            resolved_query="wind speed forecasting methods",
            semantic_query_en="wind speed forecasting methods",
            lexical_keywords_en=("wind speed", "forecasting"),
            intent=Intent.SYNTHESIZE,
            scope=QueryScope.SELECTED_PAPERS,
            target_paper_ids=("A", "B"),
            scope_source=ScopeSource.UI_SELECTION,
            retrieval_goal=ResearchGoal.SYNTHESIS_SUPPORT,
            default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        )
        plan = ResearchPlan("synthesize")

        def analyzed(evidence_id: str, paper_id: str, text: str, score: float):
            return EvidenceRecord(
                evidence_id=evidence_id,
                evidence=RetrievedEvidence(
                    source_type=EvidenceSourceType.CONTENT,
                    paper_id=paper_id,
                    seed_chunk_ids=(evidence_id,),
                    source_chunk_ids=(evidence_id,),
                    content=text,
                    score=score,
                ),
                supporting_queries=(text,),
                first_seen_iteration=1,
                last_seen_iteration=1,
                assessment=EvidenceAnalysis(
                    evidence_id=evidence_id,
                    status=EvidenceAnalysisStatus.QUALIFIED,
                    digest=text,
                    claims=(
                    EvidenceClaim(
                        f"{evidence_id}:C1",
                        text,
                        EvidenceRelation.SUPPORTS,
                        EvidenceOwnership.THIS_PAPER,
                        "直接相关",
                        text,
                    ),
                    ),
                ),
            )

        records = (
            analyzed("E1", "A", "air pollution prediction", 1.0),
            analyzed("E2", "A", "wind speed forecasting transformer", 0.8),
            analyzed("E3", "B", "wind forecasting statistical method", 0.4),
        )
        visible = _visible_evidence(
            DecisionState(context, plan, records, (), BudgetUsage()), limit=2
        )
        self.assertEqual([item.evidence_id for item in visible], ["E2", "E3"])


    def test_compact_decision_omits_legacy_digest_claims_and_raw_excerpt(self) -> None:
        raw_marker = "RAW_EVIDENCE_MUST_NOT_ENTER_DECISION"
        pending = _record(raw_marker)
        claim = EvidenceClaim(
            claim_id="E1:C1",
            statement="本文使用 EMVE。",
            relation=EvidenceRelation.SUPPORTS,
            ownership=EvidenceOwnership.THIS_PAPER,
            rationale="原文明确陈述。",
            source_excerpt=raw_marker,
        )
        analyzed = EvidenceRecord(
            evidence_id=pending.evidence_id,
            evidence=pending.evidence,
            supporting_queries=pending.supporting_queries,
            first_seen_iteration=1,
            last_seen_iteration=1,
            requested_requirement_ids=("R1",),
            assessment=EvidenceAnalysis(
                evidence_id=pending.evidence_id,
                status=EvidenceAnalysisStatus.QUALIFIED,
                digest="本文方法摘要",
                claims=(claim,),
            ),
        )
        context = QueryContext(
            original_query="方法是什么？",
            resolved_query="方法是什么？",
            intent=Intent.RETRIEVE,
            scope=QueryScope.SINGLE_PAPER,
            target_paper_ids=("A",),
            scope_source=ScopeSource.UI_SELECTION,
            retrieval_goal=ResearchGoal.FIND_EVIDENCE,
            default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        )
        plan = ResearchPlan("查找方法")

        payload = json.loads(
            _state_prompt(
                DecisionState(context, plan, (analyzed,), (), BudgetUsage()), []
            )
        )

        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(raw_marker, serialized)
        self.assertNotIn("本文方法摘要", serialized)
        self.assertNotIn("本文使用 EMVE", serialized)
        self.assertEqual(payload["view_type"], "AtomicDecisionView")
        from paperbase.agentic.decision import _legacy_state_prompt
        legacy = _legacy_state_prompt(DecisionState(context, plan, (analyzed,), (), BudgetUsage()), [])
        self.assertIn("本文方法摘要", legacy)
        self.assertIn("本文使用 EMVE", legacy)
        self.assertNotIn(raw_marker, legacy)


if __name__ == "__main__":
    unittest.main()
