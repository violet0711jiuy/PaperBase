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
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchSubtask,
    ScopeSource,
)
from paperbase.agentic.decision import DecisionState, _state_prompt, _visible_evidence
from paperbase.agentic.evidence_analysis import LLMEvidenceAnalyzer
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


class _QueuedClient:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self._responses.pop(0)


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
        target_subtask_ids=("S1",),
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
        plan = ResearchPlan(
            "synthesize",
            (
                ResearchSubtask(
                    "S1",
                    "collect evidence",
                    ("A", "B"),
                    frozenset({EvidenceTarget.METHOD}),
                    "covered",
                ),
            ),
        )

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
                analysis_status=EvidenceAnalysisStatus.QUALIFIED,
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

    def test_one_evidence_can_separate_prior_work_and_this_paper_claims(self) -> None:
        response = json.dumps(
            {
                "analyses": [
                    {
                        "evidence_id": "E1",
                        "status": "qualified",
                        "digest": "该片段先介绍既有方法，再陈述本文方法。",
                        "claims": [
                            {
                                "statement": "既有研究使用 ARIMA。",
                                "relation": "background_only",
                                "ownership": "prior_work",
                                "rationale": "原文明确使用 Existing studies。",
                                "source_excerpt": "Existing studies use ARIMA.",
                            },
                            {
                                "statement": "本文提出 EMVE。",
                                "relation": "supports",
                                "ownership": "this_paper",
                                "rationale": "原文使用 We propose。",
                                "source_excerpt": "We propose EMVE.",
                            },
                        ],
                    }
                ]
            }
        )
        client = _QueuedClient((response,))

        result = LLMEvidenceAnalyzer(client, max_attempts=1).analyze(
            question="这篇论文提出了什么方法？",
            evidence=(_record(),),
        )

        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.analyses[0].status, EvidenceAnalysisStatus.QUALIFIED)
        self.assertEqual(
            [claim.claim_id for claim in result.analyses[0].claims],
            ["E1:C1", "E1:C2"],
        )
        self.assertEqual(
            [claim.ownership for claim in result.analyses[0].claims],
            [EvidenceOwnership.PRIOR_WORK, EvidenceOwnership.THIS_PAPER],
        )

    def test_invalid_source_excerpt_retries_then_fails_closed(self) -> None:
        invalid = json.dumps(
            {
                "analyses": [
                    {
                        "evidence_id": "E1",
                        "status": "qualified",
                        "digest": "看似相关。",
                        "claims": [
                            {
                                "statement": "本文提出不存在的方法。",
                                "relation": "supports",
                                "ownership": "this_paper",
                                "rationale": "模型产生了错误锚点。",
                                "source_excerpt": "This sentence is not in the evidence.",
                            }
                        ],
                    }
                ]
            }
        )

        result = LLMEvidenceAnalyzer(
            _QueuedClient((invalid, invalid)), max_attempts=2
        ).analyze(question="方法是什么？", evidence=(_record(),))

        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.analyses[0].status, EvidenceAnalysisStatus.FAILED)
        self.assertEqual(result.analyses[0].claims, ())
        self.assertFalse(FinalEvidenceSelector._eligible(_record()))

    def test_decision_receives_digest_and_claims_but_not_raw_excerpt(self) -> None:
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
            target_subtask_ids=("S1",),
            analysis_status=EvidenceAnalysisStatus.QUALIFIED,
            digest="本文方法摘要",
            claims=(claim,),
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
        plan = ResearchPlan(
            "查找方法",
            (
                ResearchSubtask(
                    "S1", "查找方法", ("A",), frozenset({EvidenceTarget.METHOD}), "找到证据"
                ),
            ),
        )

        payload = json.loads(
            _state_prompt(
                DecisionState(context, plan, (analyzed,), (), BudgetUsage()), []
            )
        )

        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(raw_marker, serialized)
        self.assertIn("本文方法摘要", serialized)
        self.assertIn("本文使用 EMVE", serialized)


if __name__ == "__main__":
    unittest.main()
