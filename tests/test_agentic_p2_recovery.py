from __future__ import annotations

import json
import unittest

from paperbase.agentic.contracts import (
    BudgetUsage,
    CorpusScopeSnapshot,
    EvidenceRecord,
    EvidenceTarget,
    FinishAction,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchSubtask,
    ScopeSource,
)
from paperbase.agentic.candidate_selection import LLMCandidateSelector, PaperMetadata
from paperbase.agentic.decision import DecisionState, LLMDecisionAgent
from paperbase.agentic.evidence_analysis import LLMEvidenceAnalyzer
from paperbase.llm.client import LLMCompletion, LLMRequestError
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


class _DiagnosticClient:
    def __init__(self, outcomes: tuple[object, ...]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def complete_json_diagnostic(self, **kwargs) -> LLMCompletion:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


class _MetadataSource:
    def get_many(self, paper_ids):
        return tuple(PaperMetadata(paper_id, f"Paper {paper_id}") for paper_id in paper_ids)


def _record(
    index: int, *, content: str, paper_id: str = "A"
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"E{index}",
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id=paper_id,
            seed_chunk_ids=(f"c{index}",),
            source_chunk_ids=(f"c{index}",),
            content=content,
        ),
        supporting_queries=("method",),
        first_seen_iteration=1,
        last_seen_iteration=1,
    )


def _state(evidence: tuple[EvidenceRecord, ...]) -> DecisionState:
    paper_ids = tuple(dict.fromkeys(item.evidence.paper_id for item in evidence))
    context = QueryContext(
        original_query="A 使用什么方法？",
        resolved_query="A 使用什么方法？",
        intent=Intent.RETRIEVE,
        scope=(
            QueryScope.SINGLE_PAPER
            if len(paper_ids) == 1
            else QueryScope.SELECTED_PAPERS
        ),
        target_paper_ids=paper_ids,
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    plan = ResearchPlan(
        "查找方法",
        (
            ResearchSubtask(
                "S1",
                "查找方法",
                paper_ids,
                frozenset({EvidenceTarget.METHOD}),
                "找到证据",
            ),
        ),
    )
    return DecisionState(context, plan, evidence, (), BudgetUsage())


class AgenticP2RecoveryTests(unittest.TestCase):
    def test_domain_validation_uses_domain_repair_hint(self) -> None:
        incomplete = LLMCompletion(
            json.dumps(
                {
                    "assessments": [
                        {
                            "paper_id": "A",
                            "disposition": "candidate",
                            "reason": "相关",
                        }
                    ]
                }
            ),
            "stop",
        )
        complete = LLMCompletion(
            json.dumps(
                {
                    "assessments": [
                        {
                            "paper_id": paper_id,
                            "disposition": "candidate",
                            "reason": "相关",
                        }
                        for paper_id in ("A", "B")
                    ]
                }
            ),
            "stop",
        )
        client = _DiagnosticClient((incomplete, complete))

        result = LLMCandidateSelector(
            client, _MetadataSource(), max_attempts=2
        ).select(
            question="风速预测",
            snapshot=CorpusScopeSnapshot.create(("A", "B")),
        )

        second = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(result.llm_diagnostics[0].error_stage, "domain_validation")
        self.assertEqual(
            result.llm_diagnostics[0].recovery_action, "repair_domain"
        )
        self.assertIn("业务范围", second["previous_validation_errors"][0])

    def test_length_empty_content_compacts_evidence_analysis_retry(self) -> None:
        content = "ANCHOR " + "x" * 1_400
        valid = json.dumps(
            {
                "analyses": [
                    {
                        "evidence_id": "E1",
                        "status": "qualified",
                        "digest": "方法证据",
                        "claims": [
                            {
                                "statement": "原文给出方法锚点。",
                                "relation": "supports",
                                "ownership": "this_paper",
                                "rationale": "连续原文片段。",
                                "source_excerpt": "ANCHOR",
                            }
                        ],
                    }
                ]
            }
        )
        client = _DiagnosticClient(
            (
                LLMRequestError(
                    "没有最终内容",
                    error_stage="empty_content",
                    finish_reason="length",
                    prompt_tokens=900,
                    completion_tokens=3000,
                    reasoning_chars=8000,
                ),
                LLMCompletion(valid, "stop", 520, 180, 400),
            )
        )

        result = LLMEvidenceAnalyzer(
            client, max_content_chars=1_200, max_attempts=2
        ).analyze(question="方法是什么？", evidence=(_record(1, content=content),))

        first = json.loads(client.calls[0]["user_prompt"])
        second = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(len(first["evidence"][0]["content"]), 1_200)
        self.assertEqual(len(second["evidence"][0]["content"]), 600)
        self.assertEqual(result.llm_diagnostics[0].error_stage, "empty_content")
        self.assertEqual(
            result.llm_diagnostics[0].recovery_action, "compact_context"
        )
        self.assertTrue(result.llm_diagnostics[1].success)
        self.assertEqual(result.llm_diagnostics[0].reasoning_chars, 8000)

    def test_invalid_json_uses_structure_repair_without_compacting_context(self) -> None:
        valid = json.dumps(
            {
                "analyses": [
                    {
                        "evidence_id": "E1",
                        "status": "irrelevant",
                        "digest": "与问题无关",
                        "claims": [],
                    }
                ]
            }
        )
        client = _DiagnosticClient(
            (LLMCompletion("{bad", "stop"), LLMCompletion(valid, "stop"))
        )
        result = LLMEvidenceAnalyzer(client, max_attempts=2).analyze(
            question="方法是什么？", evidence=(_record(1, content="unrelated"),)
        )

        second = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(result.llm_diagnostics[0].error_stage, "invalid_json")
        self.assertEqual(
            result.llm_diagnostics[0].validation_summary, "json_invalid@root"
        )
        self.assertEqual(
            result.llm_diagnostics[0].recovery_action, "repair_structure"
        )
        self.assertIn("合法 JSON", second["previous_validation_errors"][0])

    def test_decision_length_retry_reduces_visible_evidence(self) -> None:
        client = _DiagnosticClient(
            (
                LLMRequestError(
                    "没有最终内容",
                    error_stage="empty_content",
                    finish_reason="length",
                ),
                LLMCompletion(
                    json.dumps(
                        {"decision": {"action": "finish", "reason": "done"}}
                    ),
                    "stop",
                ),
            )
        )
        result = LLMDecisionAgent(client, max_attempts=2).decide(
            _state(
                tuple(
                    _record(
                        index,
                        content=f"evidence {index}",
                        paper_id=f"P{index}",
                    )
                    for index in range(1, 8)
                )
            )
        )

        first = json.loads(client.calls[0]["user_prompt"])
        second = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(first["evidence_visibility"]["shown"], 7)
        self.assertEqual(second["evidence_visibility"]["shown"], 6)
        self.assertIsInstance(result.action, FinishAction)
        self.assertEqual(
            result.llm_diagnostics[0].recovery_action, "compact_context"
        )


if __name__ == "__main__":
    unittest.main()
