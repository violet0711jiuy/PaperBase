from __future__ import annotations

import json
import unittest
from dataclasses import replace

from paperbase.agentic.contracts import (
    BudgetUsage,
    CorpusScopeSnapshot,
    EvidenceRecord,
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    MechanismProjection,
    RequirementSupportAnchor,
    EvidenceTarget,
    FinishAction,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
    ScopeSource,
)
from paperbase.agentic.candidate_selection import LLMCandidateSelector, PaperMetadata
from paperbase.agentic.decision import DecisionState, LLMDecisionAgent
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
        goal="查找方法",
        requirements=(
            ResearchRequirement(
                "R1", "查找方法",
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
            ),
        ),
    )
    return DecisionState(context, plan, evidence, (), BudgetUsage())


def _supported_records() -> tuple[EvidenceRecord, ...]:
    records = []
    for index in range(1, 8):
        record = _record(index, content=f"evidence {index}", paper_id=f"P{index}")
        records.append(replace(record, assessment=EvidenceAnalysis(
            record.evidence_id, EvidenceAnalysisStatus.QUALIFIED,
            projection=MechanismProjection(record.evidence_id, "mechanism",
                supported_requirement_ids=("R1",), proposed_requirement_ids=("R1",),
                support_anchors=(RequirementSupportAnchor(
                    "R1", record.evidence.content,
                    f"verified local support {index}", ("other mechanism details",)
                ),)),
        )))
    return tuple(records)


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


    def test_decision_length_retry_reduces_view_detail_preserving_all_ids_and_runtime(self) -> None:
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
        result = LLMDecisionAgent(client, max_attempts=2).decide(_state(_supported_records()))

        first = json.loads(client.calls[0]["user_prompt"])
        second = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(len(first["requirements"][0]["local_supports"]), 6)
        self.assertEqual(len(second["requirements"][0]["local_supports"]), 3)
        self.assertEqual(second["requirements"][0]["local_supports_omitted"], 4)
        self.assertNotIn("verified_support_evidence_ids", second["requirements"][0])
        self.assertEqual(first["runtime_action_capabilities"], second["runtime_action_capabilities"])
        self.assertLess(result.llm_diagnostics[1].decision_view_metrics["decision_view_chars"],
                        result.llm_diagnostics[0].decision_view_metrics["decision_view_chars"])
        self.assertIsInstance(result.action, FinishAction)
        self.assertEqual(
            result.llm_diagnostics[0].recovery_action, "compact_context"
        )

    def test_decision_timeout_retry_really_compacts_the_view(self) -> None:
        client = _DiagnosticClient((
            LLMRequestError("timeout", error_stage="timeout"),
            LLMCompletion(
                json.dumps({"decision": {"action": "finish", "reason": "done"}}),
                "stop",
            ),
        ))

        result = LLMDecisionAgent(client, max_attempts=2).decide(
            _state(_supported_records())
        )

        first = json.loads(client.calls[0]["user_prompt"])
        second = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(len(first["requirements"][0]["local_supports"]), 6)
        self.assertEqual(len(second["requirements"][0]["local_supports"]), 3)
        self.assertLess(len(client.calls[1]["user_prompt"]), len(client.calls[0]["user_prompt"]))
        self.assertEqual(result.llm_diagnostics[0].recovery_action, "retry_request")


if __name__ == "__main__":
    unittest.main()
