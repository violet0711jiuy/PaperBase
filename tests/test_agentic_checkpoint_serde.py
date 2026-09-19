from __future__ import annotations

import unittest

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    AgentAction,
    BudgetUsage,
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    FinishAction,
    GetSectionContextAction,
    GuardState,
    EvidenceTarget,
    PaperSearchAttempt,
    RegistryState,
    ResearchObservation,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)
from paperbase.retrieval.contracts import (
    KnowledgeBaseSearchPayload,
    PaperOverviewPayload,
    PaperSearchSpec,
    ReferenceSearchPayload,
    SearchPapersPayload,
    SectionContextPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


class AgenticCheckpointSerdeTests(unittest.TestCase):
    def test_all_actions_round_trip_through_json_compatible_dto(self) -> None:
        actions: tuple[AgentAction, ...] = (
            SearchPapersAction((PaperSearchSpec("A", "query"),)),
            SearchKnowledgeBaseAction("query", ("term",)),
            SearchReferencesAction("A", "citation", ("term",)),
            GetSectionContextAction("A", "section_1"),
            FinishAction("done"),
        )
        for action in actions:
            with self.subTest(action=action.action_name):
                dto = dump_contract(action, AgentAction)
                self.assertTrue(_is_json_compatible(dto))
                restored = load_contract(dto, AgentAction)
                self.assertEqual(restored, action)

    def test_all_tool_payloads_keep_their_discriminated_type(self) -> None:
        payloads = (
            SearchPapersPayload(()),
            KnowledgeBaseSearchPayload("query"),
            ReferenceSearchPayload("A", "query"),
            SectionContextPayload("A", "section_1"),
            PaperOverviewPayload("A"),
        )
        for payload in payloads:
            with self.subTest(payload=payload.kind):
                result = ToolResult(
                    status=ToolStatus.EMPTY,
                    tool_name=ToolName(payload.kind),
                    payload=payload,
                )
                dto = dump_contract(result, ToolResult)
                restored = load_contract(dto, ToolResult)
                self.assertTrue(_is_json_compatible(dto))
                self.assertIs(type(restored.payload), type(payload))

    def test_guard_tuple_state_is_restored_exactly(self) -> None:
        state = GuardState(
            usage=BudgetUsage(
                agent_iterations=2,
                tool_calls=1,
                retrieval_queries=1,
                llm_calls=2,
                invalid_actions=1,
                queries_per_paper=(("A", 1),),
            ),
            no_new_per_paper=(("A", 2),),
            suppressed_paper_ids=("A",),
            executed_action_fingerprints=("fingerprint",),
            executed_paper_query_fingerprints=("paper-query",),
        )
        dto = dump_contract(state, GuardState)
        restored = load_contract(dto, GuardState)
        self.assertEqual(restored, state)
        self.assertIsInstance(restored.usage.queries_per_paper, tuple)

    def test_empty_registry_state_is_json_compatible(self) -> None:
        dto = dump_contract(RegistryState(), RegistryState)
        self.assertTrue(_is_json_compatible(dto))
        self.assertEqual(load_contract(dto, RegistryState), RegistryState())

    def test_candidate_selection_round_trips_as_checkpoint_dto(self) -> None:
        selection = CandidateSelection(
            scope_paper_ids=("A", "B", "C"),
            assessments=(
                CandidateAssessment("A", CandidateDisposition.CANDIDATE, "直接相关"),
                CandidateAssessment("B", CandidateDisposition.UNCERTAIN, "摘要不足"),
                CandidateAssessment("C", CandidateDisposition.EXCLUDED, "任务无关"),
            ),
        )
        dto = dump_contract(selection, CandidateSelection)
        self.assertTrue(_is_json_compatible(dto))
        self.assertEqual(load_contract(dto, CandidateSelection), selection)

    def test_search_attempt_intent_round_trips_in_observation(self) -> None:
        observation = ResearchObservation(
            iteration=2,
            action_name="search_papers",
            tool_status=ToolStatus.EMPTY,
            added_evidence_ids=(),
            reused_evidence_ids=(),
            search_attempts=(
                PaperSearchAttempt(
                    paper_id="A",
                    query="方法边界",
                    lexical_terms=("limitation",),
                    evidence_targets=frozenset({EvidenceTarget.CLAIM}),
                    missing_information=("哪些场景失效",),
                    target_subtask_id="S2",
                    status="empty",
                ),
            ),
        )
        dto = dump_contract((observation,), tuple[ResearchObservation, ...])
        restored = load_contract(dto, tuple[ResearchObservation, ...])

        self.assertTrue(_is_json_compatible(dto))
        self.assertEqual(restored, (observation,))


def _is_json_compatible(value) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_compatible(item)
            for key, item in value.items()
        )
    return False


if __name__ == "__main__":
    unittest.main()
