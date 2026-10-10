from __future__ import annotations

import unittest

from paperbase.agentic.checkpoint_serde import (
    AGENTIC_CHECKPOINT_NAMESPACE,
    checkpoint_thread_id,
    dump_contract,
    load_contract,
)
from paperbase.agentic.contracts import (
    AgentAction,
    ActionRejectionContext,
    BudgetUsage,
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    CoverageState,
    CoverageStrategy,
    FinishAction,
    GetSectionContextAction,
    GuardState,
    EvidenceTarget,
    EvidenceRecord,
    Intent,
    PaperSearchAttempt,
    QueryContext,
    QueryScope,
    RegistryState,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
    ScopeSource,
)
from paperbase.retrieval.contracts import (
    KnowledgeBaseSearchPayload,
    EvidenceSourceType,
    PaperOverviewPayload,
    PaperSearchOutcome,
    PaperSearchSpec,
    PaperSearchStatus,
    ReferenceSearchPayload,
    RetrievedEvidence,
    SearchPapersPayload,
    SectionContextPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


class AgenticCheckpointSerdeTests(unittest.TestCase):
    def test_reference_namespace_contract_invalidates_legacy_checkpoint_key(self) -> None:
        self.assertEqual(
            checkpoint_thread_id("thread-1"),
            f"{AGENTIC_CHECKPOINT_NAMESPACE}:thread-1",
        )
        self.assertNotEqual(checkpoint_thread_id("thread-1"), "thread-1")

    def test_legacy_coverage_fields_are_recomputed_on_restore(self) -> None:
        coverage = CoverageState(
            scope_paper_ids=("A", "B"), assessments=(),
            strategy=CoverageStrategy.EXHAUSTIVE,
            candidate_paper_ids=("A",), excluded_paper_ids=("B",),
        )
        legacy = dump_contract(coverage, CoverageState)
        self.assertNotIn("screened_paper_ids", legacy)
        self.assertNotIn("discovery_completed", legacy)
        legacy["screened_paper_ids"] = ["A"]
        legacy["discovery_completed"] = False
        restored = load_contract(legacy, CoverageState)
        self.assertEqual(restored.screened_paper_ids, ("A", "B"))

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

    def test_legacy_search_attempt_is_upgraded_in_observation(self) -> None:
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
                    status="empty",
                ),
            ),
        )
        dto = dump_contract((observation,), tuple[ResearchObservation, ...])
        dto[0]["search_attempts"][0]["missing_information"] = ["哪些场景失效"]
        dto[0]["search_attempts"][0]["target_subtask_id"] = "S2"
        restored = load_contract(dto, tuple[ResearchObservation, ...])
        canonical = dump_contract(restored, tuple[ResearchObservation, ...])

        self.assertTrue(_is_json_compatible(dto))
        attempt_dto = canonical[0]["search_attempts"][0]
        self.assertEqual(attempt_dto["requirement_ids"], ["S2"])
        self.assertNotIn("target_subtask_id", attempt_dto)
        self.assertNotIn("missing_information", attempt_dto)
        self.assertEqual(restored[0].search_attempts[0].requirement_ids, ("S2",))

    def test_legacy_observation_drops_unread_snapshot_fields(self) -> None:
        observation = ResearchObservation(
            iteration=1,
            action_name="search_papers",
            tool_status=ToolStatus.EMPTY,
            added_evidence_ids=(),
            reused_evidence_ids=(),
        )
        dto = dump_contract((observation,), tuple[ResearchObservation, ...])
        dto[0].update(
            paper_statuses=[["A", "empty"]],
            no_new_evidence=True,
            remaining_budget={"tool_calls": 3},
        )

        restored = load_contract(dto, tuple[ResearchObservation, ...])
        canonical = dump_contract(restored, tuple[ResearchObservation, ...])

        self.assertEqual(restored, (observation,))
        for field in ("paper_statuses", "no_new_evidence", "remaining_budget"):
            self.assertNotIn(field, canonical[0])

    def test_legacy_subtask_only_plan_becomes_required_requirements(self) -> None:
        legacy_plan = {
            "goal": "总结方法",
            "subtasks": [{
                "subtask_id": "S1",
                "goal": "识别方法",
                "target_paper_ids": ["A"],
                "evidence_targets": ["method"],
                "success_criteria": "找到方法证据",
                "status": "in_progress",
            }],
        }
        plan = load_contract(legacy_plan, ResearchPlan)
        self.assertFalse(hasattr(plan, "subtasks"))
        self.assertEqual(plan.requirements[0].requirement_id, "S1")
        self.assertTrue(plan.requirements[0].required)
        self.assertNotIn("subtasks", dump_contract(plan, ResearchPlan))

    def test_mixed_plan_keeps_old_subtask_optional(self) -> None:
        legacy_plan = {
            "goal": "总结方法",
            "requirements": [{
                "requirement_id": "R1",
                "description": "核心方法",
                "priority": "required",
                "evidence_targets": ["method"],
            }],
            "subtasks": [{
                "subtask_id": "S1",
                "goal": "旧任务",
                "evidence_targets": ["claim"],
            }],
        }
        plan = load_contract(legacy_plan, ResearchPlan)
        self.assertEqual([item.requirement_id for item in plan.requirements], ["R1", "S1"])
        self.assertFalse(plan.requirements[1].required)
        dto = dump_contract(plan, ResearchPlan)
        self.assertNotIn("subtasks", dto)
        self.assertNotIn("priority", dto["requirements"][0])

    def test_legacy_action_binding_is_migrated(self) -> None:
        action = SearchPapersAction(
            (PaperSearchSpec("A", "query", requirement_ids=("S1",)),),
        )
        dto = dump_contract(action, AgentAction)
        self.assertNotIn("target_subtask_id", dto)
        self.assertEqual(dto["searches"][0]["requirement_ids"], ["S1"])
        restored = load_contract(dto, AgentAction)
        self.assertEqual(restored.searches[0].requirement_ids, ("S1",))

        raw_legacy_action = {
            "action_name": "search_papers",
            "searches": [{
                "paper_id": "A",
                "query": "query",
                "target_subtask_id": "S1",
            }],
        }
        restored_raw = load_contract(raw_legacy_action, AgentAction)
        self.assertEqual(restored_raw.searches[0].requirement_ids, ("S1",))
        self.assertFalse(hasattr(restored_raw, "target_subtask_id"))

    def test_legacy_action_arguments_are_upgraded(self) -> None:
        rejection = ActionRejectionContext(
            action_name="search_papers",
            action_arguments={
                "target_subtask_id": "S1",
                "searches": [{"paper_id": "A", "query": "method"}],
            },
            requirement_ids=(),
            rejection_code="DuplicateAction",
            rejection_reason="already tried",
            guard_name="DuplicateAction",
            iteration=1,
        )
        dto = dump_contract(rejection, ActionRejectionContext)
        arguments = dto["action_arguments"]
        self.assertNotIn("target_subtask_id", arguments)
        self.assertEqual(arguments["searches"][0]["requirement_ids"], ["S1"])

    def test_source_metadata_is_not_migrated_as_a_contract(self) -> None:
        metadata = {
            "paper_id": "A", "query": "literal", "target_subtask_id": "raw",
            "missing_information": "source text",
        }
        result = ToolResult(
            status=ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload((PaperSearchOutcome(
                search=PaperSearchSpec(paper_id="A", query="method"),
                status=PaperSearchStatus.OK,
                evidence_candidates=(RetrievedEvidence(
                    source_type=EvidenceSourceType.CONTENT,
                    paper_id="A",
                    seed_chunk_ids=("C1",),
                    source_chunk_ids=("C1",),
                    content="source text",
                    retrieval_metadata=metadata,
                ),),
            ),)),
        )
        dto = dump_contract(result, ToolResult)
        outcome = dto["payload"]["paper_results"][0]
        self.assertEqual(outcome["search"]["query"], "method")
        self.assertNotIn("query", outcome)
        self.assertEqual(
            dto["payload"]["paper_results"][0]["evidence_candidates"][0]["retrieval_metadata"],
            metadata,
        )

    def test_legacy_evidence_binding_without_new_field_is_upgraded(self) -> None:
        evidence = RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="A",
            seed_chunk_ids=("C1",),
            source_chunk_ids=("C1",),
            content="A method",
        )
        record = EvidenceRecord(
            evidence_id="E1",
            evidence=evidence,
            supporting_queries=("method",),
            first_seen_iteration=1,
            last_seen_iteration=1,
        )
        legacy_dto = dump_contract((record,), tuple[EvidenceRecord, ...])
        legacy_dto[0].pop("requested_requirement_ids")
        legacy_dto[0]["target_subtask_ids"] = ["S1"]
        restored = load_contract(legacy_dto, tuple[EvidenceRecord, ...])
        self.assertEqual(restored[0].requested_requirement_ids, ("S1",))

    def test_evidence_analysis_has_one_checkpoint_object_and_loads_flat_legacy_fields(self) -> None:
        record = EvidenceRecord(
            evidence_id="E1",
            evidence=RetrievedEvidence(
                source_type=EvidenceSourceType.CONTENT,
                paper_id="A",
                seed_chunk_ids=("C1",),
                source_chunk_ids=("C1",),
                content="A method",
            ),
            supporting_queries=("method",),
            first_seen_iteration=1,
            last_seen_iteration=1,
        )
        dto = dump_contract((record,), tuple[EvidenceRecord, ...])
        self.assertIn("assessment", dto[0])
        self.assertNotIn("analysis_status", dto[0])
        self.assertNotIn("claims", dto[0])
        self.assertNotIn("requested_missing_information", dto[0])
        self.assertNotIn("addressed_information", dto[0]["assessment"])

        legacy = dto[0].copy()
        legacy.pop("assessment")
        legacy["requested_missing_information"] = ["旧缺口"]
        legacy["analysis_status"] = "qualified"
        legacy["digest"] = "A method"
        legacy["addressed_information"] = ["旧缺口"]
        legacy["claims"] = [{
            "claim_id": "E1:C1",
            "statement": "A method",
            "relation": "supports",
            "ownership": "this_paper",
            "rationale": "direct",
            "source_excerpt": "A method",
        }]
        restored = load_contract([legacy], tuple[EvidenceRecord, ...])[0]
        self.assertEqual(restored.assessment.status.value, "qualified")
        self.assertEqual(restored.assessment.claims[0].claim_id, "E1:C1")
        self.assertFalse(hasattr(restored, "requested_missing_information"))
        self.assertFalse(hasattr(restored.assessment, "addressed_information"))

        legacy.pop("analysis_status")
        legacy.pop("digest")
        legacy.pop("claims")
        legacy["qualification"] = {
            "evidence_id": "E1",
            "relation": "supports",
            "ownership": "this_paper",
            "rationale": "legacy",
        }
        restored_qualification = load_contract([legacy], tuple[EvidenceRecord, ...])[0]
        self.assertEqual(restored_qualification.assessment.status.value, "qualified")
        self.assertEqual(restored_qualification.assessment.claims[0].rationale, "legacy")

    def test_legacy_tool_result_drops_free_text_search_gap(self) -> None:
        result = ToolResult(
            status=ToolStatus.EMPTY,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload((
                PaperSearchOutcome(
                    search=PaperSearchSpec(paper_id="A", query="method"),
                    status=PaperSearchStatus.EMPTY,
                ),
            )),
        )
        dto = dump_contract(result, ToolResult)
        old_outcome = dto["payload"]["paper_results"][0]
        old_outcome.update(old_outcome.pop("search"))
        old_outcome["missing_information"] = ["旧缺口"]
        restored = load_contract(dto, ToolResult)
        restored_outcome = restored.payload.paper_results[0]
        self.assertFalse(hasattr(restored_outcome, "missing_information"))
        self.assertEqual(restored_outcome.search.query, "method")

    def test_context_flags_are_derived_after_loading_legacy_checkpoint(self) -> None:
        context = QueryContext(
            original_query="references",
            resolved_query="references",
            intent=Intent.RETRIEVE,
            scope=QueryScope.SINGLE_PAPER,
            target_paper_ids=("A",),
            scope_source=ScopeSource.UI_SELECTION,
            retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION,
            default_evidence_targets=frozenset({EvidenceTarget.REFERENCE}),
        )
        dto = dump_contract(context, QueryContext)
        self.assertNotIn("requires_complete_coverage", dto)
        self.assertNotIn("search_bibliography", dto)
        dto.update(requires_complete_coverage=True, search_bibliography=True)
        restored = load_contract(dto, QueryContext)
        self.assertTrue(restored.requires_complete_coverage)
        self.assertTrue(restored.search_bibliography)


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
