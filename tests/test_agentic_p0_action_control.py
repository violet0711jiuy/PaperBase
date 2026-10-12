from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from paperbase.agentic.action_control import ActionCanonicalizer
from paperbase.agentic.audit import build_agentic_audit_snapshot
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    ActionCanonicalization,
    ActionRejectionContext,
    BudgetUsage,
    EvidenceRecord,
    EvidenceTarget,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchObservation,
    ResearchPlan,
    ResearchRequirement,
    RequirementState,
    RequirementStatus,
    ScopeSource,
    SearchPapersAction,
)
from paperbase.agentic.decision import (
    DecisionError,
    DecisionState,
    LLMDecisionAgent,
    _decision_json_schema,
    _repair_client_if_needed,
    _state_prompt,
)
from paperbase.agentic.guards import BudgetAndGuardTracker, action_fingerprint
from paperbase.agentic.graph import _recursion_limit
from paperbase.agentic.schemas import AGENT_DECISION_ADAPTER
from paperbase.config import AgenticResearchSettings
from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchSpec,
    RetrievedEvidence,
    ToolStatus,
)


class _QueuedClient:
    def __init__(self, values: tuple[str, ...]) -> None:
        self.values = list(values)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.values.pop(0)


class _RepairAwareClient(_QueuedClient):
    def __init__(self, values: tuple[str, ...]) -> None:
        super().__init__(values)
        self.repair_view_calls = 0

    def with_repair_temperature(self):
        self.repair_view_calls += 1
        return self


def _context() -> QueryContext:
    return QueryContext(
        original_query="A 的方法是什么？",
        resolved_query="A 的方法是什么？",
        intent=Intent.RETRIEVE,
        scope=QueryScope.SELECTED_PAPERS,
        target_paper_ids=("A", "B"),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        "识别研究方法",
        requirements=(
            ResearchRequirement(
                requirement_id="R1",
                description="识别论文采用的方法",
                required=True,
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
            ),
            ResearchRequirement(
                requirement_id="R2",
                description="识别实验设置",
                required=False,
                evidence_targets=frozenset({EvidenceTarget.RESULT}),
            ),
        ),
    )


def _search(
    *,
    query: str = "method",
    lexical_terms: list[str] | None = None,
) -> dict[str, object]:
    return {
        "paper_id": "A",
        "semantic_query": query,
        "lexical_terms": lexical_terms if lexical_terms is not None else ["method"],
    }


def _decision(search: dict[str, object], requirement_id: str = "R1") -> str:
    return json.dumps(
        {"decision": {
            "action": "search_papers",
            "requirement_id": requirement_id,
            "searches": [search],
        }},
        ensure_ascii=False,
    )


def _state(
    *,
    evidence: tuple[EvidenceRecord, ...] = (),
    observations: tuple[ResearchObservation, ...] = (),
    action_rejections: tuple[ActionRejectionContext, ...] = (),
    last_action_rejection: ActionRejectionContext | None = None,
    budget_usage: BudgetUsage | None = None,
) -> DecisionState:
    return DecisionState(
        context=_context(),
        plan=_plan(),
        evidence=evidence,
        observations=observations,
        budget_usage=budget_usage or BudgetUsage(),
        action_rejections=action_rejections,
        last_action_rejection=last_action_rejection,
        per_paper_query_limit=2,
    )


def _record_with_section() -> EvidenceRecord:
    return EvidenceRecord(
        "E1",
        RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="A",
            section_id="methods",
            seed_chunk_ids=("c1",),
            source_chunk_ids=("c1",),
            content="Method evidence.",
        ),
        ("method",),
        1,
        1,
    )


class ActionControlPlaneTests(unittest.TestCase):
    def test_cross_requirement_identifier_is_repaired_before_tool_execution(self) -> None:
        plan = ResearchPlan(
            "Explain EMVE loss components",
            requirements=(
                ResearchRequirement(
                    "R1", "Explain how MPIWcapt controls interval width in EMVE.",
                    frozenset({EvidenceTarget.METHOD}),
                ),
                ResearchRequirement(
                    "R2", "Explain how C_i controls coverage in EMVE.",
                    frozenset({EvidenceTarget.METHOD}),
                ),
            ),
        )
        broad = _search(
            query="How do MPIWcapt and C_i control width and coverage in EMVE?",
            lexical_terms=["MPIWcapt", "C_i", "EMVE"],
        )
        scoped = _search(
            query="How does MPIWcapt control interval width in EMVE?",
            lexical_terms=["MPIWcapt", "EMVE"],
        )

        result = LLMDecisionAgent(
            _QueuedClient((_decision(broad), _decision(scoped)))
        ).decide(replace(_state(), plan=plan))

        self.assertEqual(result.llm_calls, 2)
        diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
        self.assertEqual(diagnostic.error_code, "CrossRequirementSearchScope")
        self.assertEqual(diagnostic.field_path, "searches[0].semantic_query")
        self.assertEqual(result.action.searches[0].requirement_ids, ("R1",))
        self.assertNotIn("C_i", result.action.searches[0].query)

    def test_shared_context_identifier_does_not_trigger_scope_repair(self) -> None:
        plan = ResearchPlan(
            "Explain EMVE loss components",
            requirements=(
                ResearchRequirement(
                    "R1", "Explain MPIWcapt in LSTM-EMVE.",
                    frozenset({EvidenceTarget.METHOD}),
                ),
                ResearchRequirement(
                    "R2", "Explain C_i in LSTM-EMVE.",
                    frozenset({EvidenceTarget.METHOD}),
                ),
            ),
        )
        scoped = _search(
            query="How does MPIWcapt work in LSTM-EMVE?",
            lexical_terms=["MPIWcapt", "LSTM-EMVE"],
        )

        result = LLMDecisionAgent(_QueuedClient((_decision(scoped),))).decide(
            replace(_state(), plan=plan)
        )

        self.assertEqual(result.llm_calls, 1)
        self.assertFalse(result.llm_diagnostics[0].validation_diagnostics)

    def test_decision_view_omits_global_query_and_retrieval_hints(self) -> None:
        payload = json.loads(_state_prompt(_state(), []))

        self.assertNotIn("question", payload)
        self.assertNotIn("research_goal", payload)
        self.assertNotIn("initial_search_hints", payload)
        self.assertNotIn("available_retrieval_targets", payload["requirements"][0])
        self.assertIn("最多选择少量", AGENT_DECISION_SYSTEM_PROMPT)

    def test_decision_sees_global_iteration_limit_and_remaining_rounds(self) -> None:
        state = replace(
            _state(budget_usage=BudgetUsage(agent_iterations=7)),
            max_agent_iterations=8,
        )

        payload = json.loads(_state_prompt(state, []))

        self.assertEqual(
            payload["iteration_budget"],
            {"limit": 8, "used": 7, "remaining": 1},
        )
        self.assertEqual(payload["iteration_budget"]["remaining"], 1)

    def test_in_progress_requirement_does_not_invent_answer_support(self) -> None:
        state = replace(
            _state(),
            requirement_states=(
                RequirementState("R1", RequirementStatus.IN_PROGRESS, ("E1",)),
                RequirementState("R2", RequirementStatus.PENDING),
            ),
        )

        payload = json.loads(_state_prompt(state, []))

        # Retrieval state never becomes an accepted answer without Grounded Answer.
        self.assertIsNone(payload["requirements"][0]["accepted_answer"])
        self.assertNotIn("local_supports", payload["requirements"][0])
        self.assertNotIn("state_evidence_ids", payload["requirements"][0])
        self.assertNotIn("verified_support_evidence_ids", payload["requirements"][0])
        self.assertNotIn("retrieval_state", payload["requirements"][0])
        self.assertIn("Grounded Answer", AGENT_DECISION_SYSTEM_PROMPT)
        self.assertIn("Requirement", AGENT_DECISION_SYSTEM_PROMPT)
        self.assertIn(
            "\u4e0d\u590d\u8ff0\u6574\u9053\u591a\u90e8\u5206\u95ee\u9898",
            AGENT_DECISION_SYSTEM_PROMPT,
        )

    def test_overlong_lexical_terms_are_reselected_instead_of_truncated(self) -> None:
        too_many = ["2019", "成都", "重庆", "冬季", "大气消光", "2001-2016"]
        chosen = ["2019", "成都", "重庆", "大气消光", "2001-2016"]
        client = _QueuedClient((
            _decision(_search(lexical_terms=too_many)),
            _decision(_search(lexical_terms=chosen)),
        ))

        result = LLMDecisionAgent(client).decide(_state())

        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.action.searches[0].lexical_terms, tuple(chosen))
        self.assertIn("最多选择少量", client.calls[0]["system_prompt"])

    def test_schema_diagnostic_is_stable_and_repair_receives_field_constraint(self) -> None:
        invalid = _search()
        del invalid["semantic_query"]
        client = _QueuedClient((_decision(invalid), _decision(_search())))

        result = LLMDecisionAgent(client).decide(_state())

        diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
        self.assertEqual(diagnostic.error_stage, "schema_validation")
        self.assertEqual(diagnostic.error_code, "MissingRequiredField")
        self.assertEqual(diagnostic.field_path, "searches[0].semantic_query")
        repair = json.loads(client.calls[1]["user_prompt"])
        self.assertEqual(repair["previous_validation_errors"][0]["error_code"], "MissingRequiredField")
        self.assertEqual(repair["previous_validation_errors"][0]["field_path"], "searches[0].semantic_query")

    def test_domain_diagnostic_locates_invalid_requirement_id(self) -> None:
        client = _QueuedClient(
            (_decision(_search(), "R9"), _decision(_search()))
        )

        result = LLMDecisionAgent(client).decide(_state())

        diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
        self.assertEqual(diagnostic.error_stage, "domain_validation")
        self.assertEqual(diagnostic.error_code, "InvalidRequirementId")
        self.assertEqual(diagnostic.field_path, "requirement_id")
        self.assertEqual(diagnostic.actual_summary, "unknown_id=R9")

    def test_domain_diagnostic_locates_invalid_section_id(self) -> None:
        invalid = json.dumps(
            {
                "decision": {
                    "action": "get_section_context",
                    "paper_id": "A",
                    "section_id": "invented",
                    "requirement_id": "R1",
                }
            }
        )
        client = _QueuedClient((invalid, _decision(_search())))

        result = LLMDecisionAgent(client).decide(_state())

        diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
        self.assertEqual(diagnostic.error_code, "InvalidSectionId")
        self.assertEqual(diagnostic.field_path, "section_id")

    def test_validation_diagnostic_does_not_store_raw_model_content(self) -> None:
        sensitive_marker = "hidden-reasoning-or-user-private-content"
        invalid = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "searches": [_search()],
                    "unexpected_field": sensitive_marker,
                }
            }
        )
        result = LLMDecisionAgent(
            _QueuedClient((invalid, _decision(_search())))
        ).decide(_state())

        serialized = json.dumps(
            asdict(result.llm_diagnostics[0].validation_diagnostics[0])
        )
        self.assertNotIn(sensitive_marker, serialized)
        self.assertEqual(
            result.llm_diagnostics[0].validation_diagnostics[0].error_code,
            "UnknownField",
        )

    def test_canonicalizer_normalizes_safe_fields_without_dropping_unknown_ids(self) -> None:
        lexical_terms = [" Method ", "", "method", "context", "a", "b", "c", "d", "e", "f", "g"]
        raw = {
            "decision": {
                "action": "search_papers",
                "requirement_id": " R9 ",
                "target_subtask_id": "legacy",
                "searches": [
                    {
                        **_search(
                            query="  method   details ",
                            lexical_terms=lexical_terms,
                        ),
                        "missing_information": "legacy",
                    }
                ],
            }
        }

        normalized, audit = ActionCanonicalizer().canonicalize(raw)
        search = normalized["decision"]["searches"][0]

        self.assertEqual(search["semantic_query"], "method details")
        self.assertEqual(normalized["decision"]["requirement_id"], "R9")
        self.assertEqual(search["lexical_terms"], ["Method", "context", "a", "b", "c", "d", "e", "f", "g"])
        self.assertNotIn("target_subtask_id", normalized["decision"])
        self.assertNotIn("missing_information", search)
        self.assertTrue(audit.changed)
        self.assertFalse(any(item.operation == "truncated" for item in audit.operations))
        with self.assertRaises(ValidationError):
            AGENT_DECISION_ADAPTER.validate_json(json.dumps(normalized))

    def test_unknown_schema_field_is_not_silently_canonicalized_away(self) -> None:
        raw = json.loads(_decision(_search()))
        raw["decision"]["top_k"] = 99
        normalized, _ = ActionCanonicalizer().canonicalize(raw)

        with self.assertRaises(ValidationError):
            AGENT_DECISION_ADAPTER.validate_json(json.dumps(normalized))

    def test_unknown_requirement_survives_canonicalization_and_fails_domain_validation(self) -> None:
        with self.assertRaises(DecisionError) as raised:
            LLMDecisionAgent(
                _QueuedClient((_decision(_search(), " R9 "),)),
                max_attempts=1,
            ).decide(_state())

        diagnostic = raised.exception.llm_diagnostics[0].validation_diagnostics[0]
        self.assertEqual(diagnostic.error_code, "InvalidRequirementId")
        self.assertEqual(diagnostic.actual_summary, "unknown_id=R9")

    def test_fingerprint_uses_canonical_query_and_lexical_form(self) -> None:
        first = SearchPapersAction(
            (
                PaperSearchSpec(
                    "A", " Method   details ", ("method", "context"), requirement_ids=("R1",)
                ),
            )
        )
        equivalent = SearchPapersAction(
            (
                PaperSearchSpec(
                    "A", "Method details", ("CONTEXT", "METHOD"), requirement_ids=("R1",)
                ),
            )
        )

        self.assertEqual(action_fingerprint(first), action_fingerprint(equivalent))

    def test_duplicate_guard_rejects_canonical_equivalent_but_allows_material_change(self) -> None:
        tracker = BudgetAndGuardTracker(AgenticResearchSettings())
        first = SearchPapersAction(
            (PaperSearchSpec("A", "method details", ("method", "context"), requirement_ids=("R1",)),)
        )
        equivalent = SearchPapersAction(
            (PaperSearchSpec("A", " method details ", ("CONTEXT", "METHOD"), requirement_ids=("R1",)),)
        )
        changed = SearchPapersAction(
            (PaperSearchSpec("A", "experimental setup", ("experiment",), requirement_ids=("R1",)),)
        )

        self.assertTrue(tracker.validate_action(first, context=_context(), plan=_plan(), snapshot=None).accepted)
        tracker.record_tool_call(first)
        self.assertEqual(
            tracker.validate_action(equivalent, context=_context(), plan=_plan(), snapshot=None).code,
            "DuplicateAction",
        )
        self.assertTrue(tracker.validate_action(changed, context=_context(), plan=_plan(), snapshot=None).accepted)

    def test_known_duplicate_is_left_to_runtime_guard_for_replan(self) -> None:
        duplicate = _decision(_search())
        client = _QueuedClient((duplicate,))
        observation = ResearchObservation(
            iteration=1,
            action_name="search_papers",
            tool_status=ToolStatus.OK,
            added_evidence_ids=("E1",),
            reused_evidence_ids=(),
            action_arguments={"searches": [_search()]},
        )

        result = LLMDecisionAgent(client).decide(_state(observations=(observation,)))
        self.assertEqual(result.action.searches[0].query, "method")
        self.assertFalse(result.llm_diagnostics[0].validation_diagnostics)

    def test_decision_repair_preserves_zero_temperature_configuration(self) -> None:
        client = _RepairAwareClient(())
        self.assertIs(
            _repair_client_if_needed(
                client, has_validation_repair=False, has_runtime_rejection=False
            ),
            client,
        )
        self.assertEqual(client.repair_view_calls, 0)
        self.assertIs(
            _repair_client_if_needed(
                client, has_validation_repair=True, has_runtime_rejection=False
            ),
            client,
        )
        self.assertEqual(client.repair_view_calls, 0)

    def test_provider_schema_uses_known_sections_not_raw_rejection_exclusions(self) -> None:
        rejection = ActionRejectionContext(
            action_name="get_section_context",
            action_arguments={"paper_id": "A", "section_id": "methods"},
            requirement_ids=("R1",),
            rejection_code="DuplicateAction",
            rejection_reason="same action",
            guard_name="DuplicateAction",
            iteration=2,
        )

        schema = _decision_json_schema(_state(last_action_rejection=rejection))
        mapping = schema["properties"]["decision"]["discriminator"]["mapping"]
        # A rejection is feedback, not authority to invent a known section.
        self.assertNotIn("get_section_context", mapping)

    def test_provider_schema_bounds_paper_search_batch_by_valid_papers(self) -> None:
        single = _state()
        single = replace(
            single,
            context=replace(
                single.context,
                scope=QueryScope.SINGLE_PAPER,
                target_paper_ids=("A",),
            ),
        )
        single_schema = _decision_json_schema(single)
        self.assertEqual(
            single_schema["$defs"]["SearchPapersDraft"]["properties"]["searches"]["maxItems"],
            1,
        )
        self.assertEqual(
            single_schema["$defs"]["SearchSpecDraft"]["properties"]["paper_id"]["enum"],
            ["A"],
        )
        multi_schema = _decision_json_schema(_state())
        self.assertEqual(
            multi_schema["$defs"]["SearchPapersDraft"]["properties"]["searches"]["maxItems"],
            2,
        )

    def test_duplicate_paper_search_diagnostic_points_to_repeated_id(self) -> None:
        invalid = json.dumps(
            {
                "decision": {
                    "action": "search_papers",
                    "requirement_id": "R1",
                    "searches": [_search(), _search(query="experiment")],
                }
            }
        )
        client = _QueuedClient((invalid, _decision(_search())))
        result = LLMDecisionAgent(client).decide(_state())
        diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
        self.assertEqual(diagnostic.error_code, "DuplicatePaperSearch")
        self.assertEqual(diagnostic.field_path, "searches[1].paper_id")
        self.assertEqual(diagnostic.actual_summary, "duplicate_count=1")

    def test_runtime_capability_envelope_is_deterministic_and_non_prescriptive(self) -> None:
        observation = ResearchObservation(
            iteration=1,
            action_name="search_papers",
            tool_status=ToolStatus.OK,
            added_evidence_ids=("E1",),
            reused_evidence_ids=(),
            action_arguments={"searches": [_search()]},
        )
        rejection = ActionRejectionContext(
            action_name="search_papers",
            action_arguments={"searches": [_search()]},
            requirement_ids=("R1",),
            rejection_code="DuplicateAction",
            rejection_reason="same action",
            guard_name="DuplicateAction",
            iteration=2,
        )
        payload = json.loads(
            _state_prompt(
                _state(
                    evidence=(_record_with_section(),),
                    observations=(observation,),
                    action_rejections=(rejection,),
                    last_action_rejection=rejection,
                    budget_usage=BudgetUsage(queries_per_paper=(("A", 1),)),
                ),
                [],
            )
        )

        from paperbase.agentic.decision_view import expand_decision_references
        payload = expand_decision_references(payload)
        capabilities = payload["runtime_action_capabilities"]
        recovery = payload["action_recovery"]
        self.assertTrue(recovery["required"])
        self.assertEqual(recovery["rejection_code"], "DuplicateAction")
        self.assertEqual(recovery["blocked_action"]["action_name"], "search_papers")
        self.assertEqual(capabilities["valid_requirement_ids"], ["R1", "R2"])
        self.assertEqual(capabilities["search_paper_ids"], ["A", "B"])
        self.assertEqual(
            capabilities["available_section_ids_by_paper"], [["A", ["methods"]]]
        )
        self.assertNotIn("tool_budget_state", capabilities)
        self.assertNotIn("next_action", capabilities)
        self.assertNotIn("recent_unavailable_actions", capabilities)
        self.assertNotIn("unavailable_paper_queries", capabilities)

    def test_section_capabilities_and_schema_exclude_an_executed_context(self) -> None:
        record = _record_with_section()
        observation = ResearchObservation(
            iteration=1,
            action_name="get_section_context",
            tool_status=ToolStatus.OK,
            added_evidence_ids=("E1",),
            reused_evidence_ids=(),
            action_arguments={"paper_id": "A", "section_id": "methods"},
        )
        state = _state(evidence=(record,), observations=(observation,))

        payload = json.loads(_state_prompt(state, []))
        schema = _decision_json_schema(state)

        capabilities = payload["runtime_action_capabilities"]
        self.assertEqual(capabilities["available_section_ids_by_paper"], [])
        self.assertNotIn("get_section_context", capabilities["available_tools"])
        mapping = schema["properties"]["decision"]["discriminator"]["mapping"]
        self.assertNotIn("get_section_context", mapping)

    def test_recursion_limit_covers_every_successful_action_node(self) -> None:
        settings = AgenticResearchSettings(max_agent_iterations=8)

        self.assertGreaterEqual(_recursion_limit(settings), 8 * 7 + 6)

    def test_canonicalization_checkpoint_and_validation_audit_round_trip(self) -> None:
        canonicalization = ActionCanonicalization()
        self.assertEqual(
            load_contract(
                dump_contract(canonicalization, ActionCanonicalization),
                ActionCanonicalization,
            ),
            canonicalization,
        )
        invalid = _search()
        del invalid["semantic_query"]
        decision = LLMDecisionAgent(
            _QueuedClient((_decision(invalid), _decision(_search())))
        ).decide(_state())
        result = SimpleNamespace(
            status="failed",
            stop_reason="invalid_action_limit",
            thread_id="test-thread",
            query_context=_context(),
            plan=_plan(),
            evidence=(),
            selected_evidence=(),
            observations=(),
            budget_usage=BudgetUsage(),
            coverage=None,
            candidate_selection=None,
            answer=None,
            llm_diagnostics=decision.llm_diagnostics,
            action_rejections=(),
            error_codes=(),
            clarification=None,
        )

        audit = build_agentic_audit_snapshot(result)

        self.assertEqual(audit.validation_errors[0].error_stage, "schema_validation")
        self.assertEqual(audit.validation_errors[0].field_path, "searches[0].semantic_query")
        self.assertNotIn("hidden-reasoning", audit.to_json())


if __name__ == "__main__":
    unittest.main()
