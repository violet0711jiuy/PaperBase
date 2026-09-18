from __future__ import annotations

import unittest

from langgraph.checkpoint.memory import InMemorySaver

from paperbase.agentic.contracts import (
    EvidenceOwnership,
    EvidenceQualification,
    EvidenceRelation,
    EvidenceTarget,
    FinishAction,
    ResearchRunStatus,
    SearchPapersAction,
    StopReason,
)
from paperbase.agentic.decision import DecisionResult, ScriptedDecisionAgent
from paperbase.agentic.graph import LangGraphResearchWorkflow
from paperbase.agentic.planner import ResearchPlanner
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.runtime import ResearchRuntime
from paperbase.agentic.scope_resolver import ScopeResolver
from paperbase.agentic.service import AgenticResearchService
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchOutcome,
    PaperSearchSpec,
    PaperSearchStatus,
    RetrievedEvidence,
    SearchPapersPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


class _CatalogDatabase:
    def __init__(self) -> None:
        self.rows = (
            {"paper_id": "A", "paper_title": "Paper A", "source_filename": "a.pdf"},
            {"paper_id": "B", "paper_title": "Paper B", "source_filename": "b.pdf"},
        )

    def list_documents(self):
        return self.rows

    def get_document(self, paper_id: str):
        return next((row for row in self.rows if row["paper_id"] == paper_id), None)


class _Executor:
    def __init__(self, empty_queries=()) -> None:
        self.empty_queries = set(empty_queries)
        self.calls = []

    def execute(self, action, **kwargs):
        self.calls.append(action)
        outcomes = []
        for search in action.searches:
            empty = search.query in self.empty_queries
            evidence = () if empty else (
                RetrievedEvidence(
                    source_type=EvidenceSourceType.CONTENT,
                    paper_id=search.paper_id,
                    seed_chunk_ids=(f"{search.paper_id}_{search.query}",),
                    source_chunk_ids=(f"{search.paper_id}_{search.query}",),
                    content=f"{search.paper_id} evidence for {search.query}",
                ),
            )
            outcomes.append(
                PaperSearchOutcome(
                    paper_id=search.paper_id,
                    status=PaperSearchStatus.EMPTY if empty else PaperSearchStatus.OK,
                    query=search.query,
                    lexical_terms=search.lexical_terms,
                    evidence_candidates=evidence,
                )
            )
        return ToolResult(
            status=(
                ToolStatus.EMPTY
                if all(not outcome.evidence_candidates for outcome in outcomes)
                else ToolStatus.OK
            ),
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(tuple(outcomes)),
        )


class _EvidenceAwareDecisionAgent:
    """无进程内游标，恢复后只根据 checkpoint state 决定下一步。"""

    def decide(self, state):
        paper_ids = {record.evidence.paper_id for record in state.evidence}
        if "A" not in paper_ids:
            return DecisionResult(_action("A", "A method", "S1"), 1)
        if "B" not in paper_ids:
            return DecisionResult(_action("B", "B method after A", "S2"), 1)
        return DecisionResult(FinishAction("both papers covered"), 1)


class _QualifyingDecisionAgent:
    """在选择下一步动作的同一次语义调用中标注上一轮证据。"""

    def decide(self, state):
        qualifications = tuple(
            EvidenceQualification(
                record.evidence_id,
                EvidenceRelation.SUPPORTS,
                EvidenceOwnership.THIS_PAPER,
                "该片段直接描述本文方法。",
            )
            for record in state.evidence
            if record.qualification is None
        )
        paper_ids = {record.evidence.paper_id for record in state.evidence}
        if "A" not in paper_ids:
            return DecisionResult(_action("A", "A method", "S1"), 1, qualifications)
        if "B" not in paper_ids:
            return DecisionResult(_action("B", "B method", "S2"), 1, qualifications)
        return DecisionResult(FinishAction("qualified"), 1, qualifications)


def _action(paper_id: str, query: str, subtask_id: str) -> SearchPapersAction:
    return SearchPapersAction(
        searches=(
            PaperSearchSpec(
                paper_id=paper_id,
                query=query,
                evidence_targets=frozenset({EvidenceTarget.METHOD.value}),
                target_subtask_id=subtask_id,
            ),
        ),
        target_subtask_id=subtask_id,
    )


def _components():
    resolver = ScopeResolver(_CatalogDatabase())
    understanding = QueryUnderstandingService()
    planner = ResearchPlanner()
    return resolver, understanding, planner


def _workflow(decision_agent, executor, *, settings=None, checkpointer=None):
    resolver, understanding, planner = _components()
    return LangGraphResearchWorkflow(
        scope_resolver=resolver,
        query_understanding=understanding,
        planner=planner,
        decision_agent=decision_agent,
        tool_executor=executor,
        settings=settings or AgenticResearchSettings(),
        checkpointer=checkpointer,
    )


class AgenticLangGraphTests(unittest.TestCase):
    def test_dynamic_a_then_b_runs_through_compiled_graph(self) -> None:
        executor = _Executor()
        result = _workflow(_EvidenceAwareDecisionAgent(), executor).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="dynamic-a-b",
        )
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertEqual(result.stop_reason, StopReason.AGENT_FINISHED)
        self.assertEqual(
            [record.evidence.paper_id for record in result.evidence], ["A", "B"]
        )
        self.assertEqual(executor.calls[0].searches[0].query, "A method")
        self.assertEqual(executor.calls[1].searches[0].query, "B method after A")
        self.assertIsNotNone(result.coverage)
        self.assertTrue(result.coverage.coverage_complete)
        self.assertEqual(
            {record.evidence.paper_id for record in result.selected_evidence},
            {"A", "B"},
        )

    def test_finish_is_rejected_until_coverage_is_complete(self) -> None:
        executor = _Executor()
        result = _workflow(
            ScriptedDecisionAgent(
                (
                    FinishAction("too early"),
                    _action("A", "A method", "S1"),
                    _action("B", "B method", "S2"),
                    FinishAction("covered"),
                )
            ),
            executor,
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="coverage-gate",
        )
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertIn("CoverageIncomplete", result.traces[0].detail)
        self.assertFalse(result.traces[0].accepted)
        self.assertEqual(len(executor.calls), 2)

    def test_qualification_is_checkpointed_before_coverage_assessment(self) -> None:
        result = _workflow(_QualifyingDecisionAgent(), _Executor()).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="qualification-coverage",
        )
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertTrue(all(record.qualification is not None for record in result.evidence))
        self.assertEqual(set(result.coverage.positive_paper_ids), {"A", "B"})

    def test_graph_and_legacy_loop_have_equivalent_domain_results(self) -> None:
        actions = (
            _action("A", "A method", "S1"),
            _action("B", "B method after A", "S2"),
            FinishAction("both papers covered"),
        )
        graph_executor = _Executor()
        graph_result = _workflow(
            ScriptedDecisionAgent(actions), graph_executor
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="parity-graph",
        )

        resolver, understanding, planner = _components()
        resolution = resolver.resolve("q", selected_paper_ids=("A", "B"))
        understood = understanding.understand(
            "A 和 B 的方法有什么区别？", resolution
        )
        planned = planner.plan(understood.context)
        legacy_result = ResearchRuntime(
            decision_agent=ScriptedDecisionAgent(actions),
            tool_executor=_Executor(),
            settings=AgenticResearchSettings(),
        ).run(
            context=understood.context,
            plan=planned.plan,
            initial_llm_calls=understood.llm_calls + planned.llm_calls,
        )
        self.assertEqual(graph_result.status, legacy_result.status)
        self.assertEqual(graph_result.stop_reason, legacy_result.stop_reason)
        self.assertEqual(graph_result.budget_usage, legacy_result.budget_usage)
        self.assertEqual(
            [(record.evidence_id, record.evidence.paper_id) for record in graph_result.evidence],
            [(record.evidence_id, record.evidence.paper_id) for record in legacy_result.evidence],
        )
        self.assertEqual(graph_result.traces, legacy_result.traces)

    def test_checkpoint_resume_does_not_repeat_completed_tool_action(self) -> None:
        checkpointer = InMemorySaver()
        first_executor = _Executor()
        first_workflow = _workflow(
            _EvidenceAwareDecisionAgent(),
            first_executor,
            checkpointer=checkpointer,
        )
        interrupted = first_workflow.start(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="resume-run",
            interrupt_after=("execute_tool",),
        )
        self.assertTrue(interrupted.interrupted)
        self.assertEqual(interrupted.next_nodes, ("register_and_observe",))
        self.assertEqual(len(first_executor.calls), 1)
        self.assertEqual(interrupted.state["guard_state"]["usage"]["tool_calls"], 1)
        self.assertEqual(
            len(interrupted.state["guard_state"]["executed_action_fingerprints"]), 1
        )

        resumed_executor = _Executor()
        resumed_workflow = _workflow(
            _EvidenceAwareDecisionAgent(),
            resumed_executor,
            checkpointer=checkpointer,
        )
        completed = resumed_workflow.resume("resume-run")
        self.assertFalse(completed.interrupted)
        self.assertIsNotNone(completed.result)
        self.assertEqual(
            [action.searches[0].paper_id for action in resumed_executor.calls], ["B"]
        )
        self.assertEqual(
            [record.evidence_id for record in completed.result.evidence], ["E1", "E2"]
        )

    def test_invalid_scope_is_rejected_before_tool_node(self) -> None:
        executor = _Executor()
        result = _workflow(
            ScriptedDecisionAgent(
                (
                    _action("outside", "invalid", "S1"),
                    _action("A", "valid", "S1"),
                    FinishAction("done"),
                )
            ),
            executor,
        ).run("A method", selected_paper_ids=("A",), thread_id="invalid-scope")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0].searches[0].paper_id, "A")
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertTrue(any("InvalidScope" in trace.detail for trace in result.traces))

    def test_duplicate_action_is_rejected_before_tool_node(self) -> None:
        executor = _Executor()
        duplicate = _action("A", "same query", "S1")
        result = _workflow(
            ScriptedDecisionAgent((duplicate, duplicate, FinishAction("done"))),
            executor,
        ).run("A method", selected_paper_ids=("A",), thread_id="duplicate")
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertTrue(any("DuplicateAction" in trace.detail for trace in result.traces))

    def test_iteration_budget_ends_graph_without_extra_tool_call(self) -> None:
        executor = _Executor(empty_queries={"miss 1", "miss 2", "must not run"})
        result = _workflow(
            ScriptedDecisionAgent(
                (
                    _action("A", "miss 1", "S1"),
                    _action("B", "miss 2", "S2"),
                    _action("A", "must not run", "S1"),
                )
            ),
            executor,
            settings=AgenticResearchSettings(
                max_agent_iterations=2,
                max_no_new_evidence_per_paper=5,
            ),
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="iteration-budget",
        )
        self.assertEqual(result.stop_reason, StopReason.BUDGET_EXHAUSTED)
        self.assertEqual(result.budget_usage.agent_iterations, 2)
        self.assertEqual(len(executor.calls), 2)

    def test_per_paper_no_new_evidence_does_not_block_other_paper(self) -> None:
        executor = _Executor(empty_queries={"A miss 1", "A miss 2", "A suppressed"})
        actions = (
            _action("A", "A miss 1", "S1"),
            _action("A", "A miss 2", "S1"),
            _action("A", "A suppressed", "S1"),
            _action("B", "B hit", "S2"),
            FinishAction("B progressed"),
        )
        result = _workflow(
            ScriptedDecisionAgent(actions),
            executor,
            settings=AgenticResearchSettings(max_no_new_evidence_per_paper=2),
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="per-paper-no-new",
        )
        self.assertEqual([record.evidence.paper_id for record in result.evidence], ["B"])
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertEqual(len(executor.calls), 3)

    def test_heterogeneous_batch_counts_no_new_evidence_per_paper(self) -> None:
        executor = _Executor(empty_queries={"A miss 1", "A miss 2", "A suppressed"})
        batch = SearchPapersAction(
            searches=(
                PaperSearchSpec("A", "A miss 1", target_subtask_id="S1"),
                PaperSearchSpec("B", "B hit 1", target_subtask_id="S2"),
            )
        )
        result = _workflow(
            ScriptedDecisionAgent(
                (
                    batch,
                    _action("A", "A miss 2", "S1"),
                    _action("A", "A suppressed", "S1"),
                    _action("B", "B hit 2", "S2"),
                    FinishAction("done"),
                )
            ),
            executor,
            settings=AgenticResearchSettings(max_no_new_evidence_per_paper=2),
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="heterogeneous-batch",
        )
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertEqual([record.evidence.paper_id for record in result.evidence], ["B", "B"])
        self.assertEqual(len(executor.calls), 3)

    def test_service_uses_graph_and_unresolved_scope_returns_clarification(self) -> None:
        executor = _Executor()
        service = AgenticResearchService(
            workflow=_workflow(_EvidenceAwareDecisionAgent(), executor)
        )
        result = service.research("没有可识别论文的问题")
        self.assertEqual(result.status, ResearchRunStatus.NEEDS_CLARIFICATION)
        self.assertEqual(result.stop_reason, StopReason.SCOPE_UNRESOLVED)
        self.assertEqual(executor.calls, [])

    def test_checkpoint_state_contains_no_runtime_dependencies(self) -> None:
        workflow = _workflow(_EvidenceAwareDecisionAgent(), _Executor())
        execution = workflow.start(
            "A method",
            selected_paper_ids=("A",),
            thread_id="serializable-state",
            interrupt_after=("build_plan",),
        )
        self.assertTrue(_is_json_compatible(execution.state))

    def test_thread_id_cannot_silently_start_a_second_run(self) -> None:
        workflow = _workflow(_EvidenceAwareDecisionAgent(), _Executor())
        workflow.run("A method", selected_paper_ids=("A",), thread_id="same-thread")
        with self.assertRaisesRegex(ValueError, "already exists"):
            workflow.start(
                "B method", selected_paper_ids=("B",), thread_id="same-thread"
            )

    def test_unknown_thread_cannot_be_resumed(self) -> None:
        workflow = _workflow(_EvidenceAwareDecisionAgent(), _Executor())
        with self.assertRaisesRegex(LookupError, "Unknown research thread"):
            workflow.resume("missing-thread")


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
