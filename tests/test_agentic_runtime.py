from __future__ import annotations

import unittest

from paperbase.agentic.contracts import (
    EvidenceTarget,
    FinishAction,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRunStatus,
    ResearchSubtask,
    ScopeSource,
    SearchPapersAction,
    StopReason,
)
from paperbase.agentic.decision import ScriptedDecisionAgent
from paperbase.agentic.runtime import ResearchRuntime
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


def _context():
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


def _plan():
    return ResearchPlan(
        goal="compare",
        subtasks=(
            ResearchSubtask("S1", "find A", ("A",), frozenset({EvidenceTarget.METHOD}), "A evidence"),
            ResearchSubtask("S2", "find B", ("B",), frozenset({EvidenceTarget.METHOD}), "B evidence"),
        ),
    )


def _action(paper_id: str, query: str, subtask: str):
    return SearchPapersAction(
        searches=(PaperSearchSpec(paper_id, query, target_subtask_id=subtask),),
        target_subtask_id=subtask,
    )


class _Executor:
    def __init__(self, empty_queries=()):
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
            status=ToolStatus.EMPTY if all(not item.evidence_candidates for item in outcomes) else ToolStatus.OK,
            tool_name=ToolName.SEARCH_PAPERS,
            payload=SearchPapersPayload(tuple(outcomes)),
        )


class AgenticRuntimeTests(unittest.TestCase):
    def test_comparison_can_query_a_then_dynamically_query_b(self) -> None:
        executor = _Executor()
        runtime = ResearchRuntime(
            decision_agent=ScriptedDecisionAgent(
                (
                    _action("A", "A method mechanism", "S1"),
                    _action("B", "B method mechanism after observing A", "S2"),
                    FinishAction("both papers covered"),
                )
            ),
            tool_executor=executor,
            settings=AgenticResearchSettings(),
        )
        result = runtime.run(context=_context(), plan=_plan())
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertEqual([record.evidence.paper_id for record in result.evidence], ["A", "B"])
        self.assertEqual(result.budget_usage.tool_calls, 2)
        self.assertEqual(executor.calls[1].searches[0].query, "B method mechanism after observing A")

    def test_no_new_evidence_suppresses_only_that_paper(self) -> None:
        executor = _Executor(empty_queries={"A attempt 1", "A attempt 2"})
        runtime = ResearchRuntime(
            decision_agent=ScriptedDecisionAgent(
                (
                    _action("A", "A attempt 1", "S1"),
                    _action("A", "A attempt 2", "S1"),
                    _action("B", "B attempt", "S2"),
                    FinishAction("B still progressed"),
                )
            ),
            tool_executor=executor,
            settings=AgenticResearchSettings(max_no_new_evidence_per_paper=2),
        )
        result = runtime.run(context=_context(), plan=_plan())
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertEqual([record.evidence.paper_id for record in result.evidence], ["B"])
        self.assertEqual(result.budget_usage.tool_calls, 3)

    def test_batch_no_new_evidence_is_counted_per_paper(self) -> None:
        first_batch = SearchPapersAction(
            searches=(
                PaperSearchSpec("A", "A miss 1", target_subtask_id="S1"),
                PaperSearchSpec("B", "B hit 1", target_subtask_id="S2"),
            )
        )
        executor = _Executor(empty_queries={"A miss 1", "A miss 2", "A should be suppressed"})
        runtime = ResearchRuntime(
            decision_agent=ScriptedDecisionAgent(
                (
                    first_batch,
                    _action("A", "A miss 2", "S1"),
                    _action("A", "A should be suppressed", "S1"),
                    _action("B", "B hit 2", "S2"),
                    FinishAction("B remained available"),
                )
            ),
            tool_executor=executor,
            settings=AgenticResearchSettings(max_no_new_evidence_per_paper=2),
        )
        result = runtime.run(context=_context(), plan=_plan())
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertEqual(len(executor.calls), 3)
        self.assertTrue(any("NoNewEvidenceSuppressed" in trace.detail for trace in result.traces))

    def test_duplicate_action_is_rejected_before_tool_execution(self) -> None:
        action = _action("A", "same query", "S1")
        executor = _Executor()
        runtime = ResearchRuntime(
            decision_agent=ScriptedDecisionAgent((action, action, FinishAction("done"))),
            tool_executor=executor,
            settings=AgenticResearchSettings(max_invalid_actions=0),
        )
        result = runtime.run(context=_context(), plan=_plan())
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertIn("DuplicateAction", result.traces[1].detail)

    def test_iteration_budget_stops_loop_without_hanging(self) -> None:
        executor = _Executor(empty_queries={"q1", "q2", "q3"})
        runtime = ResearchRuntime(
            decision_agent=ScriptedDecisionAgent(
                (_action("A", "q1", "S1"), _action("A", "q2", "S1"), _action("B", "q3", "S2"))
            ),
            tool_executor=executor,
            settings=AgenticResearchSettings(
                max_agent_iterations=2,
                max_no_new_evidence_per_paper=5,
            ),
        )
        result = runtime.run(context=_context(), plan=_plan())
        self.assertEqual(result.stop_reason, StopReason.BUDGET_EXHAUSTED)
        self.assertEqual(result.budget_usage.agent_iterations, 2)


if __name__ == "__main__":
    unittest.main()
