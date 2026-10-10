from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.checkpoint.memory import InMemorySaver

from paperbase.agentic.contracts import (
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRelation,
    EvidenceTarget,
    MechanismProjection,
    RequirementSupportAnchor,
    FinishAction,
    ResearchRunStatus,
    SearchPapersAction,
    SearchReferencesAction,
    StopReason,
)
from paperbase.agentic.decision import DecisionResult, ScriptedDecisionAgent
from paperbase.agentic.candidate_selection import CandidateSelectionResult
from paperbase.agentic.checkpoint_serde import checkpoint_thread_id
from paperbase.agentic.evidence_analysis import EvidenceAnalysisResult
from paperbase.agentic.evidence_projection import LLMEvidenceProjector
from paperbase.agentic.graph import LangGraphResearchWorkflow
from paperbase.agentic.evaluation import evaluate_research_result
from paperbase.agentic.persistence import SQLiteCheckpointStore
from paperbase.agentic.planner import ResearchPlanner
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.scope_resolver import ScopeResolver
from paperbase.agentic.service import AgenticResearchBusyError, AgenticResearchService
from paperbase.config import AgenticResearchSettings
from paperbase.generation.answer_generator import AnswerGenerationOutcome
from paperbase.llm.client import LLMCompletion
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchOutcome,
    PaperSearchSpec,
    PaperSearchStatus,
    ReferenceSearchPayload,
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
                    search=PaperSearchSpec(paper_id=search.paper_id, query=search.query, lexical_terms=search.lexical_terms),
                    status=PaperSearchStatus.EMPTY if empty else PaperSearchStatus.OK,
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
    """Decision 只选择动作，不再负责证据分析。"""

    def decide(self, state):
        paper_ids = {record.evidence.paper_id for record in state.evidence}
        if "A" not in paper_ids:
            return DecisionResult(_action("A", "A method", "S1"), 1)
        if "B" not in paper_ids:
            return DecisionResult(_action("B", "B method", "S2"), 1)
        return DecisionResult(FinishAction("qualified"), 1)


class _SupportingEvidenceAnalyzer:
    """测试替身：把每条新证据标为本文直接支持证据。"""

    def analyze(self, *, request):
        evidence, plan = request.evidence, request.plan
        analyses = tuple(
            EvidenceAnalysis(
                evidence_id=record.evidence_id,
                status=EvidenceAnalysisStatus.QUALIFIED,
                evaluated_requirement_ids=request.scope_for(record.evidence_id),
                projection=MechanismProjection(
                    evidence_id=record.evidence_id,
                    mechanism=record.evidence.paper_id,
                    supported_requirement_ids=tuple(
                        rid for rid in request.scope_for(record.evidence_id)
                    ),
                    proposed_requirement_ids=tuple(
                        rid for rid in request.scope_for(record.evidence_id)
                    ),
                    support_anchors=tuple(
                        RequirementSupportAnchor(
                            rid, record.evidence.content,
                            record.evidence.content,
                        )
                        for rid in request.scope_for(record.evidence_id)
                    ),
                ),
            )
            for record in evidence
        )
        return EvidenceAnalysisResult(analyses=analyses, llm_calls=0)


class _StaticCandidateSelector:
    def select(self, *, question, snapshot):
        return CandidateSelectionResult(
            CandidateSelection(
                scope_paper_ids=snapshot.paper_ids,
                assessments=tuple(
                    CandidateAssessment(
                        paper_id,
                        (
                            CandidateDisposition.CANDIDATE
                            if paper_id == "A"
                            else CandidateDisposition.EXCLUDED
                        ),
                        "A 与问题相关" if paper_id == "A" else "B 与问题无关",
                    )
                    for paper_id in snapshot.paper_ids
                ),
            ),
            llm_calls=1,
        )


def _action(paper_id: str, query: str, subtask_id: str) -> SearchPapersAction:
    # Historical tests named per-paper Subtasks. The current fallback Planner
    # has one answer Requirement shared by the selected papers.
    return SearchPapersAction(
        searches=(
            PaperSearchSpec(
                paper_id=paper_id,
                query=query,
                evidence_targets=frozenset({EvidenceTarget.METHOD.value}),
                requirement_ids=("R1",),
            ),
        ),
    )


def _components():
    resolver = ScopeResolver(_CatalogDatabase())
    understanding = QueryUnderstandingService()
    planner = ResearchPlanner()
    return resolver, understanding, planner


def _workflow(
    decision_agent,
    executor,
    *,
    settings=None,
    checkpointer=None,
    answer_generator=None,
    evidence_analyzer=None,
    candidate_selector=None,
):
    resolver, understanding, planner = _components()
    return LangGraphResearchWorkflow(
        scope_resolver=resolver,
        query_understanding=understanding,
        planner=planner,
        decision_agent=decision_agent,
        evidence_analyzer=evidence_analyzer or _SupportingEvidenceAnalyzer(),
        candidate_selector=candidate_selector,
        tool_executor=executor,
        settings=settings or AgenticResearchSettings(),
        answer_generator=answer_generator,
        checkpointer=checkpointer,
    )


class _AnswerGenerator:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, *, query, evidence, conversation_context=None,
                 research_requirements=(), support_by_requirement=None):
        from paperbase.generation.answer_coverage import AnswerRequirementCoverage
        self.calls.append((query, tuple(evidence)))
        citations = tuple(item.evidence_id for item in evidence)
        coverage = tuple(
            AnswerRequirementCoverage(
                requirement_id,
                "complete" if (support_by_requirement or {}).get(requirement_id) else "unsupported",
                tuple((support_by_requirement or {}).get(requirement_id, ())),
            )
            for requirement_id, _ in research_requirements
        )
        return AnswerGenerationOutcome(
            status="success",
            answer="Evidence: " + "".join(f"[{item}]" for item in citations),
            citations=citations,
            insufficient_evidence=False,
            requirement_coverage=coverage,
        )


class _FailingAnswerGenerator:
    def generate(self, **kwargs):
        raise RuntimeError("answer provider unavailable")


class _InvalidUnderstandingClient:
    def complete_json_diagnostic(self, **kwargs):
        return LLMCompletion(
            "{invalid-json",
            finish_reason="stop",
            prompt_tokens=31,
            completion_tokens=7,
            reasoning_chars=13,
        )


class AgenticLangGraphTests(unittest.TestCase):
    def test_reference_action_binding_reaches_projection_and_final_answer(self) -> None:
        class ReferenceExecutor:
            def execute(self, action, **kwargs):
                return ToolResult(
                    status=ToolStatus.OK,
                    tool_name=ToolName.SEARCH_REFERENCES,
                    payload=ReferenceSearchPayload(
                        paper_id=action.paper_id,
                        query=action.query,
                        evidence_candidates=(RetrievedEvidence(
                            source_type=EvidenceSourceType.BIBLIOGRAPHY,
                            paper_id="A",
                            seed_chunk_ids=("ref_1",),
                            source_chunk_ids=("ref_1",),
                            content="A cites Example Study by Smith (2020).",
                        ),),
                    ),
                )

        class ProjectionClient:
            def __init__(self):
                self.prompt = None

            def complete_json(self, **kwargs):
                self.prompt = json.loads(kwargs["user_prompt"])
                return json.dumps({"projections": [{
                    "projection_type": "fact",
                    "evidence_id": "REF1",
                    "status": "qualified",
                    "fact_type": "reference",
                    "value": "Example Study by Smith (2020)",
                    "ownership": "prior_work",
                    "supported_requirement_ids": ["R1"],
                    "support_candidates": [{
                        "requirement_id": "R1",
                        "source_span_ids": ["S1"],
                        "support_scope": "Example Study by Smith (2020)",
                        "unresolved_aspects": [],
                    }],
                }]})

        client = ProjectionClient()
        result = _workflow(
            ScriptedDecisionAgent((
                SearchReferencesAction("A", "Example Study", requirement_ids=("R1",)),
                FinishAction("reference verified"),
            )),
            ReferenceExecutor(),
            evidence_analyzer=LLMEvidenceProjector(client, max_attempts=1),
            answer_generator=_AnswerGenerator(),
        ).run(
            "A 引用了什么参考文献？",
            selected_paper_ids=("A",),
            thread_id="reference-action-binding",
        )

        self.assertEqual(client.prompt["projection_requirement_ids"], ["R1"])
        self.assertEqual(client.prompt["projection_requirements"][0]["requirement_id"], "R1")
        self.assertEqual(result.evidence[0].requested_requirement_ids, ("R1",))
        self.assertEqual(result.requirement_states[0].status.value, "satisfied")
        self.assertEqual(result.selected_evidence[0].evidence_id, "REF1")
        self.assertEqual(result.answer.citations, ("REF1",))
        self.assertFalse(result.answer.insufficient_evidence)

    def test_safe_llm_diagnostics_survive_checkpoint_and_final_result(self) -> None:
        resolver, _, planner = _components()
        workflow = LangGraphResearchWorkflow(
            scope_resolver=resolver,
            query_understanding=QueryUnderstandingService(
                _InvalidUnderstandingClient()
            ),
            planner=planner,
            decision_agent=ScriptedDecisionAgent(
                (_action("A", "A method", "S1"), FinishAction("covered"))
            ),
            evidence_analyzer=_SupportingEvidenceAnalyzer(),
            tool_executor=_Executor(),
            settings=AgenticResearchSettings(),
        )

        result = workflow.run(
            "A 的方法是什么？",
            selected_paper_ids=("A",),
            thread_id="safe-diagnostics",
        )
        inspected = workflow.inspect("safe-diagnostics")

        self.assertEqual(len(result.llm_diagnostics), 1)
        record = result.llm_diagnostics[0]
        self.assertEqual(record.node, "understand_query")
        self.assertEqual(record.error_stage, "invalid_json")
        self.assertEqual(record.prompt_tokens, 31)
        self.assertEqual(record.reasoning_chars, 13)
        raw_record = inspected.state["llm_diagnostics"][0]
        self.assertNotIn("content", raw_record)
        self.assertNotIn("reasoning_content", raw_record)
        report = result.observations[0].evidence_filter_report
        self.assertIsNotNone(report)
        self.assertEqual(report.total_candidates, 1)
        self.assertEqual(report.accepted_candidates, 1)
        self.assertFalse(
            hasattr(report.decisions[0], "content")  # type: ignore[union-attr]
        )
        self.assertIsNotNone(result.observations[0].evidence_analysis_cache)

    def test_knowledge_base_candidate_selection_blocks_excluded_body_search(self) -> None:
        executor = _Executor()
        result = _workflow(
            ScriptedDecisionAgent(
                (
                    _action("B", "irrelevant deep search", "S1"),
                    _action("A", "wind forecasting method", "S1"),
                    FinishAction("candidate verified"),
                )
            ),
            executor,
            candidate_selector=_StaticCandidateSelector(),
        ).run(
            "查找风速预测方法相关论文",
            knowledge_base_all=True,
            thread_id="candidate-selection",
        )
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertEqual(
            [action.searches[0].paper_id for action in executor.calls], ["A"]
        )
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertEqual(result.candidate_selection.candidate_paper_ids, ("A",))
        self.assertEqual(result.coverage.screened_paper_ids, ("A", "B"))
        self.assertEqual(result.coverage.excluded_paper_ids, ("B",))
        self.assertEqual(result.coverage.verified_paper_ids, ("A",))

    def test_final_answer_reuses_selected_evidence_ids(self) -> None:
        generator = _AnswerGenerator()
        result = _workflow(
            _QualifyingDecisionAgent(),
            _Executor(),
            answer_generator=generator,
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="answer-node",
        )
        self.assertIsNotNone(result.answer)
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertTrue(result.selected_evidence)
        self.assertEqual(
            result.answer.citations,
            tuple(item.evidence_id for item in result.selected_evidence),
        )
        self.assertEqual(result.thread_id, "answer-node")
        self.assertEqual(result.budget_usage.llm_calls, 4)
        self.assertEqual(
            [item.evidence_id for item in generator.calls[0][1]],
            [item.evidence_id for item in result.selected_evidence],
        )
        evaluation = evaluate_research_result(result)
        self.assertTrue(evaluation.passed)
        self.assertEqual(evaluation.evidence_balance, 1.0)
        self.assertEqual(evaluation.fabricated_citations, ())

    def test_stream_exposes_business_events_and_final_result(self) -> None:
        events = list(
            _workflow(_QualifyingDecisionAgent(), _Executor()).stream(
                "A 和 B 的方法有什么区别？",
                selected_paper_ids=("A", "B"),
                thread_id="stream-run",
            )
        )
        self.assertEqual(events[0].kind, "started")
        self.assertEqual(events[-1].kind, "completed")
        self.assertIsNotNone(events[-1].result)
        self.assertIn("retrieval", [event.stage for event in events])
        self.assertIn("answer", [event.stage for event in events])

    def test_answer_failure_keeps_evidence_and_records_diagnostic_code(self) -> None:
        result = _workflow(
            _QualifyingDecisionAgent(),
            _Executor(),
            answer_generator=_FailingAnswerGenerator(),
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="answer-fallback",
        )
        self.assertEqual(result.answer.status, "fallback")
        self.assertEqual(len(result.selected_evidence), 2)
        self.assertIn("AnswerGeneration:RuntimeError", result.error_codes)

    def test_service_rejects_unbounded_concurrent_run(self) -> None:
        service = AgenticResearchService(
            workflow=_workflow(_EvidenceAwareDecisionAgent(), _Executor()),
            max_concurrent_runs=1,
        )
        self.assertTrue(service._capacity.acquire(blocking=False))
        try:
            with self.assertRaises(AgenticResearchBusyError):
                service.research("A 方法", selected_paper_ids=("A",))
        finally:
            service._capacity.release()

    def test_sqlite_checkpoint_survives_process_style_reopen(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "agentic.sqlite3"
            first_store = SQLiteCheckpointStore.open(path)
            first_executor = _Executor()
            first_workflow = _workflow(
                _EvidenceAwareDecisionAgent(),
                first_executor,
                checkpointer=first_store.saver,
            )
            interrupted = first_workflow.start(
                "A 和 B 的方法有什么区别？",
                selected_paper_ids=("A", "B"),
                thread_id="sqlite-resume",
                interrupt_after=("execute_tool",),
            )
            self.assertTrue(interrupted.interrupted)
            self.assertEqual(len(first_executor.calls), 1)
            first_store.close()

            second_store = SQLiteCheckpointStore.open(path)
            resumed_executor = _Executor()
            resumed_workflow = _workflow(
                _EvidenceAwareDecisionAgent(),
                resumed_executor,
                checkpointer=second_store.saver,
            )
            completed = resumed_workflow.resume("sqlite-resume")
            self.assertFalse(completed.interrupted)
            self.assertIsNotNone(completed.result)
            self.assertEqual(
                [item.requirement_id for item in completed.result.requirement_states],
                ["R1"],
            )
            self.assertEqual(
                [action.searches[0].paper_id for action in resumed_executor.calls],
                ["B"],
            )
            second_store.close()

    def test_service_lifecycle_tracking_preserves_interrupted_resume(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "service.sqlite3"
            first_store = SQLiteCheckpointStore.open(path)
            first_service = AgenticResearchService(
                workflow=_workflow(
                    _EvidenceAwareDecisionAgent(),
                    _Executor(),
                    checkpointer=first_store.saver,
                ),
                checkpoint_store=first_store,
            )
            events = list(
                first_service.stream_research(
                    "A 和 B 的方法有什么区别？",
                    selected_paper_ids=("A", "B"),
                    thread_id="compacted-resume",
                    interrupt_after=("execute_tool",),
                )
            )
            self.assertEqual(events[-1].kind, "interrupted")
            first_service.close()

            second_store = SQLiteCheckpointStore.open(path)
            executor = _Executor()
            second_service = AgenticResearchService(
                workflow=_workflow(
                    _EvidenceAwareDecisionAgent(),
                    executor,
                    checkpointer=second_store.saver,
                ),
                checkpoint_store=second_store,
            )
            completed = second_service.resume("compacted-resume")
            self.assertFalse(completed.interrupted)
            self.assertEqual(
                [action.searches[0].paper_id for action in executor.calls], ["B"]
            )
            second_service.close()

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
        self.assertIn("RequirementResearchIncomplete", result.traces[0].detail)
        self.assertFalse(result.traces[0].accepted)
        self.assertIn(
            "guard_rejection",
            [item.error_stage for item in result.llm_diagnostics],
        )
        self.assertEqual(len(executor.calls), 2)

    def test_evidence_analysis_is_checkpointed_before_coverage_assessment(self) -> None:
        result = _workflow(_QualifyingDecisionAgent(), _Executor()).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="evidence-analysis-coverage",
        )
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertTrue(
            all(
                record.analysis_status is EvidenceAnalysisStatus.QUALIFIED
                and record.assessment.projection is not None
                for record in result.evidence
            )
        )
        self.assertEqual(set(result.coverage.positive_paper_ids), {"A", "B"})

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
        self.assertEqual(interrupted.next_nodes, ("filter_evidence",))
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

    def test_consumed_tool_payload_is_cleared_before_analysis_resume(self) -> None:
        executor = _Executor()
        workflow = _workflow(
            ScriptedDecisionAgent((
                _action("A", "A method", "S1"), FinishAction("done")
            )),
            executor,
            checkpointer=InMemorySaver(),
        )
        interrupted = workflow.start(
            "A method",
            selected_paper_ids=("A",),
            thread_id="consumed-tool-payload",
            interrupt_after=("register_and_observe",),
        )
        self.assertEqual(interrupted.next_nodes, ("analyze_evidence",))
        self.assertIsNone(interrupted.state["tool_result"])
        self.assertIsNone(interrupted.state["evidence_filter_report"])
        self.assertNotIn("action_fingerprint", interrupted.state)
        self.assertNotIn("coverage", interrupted.state)
        self.assertEqual(interrupted.state["pending_analysis_ids"], ["E1"])
        completed = workflow.resume("consumed-tool-payload")
        self.assertEqual(completed.result.status, ResearchRunStatus.COMPLETED)
        self.assertTrue(completed.result.coverage.coverage_complete)
        self.assertEqual(len(executor.calls), 1)

    def test_legacy_checkpoint_plan_and_action_resume_as_requirements(self) -> None:
        executor = _Executor()
        workflow = _workflow(
            ScriptedDecisionAgent((
                _action("A", "A method", "S1"),
                FinishAction("done"),
            )),
            executor,
            checkpointer=InMemorySaver(),
        )
        interrupted = workflow.start(
            "A method",
            selected_paper_ids=("A",),
            thread_id="legacy-checkpoint",
            interrupt_after=("decide_action",),
        )
        self.assertEqual(interrupted.next_nodes, ("assess_action",))
        legacy_plan = {
            "goal": "A method",
            "subtasks": [{
                "subtask_id": "S1",
                "goal": "查找 A 的方法",
                "target_paper_ids": ["A"],
                "evidence_targets": ["method"],
                "success_criteria": "找到方法证据",
                "status": "in_progress",
            }],
        }
        legacy_action = {
            "action_name": "search_papers",
            "searches": [{
                "paper_id": "A",
                "query": "A method",
                "target_subtask_id": "S1",
            }],
            "target_subtask_id": "S1",
        }
        workflow.graph.update_state(
            {"configurable": {
                "thread_id": checkpoint_thread_id("legacy-checkpoint"),
            }},
            {"plan": legacy_plan, "action": legacy_action},
            as_node="decide_action",
        )
        completed = workflow.resume("legacy-checkpoint")
        self.assertFalse(completed.interrupted)
        self.assertEqual([call.searches[0].requirement_ids for call in executor.calls], [("S1",)])
        self.assertEqual(completed.result.status, ResearchRunStatus.COMPLETED)
        self.assertFalse(hasattr(completed.result.plan, "subtasks"))
        self.assertEqual(completed.result.plan.requirements[0].requirement_id, "S1")

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

    def test_per_paper_budget_rejection_replans_instead_of_ending_run(self) -> None:
        executor = _Executor()
        result = _workflow(
            ScriptedDecisionAgent(
                (
                    _action("A", "A first method", "S1"),
                    _action("A", "A forbidden second method", "S1"),
                    _action("B", "B method after A budget", "S2"),
                    FinishAction("both papers researched"),
                )
            ),
            executor,
            settings=AgenticResearchSettings(max_queries_per_paper=1),
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="local-paper-budget-replan",
        )
        self.assertEqual(result.status, ResearchRunStatus.COMPLETED)
        self.assertEqual(result.stop_reason, StopReason.AGENT_FINISHED)
        self.assertEqual([call.searches[0].paper_id for call in executor.calls], ["A", "B"])
        self.assertEqual(result.budget_usage.invalid_actions, 1)
        self.assertEqual(result.action_rejections[-1].rejection_stage, "budget")
        self.assertEqual(result.action_rejections[-1].rejection_code, "PerPaperBudget")
        self.assertEqual(
            result.action_rejections[-1].blockers[0]["paper_id"], "A"
        )

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
            ),
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="iteration-budget",
        )
        self.assertEqual(result.stop_reason, StopReason.BUDGET_EXHAUSTED)
        self.assertEqual(result.budget_usage.agent_iterations, 2)
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(result.selected_evidence, ())
        self.assertTrue(result.answer.insufficient_evidence)

    def test_budget_stop_keeps_verified_evidence_and_marks_answer_partial(self) -> None:
        generator = _AnswerGenerator()
        result = _workflow(
            ScriptedDecisionAgent((_action("A", "A method", "S1"),)),
            _Executor(),
            settings=AgenticResearchSettings(max_agent_iterations=1),
            answer_generator=generator,
        ).run(
            "A 的方法是什么？",
            selected_paper_ids=("A",),
            thread_id="budget-support-preserved",
        )

        self.assertEqual(result.status, ResearchRunStatus.PARTIAL)
        self.assertEqual(result.stop_reason, StopReason.BUDGET_EXHAUSTED)
        self.assertEqual(result.requirement_states[0].status.value, "satisfied")
        self.assertEqual(len(result.selected_evidence), 1)
        self.assertEqual(result.answer.citations, (result.selected_evidence[0].evidence_id,))
        self.assertFalse(result.answer.partial_answer)
        self.assertFalse(result.answer.insufficient_evidence)
        self.assertEqual(result.answer.overall_status, "complete")
        self.assertEqual(len(generator.calls), 1)

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
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="per-paper-no-new",
        )
        self.assertEqual([record.evidence.paper_id for record in result.evidence], ["B"])
        self.assertEqual(result.budget_usage.invalid_actions, 0)
        self.assertEqual(len(executor.calls), 4)

    def test_heterogeneous_batch_counts_no_new_evidence_per_paper(self) -> None:
        executor = _Executor(empty_queries={"A miss 1", "A miss 2", "A suppressed"})
        batch = SearchPapersAction(
            searches=(
                PaperSearchSpec("A", "A miss 1", requirement_ids=("R1",)),
                PaperSearchSpec("B", "B hit 1", requirement_ids=("R1",)),
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
        ).run(
            "A 和 B 的方法有什么区别？",
            selected_paper_ids=("A", "B"),
            thread_id="heterogeneous-batch",
        )
        self.assertEqual(result.budget_usage.invalid_actions, 0)
        self.assertEqual([record.evidence.paper_id for record in result.evidence], ["B", "B"])
        self.assertEqual(len(executor.calls), 4)

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
