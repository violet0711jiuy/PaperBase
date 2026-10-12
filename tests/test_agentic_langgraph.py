from __future__ import annotations

from paperbase.agentic.contracts import (
    EvidenceTarget,
    GroundedAnswerAttempt,
    GroundedCitation,
    RequirementAnswer,
    ResearchPlan,
    ResearchRequirement,
    ResearchRunStatus,
    SearchPapersAction,
    StopReason,
    StopResearchAction,
    TryAnswerAction,
)
from paperbase.agentic.decision import ScriptedDecisionAgent
from paperbase.agentic.grounded_answer import GroundedAnswerAttemptsExhausted
from paperbase.agentic.graph import LangGraphResearchWorkflow
from paperbase.agentic.planner import PlannerResult, ResearchPlanner
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.scope_resolver import ScopeResolver
from paperbase.config import AgenticResearchSettings
from paperbase.llm.client import LLMCallRecord
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


class _Database:
    def list_documents(self):
        return ({"paper_id": "P1", "paper_title": "Paper One", "source_filename": "p1.pdf"},)

    def get_document(self, paper_id):
        return self.list_documents()[0] if paper_id == "P1" else None


class _Executor:
    def __init__(self): self.calls = []

    def execute(self, action, **kwargs):
        self.calls.append(action)
        search = action.searches[0]
        source = RetrievedEvidence(
            EvidenceSourceType.CONTENT, "P1", ("chunk_gold",), ("chunk_gold",),
            "MPIWcapt averages intervals that capture the target.",
            paper_title="Paper One", section="Method", section_id="method",
        )
        outcome = PaperSearchOutcome(
            search=PaperSearchSpec(
                paper_id=search.paper_id, query=search.query,
                lexical_terms=search.lexical_terms,
            ),
            status=PaperSearchStatus.OK,
            evidence_candidates=(source,),
        )
        return ToolResult(
            ToolStatus.OK, ToolName.SEARCH_PAPERS, SearchPapersPayload((outcome,))
        )


class _Answerer:
    def __init__(self, *, gap_first=False):
        self.gap_first = gap_first
        self.calls = []

    def answer(self, *, question, plan, evidence, accepted_answers=()):
        self.calls.append((tuple(record.evidence_id for record in evidence), tuple(accepted_answers)))
        rid = plan.requirements[0].requirement_id
        if self.gap_first and len(self.calls) == 1:
            return GroundedAnswerAttempt((
                RequirementAnswer(rid, None, remaining_gap="Re-read the retrieved method evidence."),
            ), tuple(record.evidence_id for record in evidence))
        return GroundedAnswerAttempt((
            RequirementAnswer(
                rid, "MPIWcapt averages intervals that capture the target.",
                (GroundedCitation(
                    "E1", ("S1",),
                    "MPIWcapt averages intervals that capture the target.",
                ),),
            ),
        ), tuple(record.evidence_id for record in evidence))


def _search(
    query: str = "MPIWcapt mechanism",
    lexical_terms: tuple[str, ...] = ("MPIWcapt", "capture"),
) -> SearchPapersAction:
    return SearchPapersAction((PaperSearchSpec(
        paper_id="P1", query=query,
        lexical_terms=lexical_terms, requirement_ids=("R1",),
    ),))


def _workflow(actions, answerer, *, planner=None, executor=None):
    return LangGraphResearchWorkflow(
        scope_resolver=ScopeResolver(_Database()),
        query_understanding=QueryUnderstandingService(),
        planner=planner or ResearchPlanner(),
        decision_agent=ScriptedDecisionAgent(tuple(actions)),
        tool_executor=executor or _Executor(),
        grounded_answerer=answerer,
        settings=AgenticResearchSettings(evidence_admission_enabled=False),
    )


def test_graph_runs_raw_evidence_to_grounded_complete_answer() -> None:
    answerer = _Answerer()
    result = _workflow((_search(),), answerer).run(
        "Paper One 的方法是什么？", selected_paper_ids=("P1",)
    )
    assert result.status is ResearchRunStatus.COMPLETED
    assert result.stop_reason is StopReason.AGENT_FINISHED
    assert [record.evidence_id for record in result.evidence] == ["E1"]
    assert [record.evidence_id for record in result.selected_evidence] == ["E1"]
    assert result.answer is not None and result.answer.overall_status == "complete"
    assert answerer.calls == [(('E1',), ())]
    assert result.budget_usage.llm_calls == 1


def test_answer_gap_gets_one_same_pool_reread_before_more_retrieval() -> None:
    answerer = _Answerer(gap_first=True)
    result = _workflow(
        (_search(), TryAnswerAction("focused reread")),
        answerer,
    ).run("Paper One 的方法是什么？", selected_paper_ids=("P1",))
    assert result.status is ResearchRunStatus.COMPLETED
    assert [call[0] for call in answerer.calls] == [("E1",), ("E1",)]
    assert len(result.observations) == 1


def test_reused_only_tool_result_returns_to_decision_without_automatic_answer() -> None:
    answerer = _Answerer(gap_first=True)
    result = _workflow(
        (
            _search(),
            _search("captured interval width", ("captured", "width")),
            TryAnswerAction("focused reread after no new evidence"),
        ),
        answerer,
    ).run("Paper One 的方法是什么？", selected_paper_ids=("P1",))

    assert result.status is ResearchRunStatus.COMPLETED
    assert [call[0] for call in answerer.calls] == [("E1",), ("E1",)]
    assert len(result.observations) == 2
    assert result.observations[1].added_evidence_ids == ()
    assert result.observations[1].reused_evidence_ids == ("E1",)
    assert result.budget_usage.llm_calls == 3


def test_new_evidence_after_gap_reassesses_all_requirements() -> None:
    plan = ResearchPlan("Explain width and coverage", (
        ResearchRequirement("R1", "Explain width", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain coverage", frozenset({EvidenceTarget.METHOD})),
    ))

    class Planner:
        def plan(self, context):
            return PlannerResult(plan, False, 0)

    class Executor(_Executor):
        def execute(self, action, **kwargs):
            self.calls.append(action)
            search = action.searches[0]
            coverage = "coverage" in search.query
            source = RetrievedEvidence(
                EvidenceSourceType.CONTENT, "P1",
                (("chunk_coverage",) if coverage else ("chunk_width",)),
                (("chunk_coverage",) if coverage else ("chunk_width",)),
                ("C_i penalizes misses outside the interval."
                 if coverage else "MPIWcapt narrows captured intervals."),
                paper_title="Paper One", section="Method", section_id="method",
            )
            outcome = PaperSearchOutcome(
                search=PaperSearchSpec(
                    paper_id=search.paper_id, query=search.query,
                    lexical_terms=search.lexical_terms,
                ),
                status=PaperSearchStatus.OK,
                evidence_candidates=(source,),
            )
            return ToolResult(
                ToolStatus.OK, ToolName.SEARCH_PAPERS, SearchPapersPayload((outcome,))
            )

    class Answerer:
        def __init__(self):
            self.calls = []

        def answer(self, *, question, plan, evidence, accepted_answers=()):
            self.calls.append((
                tuple(record.evidence_id for record in evidence),
                tuple(item.requirement_id for item in accepted_answers),
            ))
            r1 = RequirementAnswer(
                "R1", "MPIWcapt narrows captured intervals.",
                (GroundedCitation("E1", ("S1",), "MPIWcapt narrows captured intervals."),),
            )
            if len(evidence) == 1:
                return GroundedAnswerAttempt((
                    r1, RequirementAnswer("R2", None, remaining_gap="Need C_i coverage evidence."),
                ), ("E1",))
            return GroundedAnswerAttempt((
                r1,
                RequirementAnswer(
                    "R2", "C_i penalizes interval misses.",
                    (GroundedCitation("E2", ("S1",),
                                      "C_i penalizes misses outside the interval."),),
                ),
            ), ("E1", "E2"))

    answerer = Answerer()
    result = _workflow(
        (
            _search("MPIWcapt interval width", ("MPIWcapt", "width")),
            SearchPapersAction((PaperSearchSpec(
                paper_id="P1", query="C_i coverage penalty",
                lexical_terms=("C_i", "coverage"), requirement_ids=("R2",),
            ),)),
        ),
        answerer,
        planner=Planner(),
        executor=Executor(),
    ).run("Explain width and coverage", selected_paper_ids=("P1",))

    assert result.status is ResearchRunStatus.COMPLETED
    assert answerer.calls == [(('E1',), ()), (('E1', 'E2'), ())]
    assert result.answer_input_evidence_ids == ("E1", "E2")
    assert result.budget_usage.llm_calls == 2


def test_same_evidence_reread_preserves_complete_siblings() -> None:
    plan = ResearchPlan("Explain width and coverage", (
        ResearchRequirement("R1", "Explain width", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain coverage", frozenset({EvidenceTarget.METHOD})),
    ))

    class Planner:
        def plan(self, context):
            return PlannerResult(plan, False, 0)

    class Answerer:
        def __init__(self):
            self.calls = []

        def answer(self, *, question, plan, evidence, accepted_answers=()):
            self.calls.append(tuple(item.requirement_id for item in accepted_answers))
            r1 = RequirementAnswer(
                "R1", "MPIWcapt narrows captured intervals.",
                (GroundedCitation(
                    "E1", ("S1",),
                    "MPIWcapt averages intervals that capture the target.",
                ),),
            )
            if len(self.calls) == 1:
                return GroundedAnswerAttempt((
                    r1,
                    RequirementAnswer("R2", None, remaining_gap="Need coverage evidence."),
                ), ("E1",))
            return GroundedAnswerAttempt((
                r1,
                RequirementAnswer(
                    "R2", "The same evidence describes target capture.",
                    (GroundedCitation(
                        "E1", ("S1",),
                        "MPIWcapt averages intervals that capture the target.",
                    ),),
                ),
            ), ("E1",))

    answerer = Answerer()
    result = _workflow(
        (_search(), TryAnswerAction("focused reread")),
        answerer,
        planner=Planner(),
    ).run("Explain width and coverage", selected_paper_ids=("P1",))

    assert result.status is ResearchRunStatus.COMPLETED
    assert answerer.calls == [(), ("R1",)]


def test_stop_research_uses_answerer_to_return_explicit_gap() -> None:
    class GapAnswerer:
        def answer(self, *, question, plan, evidence, accepted_answers=()):
            return GroundedAnswerAttempt((
                RequirementAnswer(plan.requirements[0].requirement_id, None,
                                  remaining_gap="No evidence was retrieved."),
            ), ())

    result = _workflow((StopResearchAction("nothing useful to search"),), GapAnswerer()).run(
        "Paper One 的方法是什么？", selected_paper_ids=("P1",)
    )
    assert result.status is ResearchRunStatus.NO_EVIDENCE
    assert result.answer is not None and result.answer.insufficient_evidence


def test_failed_answer_attempts_remain_auditable_without_selecting_all_evidence() -> None:
    class InvalidAnswerer:
        def answer(self, *, question, plan, evidence, accepted_answers=()):
            diagnostic = LLMCallRecord(
                node="grounded_answer", attempt=2, success=False,
                error_stage="domain_validation", recovery_action="fallback",
            )
            raise GroundedAnswerAttemptsExhausted(ValueError("invalid spans"), (diagnostic,))

    result = _workflow(
        (_search(),), InvalidAnswerer()
    ).run("Paper One 的方法是什么？", selected_paper_ids=("P1",))

    assert result.stop_reason is StopReason.FATAL_ERROR
    assert result.selected_evidence == ()
    assert result.answer_input_evidence_ids == ("E1",)
    assert result.llm_diagnostics[-1].node == "grounded_answer"
