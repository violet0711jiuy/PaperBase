from __future__ import annotations

from paperbase.agentic.contracts import (
    GroundedAnswerAttempt,
    GroundedCitation,
    RequirementAnswer,
    ResearchRunStatus,
    SearchPapersAction,
    StopReason,
    StopResearchAction,
    TryAnswerAction,
)
from paperbase.agentic.decision import ScriptedDecisionAgent
from paperbase.agentic.grounded_answer import GroundedAnswerAttemptsExhausted
from paperbase.agentic.graph import LangGraphResearchWorkflow
from paperbase.agentic.planner import ResearchPlanner
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


def _search() -> SearchPapersAction:
    return SearchPapersAction((PaperSearchSpec(
        paper_id="P1", query="MPIWcapt mechanism",
        lexical_terms=("MPIWcapt", "capture"), requirement_ids=("R1",),
    ),))


def _workflow(actions, answerer):
    return LangGraphResearchWorkflow(
        scope_resolver=ScopeResolver(_Database()),
        query_understanding=QueryUnderstandingService(),
        planner=ResearchPlanner(),
        decision_agent=ScriptedDecisionAgent(tuple(actions)),
        tool_executor=_Executor(),
        grounded_answerer=answerer,
        settings=AgenticResearchSettings(evidence_admission_enabled=False),
    )


def test_graph_runs_raw_evidence_to_grounded_complete_answer() -> None:
    answerer = _Answerer()
    result = _workflow((_search(), TryAnswerAction("evidence retrieved")), answerer).run(
        "Paper One 的方法是什么？", selected_paper_ids=("P1",)
    )
    assert result.status is ResearchRunStatus.COMPLETED
    assert result.stop_reason is StopReason.AGENT_FINISHED
    assert [record.evidence_id for record in result.evidence] == ["E1"]
    assert [record.evidence_id for record in result.selected_evidence] == ["E1"]
    assert result.answer is not None and result.answer.overall_status == "complete"
    assert answerer.calls == [(('E1',), ())]


def test_answer_gap_gets_one_same_pool_reread_before_more_retrieval() -> None:
    answerer = _Answerer(gap_first=True)
    result = _workflow(
        (_search(), TryAnswerAction("first answer pass"), TryAnswerAction("focused reread")),
        answerer,
    ).run("Paper One 的方法是什么？", selected_paper_ids=("P1",))
    assert result.status is ResearchRunStatus.COMPLETED
    assert [call[0] for call in answerer.calls] == [("E1",), ("E1",)]
    assert len(result.observations) == 1


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
        (_search(), TryAnswerAction("evidence retrieved")), InvalidAnswerer()
    ).run("Paper One 的方法是什么？", selected_paper_ids=("P1",))

    assert result.stop_reason is StopReason.FATAL_ERROR
    assert result.selected_evidence == ()
    assert result.answer_input_evidence_ids == ("E1",)
    assert result.llm_diagnostics[-1].node == "grounded_answer"
