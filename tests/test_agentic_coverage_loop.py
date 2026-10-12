"""Offline loop contracts, without treating canned outputs as semantic evaluation."""

import json

import pytest

from paperbase.agentic.contracts import (
    EvidenceTarget, ResearchPlan, ResearchRequirement, ResearchRunStatus,
    SearchPapersAction,
)
from paperbase.agentic.decision import DecisionResult
from paperbase.agentic.grounded_answer import LLMGroundedAnswerer
from paperbase.agentic.graph import LangGraphResearchWorkflow
from paperbase.agentic.planner import PlannerResult
from paperbase.agentic.query_understanding import QueryUnderstandingService
from paperbase.agentic.scope_resolver import ScopeResolver
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import (
    EvidenceSourceType, PaperSearchOutcome, PaperSearchSpec, PaperSearchStatus,
    RetrievedEvidence, SearchPapersPayload, ToolName, ToolResult, ToolStatus,
)
from test_agentic_grounded_answer import QueueClient
from test_agentic_langgraph import _Database


def item(rid, answer, eid, gap=None):
    return {
        "requirement_id": rid, "evidence_ids": [eid],
        "remaining_gap": gap,
    }


def draft(answer, coverage):
    ids = list(dict.fromkeys(eid for item in coverage for eid in item["evidence_ids"]))
    return {"answer": answer + " " + "".join(f"[{eid}]" for eid in ids),
            "citations": ids, "requirements": coverage}


class PoolExecutor:
    def __init__(self, sources, chunk_windows=None):
        self.sources = iter(sources)
        self.chunk_windows = iter(chunk_windows) if chunk_windows is not None else None
        self.calls = []

    def execute(self, action, **kwargs):
        self.calls.append(action)
        index = len(self.calls)
        chunks = next(self.chunk_windows) if self.chunk_windows is not None else (f"chunk_{index}",)
        source = RetrievedEvidence(
            EvidenceSourceType.CONTENT, "P1", (chunks[0],),
            chunks, next(self.sources),
            paper_title="Paper One", section="Method", section_id="method",
        )
        return ToolResult(
            ToolStatus.OK, ToolName.SEARCH_PAPERS,
            SearchPapersPayload((PaperSearchOutcome(
                search=action.searches[0], status=PaperSearchStatus.OK,
                evidence_candidates=(source,),
            ),)),
        )


class GapDecision:
    """Observe actual answer gaps; no fixed Requirement order."""

    def __init__(self):
        self.states = []

    def decide(self, state):
        self.states.append(state)
        if state.grounded_answer is None:
            rid, query = state.plan.requirements[0].requirement_id, "Find method overview"
        else:
            rid, query = state.grounded_answer.open_gaps[0]
        return DecisionResult(SearchPapersAction((PaperSearchSpec(
            paper_id="P1", query=query, lexical_terms=("method",),
            requirement_ids=(rid,),
        ),)), 1)


def run_loop(descriptions, sources, responses, *, chunk_windows=None):
    plan = ResearchPlan("Explain the method", tuple(
        ResearchRequirement(f"R{index}", description, frozenset({EvidenceTarget.METHOD}))
        for index, description in enumerate(descriptions, start=1)
    ))

    class Planner:
        def plan(self, context):
            return PlannerResult(plan, False, 0)

    client = QueueClient(*responses)
    decision = GapDecision()
    executor = PoolExecutor(sources, chunk_windows)
    question = "Explain the signal and uncertainty paths and how their outputs are computed."
    workflow = LangGraphResearchWorkflow(
        scope_resolver=ScopeResolver(_Database()),
        query_understanding=QueryUnderstandingService(), planner=Planner(),
        decision_agent=decision, tool_executor=executor,
        grounded_answerer=LLMGroundedAnswerer(client),
        settings=AgenticResearchSettings(evidence_admission_enabled=True),
    )
    result = workflow.run(question, selected_paper_ids=("P1",))
    return result, client, decision, executor, question


@pytest.mark.parametrize("split", [False, True], ids=["broad-plan", "split-plan"])
def test_cited_partial_gap_drives_search_then_all_requirements_see_cumulative_pool(split):
    overview = "The signal path predicts a point forecast and the uncertainty path uses an ensemble."
    mechanism = "The signal path uses an identity transform and the uncertainty path averages three positive transforms."
    gap = "Which transforms and combination rule compute the uncertainty estimate?"
    if split:
        descriptions = ("Explain the role of the signal path", "Explain how uncertainty is computed")
        first = [item("R1", "The signal path predicts a point forecast.", "E1"),
                 item("R2", "The uncertainty path uses an ensemble.", "E1", gap)]
        final = [item("R1", "The signal path predicts a point forecast.", "E1"),
                 item("R2", "Three positive transforms are averaged.", "E2")]
    else:
        descriptions = ("Explain both path roles and how their outputs are computed",)
        first = [item("R1", overview, "E1", gap)]
        final_item = item("R1", overview + " " + mechanism, "E1")
        final_item["evidence_ids"].append("E2")
        final = [final_item]

    result, client, decision, executor, question = run_loop(
        descriptions, (overview, mechanism), (draft(overview, first), draft(overview + " " + mechanism, final)),
    )

    assert result.status is ResearchRunStatus.COMPLETED
    assert len(decision.states) == len(executor.calls) == len(client.calls) == 2
    intermediate = decision.states[1].grounded_answer
    assert not intermediate.complete
    assert intermediate.answer is not None
    assert intermediate.requirements[-1].evidence_ids
    assert intermediate.open_gaps == (("R2" if split else "R1", gap),)
    assert executor.calls[1].searches[0].query == gap
    # Both searches target the gap; earlier evidence remains usable by every item.
    payloads = [json.loads(call["user_prompt"]) for call in client.calls]
    assert [[e["evidence_id"] for e in p["evidence"]] for p in payloads] == [["E1"], ["E1", "E2"]]
    assert all(p["question"] == question for p in payloads)
    expected_ids = [f"R{i}" for i in range(1, len(descriptions) + 1)]
    assert all([r["requirement_id"] for r in p["requirements"]] == expected_ids for p in payloads)
    assert result.answer_input_evidence_ids == ("E1", "E2")
    assert result.answer is not None and result.answer.overall_status == "complete"


def test_fresh_evidence_can_reopen_complete_requirement_and_drive_another_search():
    reopened_gap = "Which processing paths are required beyond the illustrative example?"
    initial = [item("R1", "The example has two processing paths.", "E1"),
               item("R2", "The example reports a forecast.", "E1", "How is uncertainty computed?")]
    reassessed = [item("R1", "The two-path diagram is only an illustration.", "E2", reopened_gap),
                  item("R2", "Uncertainty is computed by averaging the ensemble.", "E2")]
    final = [item("R1", "The required paths are specified by the general construction.", "E3"),
             item("R2", "Uncertainty is computed by averaging the ensemble.", "E2")]
    result, client, decision, executor, _ = run_loop(
        ("Explain the processing paths", "Explain the uncertainty calculation"),
        ("The example has two processing paths and reports a forecast.",
         "The two-path diagram is illustrative and uncertainty uses ensemble averaging.",
         "The general construction specifies the required processing paths."),
        (draft("The example has two paths and reports a forecast.", initial),
         draft("The diagram is illustrative; uncertainty averages the ensemble.", reassessed),
         draft("The general construction specifies paths and uncertainty averages the ensemble.", final)),
    )

    assert result.status is ResearchRunStatus.COMPLETED
    assert decision.states[1].grounded_answer.requirements[0].complete
    reopened = decision.states[2].grounded_answer
    assert not reopened.requirements[0].complete
    assert reopened.requirements[1].complete
    assert reopened.open_gaps == (("R1", reopened_gap),)
    assert executor.calls[2].searches[0].requirement_ids == ("R1",)
    assert executor.calls[2].searches[0].query == reopened_gap
    payloads = [json.loads(call["user_prompt"]) for call in client.calls]
    assert all([r["requirement_id"] for r in p["requirements"]] == ["R1", "R2"] for p in payloads)
    assert [e["evidence_id"] for e in payloads[-1]["evidence"]] == ["E1", "E2", "E3"]


def test_expanded_window_with_new_chunk_reaches_answer_and_closes_gap():
    overview = "The uncertainty path uses an ensemble."
    detail = "Three positive transforms are averaged to compute uncertainty."
    gap = "What combination rule computes the uncertainty estimate?"
    first = item("R1", overview, "E1", gap)
    final = item("R1", overview + " " + detail, "E2")
    result, client, decision, _, _ = run_loop(
        ("Explain how uncertainty is computed",),
        (overview, overview + " " + detail),
        (draft(overview, [first]), draft(overview + " " + detail, [final])),
        chunk_windows=(("c1", "c2", "c3"), ("c1", "c2", "c3", "c4")),
    )
    assert len(decision.states) == 2
    assert result.status is ResearchRunStatus.COMPLETED
    assert result.observations[1].added_evidence_ids == ("E2",)
    assert result.observations[1].reused_evidence_ids == ()
    assert result.answer_input_evidence_ids == ("E1", "E2")
    payload = json.loads(client.calls[-1]["user_prompt"])
    assert payload["evidence"][1]["source_chunk_ids"] == ["c1", "c2", "c3", "c4"]
    assert detail in payload["evidence"][1]["content"]
    assert detail in result.answer.answer
