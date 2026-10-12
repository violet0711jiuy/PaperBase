from __future__ import annotations

from paperbase.agentic.contracts import (
    EvidenceTarget,
    ResearchPlan,
    ResearchRequirement,
    SearchPapersAction,
)
from paperbase.agentic.evidence_registry import EvidenceRegistry
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


def _plan() -> ResearchPlan:
    return ResearchPlan("Explain", (
        ResearchRequirement("R1", "Explain A", frozenset({EvidenceTarget.METHOD})),
        ResearchRequirement("R2", "Explain B", frozenset({EvidenceTarget.RESULT})),
    ))


def _action(query: str, rid: str) -> SearchPapersAction:
    return SearchPapersAction((PaperSearchSpec(
        paper_id="P1", query=query, lexical_terms=(query,), requirement_ids=(rid,)
    ),))


def _result(query: str, chunk_ids=("c1",)) -> ToolResult:
    evidence = RetrievedEvidence(
        EvidenceSourceType.CONTENT, "P1", (chunk_ids[0],), tuple(chunk_ids),
        "raw evidence", section_id="sec1",
    )
    outcome = PaperSearchOutcome(
        search=PaperSearchSpec("P1", query, (query,)),
        status=PaperSearchStatus.OK, evidence_candidates=(evidence,),
    )
    return ToolResult(ToolStatus.OK, ToolName.SEARCH_PAPERS, SearchPapersPayload((outcome,)))


def test_registry_keeps_raw_evidence_and_merges_retrieval_provenance() -> None:
    registry = EvidenceRegistry()
    first = registry.register_tool_result(
        _result("method"), action=_action("method", "R1"), iteration=1, plan=_plan()
    )
    second = registry.register_tool_result(
        _result("result"), action=_action("result", "R2"), iteration=2, plan=_plan()
    )
    assert first.added_ids == ("E1",)
    assert second.reused_ids == ("E1",)
    record = registry.records[0]
    assert record.evidence.content == "raw evidence"
    assert record.supporting_queries == ("method", "result")
    assert record.requested_requirement_ids == ("R1", "R2")
    assert record.requested_evidence_targets == frozenset({EvidenceTarget.METHOD, EvidenceTarget.RESULT})


def test_registry_preserves_new_chunks_in_overlapping_expansion_windows() -> None:
    registry = EvidenceRegistry()
    registry.register_tool_result(
        _result("first", ("c1", "c2", "c3")),
        action=_action("first", "R1"), iteration=1, plan=_plan(),
    )
    delta = registry.register_tool_result(
        _result("expanded", ("c1", "c2", "c3", "c4")),
        action=_action("expanded", "R1"), iteration=2, plan=_plan(),
    )
    assert delta.added_ids == ("E2",)
    assert delta.reused_ids == ()
    assert len(registry.records) == 2
    assert registry.get("E1").evidence.source_chunk_ids == ("c1", "c2", "c3")
    assert registry.get("E2").evidence.source_chunk_ids == ("c1", "c2", "c3", "c4")


def test_registry_reuses_fully_covered_window_and_keeps_existing_spans_stable() -> None:
    registry = EvidenceRegistry()
    registry.register_tool_result(
        _result("first", ("c1", "c2", "c3", "c4")),
        action=_action("first", "R1"), iteration=1, plan=_plan(),
    )
    original = registry.get("E1").evidence
    delta = registry.register_tool_result(
        _result("subset", ("c2", "c3")),
        action=_action("subset", "R2"), iteration=2, plan=_plan(),
    )
    assert delta.added_ids == ()
    assert delta.reused_ids == ("E1",)
    assert registry.get("E1").evidence == original
    assert registry.get("E1").requested_requirement_ids == ("R1", "R2")
