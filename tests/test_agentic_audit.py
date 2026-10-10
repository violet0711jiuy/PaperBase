from __future__ import annotations

import json

import pytest

from paperbase.agentic.audit import AUDIT_SCHEMA_VERSION, AgenticAuditSnapshot, build_agentic_audit_snapshot
from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceRecord,
    EvidenceTarget,
    GroundedCitation,
    QueryContext,
    QueryScope,
    RequirementAnswer,
    RequirementState,
    RequirementStatus,
    ResearchAnswer,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
    ResearchRunResult,
    ResearchRunStatus,
    ScopeSource,
    StopReason,
)
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


def _result() -> ResearchRunResult:
    source = RetrievedEvidence(
        EvidenceSourceType.CONTENT, "P1", ("secret-chunk",), ("secret-chunk",),
        "C_i indicates whether the interval captures the target.",
        paper_title="Paper One", section="Method", section_id="method",
    )
    record = EvidenceRecord(
        "E1", source, ("capture indicator",), 1, 1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requested_requirement_ids=("R1",),
    )
    requirement = ResearchRequirement("R1", "Explain C_i", frozenset({EvidenceTarget.METHOD}))
    grounded = RequirementAnswer(
        "R1", "C_i is a capture indicator.",
        (GroundedCitation("E1", ("S1",), source.content),),
    )
    answer = ResearchAnswer(
        "success", "C_i is a capture indicator. [E1]", ("E1",),
        grounded_requirements=(grounded,), overall_status="complete",
    )
    context = QueryContext(
        "Explain C_i", "Explain C_i", "explain", QueryScope.SINGLE_PAPER,
        ("P1",), ScopeSource.UI_SELECTION, ResearchGoal.FIND_EVIDENCE,
        frozenset({EvidenceTarget.METHOD}),
    )
    return ResearchRunResult(
        ResearchRunStatus.COMPLETED, StopReason.AGENT_FINISHED, context,
        ResearchPlan("Explain", (requirement,)), (record,), (), (), BudgetUsage(),
        selected_evidence=(record,), answer=answer, thread_id="thread-1",
        requirement_states=(RequirementState("R1", RequirementStatus.SATISFIED, ("E1",)),),
    )


def test_grounded_audit_round_trips_without_projector_or_hidden_content() -> None:
    snapshot = build_agentic_audit_snapshot(_result())
    restored = AgenticAuditSnapshot.from_json(snapshot.to_json())
    assert restored == snapshot
    assert restored.schema_version == AUDIT_SCHEMA_VERSION
    assert restored.selected_evidence_ids == restored.cited_evidence_ids == ("E1",)
    assert restored.evidence[0].citation_span_ids == ("S1",)
    assert restored.requirements[0].answer == "C_i is a capture indicator."
    payload = snapshot.to_dict()
    assert "projection_type" not in payload["evidence"][0]
    assert "analysis_status" not in payload["evidence"][0]
    assert "secret-chunk" in snapshot.to_json()  # provenance is public; raw body is previewed only


def test_audit_snapshot_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema version"):
        AgenticAuditSnapshot.from_json(json.dumps({"schema_version": "old-projector-v4"}))


def test_requirement_status_and_grounded_citation_share_one_authority() -> None:
    snapshot = build_agentic_audit_snapshot(_result())
    requirement = snapshot.requirements[0]
    assert requirement.status == "satisfied"
    assert requirement.evidence_ids == ("E1",)
    assert requirement.citations == ({"evidence_id": "E1", "source_span_ids": ("S1",)},)
