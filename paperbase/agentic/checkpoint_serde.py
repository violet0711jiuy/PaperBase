"""Current Agentic domain contracts at the LangGraph checkpoint boundary.

The Grounded Answer architecture uses a new namespace and intentionally does
not migrate semantic state from the removed Projector architecture.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import TypeAdapter

from paperbase.retrieval.contracts import ToolResult
from paperbase.retrieval.requests import CanonicalSearchRequest
from .contracts import (
    ActionCanonicalization,
    ActionRejectionContext,
    ActionTrace,
    AgentAction,
    CandidateSelection,
    CorpusScopeSnapshot,
    CoverageState,
    EvidenceFilterReport,
    EvidenceRecord,
    GroundedAnswerAttempt,
    GuardState,
    LLMCallRecord,
    QueryContext,
    RegistryState,
    RequirementState,
    ResearchAnswer,
    ResearchObservation,
    ResearchPlan,
    ResearchRunResult,
    ScopeResolution,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
    GetSectionContextAction,
    GetPaperOverviewAction,
    StopResearchAction,
    TryAnswerAction,
)
from .scope_resolver import TrustedWorkspaceScope


AGENTIC_CHECKPOINT_NAMESPACE = "paperbase-agentic-grounded-answer-v1"


def checkpoint_thread_id(thread_id: str) -> str:
    return f"{AGENTIC_CHECKPOINT_NAMESPACE}:{thread_id}"


CheckpointValue = dict[str, Any] | list[Any] | str | int | float | bool | None
T = TypeVar("T")


_ADAPTERS: dict[object, TypeAdapter[Any]] = {
    tuple[CanonicalSearchRequest, ...]: TypeAdapter(tuple[CanonicalSearchRequest, ...]),
    TrustedWorkspaceScope: TypeAdapter(TrustedWorkspaceScope),
    ScopeResolution: TypeAdapter(ScopeResolution),
    CandidateSelection: TypeAdapter(CandidateSelection),
    CorpusScopeSnapshot: TypeAdapter(CorpusScopeSnapshot),
    CoverageState: TypeAdapter(CoverageState),
    tuple[EvidenceRecord, ...]: TypeAdapter(tuple[EvidenceRecord, ...]),
    EvidenceFilterReport: TypeAdapter(EvidenceFilterReport),
    QueryContext: TypeAdapter(QueryContext),
    ResearchPlan: TypeAdapter(ResearchPlan),
    RequirementState: TypeAdapter(RequirementState),
    tuple[RequirementState, ...]: TypeAdapter(tuple[RequirementState, ...]),
    AgentAction: TypeAdapter(AgentAction),
    ActionCanonicalization: TypeAdapter(ActionCanonicalization),
    ToolResult: TypeAdapter(ToolResult),
    RegistryState: TypeAdapter(RegistryState),
    GuardState: TypeAdapter(GuardState),
    ActionRejectionContext: TypeAdapter(ActionRejectionContext),
    tuple[ActionRejectionContext, ...]: TypeAdapter(tuple[ActionRejectionContext, ...]),
    tuple[ResearchObservation, ...]: TypeAdapter(tuple[ResearchObservation, ...]),
    tuple[ActionTrace, ...]: TypeAdapter(tuple[ActionTrace, ...]),
    tuple[LLMCallRecord, ...]: TypeAdapter(tuple[LLMCallRecord, ...]),
    ResearchRunResult: TypeAdapter(ResearchRunResult),
    ResearchAnswer: TypeAdapter(ResearchAnswer),
    GroundedAnswerAttempt: TypeAdapter(GroundedAnswerAttempt),
}


def dump_contract(value: T, contract_type: object) -> CheckpointValue:
    return _ADAPTERS[contract_type].dump_python(value, mode="json")


def load_contract(value: CheckpointValue, contract_type: object) -> T:
    if contract_type is AgentAction:
        if not isinstance(value, dict):
            raise ValueError("AgentAction checkpoint payload must be an object.")
        action_types = {
            "search_papers": SearchPapersAction,
            "search_knowledge_base": SearchKnowledgeBaseAction,
            "search_references": SearchReferencesAction,
            "get_section_context": GetSectionContextAction,
            "get_paper_overview": GetPaperOverviewAction,
            "try_answer": TryAnswerAction,
            "stop_research": StopResearchAction,
        }
        action_type = action_types.get(value.get("action_name"))
        if action_type is None:
            raise ValueError(f"Unknown AgentAction checkpoint kind: {value.get('action_name')!r}.")
        return TypeAdapter(action_type).validate_python(value)
    return _ADAPTERS[contract_type].validate_python(value)
