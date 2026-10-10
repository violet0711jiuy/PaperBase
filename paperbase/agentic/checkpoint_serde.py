"""领域契约与 LangGraph checkpoint DTO 之间的唯一版本兼容边界。"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import TypeAdapter

from paperbase.retrieval.contracts import ToolResult
from paperbase.retrieval.requests import CanonicalSearchRequest

from .contracts import (
    ActionTrace,
    AgentAction,
    ActionCanonicalization,
    CandidateSelection,
    CorpusScopeSnapshot,
    CoverageState,
    EvidenceQualification,
    EvidenceFilterReport,
    EvidenceRecord,
    GuardState,
    ActionRejectionContext,
    LLMCallRecord,
    QueryContext,
    RegistryState,
    ResearchObservation,
    ResearchPlan,
    RequirementState,
    ResearchAnswer,
    ResearchRunResult,
    ScopeResolution,
)
from .scope_resolver import TrustedWorkspaceScope


# Evidence IDs are run-scoped locators embedded throughout Registry, RequirementState,
# observations, selection, and Answer state.  REF# therefore starts a fresh
# LangGraph namespace instead of keeping two runtime Evidence ID formats alive.
AGENTIC_CHECKPOINT_NAMESPACE = "paperbase-agentic-ref-evidence-v2"


def checkpoint_thread_id(thread_id: str) -> str:
    """Map a public thread ID to the current checkpoint contract namespace."""
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
    tuple[EvidenceQualification, ...]: TypeAdapter(tuple[EvidenceQualification, ...]),
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
}

_LEGACY_ANALYSIS_FIELDS = {
    "analysis_status": "status",
    "digest": "digest",
    "claims": "claims",
    "projection": "projection",
    "covered_targets": "covered_targets",
    "covered_requirement_ids": "covered_requirement_ids",
    "method_entities": "method_entities",
    "method_families": "method_families",
    "baseline_entities": "baseline_entities",
    "projection_rejections": "projection_rejections",
    "analysis_error": "error_code",
}

_DERIVED_ASSESSMENT_FIELDS = {
    "covered_targets", "covered_requirement_ids", "method_entities",
    "method_families", "baseline_entities",
}


def dump_contract(value: T, contract_type: object) -> CheckpointValue:
    """新 checkpoint 只写 Requirement-first 字段。"""
    payload = _ADAPTERS[contract_type].dump_python(value, mode="json")
    return _canonical_checkpoint_payload(payload)


def load_contract(value: CheckpointValue, contract_type: object) -> T:
    """集中升级旧 DTO，再由 Pydantic 执行领域约束。"""
    return _ADAPTERS[contract_type].validate_python(
        _canonical_checkpoint_payload(value)
    )


def _canonical_checkpoint_payload(value: Any) -> Any:
    """One-way migration for legacy Subtask and free-text search hints.

    Historical checkpoints may be resumed in the middle of a graph edge, so
    plans, actions, tool results and observations are upgraded independently.
    A legacy Subtask ID is retained as an optional Requirement ID when a plan
    already has Requirements; it cannot silently add a new answer obligation.
    """
    if isinstance(value, list):
        return [_canonical_checkpoint_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    result = {
        key: (
            item
            if key in {"retrieval_metadata", "metadata", "raw_action_arguments", "canonical_effective_requests"}
            else _canonical_checkpoint_payload(item)
        )
        for key, item in value.items()
    }
    if "evidence_id" in result and "status" in result and "projections" in result:
        if result["projections"]:
            if result.get("projection") is not None and result["projections"] != [result["projection"]]:
                raise ValueError("Conflicting checkpoint projection aliases.")
            result.pop("projection", None)
        elif result.get("projection") is not None:
            result["projections"] = [result.pop("projection")]
        else:
            result.pop("projection", None)
    if result.get("projection_type") == "fact" and "fact_type" in result:
        result["facts"] = [{
            "fact_type": result.pop("fact_type"),
            "value": result.pop("value"),
            "ownership": result.pop("ownership"),
            "supported_requirement_ids": result.pop("supported_requirement_ids", []),
            "proposed_requirement_ids": result.pop("proposed_requirement_ids", []),
            "support_anchors": result.pop("support_anchors", []),
        }]
    if "action_name" in result and isinstance(result.get("action_arguments"), dict):
        arguments = result["action_arguments"]
        result["action_arguments"] = _canonical_checkpoint_payload(
            {**arguments, "action_name": result["action_name"]}
        )
        result["action_arguments"].pop("action_name", None)
    if "original_query" in result and "retrieval_goal" in result and "default_evidence_targets" in result:
        result.pop("requires_complete_coverage", None)
        result.pop("search_bibliography", None)
    if (
        "requirement_id" in result
        and "description" in result
        and "evidence_targets" in result
        and "priority" in result
    ):
        priority = result.pop("priority")
        if priority not in (None, "required", "optional"):
            raise ValueError(f"Invalid legacy Requirement priority: {priority!r}")
        result.setdefault("required", priority != "optional")

    if "goal" in result and "subtasks" in result:
        subtasks = result.pop("subtasks") or []
        if not isinstance(subtasks, list):
            raise ValueError("Legacy plan subtasks must be a list.")
        requirements = result.setdefault("requirements", [])
        if not isinstance(requirements, list):
            raise ValueError("Plan requirements must be a list.")
        legacy_only = not requirements
        known_ids = {
            item["requirement_id"]
            for item in requirements
            if isinstance(item, dict) and "requirement_id" in item
        }
        seen_subtask_ids: set[str] = set()
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                raise ValueError("Legacy plan subtask must be an object.")
            subtask_id = subtask.get("subtask_id")
            if not isinstance(subtask_id, str) or not subtask_id.strip():
                raise ValueError("Legacy plan subtask needs a non-empty ID.")
            if subtask_id in seen_subtask_ids:
                raise ValueError("Legacy plan subtask IDs must be unique.")
            seen_subtask_ids.add(subtask_id)
            if subtask_id in known_ids:
                continue
            requirements.append({
                "requirement_id": subtask_id,
                "description": subtask.get("goal") or subtask.get("success_criteria"),
                "required": legacy_only,
                "evidence_targets": subtask.get("evidence_targets") or ["general"],
            })
            known_ids.add(subtask_id)

    is_search_item = "paper_id" in result and "query" in result
    is_action = "action_name" in result
    if is_search_item or is_action:
        legacy_id = result.pop("target_subtask_id", None)
    else:
        legacy_id = None
    if legacy_id is not None and (
        not isinstance(legacy_id, str) or not legacy_id.strip()
    ):
        raise ValueError("Legacy target_subtask_id must be a non-empty string.")
    if legacy_id:
        if result.get("action_name") == "search_papers" or result.get("action") == "search_papers":
            for search in result.get("searches", []):
                if isinstance(search, dict) and not search.get("requirement_ids"):
                    search["requirement_ids"] = [legacy_id]
        elif not result.get("requirement_ids"):
            result["requirement_ids"] = [legacy_id]

    legacy_ids = result.pop("target_subtask_ids", None) if "evidence_id" in result else None
    if legacy_ids:
        result["requested_requirement_ids"] = list(dict.fromkeys(
            [*result.get("requested_requirement_ids", []), *legacy_ids]
        ))
    if is_search_item:
        result.pop("missing_information", None)
    if "evidence_id" in result:
        result.pop("requested_missing_information", None)
        result.pop("addressed_information", None)
    if "evidence_id" in result and "evidence" in result:
        assessment = result.pop("assessment", None)
        legacy_assessment = result.pop("analysis", None)
        if assessment is None:
            assessment = legacy_assessment
        legacy_analysis = {
            target: result.pop(source)
            for source, target in _LEGACY_ANALYSIS_FIELDS.items()
            if source in result
        }
        qualification = result.pop("qualification", None)
        if assessment is None and qualification is not None and (
            not legacy_analysis or legacy_analysis.get("status", "pending") == "pending"
        ):
            relation = qualification["relation"]
            evidence_content = result["evidence"]["content"]
            digest = evidence_content[:500]
            legacy_analysis = {
                "status": "irrelevant" if relation == "irrelevant" else "qualified",
                "digest": digest,
                "claims": [] if relation == "irrelevant" else [{
                    "claim_id": f"{result['evidence_id']}:C1",
                    "statement": digest,
                    "relation": relation,
                    "ownership": qualification["ownership"],
                    "rationale": qualification["rationale"],
                    "source_excerpt": evidence_content[:300],
                }],
            }
        if assessment is None and legacy_analysis:
            assessment = {
                "evidence_id": result["evidence_id"],
                "status": "pending",
                **legacy_analysis,
            }
        if isinstance(assessment, dict):
            for field in _DERIVED_ASSESSMENT_FIELDS:
                assessment.pop(field, None)
            if assessment.get("projection") is not None or assessment.get("projections"):
                # 旧分析同时存了 Claims 和 Projection；新结果只以核验投影为准。
                assessment.pop("claims", None)
                assessment.pop("digest", None)
            result["assessment"] = assessment
    if (
        "search" not in result
        and is_search_item
        and "status" in result
        and "evidence_candidates" in result
    ):
        # 旧 ToolResult 在 Outcome 顶层重复保存了检索请求字段。
        result["search"] = {
            key: result.pop(key)
            for key in (
                "paper_id", "query", "lexical_terms", "evidence_targets", "requirement_ids"
            )
            if key in result
        }
    return result
