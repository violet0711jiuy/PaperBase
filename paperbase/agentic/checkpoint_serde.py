"""领域契约与 LangGraph JSON-compatible checkpoint DTO 之间的唯一转换层。"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import TypeAdapter

from paperbase.retrieval.contracts import ToolResult

from .contracts import (
    ActionTrace,
    AgentAction,
    CorpusScopeSnapshot,
    CoverageState,
    EvidenceQualification,
    GuardState,
    QueryContext,
    RegistryState,
    ResearchObservation,
    ResearchPlan,
    ResearchRunResult,
    ScopeResolution,
)
from .scope_resolver import TrustedWorkspaceScope


CheckpointValue = dict[str, Any] | list[Any] | str | int | float | bool | None
T = TypeVar("T")


_ADAPTERS: dict[object, TypeAdapter[Any]] = {
    TrustedWorkspaceScope: TypeAdapter(TrustedWorkspaceScope),
    ScopeResolution: TypeAdapter(ScopeResolution),
    CorpusScopeSnapshot: TypeAdapter(CorpusScopeSnapshot),
    CoverageState: TypeAdapter(CoverageState),
    tuple[EvidenceQualification, ...]: TypeAdapter(tuple[EvidenceQualification, ...]),
    QueryContext: TypeAdapter(QueryContext),
    ResearchPlan: TypeAdapter(ResearchPlan),
    AgentAction: TypeAdapter(AgentAction),
    ToolResult: TypeAdapter(ToolResult),
    RegistryState: TypeAdapter(RegistryState),
    GuardState: TypeAdapter(GuardState),
    tuple[ResearchObservation, ...]: TypeAdapter(tuple[ResearchObservation, ...]),
    tuple[ActionTrace, ...]: TypeAdapter(tuple[ActionTrace, ...]),
    ResearchRunResult: TypeAdapter(ResearchRunResult),
}


def dump_contract(value: T, contract_type: object) -> CheckpointValue:
    """输出只含 dict/list/scalar 的稳定 checkpoint 表示。"""
    return _ADAPTERS[contract_type].dump_python(value, mode="json")


def load_contract(value: CheckpointValue, contract_type: object) -> T:
    """通过 Pydantic 重新执行类型恢复和 dataclass 约束。"""
    return _ADAPTERS[contract_type].validate_python(value)
