"""Scope → Understanding → Plan → Agent Loop 的 Phase 2 应用入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import (
    BudgetUsage,
    ResearchRunResult,
    ResearchRunStatus,
    ScopeResolutionStatus,
    StopReason,
)
from .planner import ResearchPlanner
from .query_understanding import QueryUnderstandingService
from .runtime import ResearchRuntime
from .scope_resolver import ScopeResolver, TrustedWorkspaceScope

if TYPE_CHECKING:
    from .graph import LangGraphResearchWorkflow


class AgenticResearchService:
    def __init__(
        self,
        *,
        workflow: "LangGraphResearchWorkflow | None" = None,
        scope_resolver: ScopeResolver | None = None,
        query_understanding: QueryUnderstandingService | None = None,
        planner: ResearchPlanner | None = None,
        runtime: ResearchRuntime | None = None,
    ) -> None:
        if workflow is None and any(
            component is None
            for component in (scope_resolver, query_understanding, planner, runtime)
        ):
            raise ValueError(
                "Provide a LangGraph workflow, or every legacy Phase 2 component."
            )
        self._workflow = workflow
        self._scope_resolver = scope_resolver
        self._query_understanding = query_understanding
        self._planner = planner
        self._runtime = runtime

    def research(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
    ) -> ResearchRunResult:
        if self._workflow is not None:
            return self._workflow.run(
                query,
                trusted_workspace=trusted_workspace,
                selected_paper_ids=selected_paper_ids,
                knowledge_base_all=knowledge_base_all,
            )
        assert self._scope_resolver is not None
        assert self._query_understanding is not None
        assert self._planner is not None
        assert self._runtime is not None
        resolution = self._scope_resolver.resolve(
            query,
            trusted_workspace=trusted_workspace,
            selected_paper_ids=selected_paper_ids,
            knowledge_base_all=knowledge_base_all,
        )
        if resolution.status is not ScopeResolutionStatus.RESOLVED:
            return ResearchRunResult(
                status=ResearchRunStatus.NEEDS_CLARIFICATION,
                stop_reason=StopReason.SCOPE_UNRESOLVED,
                query_context=None,
                plan=None,
                evidence=(),
                observations=(),
                traces=(),
                budget_usage=BudgetUsage(),
                clarification=resolution.clarification,
            )
        understanding = self._query_understanding.understand(query, resolution)
        planned = self._planner.plan(understanding.context)
        return self._runtime.run(
            context=understanding.context,
            plan=planned.plan,
            corpus_snapshot=resolution.corpus_snapshot,
            initial_llm_calls=understanding.llm_calls + planned.llm_calls,
        )
