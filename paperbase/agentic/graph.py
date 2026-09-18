"""Phase 2.5 LangGraph workflow；框架只负责编排，业务规则仍由领域组件实现。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import ToolResult

from .contracts import (
    ActionTrace,
    AgentAction,
    BudgetUsage,
    CorpusScopeSnapshot,
    CoverageState,
    EvidenceQualification,
    FinishAction,
    GuardState,
    QueryContext,
    RegistryState,
    ResearchObservation,
    ResearchPlan,
    ResearchRunResult,
    ResearchRunStatus,
    ScopeResolution,
    ScopeResolutionStatus,
    StopReason,
)
from .checkpoint_serde import CheckpointValue, dump_contract, load_contract
from .coverage import CoverageManager, CoveragePolicy
from .decision import DecisionAgent, DecisionError, DecisionState
from .evidence_registry import EvidenceRegistry
from .guards import (
    BudgetAndGuardTracker,
    action_fingerprint,
    action_subtask_ids,
)
from .observation import build_observation
from .planner import ResearchPlanner
from .query_understanding import QueryUnderstandingService
from .runtime import ToolExecutor
from .selector import FinalEvidenceSelector
from .scope_resolver import ScopeResolver, TrustedWorkspaceScope
from .transitions import finish_plan, mark_subtasks_started


NextNode = Literal[
    "understand_query",
    "build_plan",
    "decide_action",
    "qualify_and_assess",
    "validate_action",
    "execute_tool",
    "register_and_observe",
    "finish_run",
]


class AgentGraphState(TypedDict, total=False):
    """Checkpoint 中只保存领域 DTO；所有服务和连接通过 Runtime Context 注入。"""

    original_query: str
    trusted_workspace: CheckpointValue
    selected_paper_ids: list[str]
    knowledge_base_all: bool
    scope_resolution: CheckpointValue
    corpus_snapshot: CheckpointValue
    query_context: CheckpointValue
    plan: CheckpointValue
    action: CheckpointValue
    qualifications: CheckpointValue
    action_fingerprint: str
    tool_result: CheckpointValue
    registry_state: CheckpointValue
    guard_state: CheckpointValue
    coverage: CheckpointValue
    observations: CheckpointValue
    traces: CheckpointValue
    terminal_status: str
    stop_reason: str
    clarification: str | None
    error_codes: list[str]
    next_node: NextNode
    result: CheckpointValue


@dataclass(frozen=True)
class AgentGraphContext:
    scope_resolver: ScopeResolver
    query_understanding: QueryUnderstandingService
    planner: ResearchPlanner
    decision_agent: DecisionAgent
    tool_executor: ToolExecutor
    settings: AgenticResearchSettings


@dataclass(frozen=True)
class GraphExecution:
    thread_id: str
    interrupted: bool
    next_nodes: tuple[str, ...]
    state: AgentGraphState
    result: ResearchRunResult | None


class LangGraphResearchWorkflow:
    """编译并执行 PaperBase 单 Agent 图，支持内存 checkpoint 与恢复。"""

    def __init__(
        self,
        *,
        scope_resolver: ScopeResolver,
        query_understanding: QueryUnderstandingService,
        planner: ResearchPlanner,
        decision_agent: DecisionAgent,
        tool_executor: ToolExecutor,
        settings: AgenticResearchSettings,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._context = AgentGraphContext(
            scope_resolver=scope_resolver,
            query_understanding=query_understanding,
            planner=planner,
            decision_agent=decision_agent,
            tool_executor=tool_executor,
            settings=settings,
        )
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = build_agentic_research_graph().compile(
            checkpointer=self._checkpointer,
            name="paperbase_agentic_research",
        )

    @property
    def graph(self):
        return self._graph

    def run(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
        thread_id: str | None = None,
    ) -> ResearchRunResult:
        execution = self.start(
            query,
            trusted_workspace=trusted_workspace,
            selected_paper_ids=selected_paper_ids,
            knowledge_base_all=knowledge_base_all,
            thread_id=thread_id,
        )
        if execution.result is None:
            raise RuntimeError("LangGraph run stopped before producing ResearchRunResult.")
        return execution.result

    def start(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
        thread_id: str | None = None,
        interrupt_after: tuple[str, ...] = (),
    ) -> GraphExecution:
        resolved_thread_id = thread_id or f"research_{uuid4().hex}"
        config = _thread_config(
            resolved_thread_id,
            recursion_limit=_recursion_limit(self._context.settings),
        )
        if self._graph.get_state(config).values:
            raise ValueError(
                f"Research thread already exists: {resolved_thread_id}. Use resume()."
            )
        initial_state: AgentGraphState = {
            "original_query": query,
            "trusted_workspace": (
                dump_contract(trusted_workspace, TrustedWorkspaceScope)
                if trusted_workspace is not None
                else None
            ),
            "selected_paper_ids": list(selected_paper_ids),
            "knowledge_base_all": knowledge_base_all,
            "registry_state": dump_contract(RegistryState(), RegistryState),
            "guard_state": dump_contract(GuardState(), GuardState),
            "observations": dump_contract((), tuple[ResearchObservation, ...]),
            "traces": dump_contract((), tuple[ActionTrace, ...]),
            "error_codes": [],
        }
        self._graph.invoke(
            initial_state,
            config=config,
            context=self._context,
            interrupt_after=interrupt_after or None,
        )
        return self.inspect(resolved_thread_id)

    def resume(
        self,
        thread_id: str,
        *,
        interrupt_after: tuple[str, ...] = (),
    ) -> GraphExecution:
        existing = self._graph.get_state(_thread_config(thread_id))
        if not existing.values:
            raise LookupError(f"Unknown research thread: {thread_id}")
        if not existing.next:
            return self.inspect(thread_id)
        self._graph.invoke(
            None,
            config=_thread_config(
                thread_id,
                recursion_limit=_recursion_limit(self._context.settings),
            ),
            context=self._context,
            interrupt_after=interrupt_after or None,
        )
        return self.inspect(thread_id)

    def inspect(self, thread_id: str) -> GraphExecution:
        snapshot = self._graph.get_state(_thread_config(thread_id))
        values: AgentGraphState = dict(snapshot.values)  # type: ignore[assignment]
        next_nodes = tuple(snapshot.next)
        raw_result = values.get("result")
        return GraphExecution(
            thread_id=thread_id,
            interrupted=bool(next_nodes),
            next_nodes=next_nodes,
            state=values,
            result=(
                load_contract(raw_result, ResearchRunResult)
                if raw_result is not None
                else None
            ),
        )


def build_agentic_research_graph() -> StateGraph:
    graph = StateGraph(AgentGraphState, context_schema=AgentGraphContext)
    graph.add_node("resolve_scope", _resolve_scope)
    graph.add_node("understand_query", _understand_query)
    graph.add_node("build_plan", _build_plan)
    graph.add_node("decide_action", _decide_action)
    graph.add_node("qualify_and_assess", _qualify_and_assess)
    graph.add_node("validate_action", _validate_action)
    graph.add_node("execute_tool", _execute_tool)
    graph.add_node("register_and_observe", _register_and_observe)
    graph.add_node("finish_run", _finish_run)

    graph.add_edge(START, "resolve_scope")
    graph.add_conditional_edges(
        "resolve_scope",
        _route,
        {"understand_query": "understand_query", "finish_run": "finish_run"},
    )
    graph.add_conditional_edges(
        "understand_query",
        _route,
        {"build_plan": "build_plan", "finish_run": "finish_run"},
    )
    graph.add_conditional_edges(
        "build_plan",
        _route,
        {"decide_action": "decide_action", "finish_run": "finish_run"},
    )
    graph.add_conditional_edges(
        "decide_action",
        _route,
        {
            "decide_action": "decide_action",
            "qualify_and_assess": "qualify_and_assess",
            "finish_run": "finish_run",
        },
    )
    graph.add_conditional_edges(
        "qualify_and_assess",
        _route,
        {
            "decide_action": "decide_action",
            "validate_action": "validate_action",
            "finish_run": "finish_run",
        },
    )
    graph.add_conditional_edges(
        "validate_action",
        _route,
        {
            "decide_action": "decide_action",
            "execute_tool": "execute_tool",
            "finish_run": "finish_run",
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        _route,
        {
            "register_and_observe": "register_and_observe",
            "finish_run": "finish_run",
        },
    )
    graph.add_edge("register_and_observe", "decide_action")
    graph.add_edge("finish_run", END)
    return graph


def _resolve_scope(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    try:
        resolution = runtime.context.scope_resolver.resolve(
            state["original_query"],
            trusted_workspace=(
                load_contract(state["trusted_workspace"], TrustedWorkspaceScope)
                if state.get("trusted_workspace") is not None
                else None
            ),
            selected_paper_ids=tuple(state.get("selected_paper_ids", [])),
            knowledge_base_all=state.get("knowledge_base_all", False),
        )
    except Exception as error:
        return _fatal_update(state, "resolve_scope", error)
    if resolution.status is not ScopeResolutionStatus.RESOLVED:
        return {
            "scope_resolution": dump_contract(resolution, ScopeResolution),
            "clarification": resolution.clarification,
            "terminal_status": ResearchRunStatus.NEEDS_CLARIFICATION.value,
            "stop_reason": StopReason.SCOPE_UNRESOLVED.value,
            "next_node": "finish_run",
        }
    return {
        "scope_resolution": dump_contract(resolution, ScopeResolution),
        "corpus_snapshot": (
            dump_contract(resolution.corpus_snapshot, CorpusScopeSnapshot)
            if resolution.corpus_snapshot is not None
            else None
        ),
        "next_node": "understand_query",
    }


def _understand_query(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    try:
        understood = runtime.context.query_understanding.understand(
            state["original_query"],
            load_contract(state["scope_resolution"], ScopeResolution),
        )
    except Exception as error:
        return _fatal_update(state, "understand_query", error)
    tracker = _tracker(state, runtime.context.settings)
    tracker.record_preparation_llm_calls(understood.llm_calls)
    return {
        "query_context": dump_contract(understood.context, QueryContext),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": "build_plan",
    }


def _build_plan(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    try:
        planned = runtime.context.planner.plan(
            load_contract(state["query_context"], QueryContext)
        )
    except Exception as error:
        return _fatal_update(state, "build_plan", error)
    tracker = _tracker(state, runtime.context.settings)
    tracker.record_preparation_llm_calls(planned.llm_calls)
    return {
        "plan": dump_contract(planned.plan, ResearchPlan),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": "decide_action",
    }


def _decide_action(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    capacity = tracker.can_decide()
    if not capacity.accepted:
        return _terminal_update(
            state,
            tracker=tracker,
            status=_partial_or_no_evidence(registry),
            reason=StopReason.BUDGET_EXHAUSTED,
        )
    query_context = load_contract(state["query_context"], QueryContext)
    plan = load_contract(state["plan"], ResearchPlan)
    coverage = _evaluate_coverage(
        state,
        settings=runtime.context.settings,
        context=query_context,
        registry=registry,
    )
    if tracker.no_available_actions(query_context):
        return _terminal_update(
            state,
            tracker=tracker,
            status=_partial_or_no_evidence(registry),
            reason=StopReason.NO_AVAILABLE_ACTIONS,
        )
    try:
        decision = runtime.context.decision_agent.decide(
            DecisionState(
                context=query_context,
                plan=plan,
                evidence=registry.records,
                observations=_observations(state),
                budget_usage=tracker.usage(),
                suppressed_paper_ids=tuple(sorted(tracker.suppressed_papers)),
                coverage=coverage,
            )
        )
    except DecisionError as error:
        capacity = tracker.record_decision(error.llm_calls)
        traces = _append_trace(
            state,
            ActionTrace(tracker.iterations, "invalid_decision", False, str(error)),
        )
        if not capacity.accepted:
            return _terminal_update(
                state,
                tracker=tracker,
                traces=traces,
                status=(
                    ResearchRunStatus.PARTIAL
                    if registry.records
                    else ResearchRunStatus.FAILED
                ),
                reason=StopReason.BUDGET_EXHAUSTED,
            )
        if tracker.record_invalid_action():
            return _terminal_update(
                state,
                tracker=tracker,
                traces=traces,
                status=(
                    ResearchRunStatus.PARTIAL
                    if registry.records
                    else ResearchRunStatus.FAILED
                ),
                reason=StopReason.INVALID_ACTION_LIMIT,
            )
        return {
            "guard_state": dump_contract(tracker.snapshot(), GuardState),
            "traces": traces,
            "next_node": "decide_action",
        }
    capacity = tracker.record_decision(decision.llm_calls)
    if not capacity.accepted:
        traces = _append_trace(
            state,
            ActionTrace(
                tracker.iterations,
                decision.action.action_name,
                False,
                capacity.detail,
            ),
        )
        return _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=_partial_or_no_evidence(registry),
            reason=StopReason.BUDGET_EXHAUSTED,
        )
    return {
        "action": dump_contract(decision.action, AgentAction),
        "qualifications": dump_contract(
            decision.qualifications, tuple[EvidenceQualification, ...]
        ),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": "qualify_and_assess",
    }


def _qualify_and_assess(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    action = load_contract(state["action"], AgentAction)
    qualifications = load_contract(
        state["qualifications"], tuple[EvidenceQualification, ...]
    )
    query_context = load_contract(state["query_context"], QueryContext)
    plan = load_contract(state["plan"], ResearchPlan)
    try:
        registry.apply_qualifications(qualifications)
    except ValueError as error:
        traces = _append_trace(
            state,
            ActionTrace(
                tracker.iterations,
                action.action_name,
                False,
                f"InvalidQualification: {error}",
            ),
        )
        if tracker.record_invalid_action():
            return _terminal_update(
                state,
                tracker=tracker,
                traces=traces,
                status=(ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.FAILED),
                reason=StopReason.INVALID_ACTION_LIMIT,
            )
        return {
            "guard_state": dump_contract(tracker.snapshot(), GuardState),
            "traces": traces,
            "next_node": "decide_action",
        }
    coverage = _evaluate_coverage(
        state,
        settings=runtime.context.settings,
        context=query_context,
        registry=registry,
    )
    if isinstance(action, FinishAction):
        if not coverage.coverage_complete:
            traces = _append_trace(
                state,
                ActionTrace(
                    tracker.iterations,
                    action.action_name,
                    False,
                    "CoverageIncomplete: " + ", ".join(coverage.remaining_paper_ids),
                ),
            )
            if tracker.record_invalid_action():
                return _terminal_update(
                    state,
                    tracker=tracker,
                    traces=traces,
                    status=_partial_or_no_evidence(registry),
                    reason=StopReason.INVALID_ACTION_LIMIT,
                )
            return {
                "registry_state": dump_contract(registry.snapshot(), RegistryState),
                "coverage": dump_contract(coverage, CoverageState),
                "guard_state": dump_contract(tracker.snapshot(), GuardState),
                "traces": traces,
                "next_node": "decide_action",
            }
        traces = _append_trace(
            state,
            ActionTrace(
                tracker.iterations,
                action.action_name,
                True,
                action.reason,
            ),
        )
        selected = FinalEvidenceSelector(
            max_units=runtime.context.settings.final_evidence_max_units,
            max_per_paper=runtime.context.settings.final_evidence_max_per_paper,
        ).select(plan=plan, coverage=coverage, evidence=registry.records)
        update = _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            plan=finish_plan(plan, bool(selected)),
            status=(
                ResearchRunStatus.COMPLETED
                if selected
                else ResearchRunStatus.NO_EVIDENCE
            ),
            reason=StopReason.AGENT_FINISHED,
        )
        update["registry_state"] = dump_contract(registry.snapshot(), RegistryState)
        update["coverage"] = dump_contract(coverage, CoverageState)
        return update
    return {
        "registry_state": dump_contract(registry.snapshot(), RegistryState),
        "coverage": dump_contract(coverage, CoverageState),
        "next_node": "validate_action",
    }


def _validate_action(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    action = load_contract(state["action"], AgentAction)
    guard = tracker.validate_action(
        action,
        context=load_contract(state["query_context"], QueryContext),
        plan=load_contract(state["plan"], ResearchPlan),
        snapshot=(
            load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
            if state.get("corpus_snapshot") is not None
            else None
        ),
    )
    if guard.accepted:
        return {
            "guard_state": dump_contract(tracker.snapshot(), GuardState),
            "action_fingerprint": action_fingerprint(action),
            "next_node": "execute_tool",
        }
    traces = _append_trace(
        state,
        ActionTrace(
            tracker.iterations,
            action.action_name,
            False,
            f"{guard.code}: {guard.detail}",
        ),
    )
    if guard.budget_exhausted:
        return _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=_partial_or_no_evidence(registry),
            reason=StopReason.BUDGET_EXHAUSTED,
        )
    if tracker.record_invalid_action():
        return _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=(
                ResearchRunStatus.PARTIAL
                if registry.records
                else ResearchRunStatus.FAILED
            ),
            reason=StopReason.INVALID_ACTION_LIMIT,
        )
    return {
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "traces": traces,
        "next_node": "decide_action",
    }


def _execute_tool(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    action = load_contract(state["action"], AgentAction)
    try:
        result = runtime.context.tool_executor.execute(
            action,
            context=load_contract(state["query_context"], QueryContext),
            corpus_snapshot=(
                load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
                if state.get("corpus_snapshot") is not None
                else None
            ),
        )
    except Exception as error:
        traces = _append_trace(
            state,
            ActionTrace(
                tracker.iterations,
                action.action_name,
                False,
                f"{type(error).__name__}: {error}",
            ),
        )
        return _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=(
                ResearchRunStatus.PARTIAL
                if registry.records
                else ResearchRunStatus.FAILED
            ),
            reason=StopReason.FATAL_ERROR,
            error_code=type(error).__name__,
        )
    tracker.record_tool_call(action)
    return {
        "tool_result": dump_contract(result, ToolResult),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": "register_and_observe",
    }


def _register_and_observe(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    registry = _registry(state)
    tracker = _tracker(state, runtime.context.settings)
    action = load_contract(state["action"], AgentAction)
    tool_result = load_contract(state["tool_result"], ToolResult)
    delta = registry.register_tool_result(
        tool_result,
        iteration=tracker.iterations,
        target_subtask_ids=action_subtask_ids(action),
    )
    added_paper_ids = {
        record.evidence.paper_id
        for evidence_id in delta.added_ids
        if (record := registry.get(evidence_id)) is not None
    }
    tracker.record_evidence_delta(
        action, added_paper_ids=added_paper_ids
    )
    observation = build_observation(
        iteration=tracker.iterations,
        result=tool_result,
        delta=delta,
        remaining_budget=tracker.remaining(),
    )
    traces = _append_trace(
        state,
        ActionTrace(
            tracker.iterations,
            action.action_name,
            True,
            f"added={len(delta.added_ids)}, reused={len(delta.reused_ids)}",
        ),
    )
    return {
        "registry_state": dump_contract(registry.snapshot(), RegistryState),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "observations": dump_contract(
            (*_observations(state), observation),
            tuple[ResearchObservation, ...],
        ),
        "traces": traces,
        "plan": dump_contract(
            mark_subtasks_started(
                load_contract(state["plan"], ResearchPlan),
                action_subtask_ids(action),
            ),
            ResearchPlan,
        ),
        "next_node": "decide_action",
    }


def _finish_run(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    registry_state = _registry_snapshot(state)
    guard_state = _guard_snapshot(state)
    result = ResearchRunResult(
        status=ResearchRunStatus(
            state.get("terminal_status", ResearchRunStatus.FAILED.value)
        ),
        stop_reason=StopReason(
            state.get("stop_reason", StopReason.FATAL_ERROR.value)
        ),
        query_context=(
            load_contract(state["query_context"], QueryContext)
            if state.get("query_context") is not None
            else None
        ),
        plan=(
            load_contract(state["plan"], ResearchPlan)
            if state.get("plan") is not None
            else None
        ),
        evidence=registry_state.records,
        observations=_observations(state),
        traces=_traces(state),
        budget_usage=guard_state.usage,
        coverage=(
            _evaluate_coverage(
                state,
                settings=runtime.context.settings,
                context=load_contract(state["query_context"], QueryContext),
                registry=EvidenceRegistry(registry_state),
            )
            if state.get("query_context") is not None
            else None
        ),
        selected_evidence=(),
        clarification=state.get("clarification"),
    )
    if result.coverage is not None and result.plan is not None:
        result = replace(
            result,
            selected_evidence=FinalEvidenceSelector(
                max_units=runtime.context.settings.final_evidence_max_units,
                max_per_paper=runtime.context.settings.final_evidence_max_per_paper,
            ).select(
                plan=result.plan,
                coverage=result.coverage,
                evidence=result.evidence,
            ),
        )
    return {"result": dump_contract(result, ResearchRunResult)}


def _route(state: AgentGraphState) -> NextNode:
    return state["next_node"]


def _tracker(
    state: AgentGraphState, settings: AgenticResearchSettings
) -> BudgetAndGuardTracker:
    return BudgetAndGuardTracker(settings, state=_guard_snapshot(state))


def _append_trace(
    state: AgentGraphState, trace: ActionTrace
) -> CheckpointValue:
    return dump_contract((*_traces(state), trace), tuple[ActionTrace, ...])


def _partial_or_no_evidence(registry: EvidenceRegistry) -> ResearchRunStatus:
    return (
        ResearchRunStatus.PARTIAL
        if registry.records
        else ResearchRunStatus.NO_EVIDENCE
    )


def _terminal_update(
    state: AgentGraphState,
    *,
    tracker: BudgetAndGuardTracker,
    status: ResearchRunStatus,
    reason: StopReason,
    traces: CheckpointValue | None = None,
    plan: ResearchPlan | None = None,
    error_code: str | None = None,
) -> AgentGraphState:
    update: AgentGraphState = {
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "terminal_status": status.value,
        "stop_reason": reason.value,
        "next_node": "finish_run",
    }
    if traces is not None:
        update["traces"] = traces
    if plan is not None:
        update["plan"] = dump_contract(plan, ResearchPlan)
    if error_code:
        update["error_codes"] = list(
            dict.fromkeys((*state.get("error_codes", []), error_code))
        )
    return update


def _fatal_update(
    state: AgentGraphState, node_name: str, error: Exception
) -> AgentGraphState:
    guard_state = _guard_snapshot(state)
    trace = ActionTrace(
        guard_state.usage.agent_iterations,
        node_name,
        False,
        f"{type(error).__name__}: {error}",
    )
    return {
        "traces": _append_trace(state, trace),
        "terminal_status": ResearchRunStatus.FAILED.value,
        "stop_reason": StopReason.FATAL_ERROR.value,
        "error_codes": list(
            dict.fromkeys((*state.get("error_codes", []), type(error).__name__))
        ),
        "next_node": "finish_run",
    }


def _thread_config(
    thread_id: str, *, recursion_limit: int | None = None
) -> dict[str, object]:
    normalized = " ".join(thread_id.split())
    if not normalized:
        raise ValueError("thread_id must be a non-empty string.")
    config: dict[str, object] = {"configurable": {"thread_id": normalized}}
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def _recursion_limit(settings: AgenticResearchSettings) -> int:
    # 每轮最多经过 decide/qualify/validate/execute/register 五个节点，再预留准备与收尾。
    return max(25, settings.max_agent_iterations * 5 + 10)


def _registry_snapshot(state: AgentGraphState) -> RegistryState:
    raw = state.get("registry_state")
    return (
        load_contract(raw, RegistryState)
        if raw is not None
        else RegistryState()
    )


def _registry(state: AgentGraphState) -> EvidenceRegistry:
    return EvidenceRegistry(_registry_snapshot(state))


def _guard_snapshot(state: AgentGraphState) -> GuardState:
    raw = state.get("guard_state")
    return load_contract(raw, GuardState) if raw is not None else GuardState()


def _observations(state: AgentGraphState) -> tuple[ResearchObservation, ...]:
    raw = state.get("observations")
    return (
        load_contract(raw, tuple[ResearchObservation, ...])
        if raw is not None
        else ()
    )


def _evaluate_coverage(
    state: AgentGraphState,
    *,
    settings: AgenticResearchSettings,
    context: QueryContext,
    registry: EvidenceRegistry,
) -> CoverageState:
    return CoverageManager(
        CoveragePolicy(
            candidate_min_queries_per_paper=(
                settings.coverage_candidate_min_queries_per_paper
            ),
            exhaustive_min_queries_per_paper=(
                settings.coverage_exhaustive_min_queries_per_paper
            ),
        )
    ).evaluate(
        context=context,
        corpus_snapshot=(
            load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
            if state.get("corpus_snapshot") is not None
            else None
        ),
        evidence=registry.records,
        observations=_observations(state),
    )


def _traces(state: AgentGraphState) -> tuple[ActionTrace, ...]:
    raw = state.get("traces")
    return (
        load_contract(raw, tuple[ActionTrace, ...])
        if raw is not None
        else ()
    )
