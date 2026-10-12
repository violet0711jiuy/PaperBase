"""Phase 2.5 LangGraph workflow；框架只负责编排，业务规则仍由领域组件实现。"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from collections.abc import Iterator
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import ToolResult, ToolStatus

from .contracts import (
    ActionTrace,
    AgentAction,
    ActionCanonicalization,
    CandidateSelection,
    CorpusScopeSnapshot,
    CoverageState,
    EvidenceFilterReport,
    EvidenceRecord,
    GroundedAnswerAttempt,
    TryAnswerAction,
    StopResearchAction,
    ActionRejectionContext,
    GuardState,
    LLMCallRecord,
    QueryContext,
    QueryScope,
    RegistryState,
    ResearchAnswer,
    ResearchObservation,
    ResearchPlan,
    RequirementState,
    RequirementStatus,
    ResearchRunResult,
    ResearchRunStatus,
    ScopeResolution,
    ScopeResolutionStatus,
    StopReason,
)
from .grounded_answer import (
    GroundedAnswerAttemptsExhausted,
    GroundedAnswerer,
    render_grounded_answer,
)
from .candidate_selection import CandidateSelector
from .checkpoint_serde import (
    CheckpointValue,
    checkpoint_thread_id,
    dump_contract,
    load_contract,
)
from .coverage import CoverageManager
from .decision import DecisionAgent, DecisionError, DecisionState
from .evidence_admission import EvidenceAdmissionFilter, fail_open_admission
from .evidence_registry import EvidenceRegistry
from .events import ResearchProgressEvent, progress_event_for_node
from .guards import (
    BudgetAndGuardTracker,
    affected_paper_ids,
    action_requirement_ids,
    queries_for_paper,
)
from .observation import _action_arguments, build_observation
from .planner import ResearchPlanner
from .query_understanding import QueryUnderstandingService
from .requirement_state import derive_requirement_states
from .tools import ToolExecutor
from .search_requests import compile_action_requests
from paperbase.retrieval.requests import CanonicalSearchRequest, request_trace
from .scope_resolver import ScopeResolver, TrustedWorkspaceScope


NextNode = Literal[
    "understand_query",
    "select_candidates",
    "build_plan",
    "decide_action",
    "assess_action",
    "validate_action",
    "execute_tool",
    "filter_evidence",
    "register_and_observe",
    "generate_answer",
    "finish_run",
]


class AgentGraphState(TypedDict, total=False):
    semantic_authority_version: str
    """Checkpoint 中只保存领域 DTO；所有服务和连接通过 Runtime Context 注入。"""

    thread_id: str
    original_query: str
    trusted_workspace: CheckpointValue
    selected_paper_ids: list[str]
    knowledge_base_all: bool
    scope_resolution: CheckpointValue
    corpus_snapshot: CheckpointValue
    query_context: CheckpointValue
    candidate_selection: CheckpointValue
    plan: CheckpointValue
    requirement_states: CheckpointValue
    action: CheckpointValue
    canonical_search_requests: CheckpointValue
    action_canonicalization: CheckpointValue
    tool_result: CheckpointValue
    evidence_filter_report: CheckpointValue
    registry_state: CheckpointValue
    guard_state: CheckpointValue
    # New checkpoints use these names.  The helpers below read Phase A.1
    # aliases as a one-way compatibility fallback.
    last_action_rejection: CheckpointValue
    action_rejections: CheckpointValue
    observations: CheckpointValue
    traces: CheckpointValue
    terminal_status: str
    stop_reason: str
    clarification: str | None
    error_codes: list[str]
    selected_evidence: CheckpointValue
    grounded_answer_attempt: CheckpointValue
    answer_evidence_ids: list[str]
    answer_attempt_count: int
    answer_mode: str
    answer: CheckpointValue
    next_node: NextNode
    result: CheckpointValue
    llm_diagnostics: CheckpointValue


_SEMANTIC_AUTHORITY_VERSION = "raw-evidence-grounded-answer-v1"


def _require_current_semantic_authority(values: dict[str, object]) -> None:
    if values.get("semantic_authority_version") != _SEMANTIC_AUTHORITY_VERSION:
        raise ValueError(
            "Incompatible checkpoint semantic authority: "
            f"{values.get('semantic_authority_version')!r}; expected "
            f"{_SEMANTIC_AUTHORITY_VERSION!r}. Grounded Answer state "
            "cannot safely reuse legacy Projector checkpoints; start a new research thread "
            "instead of resuming it."
        )


@dataclass(frozen=True)
class AgentGraphContext:
    scope_resolver: ScopeResolver
    query_understanding: QueryUnderstandingService
    planner: ResearchPlanner
    decision_agent: DecisionAgent
    candidate_selector: CandidateSelector | None
    evidence_filter: EvidenceAdmissionFilter
    tool_executor: ToolExecutor
    grounded_answerer: GroundedAnswerer | None
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
        evidence_filter: EvidenceAdmissionFilter | None = None,
        tool_executor: ToolExecutor,
        settings: AgenticResearchSettings,
        candidate_selector: CandidateSelector | None = None,
        grounded_answerer: GroundedAnswerer | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._context = AgentGraphContext(
            scope_resolver=scope_resolver,
            query_understanding=query_understanding,
            planner=planner,
            decision_agent=decision_agent,
            candidate_selector=candidate_selector,
            evidence_filter=evidence_filter or EvidenceAdmissionFilter(
                enabled=settings.evidence_admission_enabled,
                min_rerank_score=settings.evidence_min_rerank_score,
                source_overlap_threshold=settings.evidence_source_overlap_threshold,
            ),
            tool_executor=tool_executor,
            grounded_answerer=grounded_answerer,
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
        resolved_thread_id, config, initial_state = self._prepare_start(
            query,
            trusted_workspace=trusted_workspace,
            selected_paper_ids=selected_paper_ids,
            knowledge_base_all=knowledge_base_all,
            thread_id=thread_id,
        )
        self._graph.invoke(
            initial_state,
            config=config,
            context=self._context,
            interrupt_after=interrupt_after or None,
        )
        return self.inspect(resolved_thread_id)

    def stream(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
        thread_id: str | None = None,
        interrupt_after: tuple[str, ...] = (),
    ) -> Iterator[ResearchProgressEvent]:
        """以稳定业务事件流执行新任务，隐藏 LangGraph 的原始状态更新。"""
        resolved_thread_id, config, initial_state = self._prepare_start(
            query,
            trusted_workspace=trusted_workspace,
            selected_paper_ids=selected_paper_ids,
            knowledge_base_all=knowledge_base_all,
            thread_id=thread_id,
        )
        yield ResearchProgressEvent(
            kind="started",
            stage="start",
            message="研究任务已开始",
            thread_id=resolved_thread_id,
        )
        yield from self._stream_graph(
            initial_state,
            config=config,
            thread_id=resolved_thread_id,
            interrupt_after=interrupt_after,
        )

    def resume_stream(
        self,
        thread_id: str,
        *,
        interrupt_after: tuple[str, ...] = (),
    ) -> Iterator[ResearchProgressEvent]:
        """从已持久化 checkpoint 继续任务，并继续输出业务进度事件。"""
        config = _thread_config(
            thread_id,
            recursion_limit=_recursion_limit(self._context.settings),
        )
        existing = self._graph.get_state(config)
        if not existing.values:
            raise LookupError(f"Unknown research thread: {thread_id}")
        _require_current_semantic_authority(dict(existing.values))
        if not existing.next:
            execution = self.inspect(thread_id)
            yield ResearchProgressEvent(
                kind="completed",
                stage="completed",
                message="研究任务已经完成",
                thread_id=thread_id,
                result=execution.result,
            )
            return
        yield ResearchProgressEvent(
            kind="started",
            stage="resume",
            message="正在恢复研究任务",
            thread_id=thread_id,
        )
        yield from self._stream_graph(
            None,
            config=config,
            thread_id=thread_id,
            interrupt_after=interrupt_after,
        )

    def resume(
        self,
        thread_id: str,
        *,
        interrupt_after: tuple[str, ...] = (),
    ) -> GraphExecution:
        existing = self._graph.get_state(_thread_config(thread_id))
        if not existing.values:
            raise LookupError(f"Unknown research thread: {thread_id}")
        _require_current_semantic_authority(dict(existing.values))
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

    def _prepare_start(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None,
        selected_paper_ids: tuple[str, ...],
        knowledge_base_all: bool,
        thread_id: str | None,
    ) -> tuple[str, dict[str, object], AgentGraphState]:
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
            "semantic_authority_version": _SEMANTIC_AUTHORITY_VERSION,
            "thread_id": resolved_thread_id,
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
            "llm_diagnostics": dump_contract((), tuple[LLMCallRecord, ...]),
            "error_codes": [],
        }
        return resolved_thread_id, config, initial_state

    def _stream_graph(
        self,
        input_state: AgentGraphState | None,
        *,
        config: dict[str, object],
        thread_id: str,
        interrupt_after: tuple[str, ...],
    ) -> Iterator[ResearchProgressEvent]:
        for part in self._graph.stream(
            input_state,
            config=config,
            context=self._context,
            stream_mode="updates",
            version="v2",
            interrupt_after=interrupt_after or None,
        ):
            for node in _stream_update_nodes(part):
                yield progress_event_for_node(node=node, thread_id=thread_id)
        execution = self.inspect(thread_id)
        if execution.interrupted:
            yield ResearchProgressEvent(
                kind="interrupted",
                stage="interrupted",
                message="研究任务已暂停，可从当前检查点继续",
                thread_id=thread_id,
            )
        else:
            yield ResearchProgressEvent(
                kind="completed",
                stage="completed",
                message="研究任务已完成",
                thread_id=thread_id,
                result=execution.result,
            )


def build_agentic_research_graph() -> StateGraph:
    graph = StateGraph(AgentGraphState, context_schema=AgentGraphContext)
    graph.add_node("resolve_scope", _resolve_scope)
    graph.add_node("understand_query", _understand_query)
    graph.add_node("select_candidates", _select_candidates)
    graph.add_node("build_plan", _build_plan)
    graph.add_node("decide_action", _decide_action)
    graph.add_node("assess_action", _assess_action)
    graph.add_node("validate_action", _validate_action)
    graph.add_node("execute_tool", _execute_tool)
    graph.add_node("filter_evidence", _filter_evidence)
    graph.add_node("register_and_observe", _register_and_observe)
    graph.add_node("generate_answer", _generate_answer)
    graph.add_node("finish_run", _finish_run)

    graph.add_edge(START, "resolve_scope")
    graph.add_conditional_edges(
        "resolve_scope",
        _route,
        {"understand_query": "understand_query", "generate_answer": "generate_answer"},
    )
    graph.add_conditional_edges(
        "understand_query",
        _route,
        {
            "select_candidates": "select_candidates",
            "build_plan": "build_plan",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "select_candidates",
        _route,
        {"build_plan": "build_plan", "generate_answer": "generate_answer"},
    )
    graph.add_conditional_edges(
        "build_plan",
        _route,
        {"decide_action": "decide_action", "generate_answer": "generate_answer"},
    )
    graph.add_conditional_edges(
        "decide_action",
        _route,
        {
            "decide_action": "decide_action",
            "assess_action": "assess_action",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "assess_action",
        _route,
        {
            "decide_action": "decide_action",
            "validate_action": "validate_action",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "register_and_observe",
        _route,
        {"decide_action": "decide_action", "generate_answer": "generate_answer"},
    )
    graph.add_conditional_edges(
        "validate_action",
        _route,
        {
            "decide_action": "decide_action",
            "execute_tool": "execute_tool",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        _route,
        {
            "filter_evidence": "filter_evidence",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("filter_evidence", "register_and_observe")
    graph.add_conditional_edges(
        "generate_answer",
        _route,
        {"decide_action": "decide_action", "finish_run": "finish_run"},
    )
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
            "next_node": "generate_answer",
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
        "llm_diagnostics": _append_llm_diagnostics(
            state, understood.llm_diagnostics
        ),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": (
            "select_candidates"
            if understood.context.scope is QueryScope.KNOWLEDGE_BASE
            else "build_plan"
        ),
    }


def _select_candidates(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    """全库问题必须先完成元数据筛选，再允许进入正文检索规划。"""
    selector = runtime.context.candidate_selector
    if selector is None:
        return _fatal_update(
            state,
            "select_candidates",
            RuntimeError("Knowledge-base workflow requires CandidateSelector."),
        )
    snapshot = load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
    context = load_contract(state["query_context"], QueryContext)
    try:
        result = selector.select(question=context.resolved_query, snapshot=snapshot)
    except Exception as error:
        return _fatal_update(state, "select_candidates", error)
    tracker = _tracker(state, runtime.context.settings)
    tracker.record_preparation_llm_calls(result.llm_calls)
    return {
        "candidate_selection": dump_contract(result.selection, CandidateSelection),
        "llm_diagnostics": _append_llm_diagnostics(
            state, result.llm_diagnostics
        ),
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
        "requirement_states": dump_contract(
            derive_requirement_states(
                plan=planned.plan,
                evidence=(),
                observations=(),
            ),
            tuple[RequirementState, ...],
        ),
        "llm_diagnostics": _append_llm_diagnostics(state, planned.llm_diagnostics),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": "decide_action",
    }


def _decide_action(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    registry = _registry(state)
    # Decision receives Evidence metadata, not Evidence bodies.  Route every
    # unseen Registry snapshot through the semantic authority before asking
    # Decision to choose another retrieval action.  This also protects a
    # checkpoint resumed directly at decide_action.
    if _has_unassessed_evidence(state, registry):
        return {"answer_mode": "", "next_node": "generate_answer"}
    tracker = _tracker(state, runtime.context.settings)
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
        context=query_context,
        registry=registry,
    )
    candidate_selection = _candidate_selection(state)
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
                candidate_selection=candidate_selection,
                requirement_states=_state_requirement_states(state),
                grounded_answer=(
                    load_contract(state["grounded_answer_attempt"], GroundedAnswerAttempt)
                    if state.get("grounded_answer_attempt") is not None else None
                ),
                answer_evidence_ids=tuple(state.get("answer_evidence_ids", ())),
                answer_attempt_count=int(state.get("answer_attempt_count", 0)),
                last_action_rejection=_last_action_rejection(state),
                action_rejections=_action_rejections(state),
                per_paper_query_limit=runtime.context.settings.max_queries_per_paper,
                max_agent_iterations=runtime.context.settings.max_agent_iterations,
                guard_preflight=lambda action: _assess_candidate_with_guard(
                    state, runtime, action, _tracker(state, runtime.context.settings)
                ),
                overview_status_by_paper=_overview_statuses(
                    runtime.context.tool_executor,
                    query_context,
                    _profile_paper_ids_for_context(query_context, candidate_selection, registry),
                ),
            )
        )
    except DecisionError as error:
        recovery_update = _decision_recovery_update(state, error.recovery_rejections)
        capacity = tracker.record_decision(error.llm_calls)
        traces = _append_trace(
            state,
            ActionTrace(tracker.iterations, "invalid_decision", False, str(error)),
        )
        if not capacity.accepted:
            update = _terminal_update(
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
            update["llm_diagnostics"] = _append_llm_diagnostics(
                state, error.llm_diagnostics
            )
            update.update(recovery_update)
            return update
        if tracker.record_invalid_action():
            update = _terminal_update(
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
            update["llm_diagnostics"] = _append_llm_diagnostics(
                state, error.llm_diagnostics
            )
            update.update(recovery_update)
            return update
        return {
            **recovery_update,
            "guard_state": dump_contract(tracker.snapshot(), GuardState),
            "traces": traces,
            "llm_diagnostics": _append_llm_diagnostics(
                state, error.llm_diagnostics
            ),
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
        update = _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=_partial_or_no_evidence(registry),
            reason=StopReason.BUDGET_EXHAUSTED,
        )
        update["llm_diagnostics"] = _append_llm_diagnostics(
            state, decision.llm_diagnostics
        )
        return update
    update: AgentGraphState = {
        **_decision_recovery_update(state, decision.recovery_rejections),
        "action": dump_contract(decision.action, AgentAction),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "llm_diagnostics": _append_llm_diagnostics(
            state, decision.llm_diagnostics
        ),
        "next_node": "assess_action",
    }
    update["action_canonicalization"] = (
        dump_contract(decision.action_canonicalization, ActionCanonicalization)
        if decision.action_canonicalization is not None
        and decision.action_canonicalization.changed
        else None
    )
    return update


def _assess_action(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    action = load_contract(state["action"], AgentAction)
    query_context = load_contract(state["query_context"], QueryContext)
    plan = load_contract(state["plan"], ResearchPlan)
    coverage = _evaluate_coverage(
        state,
        context=query_context,
        registry=registry,
    )
    if isinstance(action, (TryAnswerAction, StopResearchAction)):
        evidence_ids = tuple(record.evidence_id for record in registry.records)
        if (
            isinstance(action, TryAnswerAction)
            and state.get("grounded_answer_attempt") is not None
            and tuple(state.get("answer_evidence_ids", ())) == evidence_ids
            and int(state.get("answer_attempt_count", 0)) >= 2
        ):
            rejection = ActionRejectionContext(
                action_name="try_answer",
                action_arguments={"reason": action.reason},
                requirement_ids=(),
                rejection_code="AnswerRetryBudgetExhausted",
                rejection_reason="No new Evidence was added and the one focused answer retry is exhausted.",
                guard_name="answer_attempt_guard",
                iteration=tracker.iterations,
                rejection_stage="answer_admission",
                validator_name="answer_attempt_guard",
                action_canonicalization=_action_canonicalization(state),
            )
            return {
                "last_action_rejection": dump_contract(rejection, ActionRejectionContext),
                "action_rejections": dump_contract(
                    (*_action_rejections(state), rejection), tuple[ActionRejectionContext, ...]
                ),
                "traces": _append_trace(
                    state, ActionTrace(tracker.iterations, action.action_name, False, rejection.rejection_code)
                ),
                "next_node": "decide_action",
            }
        return {
            "answer_mode": action.action_name,
            "traces": _append_trace(
                state, ActionTrace(tracker.iterations, action.action_name, True, action.reason)
            ),
            "last_action_rejection": None,
            "next_node": "generate_answer",
        }
    return {
        "next_node": "validate_action",
    }


def _decision_recovery_update(state, rejections):
    """Reuse existing current/history DTOs for preflight feedback, never budgets."""
    if not rejections:
        return {}
    return {
        "last_action_rejection": dump_contract(rejections[-1], ActionRejectionContext),
        "action_rejections": dump_contract((*_action_rejections(state), *rejections), tuple[ActionRejectionContext, ...]),
    }


def _assess_candidate_with_guard(state, runtime, action, tracker):
    """Shared admission wiring; callers choose disposable or authoritative tracker.

    No duplicate/domain rule lives here. Both preflight and formal Guard use the
    executor's actual canonical compiler and the existing Guard implementation.
    """
    registry = _registry(state)
    context = load_contract(state["query_context"], QueryContext)
    snapshot = load_contract(state["corpus_snapshot"], CorpusScopeSnapshot) if state.get("corpus_snapshot") is not None else None
    executor = runtime.context.tool_executor
    requests = (executor.compile_requests(action, context=context, corpus_snapshot=snapshot)
                if hasattr(executor, "compile_requests") else compile_action_requests(
                    action, context=context, snapshot=snapshot, top_k=runtime.context.settings.top_k_per_paper))
    guard = tracker.validate_action(
        action,
        canonical_requests=requests,
        context=load_contract(state["query_context"], QueryContext),
        plan=load_contract(state["plan"], ResearchPlan),
        snapshot=(
            load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
            if state.get("corpus_snapshot") is not None
            else None
        ),
        candidate_selection=_candidate_selection(state),
        requirement_states=(
            load_contract(
                state["requirement_states"], tuple[RequirementState, ...]
            )
            if state.get("requirement_states") is not None
            else ()
        ),
        known_section_pairs=frozenset(
            (record.evidence.paper_id, record.evidence.section_id)
            for record in registry.records
            if record.evidence.section_id
        ),
    )
    return guard, requests


def _validate_action(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    action = load_contract(state["action"], AgentAction)
    guard, requests = _assess_candidate_with_guard(state, runtime, action, tracker)
    if guard.accepted:
        return {
            "canonical_search_requests": dump_contract(requests, tuple[CanonicalSearchRequest, ...]),
            "guard_state": dump_contract(tracker.snapshot(), GuardState),
            "next_node": "execute_tool",
        }
    rejection = ActionRejectionContext(
        action_name=action.action_name,
        action_arguments=_action_arguments(action),
        requirement_ids=action_requirement_ids(action),
        rejection_code=guard.code,
        rejection_reason=guard.detail,
        guard_name=guard.code,
        iteration=tracker.iterations,
        rejection_stage=("budget" if guard.code == "PerPaperBudget" else "guard"),
        blockers=_budget_blockers(action, tracker, runtime.context.settings) if guard.code == "PerPaperBudget" else (),
        validator_name=("per_paper_query_budget" if guard.code == "PerPaperBudget" else "runtime_guard"),
        action_canonicalization=_action_canonicalization(state),
    )
    rejections = (*_action_rejections(state), rejection)
    traces = _append_trace(
        state,
        ActionTrace(
            tracker.iterations,
            action.action_name,
            False,
            f"{guard.code}: {guard.detail}",
        ),
    )
    # A per-paper ceiling is a local feasibility signal, not a run-level stop.
    # Feed it back through the same rejection loop so the Decision Agent can
    # choose another paper, tool, query strategy, or a valid finish.
    if guard.code != "PerPaperBudget" and guard.budget_exhausted:
        guard_diagnostic = _guard_rejection_record(tracker, can_retry=False)
        update = _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=_partial_or_no_evidence(registry),
            reason=(
                StopReason.PER_PAPER_QUERY_LIMIT
                if guard.code == "PerPaperBudget"
                else StopReason.BUDGET_EXHAUSTED
            ),
        )
        update["llm_diagnostics"] = _append_llm_diagnostics(
            state, (guard_diagnostic,)
        )
        update["last_action_rejection"] = dump_contract(rejection, ActionRejectionContext)
        update["action_rejections"] = dump_contract(rejections, tuple[ActionRejectionContext, ...])
        return update
    invalid_limit = tracker.record_invalid_action()
    guard_diagnostic = _guard_rejection_record(
        tracker, can_retry=not invalid_limit
    )
    if invalid_limit:
        update = _terminal_update(
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
        update["llm_diagnostics"] = _append_llm_diagnostics(
            state, (guard_diagnostic,)
        )
        update["last_action_rejection"] = dump_contract(rejection, ActionRejectionContext)
        update["action_rejections"] = dump_contract(rejections, tuple[ActionRejectionContext, ...])
        return update
    return {
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "traces": traces,
        "last_action_rejection": dump_contract(rejection, ActionRejectionContext),
        "action_rejections": dump_contract(rejections, tuple[ActionRejectionContext, ...]),
        "llm_diagnostics": _append_llm_diagnostics(
            state, (guard_diagnostic,)
        ),
        "next_node": "decide_action",
    }


def _execute_tool(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    tracker = _tracker(state, runtime.context.settings)
    registry = _registry(state)
    action = load_contract(state["action"], AgentAction)
    requests = (load_contract(state["canonical_search_requests"], tuple[CanonicalSearchRequest, ...])
                if state.get("canonical_search_requests") is not None else ())
    if state.get("canonical_search_requests") is None:
        # Pending legacy Actions may resume at this edge. Compile the current
        # Action, never reconstruct old executed identities from partial traces.
        context = load_contract(state["query_context"], QueryContext)
        snapshot = load_contract(state["corpus_snapshot"], CorpusScopeSnapshot) if state.get("corpus_snapshot") is not None else None
        executor = runtime.context.tool_executor
        requests = (executor.compile_requests(action, context=context, corpus_snapshot=snapshot)
                    if hasattr(executor, "compile_requests") else compile_action_requests(
                        action, context=context, snapshot=snapshot, top_k=runtime.context.settings.top_k_per_paper))
    try:
        result = runtime.context.tool_executor.execute(
            action,
            **({"canonical_requests": requests} if hasattr(runtime.context.tool_executor, "compile_requests") else {}),
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
        tracker.request_executions.extend({"request_identity": request.identity,
            "iteration": tracker.iterations, "state": "failed", "attempted": True,
            "executed": False, "failed": True, "successful": False,
            "detail": "executor raised without execution receipt; entry unconfirmed"} for request in requests)
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
    if requests and "canonical_effective_requests" not in result.metadata:
        # Compatibility executors: mark the observed Tool boundary, not guessed
        # Dense/FTS internals. Production executor always supplies its receipt.
        result = replace(result, metadata={**result.metadata,
            "request_contract_version": "search-v1", "raw_action_arguments": asdict(action),
            "canonical_effective_requests": request_trace(requests)})
    tracker.record_tool_call(action, canonical_requests=requests, result=result)
    update: AgentGraphState = {
        "tool_result": dump_contract(result, ToolResult),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "next_node": "filter_evidence",
    }
    if result.status in {ToolStatus.OK, ToolStatus.EMPTY, ToolStatus.PARTIAL}:
        # Legality is still decided by the original Guard. Only the current
        # recovery pointer is consumed; the rejection audit/history stays intact.
        update["last_action_rejection"] = None
    return update


def _filter_evidence(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    """在 Registry 分配 E#/REF# 前执行 Agentic 专用 Evidence 准入。"""
    tool_result = load_contract(state["tool_result"], ToolResult)
    try:
        admission = runtime.context.evidence_filter.apply(tool_result)
    except Exception as error:
        admission = fail_open_admission(
            tool_result, error_code=type(error).__name__
        )
    return {
        "tool_result": dump_contract(admission.tool_result, ToolResult),
        "evidence_filter_report": dump_contract(
            admission.report, EvidenceFilterReport
        ),
        "next_node": "register_and_observe",
    }


def _register_and_observe(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    registry = _registry(state)
    tracker = _tracker(state, runtime.context.settings)
    action = load_contract(state["action"], AgentAction)
    tool_result = load_contract(state["tool_result"], ToolResult)
    plan = load_contract(state["plan"], ResearchPlan)
    filter_report = (
        load_contract(state["evidence_filter_report"], EvidenceFilterReport)
        if state.get("evidence_filter_report") is not None
        else None
    )
    delta = registry.register_tool_result(
        tool_result,
        action=action,
        plan=plan,
        iteration=tracker.iterations,
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
        action=action,
        plan=plan,
        action_canonicalization=_action_canonicalization(state),
        evidence_filter_report=filter_report,
    )
    observation = replace(
        observation,
        requirement_progress_before=tuple(
            (item.requirement_id, item.status.value)
            for item in _state_requirement_states(state)
        ),
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
    updated_observations = (*_observations(state), observation)
    updated_coverage = CoverageManager().evaluate(
        context=load_contract(state["query_context"], QueryContext),
        corpus_snapshot=(
            load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
            if state.get("corpus_snapshot") is not None
            else None
        ),
        candidate_selection=_candidate_selection(state),
        evidence=registry.records,
        observations=updated_observations,
    )
    return {
        "tool_result": None,
        "evidence_filter_report": None,
        "registry_state": dump_contract(registry.snapshot(), RegistryState),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "observations": dump_contract(
            updated_observations,
            tuple[ResearchObservation, ...],
        ),
        "traces": traces,
        "requirement_states": dump_contract(
            derive_requirement_states(
                plan=load_contract(state["plan"], ResearchPlan),
                evidence=registry.records,
                observations=updated_observations,
                context=load_contract(state["query_context"], QueryContext),
                coverage=updated_coverage,
                previous_states=_state_requirement_states(state),
            ),
            tuple[RequirementState, ...],
        ),
        "next_node": (
            "generate_answer"
            if _has_unassessed_evidence(state, registry)
            else "decide_action"
        ),
    }


def _generate_answer(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    """Attempt a span-grounded answer; unresolved gaps return to Decision."""
    registry = _registry(state)
    tracker = _tracker(state, runtime.context.settings)
    status = ResearchRunStatus(state.get("terminal_status", ResearchRunStatus.FAILED.value))
    context = (load_contract(state["query_context"], QueryContext)
               if state.get("query_context") is not None else None)
    plan = (load_contract(state["plan"], ResearchPlan)
            if state.get("plan") is not None else None)
    if status is ResearchRunStatus.NEEDS_CLARIFICATION:
        answer = ResearchAnswer(
            status="needs_clarification",
            answer=state.get("clarification") or "请补充需要检索的论文范围。",
            insufficient_evidence=True,
        )
        return {"answer": dump_contract(answer, ResearchAnswer), "next_node": "finish_run"}
    if context is None or plan is None:
        answer = ResearchAnswer(
            status="failed",
            answer="研究流程未能完成；请稍后重试，或缩小问题与论文范围。",
            insufficient_evidence=True,
        )
        return {"answer": dump_contract(answer, ResearchAnswer), "next_node": "finish_run"}
    if runtime.context.grounded_answerer is None:
        answer = ResearchAnswer(
            status="disabled",
            answer="回答生成功能当前未启用；请查看已检索到的论文证据。",
            insufficient_evidence=True,
            coverage_note="系统已完成证据检索，但未配置最终回答生成器。",
        )
        return {"answer": dump_contract(answer, ResearchAnswer), "next_node": "finish_run"}

    previous = (load_contract(state["grounded_answer_attempt"], GroundedAnswerAttempt)
                if state.get("grounded_answer_attempt") is not None else None)
    current_evidence_ids = tuple(record.evidence_id for record in registry.records)
    previous_evidence_ids = tuple(state.get("answer_evidence_ids", ()))
    # A validated sibling is stable only while the semantic input snapshot is
    # unchanged.  Once new Evidence arrives, every required Requirement must
    # be reassessed: a formerly plausible "complete" answer may have hidden a
    # retrieval gap that the expanded pool now makes visible.  Requirement-
    # scoped repair inside LLMGroundedAnswerer still preserves valid siblings.
    same_evidence_snapshot = (
        previous is not None
        and previous_evidence_ids == current_evidence_ids
    )
    accepted = (
        tuple(item for item in previous.requirements if item.complete)
        if same_evidence_snapshot
        else ()
    )
    try:
        attempt = runtime.context.grounded_answerer.answer(
            question=context.resolved_query,
            plan=plan,
            evidence=registry.records,
            accepted_answers=accepted,
        )
        tracker.record_preparation_llm_calls(len(attempt.llm_diagnostics))
        answer = render_grounded_answer(attempt, plan)
    except Exception as error:
        diagnostics = tuple(getattr(error, "diagnostics", ()) or ())
        tracker.record_preparation_llm_calls(len(diagnostics))
        root_error = error.cause if isinstance(error, GroundedAnswerAttemptsExhausted) else error
        fallback = ResearchAnswer(
            status="fallback",
            answer="回答生成未完成；系统保留了已检索到的原始论文证据，供重新尝试。",
            insufficient_evidence=True,
            coverage_note="Grounded Answer 未通过结构或 source-span 引用校验。",
        )
        return {
            "answer": dump_contract(fallback, ResearchAnswer),
            "selected_evidence": dump_contract((), tuple[EvidenceRecord, ...]),
            "answer_evidence_ids": list(current_evidence_ids),
            "llm_diagnostics": _append_llm_diagnostics(state, diagnostics),
            "error_codes": list(dict.fromkeys(
                (*state.get("error_codes", []),
                 f"GroundedAnswer:{type(root_error).__name__}")
            )),
            "guard_state": dump_contract(tracker.snapshot(), GuardState),
            "terminal_status": _partial_or_no_evidence(registry).value,
            "stop_reason": StopReason.FATAL_ERROR.value,
            "next_node": "finish_run",
        }

    by_id = {record.evidence_id: record for record in registry.records}
    selected = tuple(by_id[eid] for eid in attempt.evidence_ids if eid in by_id)
    terminal_requested = (
        state.get("answer_mode") == "stop_research"
        or state.get("terminal_status") is not None
        or not tracker.can_decide().accepted
    )
    requirement_states = _states_from_grounded_answer(plan, attempt, terminal=terminal_requested)
    update: AgentGraphState = {
        "grounded_answer_attempt": dump_contract(attempt, GroundedAnswerAttempt),
        "answer_evidence_ids": list(current_evidence_ids),
        "answer_attempt_count": (
            int(state.get("answer_attempt_count", 0)) + 1
            if tuple(state.get("answer_evidence_ids", ()))
            == current_evidence_ids
            else 1
        ),
        "answer": dump_contract(answer, ResearchAnswer),
        "selected_evidence": dump_contract(selected, tuple[EvidenceRecord, ...]),
        "requirement_states": dump_contract(requirement_states, tuple[RequirementState, ...]),
        "llm_diagnostics": _append_llm_diagnostics(state, attempt.llm_diagnostics),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "answer_mode": "",
    }
    if attempt.complete:
        update.update(
            terminal_status=ResearchRunStatus.COMPLETED.value,
            stop_reason=StopReason.AGENT_FINISHED.value,
            next_node="finish_run",
        )
    elif terminal_requested:
        update.update(
            terminal_status=(ResearchRunStatus.PARTIAL.value
                             if attempt.has_grounded_answer else ResearchRunStatus.NO_EVIDENCE.value),
            stop_reason=state.get("stop_reason", StopReason.AGENT_FINISHED.value),
            next_node="finish_run",
        )
    else:
        update["next_node"] = "decide_action"
    return update


def _states_from_grounded_answer(
    plan: ResearchPlan, attempt: GroundedAnswerAttempt, *, terminal: bool,
) -> tuple[RequirementState, ...]:
    by_id = {item.requirement_id: item for item in attempt.requirements}
    states = []
    for requirement in plan.requirements:
        answer = by_id.get(requirement.requirement_id)
        evidence_ids = tuple(dict.fromkeys(
            citation.evidence_id for citation in answer.citations
        )) if answer else ()
        if answer is not None and answer.complete:
            status = RequirementStatus.SATISFIED
        elif terminal:
            status = RequirementStatus.UNCERTAIN if answer and answer.remaining_gap else RequirementStatus.NO_SUPPORT_FOUND
        else:
            status = RequirementStatus.IN_PROGRESS
        states.append(RequirementState(requirement.requirement_id, status, evidence_ids))
    return tuple(states)


def _finish_run(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    registry_state = _registry_snapshot(state)
    guard_state = _guard_snapshot(state)
    plan = (
        load_contract(state["plan"], ResearchPlan)
        if state.get("plan") is not None
        else None
    )
    observations = _observations(state)
    context = (
        load_contract(state["query_context"], QueryContext)
        if state.get("query_context") is not None
        else None
    )
    coverage = (
        _evaluate_coverage(state, context=context, registry=_registry(state))
        if context is not None
        else None
    )
    # A checkpoint created before Phase 1 has no requirement_states key.  The
    # Regenerate progress from the current plan and retrieval history when a
    # fresh Grounded Answer checkpoint has not stored it yet.
    requirement_states = (
        _state_requirement_states(state)
        if state.get("requirement_states") is not None
        else derive_requirement_states(
            plan=plan,
            evidence=registry_state.records,
            observations=observations,
            context=context,
            coverage=coverage,
        )
        if plan is not None
        else ()
    )
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
        plan=plan,
        evidence=registry_state.records,
        observations=observations,
        traces=_traces(state),
        budget_usage=guard_state.usage,
        coverage=coverage,
        candidate_selection=_candidate_selection(state),
        selected_evidence=(
            load_contract(state["selected_evidence"], tuple[EvidenceRecord, ...])
            if state.get("selected_evidence") is not None
            else ()
        ),
        answer=(
            load_contract(state["answer"], ResearchAnswer)
            if state.get("answer") is not None
            else None
        ),
        thread_id=state.get("thread_id"),
        error_codes=tuple(state.get("error_codes", [])),
        clarification=state.get("clarification"),
        llm_diagnostics=_llm_diagnostics(state),
        requirement_states=requirement_states,
        action_rejections=_action_rejections(state),
        answer_input_evidence_ids=tuple(state.get("answer_evidence_ids", ())),
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


def _llm_diagnostics(state: AgentGraphState) -> tuple[LLMCallRecord, ...]:
    raw = state.get("llm_diagnostics")
    return (
        load_contract(raw, tuple[LLMCallRecord, ...])
        if raw is not None
        else ()
    )


def _append_llm_diagnostics(
    state: AgentGraphState, records: tuple[LLMCallRecord, ...]
) -> CheckpointValue:
    return dump_contract(
        (*_llm_diagnostics(state), *records), tuple[LLMCallRecord, ...]
    )


def _guard_rejection_record(
    tracker: BudgetAndGuardTracker, *, can_retry: bool
) -> LLMCallRecord:
    """把确定性 Guard 拒绝与模型/Schema 失败区分开。"""
    return LLMCallRecord(
        node="validate_action",
        attempt=max(1, tracker.iterations),
        success=False,
        error_stage="guard_rejection",
        recovery_action="replan_action" if can_retry else None,
    )


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
        "next_node": "generate_answer",
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
        "next_node": "generate_answer",
    }


def _thread_config(
    thread_id: str, *, recursion_limit: int | None = None
) -> dict[str, object]:
    normalized = " ".join(thread_id.split())
    if not normalized:
        raise ValueError("thread_id must be a non-empty string.")
    config: dict[str, object] = {
        "configurable": {"thread_id": checkpoint_thread_id(normalized)}
    }
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def _recursion_limit(settings: AgenticResearchSettings) -> int:
    # 一个成功动作会经过 decide / assess / validate / execute / filter /
    # register / analyze 七个节点；另留出初始化、finish 校验和回答收尾。
    # 这里是 LangGraph 的技术步数上界，不改变 Agent 的业务 iteration 预算。
    return max(40, settings.max_agent_iterations * 8 + 12)


def _stream_update_nodes(part: Any) -> tuple[str, ...]:
    """兼容 LangGraph v2 StreamPart，并仅提取完成更新的节点名称。"""
    if not isinstance(part, dict) or part.get("type") != "updates":
        return ()
    data = part.get("data")
    if not isinstance(data, dict):
        return ()
    return tuple(str(node) for node in data if isinstance(node, str))


def _registry_snapshot(state: AgentGraphState) -> RegistryState:
    raw = state.get("registry_state")
    return (
        load_contract(raw, RegistryState)
        if raw is not None
        else RegistryState()
    )


def _registry(state: AgentGraphState) -> EvidenceRegistry:
    return EvidenceRegistry(_registry_snapshot(state))


def _has_unassessed_evidence(
    state: AgentGraphState, registry: EvidenceRegistry | None = None,
) -> bool:
    """Whether the Registry snapshot has not yet reached Grounded Answer."""
    active = registry or _registry(state)
    current_ids = tuple(record.evidence_id for record in active.records)
    return bool(current_ids) and current_ids != tuple(
        state.get("answer_evidence_ids", ())
    )


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


def _state_requirement_states(state: AgentGraphState) -> tuple[RequirementState, ...]:
    raw = state.get("requirement_states")
    return load_contract(raw, tuple[RequirementState, ...]) if raw is not None else ()


def _last_action_rejection(state: AgentGraphState) -> ActionRejectionContext | None:
    raw = state.get("last_action_rejection")
    # Phase A.1 checkpoint compatibility only; do not write this old key.
    if raw is None:
        raw = state.get("last_guard_rejection")
    return load_contract(raw, ActionRejectionContext) if raw is not None else None


def _action_rejections(state: AgentGraphState) -> tuple[ActionRejectionContext, ...]:
    raw = state.get("action_rejections")
    if raw is None:
        raw = state.get("guard_rejections")
    return load_contract(raw, tuple[ActionRejectionContext, ...]) if raw is not None else ()


def _action_canonicalization(state: AgentGraphState) -> ActionCanonicalization | None:
    raw = state.get("action_canonicalization")
    return load_contract(raw, ActionCanonicalization) if raw is not None else None


def _budget_blockers(
    action: AgentAction,
    tracker: BudgetAndGuardTracker,
    settings: AgenticResearchSettings,
) -> tuple[dict[str, object], ...]:
    usage = dict(tracker.usage().queries_per_paper)
    limit = settings.max_queries_per_paper
    return tuple(
        {
            "type": "per_paper_query_limit",
            "paper_id": paper_id,
            "used": usage.get(paper_id, 0),
            "requested": queries_for_paper(action, paper_id),
            "limit": limit,
        }
        for paper_id in affected_paper_ids(action)
        if usage.get(paper_id, 0) + queries_for_paper(action, paper_id) > limit
    )


def _evaluate_coverage(
    state: AgentGraphState,
    *,
    context: QueryContext,
    registry: EvidenceRegistry,
) -> CoverageState:
    return CoverageManager().evaluate(
        context=context,
        corpus_snapshot=(
            load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
            if state.get("corpus_snapshot") is not None
            else None
        ),
        evidence=registry.records,
        observations=_observations(state),
        candidate_selection=_candidate_selection(state),
    )


def _candidate_selection(state: AgentGraphState) -> CandidateSelection | None:
    raw = state.get("candidate_selection")
    return (
        load_contract(raw, CandidateSelection)
        if raw is not None
        else None
    )


def _profile_paper_ids_for_context(
    context: QueryContext,
    candidate_selection: CandidateSelection | None,
    registry: EvidenceRegistry,
) -> tuple[str, ...]:
    if candidate_selection is not None:
        return tuple(
            dict.fromkeys(
                (*candidate_selection.candidate_paper_ids, *candidate_selection.uncertain_paper_ids)
            )
        )
    return tuple(
        dict.fromkeys(
            (*context.target_paper_ids, *(record.evidence.paper_id for record in registry.records))
        )
    )


def _overview_statuses(
    tool_executor: ToolExecutor,
    context: QueryContext,
    paper_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Read optional tool capability metadata without coupling test executors."""
    reader = getattr(tool_executor, "overview_statuses", None)
    if callable(reader):
        try:
            values = reader(paper_ids, context=context)
            return tuple(
                (str(paper_id), str(status))
                for paper_id, status in values
                if str(paper_id) in paper_ids
            )
        except Exception:
            pass
    return tuple((paper_id, "unknown") for paper_id in paper_ids)


def _traces(state: AgentGraphState) -> tuple[ActionTrace, ...]:
    raw = state.get("traces")
    return (
        load_contract(raw, tuple[ActionTrace, ...])
        if raw is not None
        else ()
    )
