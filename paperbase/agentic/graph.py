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
from paperbase.generation.answer_coverage import compile_answer_requirement_coverage
from paperbase.retrieval.contracts import ToolResult, ToolStatus

from .contracts import (
    ActionTrace,
    AgentAction,
    ActionCanonicalization,
    CandidateSelection,
    CorpusScopeSnapshot,
    CoverageState,
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceFilterReport,
    EvidenceRecord,
    FinishAction,
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
    ResearchRunResult,
    ResearchRunStatus,
    ScopeResolution,
    ScopeResolutionStatus,
    StopReason,
)
from .answering import (
    AgenticAnswerGenerator, build_completion_view, generate_agentic_answer,
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
from .evidence_analysis import (
    EvidenceAnalyzer, EvidenceAnalysisRequest, EvidenceAnalysisResult,
    projection_requirement_scope, requires_analysis,
)
from .evidence_admission import EvidenceAdmissionFilter, fail_open_admission
from .evidence_registry import EvidenceRegistry
from .events import ResearchProgressEvent, progress_event_for_node
from .guards import (
    BudgetAndGuardTracker,
    affected_paper_ids,
    action_requirement_ids,
    queries_for_paper,
)
from .grounding import verified_support_ids
from .observation import _action_arguments, build_observation
from .planner import ResearchPlanner
from .query_understanding import QueryUnderstandingService
from .research_profile import goal_signal_items
from .requirement_state import (
    derive_requirement_states,
    finalize_requirement_states,
    required_requirements_researched,
)
from .tools import ToolExecutor
from .search_requests import compile_action_requests
from paperbase.retrieval.requests import CanonicalSearchRequest, request_trace
from .selector import FinalEvidenceSelector
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
    "analyze_evidence",
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
    pending_analysis_ids: list[str]
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
    answer: CheckpointValue
    next_node: NextNode
    result: CheckpointValue
    llm_diagnostics: CheckpointValue


_SEMANTIC_AUTHORITY_VERSION = "projector-requirement-directed-multi-projection-v4"


def _require_current_semantic_authority(values: dict[str, object]) -> None:
    if values.get("semantic_authority_version") != _SEMANTIC_AUTHORITY_VERSION:
        raise ValueError(
            "Incompatible checkpoint semantic authority: "
            f"{values.get('semantic_authority_version')!r}; expected "
            f"{_SEMANTIC_AUTHORITY_VERSION!r}. Evidence-local Requirement scope "
            "cannot safely reuse legacy support; start a new research thread "
            "instead of resuming it."
        )


@dataclass(frozen=True)
class AgentGraphContext:
    scope_resolver: ScopeResolver
    query_understanding: QueryUnderstandingService
    planner: ResearchPlanner
    decision_agent: DecisionAgent
    candidate_selector: CandidateSelector | None
    evidence_analyzer: EvidenceAnalyzer
    evidence_filter: EvidenceAdmissionFilter
    tool_executor: ToolExecutor
    answer_generator: AgenticAnswerGenerator | None
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
        evidence_analyzer: EvidenceAnalyzer,
        evidence_filter: EvidenceAdmissionFilter | None = None,
        tool_executor: ToolExecutor,
        settings: AgenticResearchSettings,
        candidate_selector: CandidateSelector | None = None,
        answer_generator: AgenticAnswerGenerator | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._context = AgentGraphContext(
            scope_resolver=scope_resolver,
            query_understanding=query_understanding,
            planner=planner,
            decision_agent=decision_agent,
            candidate_selector=candidate_selector,
            evidence_analyzer=evidence_analyzer,
            evidence_filter=evidence_filter or EvidenceAdmissionFilter(
                enabled=settings.evidence_admission_enabled,
                min_rerank_score=settings.evidence_min_rerank_score,
                source_overlap_threshold=settings.evidence_source_overlap_threshold,
            ),
            tool_executor=tool_executor,
            answer_generator=answer_generator,
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
    graph.add_node("analyze_evidence", _analyze_evidence)
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
    graph.add_edge("register_and_observe", "analyze_evidence")
    graph.add_conditional_edges(
        "analyze_evidence",
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
    graph.add_edge("generate_answer", "finish_run")
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
    if isinstance(action, FinishAction):
        progress_states = derive_requirement_states(
            plan=plan, evidence=registry.records, observations=_observations(state),
            context=query_context, coverage=coverage,
            previous_states=_state_requirement_states(state),
        )
        if not coverage.coverage_complete or not required_requirements_researched(
            plan, progress_states
        ):
            blockers = tuple(
                value
                for value, blocked in (
                    ({"type": "required_requirement_untouched"}, not required_requirements_researched(plan, progress_states)),
                    ({"type": "candidate_coverage_incomplete"}, not coverage.coverage_complete),
                )
                if blocked
            )
            rejection = ActionRejectionContext(
                action_name="finish", action_arguments={}, requirement_ids=(),
                rejection_code="FinishClosureIncomplete",
                rejection_reason="Runtime finish closure validation is incomplete.",
                guard_name="finish_closure", iteration=tracker.iterations,
            rejection_stage="finish_validation", blockers=blockers,
                validator_name="finish_closure_validation",
                action_canonicalization=_action_canonicalization(state),
            )
            rejections = (*_action_rejections(state), rejection)
            traces = _append_trace(
                state,
                ActionTrace(
                    tracker.iterations,
                    action.action_name,
                    False,
                    (
                        "RequirementResearchIncomplete"
                        if not required_requirements_researched(plan, progress_states)
                        else "CoverageIncomplete: "
                        + ", ".join(coverage.remaining_paper_ids)
                    ),
                ),
            )
            invalid_limit = tracker.record_invalid_action()
            guard_diagnostic = _guard_rejection_record(
                tracker, can_retry=not invalid_limit
            )
            if invalid_limit:
                update = _terminal_update(
                    state,
                    tracker=tracker,
                    traces=traces,
                    status=_partial_or_no_evidence(registry),
                    reason=StopReason.INVALID_ACTION_LIMIT,
                )
                update["llm_diagnostics"] = _append_llm_diagnostics(
                    state, (guard_diagnostic,)
                )
                update["last_action_rejection"] = dump_contract(rejection, ActionRejectionContext)
                update["action_rejections"] = dump_contract(rejections, tuple[ActionRejectionContext, ...])
                return update
            return {
                "registry_state": dump_contract(registry.snapshot(), RegistryState),
                "guard_state": dump_contract(tracker.snapshot(), GuardState),
                "traces": traces,
                "last_action_rejection": dump_contract(rejection, ActionRejectionContext),
                "action_rejections": dump_contract(rejections, tuple[ActionRejectionContext, ...]),
                "llm_diagnostics": _append_llm_diagnostics(
                    state, (guard_diagnostic,)
                ),
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
        finalized_states = finalize_requirement_states(
            plan=plan,
            evidence=registry.records,
            observations=_observations(state),
            previous_states=_state_requirement_states(state),
        ).states
        selected = FinalEvidenceSelector(
            max_units=runtime.context.settings.final_evidence_max_units,
        ).select(
            context=query_context,
            plan=plan,
            coverage=coverage,
            evidence=registry.records,
            requirement_states=finalized_states,
        )
        update = _terminal_update(
            state,
            tracker=tracker,
            traces=traces,
            status=(
                ResearchRunStatus.COMPLETED
                if selected
                else ResearchRunStatus.NO_EVIDENCE
            ),
            reason=StopReason.AGENT_FINISHED,
        )
        update["registry_state"] = dump_contract(registry.snapshot(), RegistryState)
        update["requirement_states"] = dump_contract(finalized_states, tuple[RequirementState, ...])
        update["last_action_rejection"] = None
        return update
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
        "pending_analysis_ids": list(delta.analysis_required_ids),
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
        "next_node": "analyze_evidence",
    }


def _analyze_evidence(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    """Analyze/reanalyze Registry-routed Evidence; preserve verified support on failure."""
    registry = _registry(state)
    pending_ids = tuple(state.get("pending_analysis_ids", []))
    plan = load_contract(state["plan"], ResearchPlan)
    pending = tuple(
        record
        for evidence_id in pending_ids
        if (record := registry.get(evidence_id)) is not None
        and requires_analysis(record, plan)
    )
    if not pending:
        return {"pending_analysis_ids": [], "next_node": "decide_action"}
    tracker = _tracker(state, runtime.context.settings)
    query_context = load_contract(state["query_context"], QueryContext)
    previous_goal_items = goal_signal_items(registry.records)
    previous_support = {record.evidence_id: set(verified_support_ids(record, plan.requirements))
                        for record in pending}
    scoped = tuple(record for record in pending if projection_requirement_scope(record, plan))
    unbound_failures = tuple(EvidenceAnalysis(record.evidence_id, EvidenceAnalysisStatus.FAILED,
        error_code="MissingEvidenceProjectionScope") for record in pending if record not in scoped)
    try:
        request = EvidenceAnalysisRequest(
            question=query_context.resolved_query, evidence=scoped,
            context=query_context, plan=plan,
            requirement_scopes=tuple((record.evidence_id, projection_requirement_scope(record, plan))
                                     for record in scoped),
        )
        result = (runtime.context.evidence_analyzer.analyze(request=request)
                  if scoped else EvidenceAnalysisResult((), 0))
        analyses = (*result.analyses, *unbound_failures)
        llm_calls = result.llm_calls
        llm_diagnostics = result.llm_diagnostics
        cache_stats = result.cache_stats
    except Exception as error:
        analyses = tuple(
            EvidenceAnalysis(
                evidence_id=record.evidence_id,
                status=EvidenceAnalysisStatus.FAILED,
                error_code=f"{type(error).__name__}: {error}"[:300],
            )
            for record in pending
        )
        llm_calls = 0
        llm_diagnostics = ()
        cache_stats = None
    try:
        registry.apply_analyses(analyses, plan=plan)
    except ValueError as error:
        return _fatal_update(state, "analyze_evidence", error)
    # apply_analyses replaces immutable records; do not calculate the search
    # outcome from the pre-projection `pending` snapshot.
    analyzed_records = tuple(
        record
        for evidence_id in pending_ids
        if (record := registry.get(evidence_id)) is not None
    )
    tracker.record_preparation_llm_calls(llm_calls)
    current_goal_items = goal_signal_items(registry.records)
    new_goal_items = tuple(sorted(current_goal_items - previous_goal_items))
    newly_supported_requirement_ids = tuple(
        sorted(
            {
                requirement_id
                for record in analyzed_records
                if bool(record.projections)
                for requirement_id in set(verified_support_ids(record, plan.requirements)) - previous_support.get(record.evidence_id, set())
            }
        )
    )
    new_supporting_evidence_count = sum(
        1
        for record in analyzed_records
        if bool(record.projections)
        and set(verified_support_ids(record, plan.requirements)) - previous_support.get(record.evidence_id, set())
    )
    newly_exposed_section_ids = tuple(
        dict.fromkeys(
            record.evidence.section_id
            for record in analyzed_records
            if record.evidence.section_id
        )
    )
    if query_context.retrieval_goal.value == "exhaustive_enumeration":
        tracker.record_goal_progress(
            tuple(record.evidence.paper_id for record in pending),
            progressed=bool(new_goal_items),
        )
    update: AgentGraphState = {
        "registry_state": dump_contract(registry.snapshot(), RegistryState),
        "guard_state": dump_contract(tracker.snapshot(), GuardState),
        "pending_analysis_ids": [],
        "next_node": "decide_action",
        "llm_diagnostics": _append_llm_diagnostics(
            state, llm_diagnostics
        ),
    }
    observations = _observations(state)
    if observations:
        update["observations"] = dump_contract(
            (
                *observations[:-1],
                replace(
                    observations[-1],
                    evidence_analysis_cache=cache_stats,
                    goal_progress=bool(new_goal_items),
                    new_goal_items=new_goal_items,
                    new_supporting_evidence_count=new_supporting_evidence_count,
                    newly_supported_requirement_ids=newly_supported_requirement_ids,
                    newly_exposed_section_ids=newly_exposed_section_ids,
                    requirement_progress_after=tuple(
                        (item.requirement_id, item.status.value)
                        for item in derive_requirement_states(
                            plan=load_contract(state["plan"], ResearchPlan),
                            evidence=registry.records,
                            observations=observations,
                            context=query_context,
                            # The post-projection coverage snapshot is built
                            # below, after this observation is updated.  The
                            # progress tracker does not use coverage to infer
                            # a terminal state during the research loop.
                            coverage=None,
                            previous_states=_state_requirement_states(state),
                        )
                    ),
                ),
            ),
            tuple[ResearchObservation, ...],
        )
    plan = load_contract(state["plan"], ResearchPlan)
    updated_observations = (
        load_contract(update["observations"], tuple[ResearchObservation, ...])
        if update.get("observations") is not None
        else observations
    )
    updated_coverage = CoverageManager().evaluate(
        context=query_context,
        corpus_snapshot=(
            load_contract(state["corpus_snapshot"], CorpusScopeSnapshot)
            if state.get("corpus_snapshot") is not None
            else None
        ),
        candidate_selection=_candidate_selection(state),
        evidence=registry.records,
        observations=updated_observations,
    )
    update["requirement_states"] = dump_contract(
        derive_requirement_states(
            plan=plan,
            evidence=registry.records,
            observations=updated_observations,
            context=query_context,
            coverage=updated_coverage,
            previous_states=_state_requirement_states(state),
        ),
        tuple[RequirementState, ...],
    )
    return update


def _generate_answer(
    state: AgentGraphState, runtime: Runtime[AgentGraphContext]
) -> AgentGraphState:
    """在 Agent 循环终止后，统一选择证据并生成用户可见回答。"""
    status = ResearchRunStatus(
        state.get("terminal_status", ResearchRunStatus.FAILED.value)
    )
    registry = _registry(state)
    coverage: CoverageState | None = None
    selected: tuple[EvidenceRecord, ...] = ()
    context: QueryContext | None = None
    plan: ResearchPlan | None = None
    requirement_states: tuple[RequirementState, ...] = ()
    terminal_requirement_update: AgentGraphState = {}

    if state.get("query_context") is not None:
        context = load_contract(state["query_context"], QueryContext)
        coverage = _evaluate_coverage(
            state,
            context=context,
            registry=registry,
        )
    if state.get("plan") is not None:
        plan = load_contract(state["plan"], ResearchPlan)
    if coverage is not None and plan is not None:
        requirement_states = load_contract(
            state["requirement_states"], tuple[RequirementState, ...]
        ) if state.get("requirement_states") is not None else derive_requirement_states(
            plan=plan, evidence=registry.records, observations=_observations(state),
            context=context, coverage=coverage,
        )
        # Refresh research progress against integrity-valid Projector support.
        requirement_states = derive_requirement_states(
            plan=plan, evidence=registry.records, observations=_observations(state),
            context=context, coverage=coverage, previous_states=requirement_states,
        )
        if state.get("stop_reason") in {
            StopReason.AGENT_FINISHED.value, StopReason.BUDGET_EXHAUSTED.value,
        }:
            requirement_states = finalize_requirement_states(
                plan=plan,
                evidence=registry.records,
                observations=_observations(state),
                previous_states=requirement_states,
                research_complete=state.get("stop_reason") == StopReason.AGENT_FINISHED.value,
            ).states
            terminal_requirement_update["requirement_states"] = dump_contract(
                requirement_states, tuple[RequirementState, ...]
            )
        selected = FinalEvidenceSelector(
            max_units=runtime.context.settings.final_evidence_max_units,
        ).select(
            context=context,
            plan=plan,
            coverage=coverage,
            evidence=registry.records,
            requirement_states=requirement_states,
        )
    completion_view = (
        build_completion_view(plan=plan, states=requirement_states, selected=selected)
        if plan is not None else None
    )

    if status is ResearchRunStatus.NEEDS_CLARIFICATION:
        answer = ResearchAnswer(
            status="needs_clarification",
            answer=state.get("clarification") or "请补充需要检索的论文范围。",
            insufficient_evidence=True,
        )
    elif status is ResearchRunStatus.FAILED:
        answer = ResearchAnswer(
            status="failed",
            answer="研究流程未能完成；请稍后重试，或缩小问题与论文范围。",
            insufficient_evidence=True,
        )
    elif not selected or context is None or (
        completion_view is not None
        and completion_view.required_requirement_ids
        and not completion_view.supported_requirement_ids
    ):
        answer = ResearchAnswer(
            status="insufficient_evidence",
            answer="当前检索到的论文证据不足以回答该问题。",
            insufficient_evidence=True,
            overall_status="insufficient" if plan is not None else None,
            requirement_coverage=compile_answer_requirement_coverage(
                required_requirement_ids=tuple(item.requirement_id for item in plan.requirements if item.required),
                allowed_support_by_requirement={}, semantic_status_by_requirement={},
                inline_citations=(),
            ) if plan is not None else (),
        )
    elif runtime.context.answer_generator is None:
        answer = ResearchAnswer(
            status="disabled",
            answer="回答生成功能当前未启用；请查看已检索到的论文证据。",
            insufficient_evidence=True,
            coverage_note="系统已完成证据检索，但未配置最终回答生成器。",
        )
    else:
        tracker = _tracker(state, runtime.context.settings)
        tracker.record_preparation_llm_calls(1)
        try:
            answer = generate_agentic_answer(
                generator=runtime.context.answer_generator,
                query=context.resolved_query,
                evidence=selected,
                completion_view=completion_view,
                plan=plan,
                stopped_by_budget=(
                    state.get("stop_reason") == StopReason.BUDGET_EXHAUSTED.value
                ),
            )
        except Exception as error:
            error_codes = list(
                dict.fromkeys(
                    (*state.get("error_codes", []), f"AnswerGeneration:{type(error).__name__}")
                )
            )
            return {
                **terminal_requirement_update,
                "selected_evidence": dump_contract(
                    selected, tuple[EvidenceRecord, ...]
                ),
                "answer": dump_contract(
                    ResearchAnswer(
                        status="fallback",
                        answer="回答生成未完成；系统保留了已检索到的论文证据，供你核验或重新提问。",
                        insufficient_evidence=True,
                        coverage_note="最终回答生成失败，但检索证据与来源信息仍然可用。",
                    ),
                    ResearchAnswer,
                ),
                "error_codes": error_codes,
                "guard_state": dump_contract(tracker.snapshot(), GuardState),
                "next_node": "finish_run",
            }

    return {
        **terminal_requirement_update,
        "selected_evidence": dump_contract(selected, tuple[EvidenceRecord, ...]),
        "answer": dump_contract(answer, ResearchAnswer),
        "llm_diagnostics": _append_llm_diagnostics(
            state, answer.llm_diagnostics
        ),
        "guard_state": (
            dump_contract(tracker.snapshot(), GuardState)
            if runtime.context.answer_generator is not None and selected and context
            else state.get("guard_state")
        ),
        "next_node": "finish_run",
    }


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
    # final projection is therefore regenerated from its still-compatible
    # plan/evidence/observation fields instead of rejecting the task.
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
