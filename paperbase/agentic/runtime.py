"""有预算、可停止、可审计的单 Agent 研究循环。"""

from __future__ import annotations

from typing import Protocol

from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import ToolResult

from .contracts import (
    ActionTrace,
    CorpusScopeSnapshot,
    FinishAction,
    QueryContext,
    ResearchPlan,
    ResearchRunResult,
    ResearchRunStatus,
    StopReason,
)
from .decision import DecisionAgent, DecisionError, DecisionState
from .coverage import CoverageManager, CoveragePolicy
from .evidence_analysis import EvidenceAnalyzer
from .evidence_registry import EvidenceRegistry
from .guards import BudgetAndGuardTracker, action_subtask_ids
from .observation import build_observation
from .selector import FinalEvidenceSelector
from .transitions import finish_plan, mark_subtasks_started


class ToolExecutor(Protocol):
    def execute(
        self,
        action: object,
        *,
        context: QueryContext,
        corpus_snapshot: CorpusScopeSnapshot | None = None,
    ) -> ToolResult: ...


class ResearchRuntime:
    def __init__(
        self,
        *,
        decision_agent: DecisionAgent,
        evidence_analyzer: EvidenceAnalyzer | None = None,
        tool_executor: ToolExecutor,
        settings: AgenticResearchSettings,
    ) -> None:
        self._decision_agent = decision_agent
        self._evidence_analyzer = evidence_analyzer
        self._tool_executor = tool_executor
        self._settings = settings

    def run(
        self,
        *,
        context: QueryContext,
        plan: ResearchPlan,
        corpus_snapshot: CorpusScopeSnapshot | None = None,
        initial_llm_calls: int = 0,
    ) -> ResearchRunResult:
        registry = EvidenceRegistry()
        tracker = BudgetAndGuardTracker(
            self._settings, initial_llm_calls=initial_llm_calls
        )
        observations = []
        traces: list[ActionTrace] = []
        current_plan = plan
        coverage_manager = CoverageManager(
            CoveragePolicy(
                candidate_min_queries_per_paper=(
                    self._settings.coverage_candidate_min_queries_per_paper
                ),
                exhaustive_min_queries_per_paper=(
                    self._settings.coverage_exhaustive_min_queries_per_paper
                ),
            )
        )

        while True:
            coverage = coverage_manager.evaluate(
                context=context,
                corpus_snapshot=corpus_snapshot,
                evidence=registry.records,
                observations=tuple(observations),
            )
            capacity = tracker.can_decide()
            if not capacity.accepted:
                return _result(
                    status=(
                        ResearchRunStatus.PARTIAL
                        if registry.records
                        else ResearchRunStatus.NO_EVIDENCE
                    ),
                    reason=StopReason.BUDGET_EXHAUSTED,
                    context=context,
                    plan=current_plan,
                    registry=registry,
                    observations=observations,
                    traces=traces,
                    tracker=tracker,
                    corpus_snapshot=corpus_snapshot,
                )
            if tracker.no_available_actions(context):
                return _result(
                    status=(
                        ResearchRunStatus.PARTIAL
                        if registry.records
                        else ResearchRunStatus.NO_EVIDENCE
                    ),
                    reason=StopReason.NO_AVAILABLE_ACTIONS,
                    context=context,
                    plan=current_plan,
                    registry=registry,
                    observations=observations,
                    traces=traces,
                    tracker=tracker,
                    corpus_snapshot=corpus_snapshot,
                )
            try:
                decision = self._decision_agent.decide(
                    DecisionState(
                        context=context,
                        plan=current_plan,
                        evidence=registry.records,
                        observations=tuple(observations),
                        budget_usage=tracker.usage(),
                        suppressed_paper_ids=tuple(sorted(tracker.suppressed_papers)),
                        coverage=coverage,
                    )
                )
            except DecisionError as error:
                capacity = tracker.record_decision(error.llm_calls)
                traces.append(
                    ActionTrace(
                        iteration=tracker.iterations,
                        action_name="invalid_decision",
                        accepted=False,
                        detail=str(error),
                    )
                )
                if not capacity.accepted:
                    return _result(
                        status=ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.FAILED,
                        reason=StopReason.BUDGET_EXHAUSTED,
                        context=context,
                        plan=current_plan,
                        registry=registry,
                        observations=observations,
                        traces=traces,
                        tracker=tracker,
                        corpus_snapshot=corpus_snapshot,
                    )
                if tracker.record_invalid_action():
                    return _result(
                        status=ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.FAILED,
                        reason=StopReason.INVALID_ACTION_LIMIT,
                        context=context,
                        plan=current_plan,
                        registry=registry,
                        observations=observations,
                        traces=traces,
                        tracker=tracker,
                        corpus_snapshot=corpus_snapshot,
                    )
                continue
            capacity = tracker.record_decision(decision.llm_calls)
            if not capacity.accepted:
                traces.append(
                    ActionTrace(
                        tracker.iterations,
                        decision.action.action_name,
                        False,
                        capacity.detail,
                    )
                )
                return _result(
                    status=ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.NO_EVIDENCE,
                    reason=StopReason.BUDGET_EXHAUSTED,
                    context=context,
                    plan=current_plan,
                    registry=registry,
                    observations=observations,
                    traces=traces,
                    tracker=tracker,
                    corpus_snapshot=corpus_snapshot,
                )
            try:
                registry.apply_qualifications(decision.qualifications)
            except ValueError as error:
                traces.append(
                    ActionTrace(
                        tracker.iterations,
                        decision.action.action_name,
                        False,
                        f"InvalidQualification: {error}",
                    )
                )
                if tracker.record_invalid_action():
                    return _result(
                        status=(
                            ResearchRunStatus.PARTIAL
                            if registry.records
                            else ResearchRunStatus.FAILED
                        ),
                        reason=StopReason.INVALID_ACTION_LIMIT,
                        context=context,
                        plan=current_plan,
                        registry=registry,
                        observations=observations,
                        traces=traces,
                        tracker=tracker,
                        corpus_snapshot=corpus_snapshot,
                    )
                continue
            coverage = coverage_manager.evaluate(
                context=context,
                corpus_snapshot=corpus_snapshot,
                evidence=registry.records,
                observations=tuple(observations),
            )
            if isinstance(decision.action, FinishAction):
                if not coverage.coverage_complete:
                    traces.append(
                        ActionTrace(
                            tracker.iterations,
                            decision.action.action_name,
                            False,
                            "CoverageIncomplete: " + ", ".join(coverage.remaining_paper_ids),
                        )
                    )
                    if tracker.record_invalid_action():
                        return _result(
                            status=(
                                ResearchRunStatus.PARTIAL
                                if registry.records
                                else ResearchRunStatus.NO_EVIDENCE
                            ),
                            reason=StopReason.INVALID_ACTION_LIMIT,
                            context=context,
                            plan=current_plan,
                            registry=registry,
                            observations=observations,
                            traces=traces,
                            tracker=tracker,
                            corpus_snapshot=corpus_snapshot,
                        )
                    continue
                traces.append(
                    ActionTrace(
                        tracker.iterations,
                        decision.action.action_name,
                        True,
                        decision.action.reason,
                    )
                )
                selected = FinalEvidenceSelector(
                    max_units=self._settings.final_evidence_max_units,
                    max_per_paper=self._settings.final_evidence_max_per_paper,
                ).select(
                    plan=current_plan,
                    coverage=coverage,
                    evidence=registry.records,
                )
                current_plan = finish_plan(current_plan, bool(selected))
                return _result(
                    status=(
                        ResearchRunStatus.COMPLETED
                        if selected
                        else ResearchRunStatus.NO_EVIDENCE
                    ),
                    reason=StopReason.AGENT_FINISHED,
                    context=context,
                    plan=current_plan,
                    registry=registry,
                    observations=observations,
                    traces=traces,
                    tracker=tracker,
                    corpus_snapshot=corpus_snapshot,
                )
            guard = tracker.validate_action(
                decision.action,
                context=context,
                plan=current_plan,
                snapshot=corpus_snapshot,
            )
            if not guard.accepted:
                traces.append(
                    ActionTrace(
                        tracker.iterations,
                        decision.action.action_name,
                        False,
                        f"{guard.code}: {guard.detail}",
                    )
                )
                if guard.budget_exhausted:
                    return _result(
                        status=ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.NO_EVIDENCE,
                        reason=StopReason.BUDGET_EXHAUSTED,
                        context=context,
                        plan=current_plan,
                        registry=registry,
                        observations=observations,
                        traces=traces,
                        tracker=tracker,
                        corpus_snapshot=corpus_snapshot,
                    )
                if tracker.record_invalid_action():
                    return _result(
                        status=ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.FAILED,
                        reason=StopReason.INVALID_ACTION_LIMIT,
                        context=context,
                        plan=current_plan,
                        registry=registry,
                        observations=observations,
                        traces=traces,
                        tracker=tracker,
                        corpus_snapshot=corpus_snapshot,
                    )
                continue
            try:
                tool_result = self._tool_executor.execute(
                    decision.action,
                    context=context,
                    corpus_snapshot=corpus_snapshot,
                )
            except Exception as error:
                traces.append(
                    ActionTrace(
                        tracker.iterations,
                        decision.action.action_name,
                        False,
                        f"{type(error).__name__}: {error}",
                    )
                )
                return _result(
                    status=ResearchRunStatus.PARTIAL if registry.records else ResearchRunStatus.FAILED,
                    reason=StopReason.FATAL_ERROR,
                    context=context,
                    plan=current_plan,
                    registry=registry,
                    observations=observations,
                    traces=traces,
                    tracker=tracker,
                    corpus_snapshot=corpus_snapshot,
                )
            tracker.record_tool_call(decision.action)
            delta = registry.register_tool_result(
                tool_result,
                iteration=tracker.iterations,
                target_subtask_ids=action_subtask_ids(decision.action),
            )
            if self._evidence_analyzer is not None and delta.added_ids:
                pending = tuple(
                    record
                    for evidence_id in delta.added_ids
                    if (record := registry.get(evidence_id)) is not None
                )
                analysis = self._evidence_analyzer.analyze(
                    question=context.resolved_query,
                    evidence=pending,
                )
                registry.apply_analyses(analysis.analyses)
                tracker.record_preparation_llm_calls(analysis.llm_calls)
            added_paper_ids = {
                record.evidence.paper_id
                for evidence_id in delta.added_ids
                if (record := registry.get(evidence_id)) is not None
            }
            tracker.record_evidence_delta(
                decision.action, added_paper_ids=added_paper_ids
            )
            observations.append(
                build_observation(
                    iteration=tracker.iterations,
                    result=tool_result,
                    delta=delta,
                    remaining_budget=tracker.remaining(),
                )
            )
            traces.append(
                ActionTrace(
                    tracker.iterations,
                    decision.action.action_name,
                    True,
                    f"added={len(delta.added_ids)}, reused={len(delta.reused_ids)}",
                )
            )
            current_plan = mark_subtasks_started(
                current_plan, action_subtask_ids(decision.action)
            )


def _result(
    *,
    status: ResearchRunStatus,
    reason: StopReason,
    context: QueryContext,
    plan: ResearchPlan,
    registry: EvidenceRegistry,
    observations: list,
    traces: list[ActionTrace],
    tracker: BudgetAndGuardTracker,
    corpus_snapshot: CorpusScopeSnapshot | None = None,
) -> ResearchRunResult:
    coverage = CoverageManager(
        CoveragePolicy(
            candidate_min_queries_per_paper=(
                tracker.settings.coverage_candidate_min_queries_per_paper
            ),
            exhaustive_min_queries_per_paper=(
                tracker.settings.coverage_exhaustive_min_queries_per_paper
            ),
        )
    ).evaluate(
        context=context,
        corpus_snapshot=corpus_snapshot,
        evidence=registry.records,
        observations=tuple(observations),
    )
    selected = FinalEvidenceSelector(
        max_units=tracker.settings.final_evidence_max_units,
        max_per_paper=tracker.settings.final_evidence_max_per_paper,
    ).select(plan=plan, coverage=coverage, evidence=registry.records)
    return ResearchRunResult(
        status=status,
        stop_reason=reason,
        query_context=context,
        plan=plan,
        evidence=registry.records,
        observations=tuple(observations),
        traces=tuple(traces),
        budget_usage=tracker.usage(),
        coverage=coverage,
        selected_evidence=selected,
    )
