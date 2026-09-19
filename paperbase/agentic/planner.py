"""把 QueryContext 转成可审计的 research subtasks。"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from paperbase.llm.client import ChatCompletionClient, LLMRequestError
from paperbase.prompts.agentic_planner import (
    RESEARCH_PLANNER_SYSTEM_PROMPT,
    build_research_planner_user_prompt,
)

from .contracts import EvidenceTarget, Intent, QueryContext, QueryScope, ResearchPlan, ResearchSubtask
from .schemas import ResearchPlanDraft


@dataclass(frozen=True)
class PlannerResult:
    plan: ResearchPlan
    used_fallback: bool
    llm_calls: int


class ResearchPlanner:
    def __init__(self, client: ChatCompletionClient | None = None) -> None:
        self._client = client

    def plan(self, context: QueryContext) -> PlannerResult:
        fallback = _fallback_plan(context)
        if self._client is None:
            return PlannerResult(fallback, True, 0)
        try:
            raw = self._client.complete_json(
                system_prompt=RESEARCH_PLANNER_SYSTEM_PROMPT,
                user_prompt=build_research_planner_user_prompt(
                    question=context.resolved_query,
                    intent=context.intent.value,
                    scope=context.scope.value,
                    trusted_paper_ids=context.target_paper_ids,
                    evidence_targets=sorted(
                        target.value for target in context.default_evidence_targets
                    ),
                ),
                json_schema=ResearchPlanDraft.model_json_schema(),
                schema_name="paperbase_research_plan",
            )
            draft = ResearchPlanDraft.model_validate_json(raw)
            plan = _validate_and_build(draft, context)
            return PlannerResult(plan, False, 1)
        except (LLMRequestError, ValidationError, ValueError, TypeError):
            return PlannerResult(fallback, True, 1)


def _validate_and_build(draft: ResearchPlanDraft, context: QueryContext) -> ResearchPlan:
    if len(draft.subtasks) > 3:
        raise ValueError("Compact Planner 不能生成超过 3 个 Subtask。")
    allowed = set(context.target_paper_ids)
    tasks: list[ResearchSubtask] = []
    covered: set[str] = set()
    for index, item in enumerate(draft.subtasks, start=1):
        targets = tuple(dict.fromkeys(item.target_paper_ids))
        if context.scope is QueryScope.KNOWLEDGE_BASE:
            if targets:
                raise ValueError("Knowledge-base subtasks must not invent paper IDs.")
        elif not set(targets).issubset(allowed) or not targets:
            raise ValueError("Planner used an untrusted or empty paper scope.")
        covered.update(targets)
        tasks.append(
            ResearchSubtask(
                subtask_id=f"S{index}",
                goal=item.goal,
                target_paper_ids=targets,
                evidence_targets=frozenset(item.evidence_targets),
                success_criteria=item.success_criteria,
            )
        )
    if context.intent is Intent.COMPARE and covered != allowed:
        raise ValueError("A comparison plan must cover every selected paper.")
    return ResearchPlan(goal=draft.goal, subtasks=tuple(tasks))


def _fallback_plan(context: QueryContext) -> ResearchPlan:
    targets = (
        () if context.scope is QueryScope.KNOWLEDGE_BASE else context.target_paper_ids
    )
    if context.scope is QueryScope.KNOWLEDGE_BASE:
        tasks = [
            ResearchSubtask(
                subtask_id="S1",
                goal="从已筛选的候选论文中逐篇收集与问题直接相关的证据",
                target_paper_ids=(),
                evidence_targets=context.default_evidence_targets,
                success_criteria="每篇候选论文均已获得直接证据或形成可审计的未找到记录",
            ),
            ResearchSubtask(
                subtask_id="S2",
                goal="合并证据并检查候选覆盖、归属与信息缺口",
                target_paper_ids=(),
                evidence_targets=context.default_evidence_targets,
                success_criteria="结论能够说明已覆盖候选、排除范围和剩余不确定性",
            ),
        ]
    else:
        tasks = [
            ResearchSubtask(
                subtask_id="S1",
                goal="逐篇收集与问题直接相关且可归属的论文证据",
                target_paper_ids=targets,
                evidence_targets=context.default_evidence_targets,
                success_criteria="每篇目标论文均获得直接证据或可审计的未找到记录",
            ),
            ResearchSubtask(
                subtask_id="S2",
                goal=(
                    "按共同维度比较各论文证据并检查信息缺口"
                    if context.intent is Intent.COMPARE
                    else "核对证据归属、覆盖范围与尚缺信息"
                ),
                target_paper_ids=targets,
                evidence_targets=context.default_evidence_targets,
                success_criteria=(
                    "共同点、差异和证据缺口均能追溯到对应论文"
                    if context.intent is Intent.COMPARE
                    else "回答边界与证据支持范围清晰可审计"
                ),
            ),
        ]
    return ResearchPlan(goal=context.resolved_query, subtasks=tuple(tasks))
