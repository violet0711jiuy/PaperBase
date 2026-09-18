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
    tasks: list[ResearchSubtask] = []
    if context.scope is QueryScope.KNOWLEDGE_BASE:
        tasks.append(
            ResearchSubtask(
                subtask_id="S1",
                goal="在冻结的知识库范围内寻找支持问题的论文证据",
                target_paper_ids=(),
                evidence_targets=context.default_evidence_targets,
                success_criteria="获得至少一条与问题直接相关、可追溯到论文的证据",
            )
        )
    else:
        for index, paper_id in enumerate(context.target_paper_ids, start=1):
            tasks.append(
                ResearchSubtask(
                    subtask_id=f"S{index}",
                    goal=f"检索论文 {paper_id} 中与问题相关的证据",
                    target_paper_ids=(paper_id,),
                    evidence_targets=context.default_evidence_targets,
                    success_criteria=f"为论文 {paper_id} 获得可追溯的直接证据",
                )
            )
        if context.intent is Intent.COMPARE:
            tasks.append(
                ResearchSubtask(
                    subtask_id=f"S{len(tasks) + 1}",
                    goal="检查各论文证据是否覆盖共同的比较维度",
                    target_paper_ids=context.target_paper_ids,
                    evidence_targets=context.default_evidence_targets,
                    success_criteria="每篇目标论文均有足以支撑比较的证据",
                )
            )
    return ResearchPlan(goal=context.resolved_query, subtasks=tuple(tasks))
