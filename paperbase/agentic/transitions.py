"""手写循环与 LangGraph 共享的纯领域状态转换。"""

from __future__ import annotations

from dataclasses import replace

from .contracts import ResearchPlan, SubtaskStatus


def mark_subtasks_started(
    plan: ResearchPlan, subtask_ids: tuple[str, ...]
) -> ResearchPlan:
    if not subtask_ids:
        return plan
    return replace(
        plan,
        subtasks=tuple(
            replace(task, status=SubtaskStatus.IN_PROGRESS)
            if task.subtask_id in subtask_ids and task.status is SubtaskStatus.PENDING
            else task
            for task in plan.subtasks
        ),
    )


def finish_plan(plan: ResearchPlan, has_evidence: bool) -> ResearchPlan:
    return replace(
        plan,
        subtasks=tuple(
            replace(
                task,
                status=(
                    SubtaskStatus.COMPLETED
                    if has_evidence and task.status is SubtaskStatus.IN_PROGRESS
                    else SubtaskStatus.BLOCKED
                    if task.status is SubtaskStatus.PENDING
                    else task.status
                ),
            )
            for task in plan.subtasks
        ),
    )
