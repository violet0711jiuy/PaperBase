"""Agentic Research 对应用层稳定暴露的业务进度事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import ResearchRunResult


ProgressEventKind = Literal["started", "progress", "interrupted", "completed"]


@dataclass(frozen=True)
class ResearchProgressEvent:
    """屏蔽 LangGraph 内部状态结构的流式事件契约。"""

    kind: ProgressEventKind
    stage: str
    message: str
    thread_id: str
    node: str | None = None
    result: ResearchRunResult | None = None


_NODE_MESSAGES: dict[str, tuple[str, str]] = {
    "resolve_scope": ("scope", "正在确认论文范围"),
    "understand_query": ("understanding", "正在理解研究问题"),
    "build_plan": ("planning", "正在拆分研究任务"),
    "decide_action": ("decision", "正在决定下一步检索动作"),
    "assess_action": ("assessment", "正在评估论文覆盖度"),
    "validate_action": ("validation", "正在校验检索动作"),
    "execute_tool": ("retrieval", "正在检索论文证据"),
    "register_and_observe": ("evidence", "正在登记并去重证据"),
    "analyze_evidence": ("evidence_analysis", "正在提取证据摘要、Claim 与归属"),
    "generate_answer": ("answer", "正在基于证据生成回答"),
    "finish_run": ("finish", "正在整理最终结果"),
}


def progress_event_for_node(*, node: str, thread_id: str) -> ResearchProgressEvent:
    stage, message = _NODE_MESSAGES.get(node, ("workflow", f"正在执行：{node}"))
    return ResearchProgressEvent(
        kind="progress",
        stage=stage,
        message=message,
        thread_id=thread_id,
        node=node,
    )
