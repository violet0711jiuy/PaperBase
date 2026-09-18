"""Agentic Research Planner 使用的提示词。"""

from __future__ import annotations

import json
from typing import Sequence


RESEARCH_PLANNER_SYSTEM_PROMPT = """\
你是学术证据检索系统的“研究规划”阶段。请生成规模小、可审计的证据收集计划，不要回答
研究问题。

只能使用输入中的 trusted_paper_ids。对于 knowledge_base 全库范围，每个 Subtask 的
target_paper_ids 必须是空数组，因为语料成员由程序维护。对于 selected_papers 比较任务，
必须覆盖每一篇已选论文，并明确每篇论文分别需要什么证据。

Subtask 只描述研究目标、证据类型和成功条件。不要选择 Tool，不要设置 Top-K，不要设置
模型参数，也不要生成不存在的 paper_id。只能返回严格符合所给 Schema 的 JSON，并显式
返回 Schema 要求的全部字段。
"""


def build_research_planner_user_prompt(
    *,
    question: str,
    intent: str,
    scope: str,
    trusted_paper_ids: Sequence[str],
    evidence_targets: Sequence[str],
) -> str:
    """构造 Planner 输入，不把用户文本插入系统规则。"""
    return json.dumps(
        {
            "question": question,
            "intent": intent,
            "trusted_scope": scope,
            "trusted_paper_ids": list(trusted_paper_ids),
            "default_evidence_targets": list(evidence_targets),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
