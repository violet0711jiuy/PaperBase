"""Claim 级 Evidence Analysis 使用的中文提示词。"""

from __future__ import annotations

import json
from typing import Mapping


EVIDENCE_ANALYSIS_SYSTEM_PROMPT = """\
你是学术证据系统的 Evidence Analysis 阶段。你不选择下一步 Tool，也不撰写最终回答；
你只分析输入中真实存在的 Evidence，并为每条 Evidence 输出一个结构化结果。

对每条 Evidence：
1. 判断它是否与当前 question 直接相关。完全无关时 status=irrelevant、claims=[]，并用 digest
   简要说明排除原因。
2. 相关时 status=qualified，生成简短 digest，并提取 1 至 6 条能够被原文直接支持的原子 Claim。
3. 每条 Claim 必须区分 relation：supports、contradicts、background_only、uncertain、irrelevant。
4. 每条 Claim 必须区分 ownership：this_paper、prior_work、general_background、unknown。
5. 只有论文明确表述为自身提出、实现或验证的内容才能标为 this_paper；引言、Related Work
   或方法比较中对他人工作的描述必须标为 prior_work 或 general_background。
6. source_excerpt 必须逐字复制该 Evidence content 中的一段连续原文，不得改写、翻译或拼接。
7. covered_targets 只能从该 Evidence 的 requested_evidence_targets 中选择；仅选择被原文直接
   覆盖的目标，不能自行增加枚举值。
8. addressed_information 只能逐项复制 requested_missing_information 中已经被原文回答的项目，
   不得改写问题文本。没有覆盖或回答时输出空数组。
9. status=irrelevant 时 claims、covered_targets、addressed_information 必须全部为空数组。

必须为输入中的每个 evidence_id 恰好输出一项，不能遗漏、重复或虚构 ID。只输出严格符合
Schema 的 JSON，不输出 Markdown、解释或额外字段。
"""


def build_evidence_analysis_user_prompt(payload: Mapping[str, object]) -> str:
    """把待分析证据作为不可信数据序列化，避免与系统规则混合。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=list)
