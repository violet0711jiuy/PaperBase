"""Agentic Query Understanding 使用的提示词。\n\n可信 Scope 与用户问题分开序列化，模型只能补充检索语义，不能改写程序维护的论文范围。\n"""

from __future__ import annotations

import json
from typing import Sequence


QUERY_UNDERSTANDING_SYSTEM_PROMPT = """\
你是学术证据检索系统的“问题理解”阶段。

请识别并规范化研究问题，但绝不能增加、删除、重命名或推测 paper_id。Scope 和
paper_id 是程序提供的可信状态。必须保留问题中明确出现的论文名称、技术术语、比较维度
以及用户要求的覆盖范围。

你需要输出：
- 一条自包含、消除指代后的 resolved_query；
- 一条可选的英文语义检索问题 semantic_query_en；没有合适改写时输出 null；
- 一组简短的英文词法检索词 lexical_keywords_en；允许为空数组；
- Schema 允许的一个 intent 和一个 retrieval_goal；
- Schema 允许的一个或多个 evidence_targets。

不要选择 Tool，不要设置检索条数，不要回答研究问题。只能返回严格符合所给 Schema 的
JSON，并显式返回 Schema 要求的全部字段。
"""


def build_query_understanding_user_prompt(
    *,
    query: str,
    scope: str,
    trusted_paper_ids: Sequence[str],
) -> str:
    """把不可信问题文本与可信 Scope 序列化为稳定输入。"""
    return json.dumps(
        {
            "question": query,
            "trusted_scope": scope,
            "trusted_paper_ids": list(trusted_paper_ids),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
