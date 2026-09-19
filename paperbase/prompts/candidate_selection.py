"""全库 Candidate Selection 使用的中文提示词。"""

from __future__ import annotations

import json
from typing import Mapping


CANDIDATE_SELECTION_SYSTEM_PROMPT = """\
你是学术论文元数据筛选器。你只判断冻结知识库范围中的论文是否值得为当前问题进入正文检索，
不回答问题、不选择 Tool，也不扩展 Scope。

必须为输入中的每个 paper_id 恰好输出一项：
- candidate：标题、摘要、关键词或 Overview 明确表明论文直接研究当前问题；
- uncertain：元数据不足、表述间接，或不能安全排除，需要一次低成本正文检索确认；
- excluded：元数据明确表明研究对象或任务与问题无关。

宁可把信息不足的论文标为 uncertain，也不能仅因标题没有出现关键词就错误排除。reason 必须简洁说明
使用了哪些元数据。不能增加、删除、重复或修改 paper_id。只输出严格符合 Schema 的 JSON。
"""


def build_candidate_selection_user_prompt(payload: Mapping[str, object]) -> str:
    """把问题与论文元数据作为不可信数据序列化。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=list)
