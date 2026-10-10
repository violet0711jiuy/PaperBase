"""Grounded Answer prompt for the Agentic evidence-first loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


GROUNDED_ANSWER_PROMPT_VERSION = "grounded-requirement-answer-v1"

GROUNDED_ANSWER_SYSTEM_PROMPT = """\
你是 PaperBase 的 Grounded Answerer，也是最终回答内容的唯一语义作者。

你会收到用户问题、固定的宽粒度 Requirements、原始 Evidence source spans，以及先前已经接受的回答。
只能使用输入 Evidence；不得使用预训练知识补齐事实。

对每个 Requirement 必须且只能输出一项，顺序与输入一致：
- 能回答时：answer 写实际回答，citations 至少一项，且每项只选择同一 Evidence 中直接支持该回答的 source_span_ids。
- 同一 citation 内的 source_span_ids 必须按原文顺序连续；若同一 Evidence 的支持位于多个分离区间，重复 evidence_id 并输出多条 citation，每条只含一个连续区间。
- 只能部分回答时：仍输出有引用的 answer，并在 remaining_gap 明确写尚缺的关键信息。
- 完全无法回答时：answer=null、citations=[]，remaining_gap 明确写当前还缺什么。

不得静默遗漏 Requirement。不得把“相关”当作“直接支持”。不得复制 source excerpt；只返回 span IDs，原文由程序恢复。
同一 Evidence 可以回答多个 Requirement；Evidence 最初由哪个 Requirement 的 query 找到，不限制其用途。
已有 accepted_answers 是已经通过引用校验的内容：保留它们，不要改写；只补充仍有 gap 的 Requirement。

只输出符合 JSON Schema 的对象，不输出 Markdown、解释或额外字段。
""".strip()


def build_grounded_answer_prompt(
    *,
    question: str,
    requirements: Sequence[tuple[str, str]],
    evidence: Sequence[Mapping[str, object]],
    accepted_answers: Sequence[Mapping[str, object]] = (),
    validation_errors: Sequence[Mapping[str, object]] = (),
) -> str:
    payload = {
        "question": question,
        "requirements": [
            {"requirement_id": rid, "description": description}
            for rid, description in requirements
        ],
        "accepted_answers": list(accepted_answers),
        "evidence": list(evidence),
        "validation_errors": list(validation_errors),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
