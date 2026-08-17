"""以 Schema 与证据编号约束 Step 8 的最终论文问答生成。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbase.config import AnswerGenerationSettings
from paperbase.llm.client import (
    ChatCompletionClient,
    LLMRequestError,
    LLMRuntimeSettings,
    OpenAICompatibleChatClient,
    load_llm_runtime_settings,
)
from paperbase.prompts.answer_generation import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    build_answer_generation_user_prompt,
)

from .section_expander import EvidenceUnit


class AnswerGenerationDraft(BaseModel):
    """LLM 只填写语义内容；最终 ``answer`` 排版由程序确定性生成。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    # 先回答用户问题，避免后续解释淹没明确结论。
    direct_answer: str = Field(min_length=1)
    # 展开论文中的步骤、推导、实验设置或比较依据，不能只复述结论。
    evidence_explanation: str = Field(min_length=1)
    # 有证据时解释阅读意义；极简事实或证据确实不足时允许为 null，但字段必须出现。
    reading_interpretation: str | None
    # 使用到的证据编号：只能来自本次传入的 E#/R#，程序会再次验证。
    # 必须显式输出，避免模型在正文使用 [E#] 后由默认空列表掩盖遗漏。
    citations: list[str]
    # 证据不足标志：true 时不得把模型常识包装成论文结论。
    insufficient_evidence: bool


class AnswerGenerationResult(BaseModel):
    """程序拼接小标题后的最终回答契约，供 API、CLI 和前端稳定消费。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool


@dataclass(frozen=True)
class AnswerGenerationOutcome:
    """回答生成结果；失败时保持检索证据可用，而不是中断整个问答流程。"""

    status: str
    answer: str | None
    citations: tuple[str, ...]
    insufficient_evidence: bool


class GroundedAnswerGenerator:
    """调用 OpenAI-compatible LLM，并对其输出执行格式和证据编号的确定性校验。"""

    def __init__(
        self,
        *,
        settings: AnswerGenerationSettings,
        client: ChatCompletionClient,
    ) -> None:
        self._settings = settings
        self._client = client

    def generate(
        self,
        *,
        query: str,
        evidence: Sequence[EvidenceUnit],
        conversation_context: Sequence[str] | None = None,
    ) -> AnswerGenerationOutcome:
        """生成可追溯回答；模型、Schema 或引用校验失败时仅返回安全降级状态。"""
        bounded_evidence = tuple(evidence[: self._settings.max_evidence_units])
        if not self._settings.enabled:
            return AnswerGenerationOutcome(
                status="disabled",
                answer=None,
                citations=(),
                insufficient_evidence=not bool(bounded_evidence),
            )
        if not bounded_evidence:
            return AnswerGenerationOutcome(
                status="insufficient_evidence",
                answer="当前检索到的论文证据不足以回答该问题。",
                citations=(),
                insufficient_evidence=True,
            )
        normalized_context = tuple(
            " ".join(item.split())
            for item in (conversation_context or ())
            if item and item.strip()
        )
        if _has_ambiguous_singular_paper_reference(
            query,
            bounded_evidence,
            has_conversation_context=bool(normalized_context),
        ):
            # “这篇论文/本文”在没有会话态论文选择器的 CLI 中没有可验证指向；
            # 多篇候选并存时不能让 LLM 任选一篇作答，必须要求用户明确标题。
            return AnswerGenerationOutcome(
                status="ambiguous_paper",
                answer="问题中的论文对象不明确；当前检索到多个论文来源，请指定论文标题后再提问。",
                citations=(),
                insufficient_evidence=True,
            )
        user_prompt = build_answer_generation_user_prompt(
            query=query,
            evidence=bounded_evidence,
            conversation_context=normalized_context,
        )
        try:
            raw_response = self._client.complete_json(
                system_prompt=ANSWER_GENERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_schema=AnswerGenerationDraft.model_json_schema(),
                schema_name="paperbase_grounded_answer",
            )
            result = _normalize_and_validate_answer(
                _compose_answer(AnswerGenerationDraft.model_validate_json(raw_response)),
                allowed_evidence_ids={item.evidence_id for item in bounded_evidence},
            )
            return _to_outcome(result, status="success")
        except (LLMRequestError, ValidationError, ValueError, TypeError):
            # 回答结构由 API 层 JSON Schema 和本地 Pydantic 模型双重保证。这里的失败既可能
            # 是 JSON 语法/类型不合法，也可能是模型给出了不存在的 E#/R#。两种情况都不应让
            # 第二次 LLM 调用来猜测如何“修复”；直接降级为可人工审阅的证据列表，才能避免
            # 修复器意外改写回答含义、伪造引用或掩盖供应商的结构化输出兼容问题。
            return self._fallback_outcome()

    def _fallback_outcome(self) -> AnswerGenerationOutcome:
        """生成环节是可选增强；失败时让调用方继续展示已审计的证据。"""
        return AnswerGenerationOutcome(
            status="fallback",
            answer=None,
            citations=(),
            insufficient_evidence=False,
        )


def create_answer_generator(
    *,
    settings: AnswerGenerationSettings,
    env_path: Path,
    client: ChatCompletionClient | None = None,
) -> GroundedAnswerGenerator:
    """构造回答生成器；显式注入 client 仅用于单元测试或未来替换 Provider。"""
    if client is None:
        # Query Rewrite 仍使用全局短输出预算；只有最终回答需要较长的论文阅读解释。
        client = OpenAICompatibleChatClient(
            _with_answer_output_budget(
                load_llm_runtime_settings(env_path), settings.max_output_tokens
            )
        )
    return GroundedAnswerGenerator(settings=settings, client=client)


def _with_answer_output_budget(
    runtime_settings: LLMRuntimeSettings, max_output_tokens: int
) -> LLMRuntimeSettings:
    """仅复制 Answer Generation 的运行配置，不能修改共享的全局 LLM 默认值。"""
    return replace(runtime_settings, max_tokens=max_output_tokens)


def _normalize_and_validate_answer(
    result: AnswerGenerationResult,
    *,
    allowed_evidence_ids: set[str],
) -> AnswerGenerationResult:
    """清洗可确定的格式问题，并拒绝模型编造的证据编号。"""
    answer = result.answer.strip()
    if not answer:
        raise ValueError("Answer must not be blank after stripping.")
    citations = _unique_nonempty_strings(result.citations)
    unknown_citations = set(citations).difference(allowed_evidence_ids)
    if unknown_citations:
        raise ValueError("Answer contains citations outside the provided evidence.")
    in_text_citations = set(re.findall(r"\[([ER]\d+)\]", answer))
    if not in_text_citations.issubset(allowed_evidence_ids):
        raise ValueError("Answer text contains fabricated evidence labels.")
    if not in_text_citations.issubset(set(citations)):
        raise ValueError("Answer text citations must also appear in citations.")
    if not result.insufficient_evidence and not citations:
        raise ValueError("A non-insufficient answer must cite provided evidence.")
    return AnswerGenerationResult(
        answer=answer,
        citations=citations,
        insufficient_evidence=result.insufficient_evidence,
    )


def _compose_answer(draft: AnswerGenerationDraft) -> AnswerGenerationResult:
    """把 LLM 的三段内容固定排版，避免模型忽略单个字符串内的小标题要求。"""
    direct_answer = draft.direct_answer.strip()
    evidence_explanation = draft.evidence_explanation.strip()
    interpretation = (draft.reading_interpretation or "").strip()
    if not direct_answer or not evidence_explanation:
        raise ValueError("Answer draft must include direct answer and evidence explanation.")
    parts = [
        f"### 直接回答\n{direct_answer}",
        f"### 论文中的依据与推导\n{evidence_explanation}",
    ]
    if interpretation:
        parts.append(f"### 如何理解\n{interpretation}")
    return AnswerGenerationResult(
        answer="\n\n".join(parts),
        citations=draft.citations,
        insufficient_evidence=draft.insufficient_evidence,
    )


def _unique_nonempty_strings(values: Sequence[str]) -> list[str]:
    """去空白、按首次出现顺序去重，保持 JSON 输出稳定且可读。"""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _has_ambiguous_singular_paper_reference(
    query: str,
    evidence: Sequence[EvidenceUnit],
    *,
    has_conversation_context: bool,
) -> bool:
    """仅在无可验证论文指向且候选跨论文时阻止“本文/这篇论文”式猜测。"""
    normalized = " ".join(query.casefold().split())
    singular_reference_patterns = (
        "这篇论文",
        "本文",
        "该论文",
        "该文",
        "this paper",
        "the paper",
    )
    return (
        not has_conversation_context
        and
        any(pattern in normalized for pattern in singular_reference_patterns)
        and len({item.paper_id for item in evidence}) > 1
    )


def _to_outcome(result: AnswerGenerationResult, *, status: str) -> AnswerGenerationOutcome:
    """将 Pydantic 对象冻结为服务层稳定返回值，避免调用方意外修改结果。"""
    return AnswerGenerationOutcome(
        status=status,
        answer=result.answer,
        citations=tuple(result.citations),
        insufficient_evidence=result.insufficient_evidence,
    )
