"""以 Schema 与证据编号约束 Step 8 的最终论文问答生成。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import time
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbase.config import AnswerGenerationSettings
from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCallRecord,
    LLMCompletion,
    LLMRequestError,
    LLMRuntimeSettings,
    ValidationDiagnostic,
    create_chat_completion_client,
    load_llm_runtime_settings,
    complete_json_with_diagnostics,
    structured_client_view,
)
from paperbase.llm.diagnostics import (
    REQUEST_ERROR_STAGES,
    classify_llm_error,
    failure_record,
    success_record,
)
from paperbase.prompts.answer_generation import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    ANSWER_SEMANTIC_SYSTEM_PROMPT,
    build_answer_generation_user_prompt,
)

from .section_expander import EvidenceUnit
from .answer_coverage import (
    AnswerRequirementCoverage, AnswerDomainValidationError,
    compile_answer_requirement_coverage, derive_overall_answer_status,
    ANSWER_SEMANTIC_SCHEMA_VERSION,
)


class AnswerSemanticDraft(BaseModel):
    """Agentic provider authors text and completeness, never bookkeeping."""

    model_config = ConfigDict(extra="forbid", strict=True)
    direct_answer: str = Field(min_length=1)
    evidence_explanation: str = Field(min_length=1)
    reading_interpretation: str | None
    requirement_status: dict[str, Literal["complete", "partial", "unsupported"]]
    coverage_note: str | None


class AnswerGenerationDraft(BaseModel):
    """LLM 只填写语义内容；最终 ``answer`` 排版由程序确定性生成。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    # Classic RAG only; never parsed as an Agentic provider response.
    # 先回答用户问题，避免后续解释淹没明确结论。
    direct_answer: str = Field(min_length=1)
    # 展开论文中的步骤、推导、实验设置或比较依据，不能只复述结论。
    evidence_explanation: str = Field(min_length=1)
    # 有证据时解释阅读意义；极简事实或证据确实不足时允许为 null，但字段必须出现。
    reading_interpretation: str | None
    # 模型填写的引用清单只作为辅助检查；最终清单由程序从正文 [E#]/[REF#] 确定性提取。
    citations: list[str] = Field(default_factory=list)
    # 证据不足表示：当前证据连问题的核心结论也无法支持。
    insufficient_evidence: bool
    # 部分回答表示：核心结论可回答，但覆盖范围、条件或次要细节不完整。
    partial_answer: bool = False
    # 仅在 partial_answer=true 时说明尚未覆盖的范围；不能写成论文事实。
    coverage_note: str | None = None


class AnswerGenerationResult(BaseModel):
    """程序拼接小标题后的最终回答契约，供 API、CLI 和前端稳定消费。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool
    partial_answer: bool = False
    coverage_note: str | None = None
    requirement_coverage: tuple[AnswerRequirementCoverage, ...] = ()


@dataclass(frozen=True)
class AnswerGenerationOutcome:
    """回答生成结果；失败时保持检索证据可用，而不是中断整个问答流程。"""

    status: str
    # 所有分支都返回可展示文本；前端无需为 answer=null 再写一套兜底逻辑。
    answer: str
    citations: tuple[str, ...]
    insufficient_evidence: bool
    partial_answer: bool = False
    coverage_note: str | None = None
    requirement_coverage: tuple[AnswerRequirementCoverage, ...] = ()
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()


def _answer_validation_diagnostics(
    error: Exception, *, attempt: int, recovery_action: str,
) -> tuple[ValidationDiagnostic, ...]:
    if isinstance(error, AnswerDomainValidationError):
        return (ValidationDiagnostic(
            node="generate_answer", attempt=attempt, error_stage="domain_validation",
            error_code=error.code, field_path=error.field_path,
            validator_name="answer_domain_validator", safe_reason=error.safe_reason,
            expected_constraint=error.expected_constraint,
            actual_summary=error.actual_summary, recovery_action=recovery_action,
        ),)
    if isinstance(error, ValidationError):
        stage = classify_llm_error(error)
        codes = {
            "json_invalid": "ANSWER_JSON_INVALID",
            "missing": "ANSWER_SCHEMA_REQUIRED_FIELD",
            "extra_forbidden": "ANSWER_SCHEMA_EXTRA_FIELD",
            "string_too_short": "ANSWER_SCHEMA_STRING_TOO_SHORT",
        }
        return tuple(ValidationDiagnostic(
            node="generate_answer", attempt=attempt, error_stage=stage,
            error_code=codes.get(str(item.get("type")), "ANSWER_SCHEMA_VALIDATION_FAILED"),
            field_path=".".join(str(part) for part in item.get("loc", ())) or "root",
            validator_name=(error.title if error.title in {
                "AnswerSemanticDraft", "AnswerGenerationDraft", "AnswerGenerationResult"
            } else "AnswerSchema"),
            safe_reason="回答草稿未通过现有 JSON Schema 校验。",
            expected_constraint="Follow the current node response JSON Schema.",
            actual_summary=str(item.get("type", "validation_error")),
            recovery_action=recovery_action,
        ) for item in error.errors()[:3])
    if isinstance(error, (ValueError, TypeError)):
        stage = classify_llm_error(error)
        return (ValidationDiagnostic(
            node="generate_answer", attempt=attempt, error_stage=stage,
            error_code=("ANSWER_DOMAIN_VALIDATION_FAILED" if stage == "domain_validation"
                        else "ANSWER_SCHEMA_VALIDATION_FAILED"),
            field_path="root", validator_name="answer_validation",
            safe_reason="回答未通过现有校验。",
            expected_constraint="Answer must satisfy its existing output contract.",
            actual_summary=type(error).__name__, recovery_action=recovery_action,
        ),)
    return ()


class GroundedAnswerGenerator:
    """调用 OpenAI-compatible LLM，并对其输出执行格式和证据编号的确定性校验。"""

    def __init__(
        self,
        *,
        settings: AnswerGenerationSettings,
        client: ChatCompletionClient,
        agentic_temperature: float = 0,
    ) -> None:
        self._settings = settings
        self._client = client
        self._agentic_temperature = agentic_temperature

    def generate(
        self,
        *,
        query: str,
        evidence: Sequence[EvidenceUnit],
        conversation_context: Sequence[str] | None = None,
        research_requirements: Sequence[tuple[str, str]] = (),
        support_by_requirement: Mapping[str, Sequence[str]] | None = None,
    ) -> AnswerGenerationOutcome:
        """生成可追溯回答；模型、Schema 或引用校验失败时仅返回安全降级状态。"""
        # Agentic selector has already applied its evidence budget and protected
        # unique Requirement support; a second truncation would silently drop it.
        bounded_evidence = tuple(
            evidence if research_requirements else evidence[: self._settings.max_evidence_units]
        )
        if not self._settings.enabled:
            return AnswerGenerationOutcome(
                status="disabled",
                answer="回答生成功能当前未启用；请查看下方已检索到的论文证据。",
                citations=(),
                # 未生成可引用的论文结论时，不能让调用方把该提示误当作回答事实。
                insufficient_evidence=True,
                coverage_note="回答生成器未启用，系统未根据检索证据形成最终回答。",
            )
        if not bounded_evidence:
            return AnswerGenerationOutcome(
                status="insufficient_evidence",
                answer="当前检索到的论文证据不足以回答该问题。",
                citations=(),
                insufficient_evidence=True,
                requirement_coverage=compile_answer_requirement_coverage(
                    required_requirement_ids=tuple(rid for rid, _ in research_requirements),
                    allowed_support_by_requirement={}, semantic_status_by_requirement={},
                    inline_citations=(),
                ),
            )
        allowed_ids = {item.evidence_id for item in bounded_evidence}
        allowed_coverage = {
            rid: tuple(item.evidence_id for item in bounded_evidence
                       if item.evidence_id in (support_by_requirement or {}).get(rid, ()))
            for rid, _ in research_requirements
        }
        if research_requirements and not any(allowed_coverage.values()):
            return AnswerGenerationOutcome(
                status="insufficient_evidence",
                answer="当前检索到的论文证据不足以回答该问题。",
                citations=(), insufficient_evidence=True,
                requirement_coverage=compile_answer_requirement_coverage(
                    required_requirement_ids=tuple(rid for rid, _ in research_requirements),
                    allowed_support_by_requirement=allowed_coverage,
                    semantic_status_by_requirement={}, inline_citations=(),
                ),
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
            research_requirements=research_requirements,
            support_by_requirement=support_by_requirement,
        )
        diagnostics: list[LLMCallRecord] = []
        # Only Agentic coverage generation changes configuration; Classic is unchanged.
        generation_client = structured_client_view(
            self._client, temperature=self._agentic_temperature, structured_output_mode="json_schema"
        ) if research_requirements else self._client
        citation_repair_used = False
        original_draft = None
        request_retries = 0
        max_attempts = 1 + self._settings.request_error_retries
        response_schema = _answer_response_schema(
            research_requirements,
            allowed_coverage,
            tuple(item.evidence_id for item in bounded_evidence),
        )
        # One citation repair in addition to the pre-existing bounded transport retries.
        for attempt in range(1, max_attempts + 1 + bool(research_requirements)):
            completion: LLMCompletion | None = None
            try:
                completion = complete_json_with_diagnostics(
                    generation_client,
                    system_prompt=(ANSWER_SEMANTIC_SYSTEM_PROMPT
                                   if research_requirements else ANSWER_GENERATION_SYSTEM_PROMPT),
                    user_prompt=user_prompt,
                    json_schema=response_schema,
                    schema_name=("paperbase_grounded_answer_semantic_v3"
                                 if research_requirements else "paperbase_grounded_answer"),
                )
                draft = (
                    AnswerSemanticDraft.model_validate_json(completion.content)
                    if research_requirements else AnswerGenerationDraft.model_validate_json(completion.content)
                )
                if original_draft is not None:
                    _validate_citation_repair(original_draft, draft)
                result = (
                    _compile_semantic_answer(
                        draft, allowed_evidence_ids=allowed_ids,
                        research_requirements=research_requirements,
                        support_by_requirement=allowed_coverage,
                    ) if research_requirements else
                    _normalize_and_validate_answer(_compose_answer(draft), allowed_evidence_ids=allowed_ids)
                )
                diagnostics.append(
                    success_record(
                        node="generate_answer",
                        attempt=attempt,
                        completion=completion,
                    )
                )
                return _to_outcome(
                    result,
                    status=("insufficient_evidence" if research_requirements and result.insufficient_evidence else "success"),
                    llm_diagnostics=tuple(diagnostics),
                )
            except (LLMRequestError, ValidationError, ValueError, TypeError) as error:
                retryable_request = (
                    isinstance(error, LLMRequestError)
                    and error.error_stage in REQUEST_ERROR_STAGES
                    and request_retries < self._settings.request_error_retries
                )
                repair_source = _semantic_repair_source(completion) if research_requirements else None
                repairable_citation = (
                    bool(research_requirements) and not citation_repair_used and repair_source is not None
                    and (isinstance(error, ValidationError) or (
                        isinstance(error, AnswerDomainValidationError)
                        and error.code in {"ANSWER_UNKNOWN_CITATION", "ANSWER_CITATION_INVALID",
                                           "ANSWER_REQUIREMENT_STATUS_INVALID"}
                    ))
                )
                recovery = "retry_request" if retryable_request else "repair_citation" if repairable_citation else "fallback"
                diagnostics.append(
                    replace(failure_record(
                        node="generate_answer",
                        attempt=attempt,
                        error=error,
                        completion=completion,
                        can_retry=retryable_request or repairable_citation,
                        validation_diagnostics=_answer_validation_diagnostics(
                            error, attempt=attempt,
                            recovery_action=recovery,
                        ),
                    ), recovery_action=recovery)
                )
                if retryable_request:
                    request_retries += 1
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                if repairable_citation:
                    citation_repair_used = True
                    original_draft = repair_source
                    user_prompt = _citation_repair_prompt(
                        user_prompt, original_draft, diagnostics[-1].validation_diagnostics,
                        allowed_ids, allowed_coverage,
                    )
                    continue
            # Repair is never a bypass: schema, provenance and body preservation
            # are checked again. A second domain failure falls back.
            return self._fallback_outcome(
                llm_diagnostics=tuple(diagnostics)
            )
        return self._fallback_outcome(llm_diagnostics=tuple(diagnostics))

    def _fallback_outcome(
        self, *, llm_diagnostics: tuple[LLMCallRecord, ...] = ()
    ) -> AnswerGenerationOutcome:
        """生成环节是可选增强；失败时让调用方继续展示已审计的证据。"""
        return AnswerGenerationOutcome(
            status="fallback",
            answer="回答生成未完成；系统保留了下方已检索到的论文证据，供你核验或重新提问。",
            citations=(),
            # 这是“生成失败”，不是把已有检索证据错误判成无关；但尚无可展示的论文结论。
            insufficient_evidence=True,
            coverage_note="回答模型未生成符合证据引用约束的内容，尚未形成可引用的论文结论。",
            llm_diagnostics=llm_diagnostics,
        )


def _answer_response_schema(requirements, allowed_coverage, allowed_evidence_ids=()):
    if not requirements:
        # Classic RAG keeps its established provider contract.
        schema = AnswerGenerationDraft.model_json_schema()
        ids = list(dict.fromkeys(allowed_evidence_ids))
        schema["properties"]["citations"] = (
            {"type": "array", "items": {"type": "string", "enum": ids}} if ids
            else {"type": "array", "maxItems": 0, "items": {"type": "string"}}
        )
        return schema
    schema = AnswerSemanticDraft.model_json_schema()
    schema["title"] = ANSWER_SEMANTIC_SCHEMA_VERSION
    judged = [rid for rid, _ in requirements if allowed_coverage.get(rid)]
    schema["properties"]["requirement_status"] = {
        "type": "object",
        "properties": {rid: {"type": "string", "enum": ["complete", "partial", "unsupported"]} for rid in judged},
        "required": judged, "additionalProperties": False,
    }
    return schema


def _semantic_repair_source(completion):
    """Preservation snapshot only, never an alternate accepted provider parser.

    A schema failure is repairable only if original fact fields can be retained.
    Invalid JSON or absent/ill-typed fact text has no safe repair snapshot.
    """
    if completion is None:
        return None
    try:
        raw = json.loads(completion.content)
        if not isinstance(raw, dict) or any(
            not isinstance(raw.get(field), str) or not raw[field].strip()
            for field in ("direct_answer", "evidence_explanation")
        ):
            return None
        reading = raw.get("reading_interpretation")
        if reading is not None and not isinstance(reading, str):
            return None
        statuses = raw.get("requirement_status", {})
        statuses = statuses if isinstance(statuses, dict) else {}
        return AnswerSemanticDraft.model_construct(
            direct_answer=raw["direct_answer"], evidence_explanation=raw["evidence_explanation"],
            reading_interpretation=reading, requirement_status=statuses,
            coverage_note=raw.get("coverage_note") if isinstance(raw.get("coverage_note"), str) else None,
        )
    except (ValueError, TypeError):
        return None


def _citation_repair_prompt(prompt, draft, errors, allowed_ids, allowed_coverage):
    """Return original Evidence context plus an explicit structural repair contract."""
    payload = {
        "instruction": (
            "仅修复 citation/domain 结构。不得增加事实，不得使用输入外的新 Evidence。"
            "修正正文 inline citation、requirement_status 的精确 keys/complete或partial或unsupported、结构字段；"
            "事实正文保持原字句，只允许删除完整句子或段落。若事实没有合法支持则删除。"
            "可将 complete 降为 partial/unsupported，partial 可降为 unsupported，不得升级或制造引用；"
            "无支持映射的 Requirement 由程序设为 unsupported，不得输出其 status key。coverage_note 与缺口一致。"
            "requirement_id 是义务 ID，不是 citation；只能从每项 allowed_coverage_evidence_ids 选择。"
            "返回完整原 Schema JSON。"
        ),
        "original_answer_json": draft.model_dump(),
        "validation_errors": [{"code": e.error_code, "field_path": e.field_path,
                               "expected_constraint": e.expected_constraint} for e in errors],
        "allowed_evidence_ids": sorted(allowed_ids),
        "allowed_coverage_evidence_ids": allowed_coverage,
    }
    return prompt + "\n\n<answer_citation_repair>\n" + json.dumps(payload, ensure_ascii=False) + "\n</answer_citation_repair>"


def _validate_citation_repair(original, repaired):
    """Structural edit guard, not an LLM/lexical semantic judge.

    Citation edits and whole-sentence deletions may retain existing wording;
    additions/rephrasings cannot quietly introduce facts in a repair request.
    """
    def plain(value):
        return re.sub(r"\[(?:E\d+|REF\d+|(?:REQ:)?R\d+)\]", "", value or "")
    for field in ("direct_answer", "evidence_explanation", "reading_interpretation"):
        before = plain(getattr(original, field)); after = " ".join(plain(getattr(repaired, field)).split())
        units = re.split(r"(?<=[。！？!?])\s*|\n+|(?<=\.)\s+", before)
        remaining = after
        for unit in units:
            normalized = " ".join(unit.split())
            if normalized and remaining.startswith(normalized) and (
                len(remaining) == len(normalized) or remaining[len(normalized)].isspace()
            ):
                remaining = remaining[len(normalized):].lstrip()
        if remaining:
            raise AnswerDomainValidationError(
                "ANSWER_REPAIR_CONTENT_CHANGED", field,
                "引用修复不得增加或改写事实正文，只允许引用编辑或完整句子删除。",
                "non_citation_text_changed", "Preserve original fact wording; only delete complete sentences."
            )
    ranks = {"unsupported": 0, "partial": 1, "complete": 2}
    original_statuses = original.requirement_status
    if any(status in ranks and isinstance(original_statuses.get(rid), str)
           and original_statuses.get(rid) in ranks
           and ranks[status] > ranks[original_statuses[rid]]
           for rid, status in repaired.requirement_status.items()):
        raise AnswerDomainValidationError(
            "ANSWER_REPAIR_COVERAGE_UPGRADED", "requirement_status",
            "引用修复不得升级原答案覆盖状态。", "coverage_upgraded", "Only preserve or downgrade coverage."
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
        client = create_chat_completion_client(
            _with_answer_runtime_settings(
                load_llm_runtime_settings(env_path), settings
            )
        )
    return GroundedAnswerGenerator(settings=settings, client=client)


def _with_answer_runtime_settings(
    runtime_settings: LLMRuntimeSettings, settings: AnswerGenerationSettings
) -> LLMRuntimeSettings:
    """为 Answer 创建独立输出预算和超时，不修改共享的全局 LLM 配置。"""
    return replace(
        runtime_settings,
        max_tokens=settings.max_output_tokens,
        timeout_seconds=settings.timeout_seconds,
    )


def _compile_semantic_answer(
    draft: AnswerSemanticDraft, *, allowed_evidence_ids: set[str],
    research_requirements: Sequence[tuple[str, str]],
    support_by_requirement: Mapping[str, Sequence[str]],
) -> AnswerGenerationResult:
    # Only these three fields establish citations/support; a note cannot supply it.
    body = _compose_body(draft.direct_answer, draft.evidence_explanation, draft.reading_interpretation)
    citations = _validated_inline_citations(
        body, allowed_evidence_ids=allowed_evidence_ids,
    )
    # Notes may explain gaps but cannot smuggle an unknown citation.
    note_citations = _validated_inline_citations(draft.coverage_note or "", allowed_evidence_ids=allowed_evidence_ids)
    coverage = compile_answer_requirement_coverage(
        required_requirement_ids=tuple(rid for rid, _ in research_requirements),
        allowed_support_by_requirement=support_by_requirement,
        semantic_status_by_requirement=draft.requirement_status,
        inline_citations=citations,
    )
    overall = derive_overall_answer_status(coverage, tuple(citations))
    if overall == "insufficient" and (citations or note_citations):
        raise AnswerDomainValidationError(
            "ANSWER_CITATION_INVALID", "answer", "无支持的回答不得引用证据作答。",
            "unsupported_answer_with_citations", "An insufficient answer must not cite evidence.",
        )
    note = (draft.coverage_note or "").strip() or None
    if overall == "partial":
        note = note or _DEFAULT_PARTIAL_COVERAGE_NOTE
    elif overall == "complete":
        note = None
    answer = body
    if note:
        answer += "\n\n### 证据覆盖说明\n" + note
    return AnswerGenerationResult(
        answer=answer, citations=citations, insufficient_evidence=overall == "insufficient",
        partial_answer=overall == "partial", coverage_note=note, requirement_coverage=coverage,
    )


def _validated_inline_citations(text, *, allowed_evidence_ids):
    citations = _extract_in_text_citations(text)
    if re.search(r"\[(?:REQ:)?R\d+\]", text):
        raise AnswerDomainValidationError(
            "ANSWER_UNKNOWN_CITATION", "answer", "Requirement ID 不能作为 Evidence 引用。",
            "requirement_namespace_citation", "Use selected E#/REF# citations, never Requirement R#.",
        )
    if not set(citations).issubset(allowed_evidence_ids):
        raise AnswerDomainValidationError(
            "ANSWER_UNKNOWN_CITATION", "answer", "正文引用了未提供的 Evidence ID。",
            "unknown_evidence_id_present", "Every inline citation must belong to selected Evidence.",
        )
    return citations


def _normalize_and_validate_answer(
    result: AnswerGenerationResult,
    *,
    allowed_evidence_ids: set[str],
) -> AnswerGenerationResult:
    """校验正文引用，并由程序确定最终引用清单与回答覆盖状态。"""
    answer = result.answer.strip()
    if not answer:
        raise AnswerDomainValidationError(
            "ANSWER_BLANK", "answer", "回答正文去除空白后为空。", "blank_answer",
            "Composed answer must be non-empty after stripping."
        )
    citations = _validated_inline_citations(answer, allowed_evidence_ids=allowed_evidence_ids)

    # ``citations`` 是可由正文确定性得到的派生数据，不能因为模型漏填重复字段而丢弃回答。
    # 因此不信任模型单独返回的 citations 数组，只以实际出现且已验证的标记为准。
    insufficient_evidence = result.insufficient_evidence
    partial_answer = result.partial_answer
    coverage_note = (result.coverage_note or "").strip() or None

    if insufficient_evidence and citations:
        # 这类输出通常代表模型已给出可核验的核心回答，却把“并未穷尽全部细节”错标成
        # evidence insufficient。将它规范为 partial，既保留回答和引用，也向 UI 交代边界。
        insufficient_evidence = False
        partial_answer = True
        coverage_note = coverage_note or _DEFAULT_PARTIAL_COVERAGE_NOTE
    elif insufficient_evidence:
        # 真正无证据回答不得携带引用，更不能同时声称“部分回答”。
        partial_answer = False
        coverage_note = None
    elif partial_answer:
        if not citations:
            raise AnswerDomainValidationError(
                "ANSWER_PARTIAL_WITHOUT_CITATION", "answer",
                "部分回答必须在正文引用已提供的证据。", "partial_answer_without_citation",
                "A partial answer must cite supplied Evidence in its text."
            )
        coverage_note = coverage_note or _DEFAULT_PARTIAL_COVERAGE_NOTE
    elif not citations:
        raise AnswerDomainValidationError(
            "ANSWER_COMPLETE_WITHOUT_CITATION", "answer",
            "非证据不足的回答必须在正文引用已提供的证据。", "complete_answer_without_citation",
            "A non-insufficient answer must cite supplied Evidence in its text."
        )
    else:
        # 完整回答不应额外声称“覆盖不完整”；忽略模型的矛盾附加字段。
        coverage_note = None

    if not partial_answer:
        answer = _remove_coverage_note_section(answer)
    # 模型误把“部分覆盖”写成 insufficient 时，compose 阶段尚未有 coverage_note；
    # 在这里补上程序生成的可见边界说明，保证返回字段与 answer 正文一致。
    if coverage_note and "### 证据覆盖说明" not in answer:
        answer = f"{answer}\n\n### 证据覆盖说明\n{coverage_note}"
    return AnswerGenerationResult(
        answer=answer,
        citations=citations,
        insufficient_evidence=insufficient_evidence,
        partial_answer=partial_answer,
        coverage_note=coverage_note,
        requirement_coverage=result.requirement_coverage,
    )


def _compose_answer(draft: AnswerGenerationDraft) -> AnswerGenerationResult:
    """把 LLM 的三段内容固定排版，避免模型忽略单个字符串内的小标题要求。"""
    coverage_note = (draft.coverage_note or "").strip()
    answer = _compose_body(draft.direct_answer, draft.evidence_explanation, draft.reading_interpretation)
    if coverage_note:
        answer += "\n\n### 证据覆盖说明\n" + coverage_note
    return AnswerGenerationResult(
        answer=answer, citations=draft.citations, insufficient_evidence=draft.insufficient_evidence,
        partial_answer=draft.partial_answer, coverage_note=coverage_note or None,
    )


def _compose_body(direct_answer: str, evidence_explanation: str, reading_interpretation: str | None) -> str:
    """Shared text formatting, independent of either provider bookkeeping DTO."""
    direct_answer = direct_answer.strip()
    evidence_explanation = evidence_explanation.strip()
    interpretation = (reading_interpretation or "").strip()
    if not direct_answer or not evidence_explanation:
        field = "direct_answer" if not direct_answer else "evidence_explanation"
        raise AnswerDomainValidationError(
            "ANSWER_REQUIRED_CONTENT_BLANK", field,
            "回答草稿的必填内容去除空白后为空。", "required_content_blank",
            "Direct answer and evidence explanation must each be non-empty after stripping."
        )
    parts = [
        f"### 直接回答\n{direct_answer}",
        f"### 论文中的依据与推导\n{evidence_explanation}",
    ]
    if interpretation:
        parts.append(f"### 如何理解\n{interpretation}")
    return "\n\n".join(parts)


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


_DEFAULT_PARTIAL_COVERAGE_NOTE = "当前证据支持上述核心回答，但未覆盖问题涉及的全部细节或范围。"


def _extract_in_text_citations(answer: str) -> list[str]:
    """按正文首次出现顺序提取引用，作为 API 的唯一 citations 来源。"""
    return _unique_nonempty_strings(re.findall(r"\[((?:E|REF)\d+)\]", answer))


def _remove_coverage_note_section(answer: str) -> str:
    """移除与最终状态矛盾的、由 compose 固定追加在末尾的覆盖说明。"""
    return re.sub(r"\n\n### 证据覆盖说明\n.*\Z", "", answer, flags=re.DOTALL)


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


def _to_outcome(
    result: AnswerGenerationResult,
    *,
    status: str,
    llm_diagnostics: tuple[LLMCallRecord, ...] = (),
) -> AnswerGenerationOutcome:
    """将 Pydantic 对象冻结为服务层稳定返回值，避免调用方意外修改结果。"""
    return AnswerGenerationOutcome(
        status=status,
        answer=result.answer,
        citations=tuple(result.citations),
        insufficient_evidence=result.insufficient_evidence,
        partial_answer=result.partial_answer,
        coverage_note=result.coverage_note,
        requirement_coverage=result.requirement_coverage,
        llm_diagnostics=llm_diagnostics,
    )
