"""LLM 调用的安全诊断、错误分类与恢复策略。"""

from __future__ import annotations

from pydantic import ValidationError

from .client import LLMCallRecord, LLMCompletion, LLMRequestError


def classify_llm_error(error: Exception) -> str:
    """把外部/语法/Schema/领域失败分开，避免上层重复同一种请求。"""
    if isinstance(error, LLMRequestError):
        return error.error_stage
    if isinstance(error, ValidationError):
        if any(item.get("type") == "json_invalid" for item in error.errors()):
            return "invalid_json"
        return "schema_validation"
    if isinstance(error, ValueError):
        return "domain_validation"
    if isinstance(error, TypeError):
        return "schema_validation"
    return "request_error"


def recovery_action_for(error: Exception, *, can_retry: bool) -> str | None:
    """根据错误类别决定下一次输入如何变化；无重试额度时明确停止。"""
    if not can_retry:
        return None
    stage = classify_llm_error(error)
    if (
        stage == "empty_content"
        and isinstance(error, LLMRequestError)
        and error.finish_reason == "length"
    ):
        return "compact_context"
    if stage in {"invalid_json", "schema_validation"}:
        return "repair_structure"
    if stage == "domain_validation":
        return "repair_domain"
    if stage == "request_error":
        return "retry_request"
    return None


def repair_hint(error: Exception) -> str:
    """只把分类后的修复提示交回模型，不回灌可能很长的原始异常。"""
    stage = classify_llm_error(error)
    hints = {
        "empty_content": "上一次没有返回最终 JSON；请直接输出完整 JSON 对象。",
        "invalid_json": "上一次输出不是合法 JSON；请修复 JSON 语法。",
        "schema_validation": "上一次 JSON 不符合字段或类型约束；请严格按 Schema 重写。",
        "domain_validation": "上一次输出违反业务范围或完整性约束；请根据输入 ID 与范围修正。",
        "request_error": "上一次请求未成功；请重新完成同一任务。",
    }
    return hints.get(stage, "上一次输出不可用；请严格按约束重写。")


def safe_validation_summary(error: Exception) -> str | None:
    """仅保留 ValidationError 的类型与字段路径，不保存 input_value。"""
    if not isinstance(error, ValidationError):
        return None
    parts: list[str] = []
    for item in error.errors()[:3]:
        location = ".".join(str(value) for value in item.get("loc", ())) or "root"
        parts.append(f"{item.get('type', 'validation_error')}@{location}")
    return ";".join(parts)[:300] or "validation_error"


def success_record(
    *, node: str, attempt: int, completion: LLMCompletion
) -> LLMCallRecord:
    return LLMCallRecord(
        node=node,
        attempt=attempt,
        success=True,
        finish_reason=completion.finish_reason,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        content_chars=completion.content_chars,
        reasoning_chars=completion.reasoning_chars,
    )


def failure_record(
    *,
    node: str,
    attempt: int,
    error: Exception,
    completion: LLMCompletion | None = None,
    can_retry: bool,
) -> LLMCallRecord:
    """从异常或已返回 completion 中提取安全统计，不保存错误正文。"""
    if isinstance(error, LLMRequestError):
        finish_reason = error.finish_reason
        prompt_tokens = error.prompt_tokens
        completion_tokens = error.completion_tokens
        content_chars = error.content_chars
        reasoning_chars = error.reasoning_chars
    else:
        finish_reason = completion.finish_reason if completion else None
        prompt_tokens = completion.prompt_tokens if completion else None
        completion_tokens = completion.completion_tokens if completion else None
        content_chars = completion.content_chars if completion else 0
        reasoning_chars = completion.reasoning_chars if completion else 0
    return LLMCallRecord(
        node=node,
        attempt=attempt,
        success=False,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        content_chars=content_chars,
        reasoning_chars=reasoning_chars,
        error_stage=classify_llm_error(error),
        recovery_action=recovery_action_for(error, can_retry=can_retry),
        validation_summary=safe_validation_summary(error),
    )
