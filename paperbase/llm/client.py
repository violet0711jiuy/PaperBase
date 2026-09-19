"""ModelScope OpenAI 兼容 LLM Client：读取本机 .env，不在业务模块散落 API 调用。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv


class LLMConfigurationError(RuntimeError):
    """.env 缺少必要项或数值格式不正确时抛出；错误消息永不包含 Token。"""


class LLMRequestError(RuntimeError):
    """LLM 调用失败；只携带可持久化的安全诊断，不保存请求或响应正文。"""

    def __init__(
        self,
        message: str,
        *,
        error_stage: str = "request_error",
        finish_reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        content_chars: int = 0,
        reasoning_chars: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_stage = error_stage
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.content_chars = content_chars
        self.reasoning_chars = reasoning_chars


@dataclass(frozen=True)
class LLMCompletion:
    """一次非流式补全及其安全元数据；正文只在当前调用栈中使用。"""

    content: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0

    @property
    def content_chars(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class LLMCallRecord:
    """可安全持久化的调用诊断，不包含提示词、正文或隐藏推理。"""

    node: str
    attempt: int
    success: bool
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    content_chars: int = 0
    reasoning_chars: int = 0
    error_stage: str | None = None
    recovery_action: str | None = None
    validation_summary: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.node, str) or not self.node.strip():
            raise ValueError("node must be a non-empty string.")
        object.__setattr__(self, "node", " ".join(self.node.split()))
        if self.attempt < 1:
            raise ValueError("attempt must be positive.")
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "content_chars",
            "reasoning_chars",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative.")


@dataclass(frozen=True)
class LLMRuntimeSettings:
    """只在运行时从 .env 读取的敏感/部署配置，不进入 config.yaml。"""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int
    structured_output_mode: str = "json_schema"


@runtime_checkable
class ChatCompletionClient(Protocol):
    """最小聊天补全接口，便于 Query 改写与后续回答生成复用及单元测试替身注入。"""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """返回模型的最终文本；调用方负责其各自的 JSON 或回答契约。"""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        """返回满足 API JSON Schema 输出约束的原始 JSON 文本，供调用方再做 Pydantic 校验。"""


def complete_json_with_diagnostics(
    client: ChatCompletionClient,
    *,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any],
    schema_name: str,
) -> LLMCompletion:
    """优先读取诊断接口；旧测试替身仍可只实现 ``complete_json``。"""
    diagnostic_call = getattr(client, "complete_json_diagnostic", None)
    if callable(diagnostic_call):
        return diagnostic_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            schema_name=schema_name,
        )
    content = client.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=json_schema,
        schema_name=schema_name,
    )
    if not isinstance(content, str) or not content.strip():
        raise LLMRequestError(
            "LLM response contains no final text content.",
            error_stage="empty_content",
            content_chars=len(content) if isinstance(content, str) else 0,
        )
    return LLMCompletion(content=content.strip())


def load_llm_runtime_settings(env_path: Path) -> LLMRuntimeSettings:
    """加载指定项目根目录的 .env，并校验需要的 ModelScope/OpenAI 兼容配置。"""
    if not env_path.is_file():
        raise LLMConfigurationError(
            f"LLM environment file not found: {env_path}. Copy .env.example to .env first."
        )
    load_dotenv(dotenv_path=env_path, override=False)
    required = {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", "").strip(),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").strip(),
        "LLM_MODEL": os.getenv("LLM_MODEL", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise LLMConfigurationError(
            "Missing required LLM environment variables: " + ", ".join(missing)
        )
    try:
        timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "500"))
    except ValueError as error:
        raise LLMConfigurationError(
            "LLM_TIMEOUT_SECONDS, LLM_TEMPERATURE, and LLM_MAX_TOKENS must be numeric."
        ) from error
    if timeout_seconds <= 0 or not 0 <= temperature <= 2 or max_tokens < 1:
        raise LLMConfigurationError("LLM runtime numeric settings are outside valid ranges.")
    structured_output_mode = os.getenv(
        "LLM_STRUCTURED_OUTPUT_MODE", "json_schema"
    ).strip().casefold()
    if structured_output_mode not in {"json_schema", "json_object"}:
        raise LLMConfigurationError(
            "LLM_STRUCTURED_OUTPUT_MODE must be json_schema or json_object."
        )
    return LLMRuntimeSettings(
        api_key=required["LLM_API_KEY"],
        base_url=required["LLM_BASE_URL"].rstrip("/"),
        model=required["LLM_MODEL"],
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        structured_output_mode=structured_output_mode,
    )


class OpenAICompatibleChatClient:
    """对 ModelScope 等 OpenAI 兼容 Chat Completions API 的非流式窄封装。

    检索改写和后续回答层都只需要完整结果，而不是逐 token 推送；因此客户端固定使用
    ``stream=False``，只读取服务端的最终 ``message.content``，不把中间推理过程暴露给用户。
    """

    def __init__(self, settings: LLMRuntimeSettings) -> None:
        self._settings = settings
        try:
            from openai import OpenAI
        except ImportError as error:
            raise LLMConfigurationError(
                "The openai package is required for the OpenAI-compatible LLM client."
            ) from error
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    def with_max_tokens(self, max_tokens: int) -> "OpenAICompatibleChatClient":
        """复用同一 SDK 连接池，创建只覆盖输出预算的轻量调用视图。"""
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        clone = object.__new__(type(self))
        clone._settings = replace(self._settings, max_tokens=max_tokens)
        clone._client = self._client
        return clone

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """发起非流式短请求；Query 改写只消费最终 content，不记录问题或响应到磁盘。"""
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                stream=False,
            )
            content = _extract_non_stream_content(response)
        except LLMRequestError:
            raise
        except Exception as error:
            # 不把底层异常文本向上原样传播，避免某些 SDK/代理错误中意外包含请求凭证。
            raise LLMRequestError(f"LLM request failed: {type(error).__name__}") from error
        return content.strip()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        """按模型能力使用严格 JSON Schema 或兼容 JSON Object 结构化输出。

        ``json_schema`` 由 Query Rewrite 的 Pydantic 模型生成；Client 只负责将其交给 API
        并取回最终文本，不在此处解释任何业务字段。即使供应商承诺遵守 Schema，调用方仍会
        再执行 Pydantic 校验，避免把外部响应直接信任为内部业务对象。
        """
        return self.complete_json_diagnostic(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            schema_name=schema_name,
        ).content

    def complete_json_diagnostic(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> LLMCompletion:
        """执行结构化补全并返回 finish reason、usage 与字符数诊断。"""
        structured_system_prompt = system_prompt
        if self._settings.structured_output_mode == "json_object":
            structured_system_prompt = (
                f"{system_prompt}\n\n"
                "你必须只输出一个 JSON 对象，且严格满足以下 JSON Schema；"
                "不要输出 Markdown 代码块或额外说明：\n"
                f"{json.dumps(json_schema, ensure_ascii=False, sort_keys=True)}"
            )
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            }
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": structured_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                # 原生结构化输出：API 层先约束字段、类型及禁止额外字段的 JSON Schema。
                response_format=response_format,
                stream=False,
            )
            completion = _extract_non_stream_completion(response)
        except LLMRequestError:
            raise
        except Exception as error:
            raise LLMRequestError(
                f"LLM structured request failed: {type(error).__name__}"
            ) from error
        return completion


def _extract_non_stream_content(response: Any) -> str:
    """从标准非流式 Chat Completion 中安全提取最终答案，不读取 reasoning_content。

    ``reasoning_content`` 可能包含模型内部推理，既不是 Query 改写的业务输入，也不应进入
    用户可见结果或日志。这里仅接受最终 ``message.content``；服务端返回空 choices 时给出
    明确错误，供健康检查或正式检索的降级逻辑处理。
    """
    return _extract_non_stream_completion(response).content


def _extract_non_stream_completion(response: Any) -> LLMCompletion:
    """提取最终内容和安全统计；隐藏推理只计算字符数，不读取或持久化文本。"""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise LLMRequestError("LLM response contains no choices.")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
    finish_reason = getattr(choice, "finish_reason", None)
    finish_reason = finish_reason if isinstance(finish_reason, str) else None
    usage = getattr(response, "usage", None)
    prompt_tokens = _optional_nonnegative_int(getattr(usage, "prompt_tokens", None))
    completion_tokens = _optional_nonnegative_int(
        getattr(usage, "completion_tokens", None)
    )
    if not isinstance(content, str) or not content.strip():
        raise LLMRequestError(
            "LLM response contains no final text content.",
            error_stage="empty_content",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            content_chars=len(content) if isinstance(content, str) else 0,
            reasoning_chars=reasoning_chars,
        )
    return LLMCompletion(
        content=content.strip(),
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_chars=reasoning_chars,
    )


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None
