"""LLM provider 接入层：读取本机 .env，不在业务模块散落 SDK 调用。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv
from .answer_output_capture import record_answer_response


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
    """一次非流式补全；独立 case capture 可保存最终正文，诊断 DTO 不保存。"""

    content: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0

    @property
    def content_chars(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class ValidationDiagnostic:
    """Safe, deterministic structured-output validation feedback.

    The object intentionally describes the failed constraint rather than
    preserving model output, prompts, source documents, or hidden reasoning.
    """

    node: str
    attempt: int
    error_stage: str
    error_code: str
    field_path: str
    validator_name: str
    safe_reason: str
    expected_constraint: str
    actual_summary: str
    recovery_action: str | None
    requirement_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "node",
            "error_stage",
            "error_code",
            "field_path",
            "validator_name",
            "safe_reason",
            "expected_constraint",
            "actual_summary",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, " ".join(value.split()))
        if self.attempt < 1:
            raise ValueError("attempt must be positive.")


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
    validation_diagnostics: tuple[ValidationDiagnostic, ...] = ()
    requested_output_tokens: int | None = None
    # Counts/sizes only, never the derived view or a second business-state copy.
    decision_iteration: int | None = None
    decision_view_metrics: dict[str, int | float] | None = None

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
        object.__setattr__(self, "validation_diagnostics", tuple(self.validation_diagnostics))


@dataclass(frozen=True)
class LLMRuntimeSettings:
    """只在运行时从 .env 读取的敏感/部署配置，不进入 config.yaml。"""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int | None
    structured_output_mode: str = "json_schema"
    strict_schema_required_fields: bool = False
    # ``zhipuai`` 使用智谱官方 Python SDK；保留旧值以兼容已有 OpenAI-compatible 部署。
    provider: str = "openai_compatible"
    # 当前 PaperBase 所有节点都要求直接输出，避免推理过程耗尽结构化节点的超时预算。
    thinking_enabled: bool = False
    # Some providers need the Schema repeated in text. Nodes using native
    # strict constrained decoding can disable this duplicate prompt copy.
    schema_in_system_prompt: bool = True


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
    """加载指定项目根目录的 .env，并按 provider 校验运行时配置。"""
    if not env_path.is_file():
        raise LLMConfigurationError(
            f"LLM environment file not found: {env_path}. Copy .env.example to .env first."
        )
    load_dotenv(dotenv_path=env_path, override=False)
    provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip().casefold()
    if provider not in {"openai_compatible", "zhipuai"}:
        raise LLMConfigurationError(
            "LLM_PROVIDER must be openai_compatible or zhipuai."
        )
    if provider == "zhipuai":
        required = {
            # 不回退到原供应商 Token，避免把 ModelScope/其他 Provider 的凭证误发给智谱。
            "ZHIPUAI_API_KEY": os.getenv("ZHIPUAI_API_KEY", "").strip(),
            "LLM_MODEL": os.getenv("LLM_MODEL", "glm-4.5-air").strip(),
        }
        base_url = os.getenv(
            "ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        ).strip()
    else:
        required = {
            "LLM_API_KEY": os.getenv("LLM_API_KEY", "").strip(),
            "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").strip(),
            "LLM_MODEL": os.getenv("LLM_MODEL", "").strip(),
        }
        base_url = required["LLM_BASE_URL"]
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
    if provider == "zhipuai" and structured_output_mode != "json_object":
        raise LLMConfigurationError(
            "ZhipuAI official SDK requires LLM_STRUCTURED_OUTPUT_MODE=json_object."
        )
    thinking_raw = os.getenv("LLM_THINKING_ENABLED", "false").strip().casefold()
    if thinking_raw not in {"true", "false"}:
        raise LLMConfigurationError("LLM_THINKING_ENABLED must be true or false.")
    return LLMRuntimeSettings(
        api_key=required["ZHIPUAI_API_KEY"] if provider == "zhipuai" else required["LLM_API_KEY"],
        base_url=base_url.rstrip("/"),
        model=required["LLM_MODEL"],
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        structured_output_mode=structured_output_mode,
        provider=provider,
        thinking_enabled=thinking_raw == "true",
    )


def create_chat_completion_client(settings: LLMRuntimeSettings) -> ChatCompletionClient:
    """按明确 provider 创建客户端，业务层不得直接选择供应商 SDK。"""
    if settings.provider == "zhipuai":
        return ZhipuAIChatClient(settings)
    return OpenAICompatibleChatClient(settings)


def structured_client_view(
    client: ChatCompletionClient, *, temperature: float = 0,
    structured_output_mode: str | None = None,
) -> ChatCompletionClient:
    """Independent schema-critical view; omit application output caps.

    Small injected test clients need not implement runtime configuration.
    Production clients implement each immutable view operation.
    """
    for method, value in (("with_max_tokens", None), ("with_temperature", temperature),
                          ("with_structured_output_mode", structured_output_mode)):
        configure = getattr(client, method, None)
        if callable(configure) and (value is not None or method == "with_max_tokens"):
            client = configure(value)
    return client


def _strict_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Materialize defaulted fields on the wire, without changing DTO semantics.

    Native strict decoding must also request support_candidates/coverage fields
    that Pydantic accepts as omitted defaults. Nullable values remain nullable.
    """
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(item) for key, item in value.items() if key != "default"}
        if result.get("type") == "object" and "properties" in result:
            result["required"] = list(result["properties"])
            result["additionalProperties"] = False
        return result
    return visit(schema)


class OpenAICompatibleChatClient:
    """对 ModelScope 等 OpenAI 兼容 Chat Completions API 的非流式窄封装。

    检索改写和后续回答层都只需要完整结果，而不是逐 token 推送；因此客户端固定使用
    ``stream=False``，只读取服务端的最终 ``message.content``，不把中间推理过程暴露给用户。
    """

    def __init__(self, settings: LLMRuntimeSettings) -> None:
        self._settings = settings
        # Qwen reasoning can consume an entire short-node timeout before the
        # final content arrives.  PaperBase requires final, parseable output
        # from every LLM call, so direct/instruct mode is the global default.
        self._extra_body: dict[str, Any] = {"enable_thinking": False}
        self._sampling_parameters: dict[str, float] = {}
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

    def with_max_tokens(self, max_tokens: int | None) -> "OpenAICompatibleChatClient":
        """复用同一 SDK 连接池，创建只覆盖输出预算的轻量调用视图。"""
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        clone = object.__new__(type(self))
        clone._settings = replace(self._settings, max_tokens=max_tokens)
        clone._client = self._client
        clone._extra_body = dict(getattr(self, "_extra_body", {}))
        clone._sampling_parameters = dict(getattr(self, "_sampling_parameters", {}))
        return clone

    def with_temperature(self, temperature: float) -> "OpenAICompatibleChatClient":
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._settings = replace(clone._settings, temperature=temperature)
        return clone

    def with_structured_output_mode(self, mode: str) -> "OpenAICompatibleChatClient":
        if mode not in {"json_object", "json_schema"}:
            raise ValueError("Invalid structured output mode.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._settings = replace(clone._settings, structured_output_mode=mode,
                                  strict_schema_required_fields=mode == "json_schema")
        return clone

    def with_schema_prompt_injection(
        self, enabled: bool
    ) -> "OpenAICompatibleChatClient":
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._settings = replace(
            clone._settings, schema_in_system_prompt=enabled
        )
        return clone

    def with_timeout_seconds(
        self, timeout_seconds: float
    ) -> "OpenAICompatibleChatClient":
        """复用连接池并为特定节点覆盖请求超时。"""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        clone = object.__new__(type(self))
        clone._settings = replace(self._settings, timeout_seconds=timeout_seconds)
        clone._client = self._client
        clone._extra_body = dict(getattr(self, "_extra_body", {}))
        clone._sampling_parameters = dict(getattr(self, "_sampling_parameters", {}))
        return clone

    def with_transport_max_retries(self, max_retries: int) -> "OpenAICompatibleChatClient":
        """Independent SDK facade sharing HTTP resources, never mutating sibling nodes."""
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._client = self._client.with_options(max_retries=max_retries)
        return clone

    def with_repair_temperature(
        self, minimum_temperature: float = 0.2
    ) -> "OpenAICompatibleChatClient":
        """Return a bounded-diversity view for a validated repair attempt."""
        if not 0 <= minimum_temperature <= 2:
            raise ValueError("minimum_temperature must be between 0 and 2.")
        clone = object.__new__(type(self))
        clone._settings = replace(
            self._settings,
            temperature=max(self._settings.temperature, minimum_temperature),
        )
        clone._client = self._client
        clone._extra_body = dict(getattr(self, "_extra_body", {}))
        clone._sampling_parameters = dict(getattr(self, "_sampling_parameters", {}))
        return clone

    def with_sampling_parameters(
        self, *, top_p: float, presence_penalty: float, top_k: int,
        min_p: float, repetition_penalty: float,
    ) -> "OpenAICompatibleChatClient":
        """Explicit sampling view; unsupported provider parameters are not dropped."""
        if not 0 < top_p <= 1 or not -2 <= presence_penalty <= 2:
            raise ValueError("Invalid top_p or presence_penalty.")
        if top_k < 1 or not 0 <= min_p <= 1 or repetition_penalty <= 0:
            raise ValueError("Invalid extended sampling parameters.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._sampling_parameters = {"top_p": top_p, "presence_penalty": presence_penalty}
        clone._extra_body = {**clone._extra_body, "top_k": top_k, "min_p": min_p,
                             "repetition_penalty": repetition_penalty}
        return clone

    def with_thinking_enabled(self, enabled: bool = True) -> "OpenAICompatibleChatClient":
        """Independent Qwen-compatible thinking switch, sharing the SDK pool."""
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._settings = replace(clone._settings, thinking_enabled=enabled)
        clone._extra_body = {**clone._extra_body, "enable_thinking": enabled}
        return clone

    def with_thinking_disabled(self) -> "OpenAICompatibleChatClient":
        return self.with_thinking_enabled(False)

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """发起非流式短请求；Query 改写只消费最终 content，不记录问题或响应到磁盘。"""
        try:
            request: dict[str, Any] = {
                "model": self._settings.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._settings.temperature,
                "timeout": self._settings.timeout_seconds,
                "stream": False,
            }
            request.update(getattr(self, "_sampling_parameters", {}))
            if self._settings.max_tokens is not None:
                request["max_tokens"] = self._settings.max_tokens
            if getattr(self, "_extra_body", None):
                request["extra_body"] = self._extra_body
            response = self._client.chat.completions.create(**request)
            content = _extract_non_stream_content(response)
        except LLMRequestError:
            raise
        except Exception as error:
            # 不把底层异常文本向上原样传播，避免某些 SDK/代理错误中意外包含请求凭证。
            raise LLMRequestError(
                f"LLM request failed: {type(error).__name__}",
                error_stage=_request_error_stage(error),
            ) from error
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
            wire_schema = (_strict_response_schema(json_schema)
                           if self._settings.strict_schema_required_fields else json_schema)
            if (self._settings.strict_schema_required_fields
                    and self._settings.schema_in_system_prompt):
                # Constrained decoding is not a substitute for showing the
                # model the field contract (notably defaulted support fields).
                structured_system_prompt = (
                    f"{system_prompt}\n\n"
                    "只输出符合以下 JSON Schema 的完整 JSON 对象，不输出 Markdown 或额外说明：\n"
                    + json.dumps(wire_schema, ensure_ascii=False, sort_keys=True)
                )
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": wire_schema,
                },
            }
        try:
            request: dict[str, Any] = {
                "model": self._settings.model,
                "messages": [
                    {"role": "system", "content": structured_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._settings.temperature,
                "response_format": response_format,
                "timeout": self._settings.timeout_seconds,
                "stream": False,
            }
            request.update(getattr(self, "_sampling_parameters", {}))
            if self._settings.max_tokens is not None:
                request["max_tokens"] = self._settings.max_tokens
            if getattr(self, "_extra_body", None):
                request["extra_body"] = self._extra_body
            response = self._client.chat.completions.create(**request)
            record_answer_response(schema_name=schema_name, request=request, response=response)
            completion = _extract_non_stream_completion(response)
        except LLMRequestError:
            raise
        except Exception as error:
            raise LLMRequestError(
                f"LLM structured request failed: {type(error).__name__}",
                error_stage=_request_error_stage(error),
            ) from error
        return completion


class ZhipuAIChatClient:
    """智谱官方 ``zhipuai.ZhipuAI`` SDK 的非流式 PaperBase 适配器。

    业务层继续只依赖 :class:`ChatCompletionClient`。GLM-4.5 的 hidden reasoning
    由 ``thinking`` 请求字段明确关闭，绝不会写入 Prompt、审计或最终答案。
    """

    def __init__(self, settings: LLMRuntimeSettings) -> None:
        if settings.provider != "zhipuai":
            raise LLMConfigurationError(
                "ZhipuAIChatClient requires LLM_PROVIDER=zhipuai."
            )
        self._settings = settings
        self._thinking: dict[str, str] = {
            "type": "enabled" if settings.thinking_enabled else "disabled"
        }
        try:
            from zhipuai import ZhipuAI
        except ImportError as error:
            raise LLMConfigurationError(
                "The zhipuai package is required for LLM_PROVIDER=zhipuai."
            ) from error
        # ZhipuAI 是官方 SDK；base_url 保留为可配置项，默认值是官方开放平台端点。
        self._client = ZhipuAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    def with_max_tokens(self, max_tokens: int | None) -> "ZhipuAIChatClient":
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        clone = object.__new__(type(self))
        clone._settings = replace(self._settings, max_tokens=max_tokens)
        clone._client = self._client
        clone._thinking = dict(self._thinking)
        return clone

    def with_temperature(self, temperature: float) -> "ZhipuAIChatClient":
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        clone = self.with_max_tokens(self._settings.max_tokens)
        clone._settings = replace(clone._settings, temperature=temperature)
        return clone

    def with_structured_output_mode(self, mode: str) -> "ZhipuAIChatClient":
        if mode != "json_object":
            raise ValueError("ZhipuAI client supports json_object only.")
        return self.with_max_tokens(self._settings.max_tokens)

    def with_timeout_seconds(self, timeout_seconds: float) -> "ZhipuAIChatClient":
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        clone = object.__new__(type(self))
        clone_settings = replace(self._settings, timeout_seconds=timeout_seconds)
        # timeout 是官方 SDK Client 的构造参数；不能只改本地 dataclass 而继续复用旧 Client。
        return type(self)(clone_settings)._with_thinking(self._thinking)

    def with_repair_temperature(
        self, minimum_temperature: float = 0.2
    ) -> "ZhipuAIChatClient":
        """Return a bounded-diversity view for a validated repair attempt."""
        if not 0 <= minimum_temperature <= 2:
            raise ValueError("minimum_temperature must be between 0 and 2.")
        clone = object.__new__(type(self))
        clone._settings = replace(
            self._settings,
            temperature=max(self._settings.temperature, minimum_temperature),
        )
        clone._client = self._client
        clone._thinking = dict(self._thinking)
        return clone

    def with_thinking_disabled(self) -> "ZhipuAIChatClient":
        clone = object.__new__(type(self))
        clone._settings = replace(self._settings, thinking_enabled=False)
        clone._client = self._client
        clone._thinking = {"type": "disabled"}
        return clone

    def _with_thinking(self, thinking: dict[str, str]) -> "ZhipuAIChatClient":
        self._thinking = dict(thinking)
        return self

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        completion = self._request_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=None,
        )
        return completion.content

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
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
        # GLM 官方 Chat Completions 使用 json_object；完整 Schema 放入系统提示，
        # 随后仍由每个业务节点的 Pydantic DTO 进行本地严格校验。
        structured_system_prompt = (
            f"{system_prompt}\n\n"
            "你必须只输出一个 JSON 对象，且严格满足以下 JSON Schema；"
            "不要输出 Markdown 代码块或额外说明：\n"
            f"{json.dumps(json_schema, ensure_ascii=False, sort_keys=True)}"
        )
        return self._request_completion(
            system_prompt=structured_system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
            schema_name=schema_name,
        )

    def _request_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, str] | None,
        schema_name: str | None = None,
    ) -> LLMCompletion:
        try:
            request: dict[str, Any] = {
                "model": self._settings.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._settings.temperature,
                "stream": False,
                "thinking": dict(self._thinking),
            }
            if self._settings.max_tokens is not None:
                request["max_tokens"] = self._settings.max_tokens
            if response_format is not None:
                request["response_format"] = response_format
            response = self._client.chat.completions.create(**request)
            record_answer_response(schema_name=schema_name, request=request, response=response)
            return _extract_non_stream_completion(response)
        except LLMRequestError:
            raise
        except Exception as error:
            raise LLMRequestError(
                f"ZhipuAI request failed: {type(error).__name__}",
                error_stage=_request_error_stage(error),
            ) from error


def _request_error_stage(error: Exception) -> str:
    """只记录安全错误类别，不持久化供应商响应正文或凭证。"""
    name = type(error).__name__.casefold()
    status_code = getattr(error, "status_code", None)
    if "timeout" in name:
        return "timeout"
    if status_code == 429 or any(
        marker in name for marker in ("ratelimit", "rate_limit", "reachlimit")
    ):
        return "rate_limit"
    if isinstance(status_code, int) and status_code >= 500:
        return "provider_5xx"
    if any(value in name for value in ("connection", "connect", "network")):
        return "connection_error"
    return "request_error"


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
