"""ModelScope OpenAI 兼容 LLM Client：读取本机 .env，不在业务模块散落 API 调用。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv


class LLMConfigurationError(RuntimeError):
    """.env 缺少必要项或数值格式不正确时抛出；错误消息永不包含 Token。"""


class LLMRequestError(RuntimeError):
    """远程 LLM 调用超时、失败或返回空文本时抛出，可由检索层安全降级。"""


@dataclass(frozen=True)
class LLMRuntimeSettings:
    """只在运行时从 .env 读取的敏感/部署配置，不进入 config.yaml。"""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    max_tokens: int


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
    return LLMRuntimeSettings(
        api_key=required["LLM_API_KEY"],
        base_url=required["LLM_BASE_URL"].rstrip("/"),
        model=required["LLM_MODEL"],
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
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
        """使用 OpenAI 兼容 ``response_format=json_schema`` 请求原生结构化结果。

        ``json_schema`` 由 Query Rewrite 的 Pydantic 模型生成；Client 只负责将其交给 API
        并取回最终文本，不在此处解释任何业务字段。即使供应商承诺遵守 Schema，调用方仍会
        再执行 Pydantic 校验，避免把外部响应直接信任为内部业务对象。
        """
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                # 原生结构化输出：API 层先约束字段、类型及禁止额外字段的 JSON Schema。
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        # Schema 名称：仅供 API 标识此结构，不参与模型业务语义。
                        "name": schema_name,
                        # strict=True：要求供应商按 JSON Schema 严格生成。
                        "strict": True,
                        # Schema 内容：由调用方的 Pydantic 模型导出。
                        "schema": json_schema,
                    },
                },
                stream=False,
            )
            content = _extract_non_stream_content(response)
        except LLMRequestError:
            raise
        except Exception as error:
            raise LLMRequestError(
                f"LLM structured request failed: {type(error).__name__}"
            ) from error
        return content.strip()


def _extract_non_stream_content(response: Any) -> str:
    """从标准非流式 Chat Completion 中安全提取最终答案，不读取 reasoning_content。

    ``reasoning_content`` 可能包含模型内部推理，既不是 Query 改写的业务输入，也不应进入
    用户可见结果或日志。这里仅接受最终 ``message.content``；服务端返回空 choices 时给出
    明确错误，供健康检查或正式检索的降级逻辑处理。
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise LLMRequestError("LLM response contains no choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise LLMRequestError("LLM response contains no final text content.")
    return content
