"""将用户问题改写为受约束的多路检索查询，而不生成问题答案。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbase.config import QueryRewriteSettings
from paperbase.prompts.query_rewrite import (
    QUERY_REWRITE_JSON_REPAIR_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_query_rewrite_user_prompt,
)
from paperbase.llm.client import (
    ChatCompletionClient,
    LLMRequestError,
    OpenAICompatibleChatClient,
    load_llm_runtime_settings,
)


class QueryRewriteError(RuntimeError):
    """保留给调用方显式使用的 Query Rewrite 领域异常类型。"""


class QueryRewriteResult(BaseModel):
    """LLM Query Rewrite 的固定结构化输出契约。

    API 层会使用此模型导出的 JSON Schema 约束输出；Python 层仍会用同一模型再次验证。
    两个字段均为必填，以便缺失字段、额外字段或错误类型不会静默混入检索流程。
    ``max_lexical_keywords_en`` 是运行时配置，不在此 Schema 中硬编码。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # 语义改写：只能是字符串或真正的 JSON null。
    semantic_query: str | None
    # 英文关键词组：只能是字符串列表；空列表代表没有有效关键词。
    lexical_keywords_en: list[str]
    # 是否需要查询参考文献索引；只有明确 citation/reference intent 才能为 true。
    search_bibliography: bool


@dataclass(frozen=True)
class QueryRewritePlan:
    """一次检索使用的改写计划。

    原始问题始终由检索器保留。LLM 只补充一条语义改写和一组英文关键词；英文关键词组
    会被合并为一条 BM25 查询，因此 LLM 不可用时仍能安全退化为原始稠密检索和原始 BM25。
    """

    original_query: str
    # 语义改写问句：用于 Rewritten Dense；无可用改写时为 None。
    semantic_query: str | None = None
    # 英文关键词组：用于编译一条 Rewritten BM25 的 OR 查询。
    lexical_keywords_en: tuple[str, ...] = ()
    # 引用检索开关：false 时严格不触碰 bibliography FTS5。
    search_bibliography: bool = False
    status: str = "success"


class QueryRewriter(Protocol):
    """可替换的 Query 规划接口，便于未来切换不同 LLM 或关闭服务。"""

    def rewrite(
        self,
        query: str,
        *,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewritePlan:
        """返回不含原始问题本身的补充检索查询。"""


class NoopQueryRewriter:
    """关闭 LLM 时使用的实现；保留统一的调用结构。"""

    def rewrite(
        self,
        query: str,
        *,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewritePlan:
        # 禁用 LLM 时不基于历史构造新检索词，始终保留原始问题路径。
        _ = conversation_context
        return QueryRewritePlan(original_query=_normalize_query(query), status="disabled")


class LLMQueryRewriter:
    """基于 OpenAI 兼容 LLM 的 JSON 改写器，带本地结构校验和安全降级。"""

    def __init__(self, *, settings: QueryRewriteSettings, client: ChatCompletionClient) -> None:
        self._settings = settings
        self._client = client

    def rewrite(
        self,
        query: str,
        *,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewritePlan:
        normalized_query = _normalize_query(query)
        normalized_context = _normalize_conversation_context(
            conversation_context,
            max_context_turns=self._settings.max_context_turns,
        )
        user_prompt = build_query_rewrite_user_prompt(
            query=normalized_query,
            conversation_context=normalized_context,
            max_lexical_keywords_en=self._settings.max_lexical_keywords_en,
        )
        raw_response: str | None = None
        try:
            raw_response = self._client.complete_json(
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                # Pydantic 导出的 JSON Schema：由 API 尽可能强制字段名、类型和 extra=forbid。
                json_schema=QueryRewriteResult.model_json_schema(),
                schema_name="query_rewrite_result",
            )
            result = normalize_query_rewrite(
                QueryRewriteResult.model_validate_json(raw_response),
                max_lexical_keywords_en=self._settings.max_lexical_keywords_en,
            )
            return QueryRewritePlan(
                original_query=normalized_query,
                semantic_query=result.semantic_query,
                lexical_keywords_en=tuple(result.lexical_keywords_en),
                search_bibliography=result.search_bibliography,
            )
        except ValidationError as error:
            # 原生 Schema 仍可能因供应商兼容差异或截断而得到无法通过 Pydantic 的文本；
            # 仅在已经获得原始文本时调用 JSON Repair，避免把网络/鉴权错误误当格式问题。
            try:
                repaired_result = self._repair_and_normalize(raw_response)
                return QueryRewritePlan(
                    original_query=normalized_query,
                    semantic_query=repaired_result.semantic_query,
                    lexical_keywords_en=tuple(repaired_result.lexical_keywords_en),
                    search_bibliography=repaired_result.search_bibliography,
                )
            except (LLMRequestError, ValidationError, ValueError, TypeError) as repair_error:
                return self._handle_failure(normalized_query, repair_error)
        except (LLMRequestError, ValueError, TypeError) as error:
            return self._handle_failure(normalized_query, error)

    def _repair_and_normalize(self, raw_response: str | None) -> QueryRewriteResult:
        """仅在第一次结构化校验失败时使用 JSON Repair，并再次执行相同 Schema 校验。"""
        if raw_response is None:
            raise ValueError("Cannot repair a missing LLM response.")
        repaired_response = self._client.complete_json(
            system_prompt=QUERY_REWRITE_JSON_REPAIR_SYSTEM_PROMPT,
            # 原始输出是 Repair 的待格式化数据；Repair Prompt 已限制不能重新理解或创造内容。
            user_prompt=raw_response,
            json_schema=QueryRewriteResult.model_json_schema(),
            schema_name="query_rewrite_result_repair",
        )
        return normalize_query_rewrite(
            QueryRewriteResult.model_validate_json(repaired_response),
            max_lexical_keywords_en=self._settings.max_lexical_keywords_en,
        )

    def _handle_failure(
        self,
        normalized_query: str,
        error: Exception,
    ) -> QueryRewritePlan:
        """改写与 Repair 均失败时无条件保留原始问题，绝不让 Query Rewrite 中断 Retriever。

        ``fallback_to_original`` 是历史配置字段；结构化输出阶段将 Query Rewrite 视为可选的
        召回增强能力，因此无论其旧配置值为何，外部 LLM、Schema 或 Repair 失败都只能降级，
        不能阻断后续 Original Dense / Original BM25 检索。
        """
        _ = error  # 失败原因由调用链保留；此处不向用户输出，避免外部服务异常泄露请求细节。
        return QueryRewritePlan(original_query=normalized_query, status="fallback")


def create_query_rewriter(
    *,
    settings: QueryRewriteSettings,
    env_path: Path,
    client: ChatCompletionClient | None = None,
) -> QueryRewriter:
    """按配置创建改写器；显式注入 client 仅用于测试或替换远程服务。"""
    if not settings.enabled:
        return NoopQueryRewriter()
    if client is None:
        client = OpenAICompatibleChatClient(load_llm_runtime_settings(env_path))
    return LLMQueryRewriter(settings=settings, client=client)


def _normalize_query(query: str) -> str:
    """折叠换行和多余空格，防止空问题及重复问题进入不同召回通道。"""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("Retrieval query must not be empty.")
    return normalized


def _normalize_conversation_context(
    conversation_context: Sequence[str] | None,
    *,
    max_context_turns: int,
) -> tuple[str, ...]:
    """规范化并截取最近历史，确保 Prompt 接收的是少量、稳定的上下文数据。

    调用方应只传入与当前检索有关的近期用户问题或已确认的助手事实，不能传入
    未校验长文档、系统提示或无关聊天记录。此函数不重新解释历史语义，只做空白
    清理、空项删除和按配置保留最近 N 条。
    """
    if max_context_turns < 0:
        raise ValueError("max_context_turns must not be negative.")
    if not conversation_context or max_context_turns == 0:
        return ()
    cleaned = tuple(
        " ".join(item.split())
        for item in conversation_context
        if item and item.strip()
    )
    return cleaned[-max_context_turns:]


def normalize_query_rewrite(
    result: QueryRewriteResult,
    *,
    max_lexical_keywords_en: int,
) -> QueryRewriteResult:
    """在 Schema 校验后执行确定性清洗，不依赖 Prompt 保证格式细节。

    - ``semantic_query``：strip 后空字符串变为 None；
    - ``lexical_keywords_en``：strip、删除空项、按首次出现顺序去重、按运行时配置截断。
    """
    if max_lexical_keywords_en < 1:
        raise ValueError("max_lexical_keywords_en must be positive.")
    semantic_query = result.semantic_query.strip() if result.semantic_query is not None else None
    if not semantic_query:
        semantic_query = None
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in result.lexical_keywords_en:
        text = item.strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) == max_lexical_keywords_en:
            break
    return QueryRewriteResult(
        semantic_query=semantic_query,
        lexical_keywords_en=cleaned,
        search_bibliography=result.search_bibliography,
    )
