"""将用户问题改写为受约束的多路检索查询，而不生成问题答案。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbase.config import QueryRewriteSettings
from paperbase.prompts.query_rewrite import (
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
        # 禁用 LLM 时不基于历史构造新检索词；但明确的引用意图仍可由确定性规则开启参考文献检索。
        _ = conversation_context
        normalized_query = _normalize_query(query)
        rule_decision = resolve_bibliography_search_rule(normalized_query)
        return QueryRewritePlan(
            original_query=normalized_query,
            search_bibliography=rule_decision is True,
            status="disabled",
        )


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
        # 引用路由采用“高精度规则优先、LLM 兜底”：规则只处理明确表达，其他问题才采用 LLM 判断。
        # 即使规则已有结果，LLM 仍继续完成英文语义改写和 BM25 关键词提取。
        rule_decision = resolve_bibliography_search_rule(normalized_query)
        normalized_context = _normalize_conversation_context(
            conversation_context,
            max_context_turns=self._settings.max_context_turns,
        )
        user_prompt = build_query_rewrite_user_prompt(
            query=normalized_query,
            conversation_context=normalized_context,
            max_lexical_keywords_en=self._settings.max_lexical_keywords_en,
        )
        try:
            # 调用封装好的LLM客户端方法，要求大模型输出严格符合schema的JSON字符串
            raw_response = self._client.complete_json(
                # 传入查询重写任务的系统提示词，定义大模型的角色、输出规则
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                # 传入组装完成的用户侧输入prompt，包含原始用户问题等信息
                user_prompt=user_prompt,
                # Pydantic 导出的 JSON Schema：由 API 尽可能强制字段名、类型和 extra=forbid。
                # 获取Pydantic模型QueryRewriteResult的JSON Schema，约束LLM输出的JSON结构
                json_schema=QueryRewriteResult.model_json_schema(),
                # 指定该schema的名称，部分LLM接口用于标识结构化输出对象
                schema_name="query_rewrite_result",
            )
            # 1.把LLM返回的原始JSON字符串校验解析为Pydantic模型实例；2.调用标准化函数做后处理
            result = normalize_query_rewrite(
                # model_validate_json：解析raw_response JSON字符串，校验字段、类型，生成QueryRewriteResult对象
                QueryRewriteResult.model_validate_json(raw_response),
                # 从配置读取英文词法关键词最大数量，标准化时用来截断、过滤关键词
                max_lexical_keywords_en=self._settings.max_lexical_keywords_en,
            )
            # 构造并返回最终的查询重写计划对象，供后续检索链路使用
            return QueryRewritePlan(
                # 经过预处理归一化后的用户原始查询文本
                original_query=normalized_query,
                # LLM生成的语义查询，用于向量/语义检索
                semantic_query=result.semantic_query,
                # 将列表转为元组，保证关键词序列不可变，避免后续意外修改
                lexical_keywords_en=tuple(result.lexical_keywords_en),
                # 调用决策函数，融合规则判断与LLM判断，得到是否需要检索参考文献的最终布尔决策
                search_bibliography=_resolve_bibliography_decision(
                    # 基于业务规则得到的是否检索文献的决策结果
                    rule_decision=rule_decision,
                    # LLM输出给出的是否检索文献的决策结果
                    llm_decision=result.search_bibliography,
                ),
            )
        except (LLMRequestError, ValidationError, ValueError, TypeError) as error:
            # API 的 response_format 已先用 JSON Schema 限制字段和类型；这里再用 Pydantic
            # 进行独立、严格的本地校验。若任一层失败，说明这次改写结果不可信。此时不再额外
            # 调用一次 LLM 去“修复 JSON”，因为修复调用仍可能改变语义、增加费用，也会让一次
            # 查询的行为更难复现。检索器会安全保留原始 Query，继续执行 Original Dense/BM25。
            return self._handle_failure(normalized_query, error, rule_decision)

    def _handle_failure(
        self,
        normalized_query: str,
        error: Exception,
        rule_decision: bool | None,
    ) -> QueryRewritePlan:
        """结构化改写失败时无条件保留原始问题，绝不让 Query Rewrite 中断 Retriever。

        ``fallback_to_original`` 是历史配置字段；结构化输出阶段将 Query Rewrite 视为可选的
        召回增强能力，因此无论其旧配置值为何，外部 LLM、API Schema 或本地 Pydantic 校验失败都只能降级，
        不能阻断后续 Original Dense / Original BM25 检索。
        """
        _ = error  # 失败原因由调用链保留；此处不向用户输出，避免外部服务异常泄露请求细节。
        return QueryRewritePlan(
            original_query=normalized_query,
            # 即使 LLM 不可用，明确的“是否引用/参考文献”问题也不应丢失 bibliography FTS5 路径。
            search_bibliography=rule_decision is True,
            status="fallback",
        )


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


def resolve_bibliography_search_rule(query: str) -> bool | None:
    """为明确表达提供 bibliography 路由的高精度规则，模糊问题返回 ``None`` 交给 LLM。

    规则不从论文正文或历史文本猜测意图，只检查当前用户问题；正向规则优先于负向规则，
    因而“作者为什么引用 Graph WaveNet”会同时保留正文和参考文献候选，而不是被“为什么”误判。
    """
    normalized = _normalize_query(query).casefold()

    # [15] 是哪篇论文？这类编号反查只能依赖参考文献条目。
    if re.search(r"\[\s*\d{1,4}\s*\]\s*(?:是|是哪|对应|什么|which|what)", normalized):
        return True

    # 仅命中明确的 citation/reference 表达；单独出现论文名、模型名或“相关工作”不会触发。
    explicit_reference_patterns = (
        r"参考文献",
        r"文献列表",
        r"参考书目",
        r"(?:有没有|是否|有无|是否有).{0,30}引用",
        r"引用了?(?:哪些|什么|哪篇|哪个|谁|.*吗|.*？|.*\?)",
        r"(?:cite|cited|cites|citation|citations|references|bibliography|works cited|literature cited)",
    )
    if any(re.search(pattern, normalized) for pattern in explicit_reference_patterns):
        return True

    # 这些模式明确要求方法、实验或机制的正文证据；只有在未命中上方引用规则时才返回 false。
    explicit_content_patterns = (
        r"(?:有什么|有何|哪些|如何|怎么).{0,20}(?:区别|不同|差异|优势|劣势)",
        r"(?:比较|对比|区别|不同|差异|优于|为什么比)",
        r"(?:baseline|baselines|基线)",
        r"(?:如何|怎么|怎样).{0,20}(?:构建|建立|训练|实现|计算|预测)",
        r"(?:方法|模型).{0,20}(?:为什么|如何|怎么)",
    )
    if any(re.search(pattern, normalized) for pattern in explicit_content_patterns):
        return False
    return None


def _resolve_bibliography_decision(*, rule_decision: bool | None, llm_decision: bool) -> bool:
    """优先采用明确规则；没有规则结论时才使用 LLM 的结构化判断。"""
    return rule_decision if rule_decision is not None else llm_decision


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
