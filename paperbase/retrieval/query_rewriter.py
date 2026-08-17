"""将查询规划拆为指代消解、检索改写与确定性参考文献路由。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from paperbase.config import QueryRewriteSettings
from paperbase.llm.client import (
    ChatCompletionClient,
    LLMRequestError,
    OpenAICompatibleChatClient,
    load_llm_runtime_settings,
)
from paperbase.prompts.query_rewrite import (
    QUERY_RESOLUTION_SYSTEM_PROMPT,
    RETRIEVAL_REWRITE_SYSTEM_PROMPT,
    build_query_resolution_user_prompt,
    build_retrieval_rewrite_user_prompt,
)


@dataclass(frozen=True)
class TrustedPaperScope:
    """由程序维护的当前论文范围，不能由用户问题或普通会话文本伪造。"""

    paper_id: str
    paper_title: str | None

    @property
    def label(self) -> str:
        """优先展示真实标题；解析标题缺失时仍可用稳定 paper_id 消解“本文”。"""
        return f"《{self.paper_title}》" if self.paper_title else self.paper_id


class QueryResolutionResult(BaseModel):
    """Resolution LLM 的最小契约：只确定问题对象，不翻译或优化检索。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    resolution_status: Literal["resolved", "unresolved"]
    resolved_query: str | None

    @model_validator(mode="after")
    def validate_contract(self) -> "QueryResolutionResult":
        if self.resolution_status == "resolved" and not _optional_normalized(self.resolved_query):
            raise ValueError("resolved_query is required when resolution succeeds.")
        if self.resolution_status == "unresolved" and self.resolved_query is not None:
            raise ValueError("unresolved query must not provide resolved_query.")
        return self


class RetrievalRewriteResult(BaseModel):
    """Retrieval Rewrite LLM 的最小契约：只产生英文检索补充。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    semantic_query_en: str = Field(min_length=1)
    lexical_keywords_en: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class QueryRewritePlan:
    """由程序拼装的最终查询计划，字段来源可独立审计。"""

    original_query: str
    resolved_query: str | None = None
    resolution_status: Literal["resolved", "unresolved"] = "resolved"
    semantic_query_en: str | None = None
    lexical_keywords_en: tuple[str, ...] = ()
    # success 表示 Retrieval Rewrite LLM 成功；degraded 不影响 resolved_query 的 Dense 检索。
    rewrite_status: Literal["success", "degraded", "not_run"] = "not_run"
    # 仅由 resolve_bibliography_search_rule 程序规则产生，绝不信任 LLM。
    search_bibliography: bool = False
    clarification_message: str | None = None


class QueryPlanner(Protocol):
    """检索器依赖的查询规划接口；可替换为离线实现。"""

    def plan(
        self,
        query: str,
        *,
        trusted_scope: TrustedPaperScope | None = None,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewritePlan:
        """返回已消解问题与可选英文检索补充。"""


class NoopQueryPlanner:
    """关闭 LLM 时的规划器：仍执行确定性 Resolution 与 bibliography 路由。"""

    def __init__(self, *, settings: QueryRewriteSettings | None = None) -> None:
        self._settings = settings or QueryRewriteSettings()

    def plan(
        self,
        query: str,
        *,
        trusted_scope: TrustedPaperScope | None = None,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewritePlan:
        normalized_query = _normalize_query(query)
        resolution = _resolve_without_llm(normalized_query, trusted_scope=trusted_scope)
        if resolution is None:
            # 离线状态不能用不可信历史猜“它”，必须明确请求用户补充对象。
            return _unresolved_plan(normalized_query)
        return QueryRewritePlan(
            original_query=normalized_query,
            resolved_query=resolution,
            semantic_query_en=None,
            lexical_keywords_en=_safe_existing_english_entities(
                resolution, max_keywords=self._settings.max_lexical_keywords_en
            ),
            rewrite_status="degraded",
            search_bibliography=resolve_bibliography_search_rule(normalized_query),
        )


class LLMQueryPlanner:
    """按 Resolution → Retrieval Rewrite 顺序执行的查询规划器。"""

    def __init__(self, *, settings: QueryRewriteSettings, client: ChatCompletionClient) -> None:
        self._settings = settings
        self._client = client

    def plan(
        self,
        query: str,
        *,
        trusted_scope: TrustedPaperScope | None = None,
        conversation_context: Sequence[str] | None = None,
    ) -> QueryRewritePlan:
        normalized_query = _normalize_query(query)
        normalized_context = _normalize_conversation_context(conversation_context)
        # 先使用当前 query 与程序可信 scope；这两者优先级高于普通会话文本。
        locally_resolved = _resolve_without_llm(normalized_query, trusted_scope=trusted_scope)
        if locally_resolved is None:
            resolution = self._resolve_with_context(
                normalized_query,
                conversation_context=normalized_context,
            )
            if resolution is None:
                return _unresolved_plan(normalized_query)
            resolved_query = resolution
        else:
            resolved_query = locally_resolved

        return self._build_resolved_plan(
            original_query=normalized_query,
            resolved_query=resolved_query,
        )

    def _resolve_with_context(
        self,
        normalized_query: str,
        *,
        conversation_context: tuple[str, ...],
    ) -> str | None:
        """仅让 LLM 处理仍有歧义的历史指代；失败即 unresolved，不做业务 fallback。"""
        try:
            raw_response = self._client.complete_json(
                system_prompt=QUERY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=build_query_resolution_user_prompt(
                    query=normalized_query,
                    conversation_context=conversation_context,
                ),
                json_schema=QueryResolutionResult.model_json_schema(),
                schema_name="query_resolution_result",
            )
            result = QueryResolutionResult.model_validate_json(raw_response)
        except (LLMRequestError, ValidationError, ValueError, TypeError):
            return None
        if result.resolution_status == "unresolved":
            return None
        return _optional_normalized(result.resolved_query)

    def _build_resolved_plan(
        self,
        *,
        original_query: str,
        resolved_query: str,
    ) -> QueryRewritePlan:
        """Resolution 成功后才调用 Retrieval Rewrite；其失败不得阻断第一条 Dense。"""
        bibliography_intent = resolve_bibliography_search_rule(original_query)
        try:
            raw_response = self._client.complete_json(
                system_prompt=RETRIEVAL_REWRITE_SYSTEM_PROMPT,
                user_prompt=build_retrieval_rewrite_user_prompt(
                    resolved_query=resolved_query,
                    max_lexical_keywords_en=self._settings.max_lexical_keywords_en,
                ),
                json_schema=RetrievalRewriteResult.model_json_schema(),
                schema_name="retrieval_rewrite_result",
            )
            result = RetrievalRewriteResult.model_validate_json(raw_response)
            semantic_query = _optional_normalized(result.semantic_query_en)
            if semantic_query is None:
                raise ValueError("semantic_query_en must not be blank.")
            return QueryRewritePlan(
                original_query=original_query,
                resolved_query=resolved_query,
                semantic_query_en=semantic_query,
                lexical_keywords_en=_normalize_keywords(
                    result.lexical_keywords_en,
                    max_keywords=self._settings.max_lexical_keywords_en,
                ),
                rewrite_status="success",
                search_bibliography=bibliography_intent,
            )
        except (LLMRequestError, ValidationError, ValueError, TypeError):
            # 不做术语表翻译或英文句式拼接；只留下原问题中可直接观察到的英文实体。
            return QueryRewritePlan(
                original_query=original_query,
                resolved_query=resolved_query,
                semantic_query_en=None,
                lexical_keywords_en=_safe_existing_english_entities(
                    resolved_query,
                    max_keywords=self._settings.max_lexical_keywords_en,
                ),
                rewrite_status="degraded",
                search_bibliography=bibliography_intent,
            )


def create_query_planner(
    *,
    settings: QueryRewriteSettings,
    env_path: Path,
    client: ChatCompletionClient | None = None,
) -> QueryPlanner:
    """构造统一 Query Planner；显式注入 client 仅用于测试。"""
    if not settings.enabled:
        return NoopQueryPlanner(settings=settings)
    if client is None:
        client = OpenAICompatibleChatClient(load_llm_runtime_settings(env_path))
    return LLMQueryPlanner(settings=settings, client=client)


def _resolve_without_llm(
    query: str,
    *,
    trusted_scope: TrustedPaperScope | None,
) -> str | None:
    """按 query 内部 → trusted scope 的优先级做确定性 Resolution。"""
    if not _has_context_dependent_reference(query):
        return query
    # 当前 query 内已有标题、缩写等先行词时，绝不能再让 scope 或历史覆盖它。
    if _has_inline_antecedent(query):
        return query
    # active paper scope 只允许消解“本文/这篇论文”，不能把“它/这个方法”猜成论文主方法。
    if trusted_scope is not None and _has_paper_reference(query):
        return re.sub(_PAPER_REFERENCE_PATTERN, f"论文{trusted_scope.label}", query)
    return None


def _unresolved_plan(original_query: str) -> QueryRewritePlan:
    """无法唯一消解时停止后续改写与检索，返回程序固定提示。"""
    return QueryRewritePlan(
        original_query=original_query,
        resolved_query=None,
        resolution_status="unresolved",
        rewrite_status="not_run",
        search_bibliography=False,
        clarification_message=(
            "当前问题中的指代无法从当前问题、当前论文范围或最近上下文唯一确定，"
            "请明确说明它指的是哪一篇论文、方法、结果或比较对象。"
        ),
    )


_PAPER_REFERENCE_PATTERN = r"(?:本文|这篇论文|该论文|该文)"


def _has_paper_reference(query: str) -> bool:
    """只识别可以由 active paper scope 消解的论文级指代。"""
    return bool(re.search(_PAPER_REFERENCE_PATTERN, query.casefold()))


def _has_context_dependent_reference(query: str) -> bool:
    """判断是否存在尚待消解的指代；未知缩写和宽泛问题不属于该类。"""
    normalized = " ".join(query.casefold().split())
    if _has_inline_antecedent(query):
        return False
    generic_patterns = (
        r"\b(this|that|it|they|former|latter)\b",
        r"(?:这个|那个|它|他们|前者|后者|上述|上面|刚才|前面)(?:方法|模型|算法|结果|实验|工作|论文|文献|指标|数据集)?",
        r"(?:这个步骤|上述步骤|第[一二三四五六七八九十\d]+步)",
    )
    return any(re.search(pattern, normalized) for pattern in generic_patterns) or _has_paper_reference(
        normalized
    )


def _has_inline_antecedent(query: str) -> bool:
    """保守识别当前 query 内的论文标题、书名号或模型缩写，确保它优先于上下文。"""
    reference = re.search(
        r"(?:本文|这篇论文|该论文|该文|这个方法|该方法|该模型|它|这个步骤|上述步骤|第[一二三四五六七八九十\d]+步|this paper|this method|it)",
        query,
        flags=re.IGNORECASE,
    )
    if reference is None:
        return False
    prefix = query[: reference.start()].strip(" ：:，,。.!？?《》")
    if re.search(r"《[^》]{2,}》\s*$", prefix):
        return True
    # 未加书名号的英文标题必须足够长；普通英文短语不能被误判为论文先行词。
    english_terms = re.findall(r"[A-Za-z][A-Za-z0-9:-]*", prefix)
    if len(english_terms) >= 5 and len(" ".join(english_terms)) >= 30:
        return True
    # 短标题或模型名（如 Graph WaveNet）也可能是同句先行词，不能只因词数少而丢给历史。
    if re.search(r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)+\b", prefix):
        return True
    # 真实模型缩写（如 ESDTW、D2STGNN）可作为“该方法/它”的同句先行词。
    return bool(re.search(r"\b[A-Z][A-Z0-9-]{1,}\b", prefix))


def _normalize_query(query: str) -> str:
    """折叠空白，避免等价问题进入不同检索路径。"""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("Retrieval query must not be empty.")
    return normalized


def _optional_normalized(value: str | None) -> str | None:
    """把空白字符串统一为缺失，避免空查询进入向量或 BM25。"""
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _normalize_conversation_context(
    conversation_context: Sequence[str] | None,
) -> tuple[str, ...]:
    """清理服务层已按 ConversationSettings 截取的上下文；其中内容仍是用户数据。"""
    if not conversation_context:
        return ()
    cleaned = tuple(" ".join(item.split()) for item in conversation_context if item and item.strip())
    return cleaned


def _normalize_keywords(values: Sequence[str], *, max_keywords: int) -> tuple[str, ...]:
    """清理模型输出；关键词允许为空，避免为满足数量制造泛词。"""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) == max_keywords:
            break
    return tuple(cleaned)


def _safe_existing_english_entities(query: str, *, max_keywords: int) -> tuple[str, ...]:
    """降级时只提取原句可见的英文实体/缩写，不翻译中文概念或补写英文问句。"""
    candidates = re.findall(
        r"(?:[A-Z][A-Za-z0-9:-]*(?:\s+[A-Z][A-Za-z0-9:-]*){0,5})",
        query,
    )
    return _normalize_keywords(candidates, max_keywords=max_keywords)


def resolve_bibliography_search_rule(query: str) -> bool:
    """仅用高精度程序规则识别 citation/reference/bibliographic metadata 意图。"""
    normalized = _normalize_query(query).casefold()
    if re.search(r"\[\s*\d{1,4}\s*\]\s*(?:是|是哪|对应|什么|which|what)", normalized):
        return True
    explicit_reference_patterns = (
        r"参考文献",
        r"文献列表",
        r"参考书目",
        r"(?:有没有|是否|有无|是否有).{0,30}引用",
        r"引用了?(?:哪些|什么|哪篇|哪个|谁|.*吗|.*？|.*\?)",
        r"(?:cite|cited|cites|citation|citations|references|bibliography|works cited|literature cited)",
        r"(?:对应|是哪).{0,20}(?:参考文献|文献|reference)",
    )
    return any(re.search(pattern, normalized) for pattern in explicit_reference_patterns)
