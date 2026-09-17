"""Agentic retrieval 的稳定请求契约。

本模块只描述检索意图，不包含 Agent、Coverage 或最终回答语义。底层适配器可以
来自正式知识库或 Temporary Workspace，但必须消费相同的 ``PaperSearchSpec``。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class QueryOrigin(str, Enum):
    """检索 Query 的来源决定是否允许再次调用 Query Rewriter。"""

    INITIAL_UNDERSTANDING = "initial_understanding"
    AGENT_SUPPLIED = "agent_supplied"


class RetrievalSource(str, Enum):
    """选择彼此隔离的正式知识库或临时工作区适配器。"""

    FORMAL_KNOWLEDGE_BASE = "formal_knowledge_base"
    WORKSPACE = "workspace"


@dataclass(frozen=True)
class PaperSearchSpec:
    """一篇论文的一次定向检索；不同论文可以拥有不同 Query。"""

    paper_id: str
    query: str
    lexical_terms: tuple[str, ...] = ()
    evidence_targets: frozenset[str] = frozenset()
    missing_information: tuple[str, ...] = ()
    target_subtask_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _required_text(self.paper_id, "paper_id"))
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        object.__setattr__(
            self,
            "lexical_terms",
            _normalized_text_tuple(self.lexical_terms, "lexical_terms"),
        )
        object.__setattr__(
            self,
            "evidence_targets",
            frozenset(_normalized_text_tuple(self.evidence_targets, "evidence_targets")),
        )
        object.__setattr__(
            self,
            "missing_information",
            _normalized_text_tuple(self.missing_information, "missing_information"),
        )
        if self.target_subtask_id is not None:
            object.__setattr__(
                self,
                "target_subtask_id",
                _required_text(self.target_subtask_id, "target_subtask_id"),
            )


@dataclass(frozen=True)
class SearchPapersRequest:
    """同一数据来源上的逐论文检索请求。

    ``top_k_per_paper`` 是由运行时配置传入的内部上限，不是 Agent Tool 参数。
    """

    query_origin: QueryOrigin
    searches: tuple[PaperSearchSpec, ...]
    top_k_per_paper: int | None = None

    def __post_init__(self) -> None:
        try:
            origin = QueryOrigin(self.query_origin)
        except ValueError as error:
            raise ValueError(f"Unsupported query_origin: {self.query_origin!r}") from error
        object.__setattr__(self, "query_origin", origin)
        object.__setattr__(self, "searches", tuple(self.searches))
        if not self.searches:
            raise ValueError("searches must contain at least one PaperSearchSpec.")
        paper_ids = [search.paper_id for search in self.searches]
        if len(set(paper_ids)) != len(paper_ids):
            raise ValueError("searches must not contain duplicate paper_id values.")
        if self.top_k_per_paper is not None and self.top_k_per_paper < 1:
            raise ValueError("top_k_per_paper must be positive when provided.")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return " ".join(value.split())


def _normalized_text_tuple(values: Iterable[object], field_name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        normalized_value = _required_text(value, field_name)
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return tuple(normalized)
