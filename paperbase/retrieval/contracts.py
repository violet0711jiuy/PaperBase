"""Agentic retrieval 的稳定请求契约。

本模块只描述检索意图，不包含 Agent、Coverage 或最终回答语义。底层适配器可以
来自正式知识库或 Temporary Workspace，但必须消费相同的 ``PaperSearchSpec``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


class ToolStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    PARTIAL = "partial"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class PaperSearchStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class ToolName(str, Enum):
    SEARCH_PAPERS = "search_papers"


class EvidenceSourceType(str, Enum):
    CONTENT = "content"
    BIBLIOGRAPHY = "bibliography"
    OVERVIEW = "overview"
    SECTION_CONTEXT = "section_context"


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


@dataclass(frozen=True)
class PaperBatchExecution:
    """Batch 内一篇论文的隔离执行结果，供 Tool 结果转换器消费。"""

    paper_id: str
    query: str
    lexical_terms: tuple[str, ...]
    result: object | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_code is None


@dataclass(frozen=True)
class RetrievedEvidence:
    """Tool 产生的无运行期编号证据；E#/R# 只由 Registry 分配。"""

    source_type: EvidenceSourceType
    paper_id: str
    seed_chunk_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    content: str
    paper_title: str | None = None
    section_id: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    retrieval_metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", EvidenceSourceType(self.source_type))
        object.__setattr__(self, "paper_id", _required_text(self.paper_id, "paper_id"))
        object.__setattr__(
            self,
            "seed_chunk_ids",
            _normalized_text_tuple(self.seed_chunk_ids, "seed_chunk_ids"),
        )
        object.__setattr__(
            self,
            "source_chunk_ids",
            _normalized_text_tuple(self.source_chunk_ids, "source_chunk_ids"),
        )
        if not self.seed_chunk_ids or not self.source_chunk_ids:
            raise ValueError("RetrievedEvidence requires seed and source chunk IDs.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string.")
        object.__setattr__(self, "content", self.content.strip())
        object.__setattr__(self, "retrieval_metadata", dict(self.retrieval_metadata))


@dataclass(frozen=True)
class PaperSearchOutcome:
    paper_id: str
    status: PaperSearchStatus
    query: str
    lexical_terms: tuple[str, ...]
    evidence_candidates: tuple[RetrievedEvidence, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PaperSearchStatus(self.status))
        if self.status is PaperSearchStatus.OK and not self.evidence_candidates:
            raise ValueError("An ok paper outcome requires evidence candidates.")
        if self.status is PaperSearchStatus.EMPTY and self.evidence_candidates:
            raise ValueError("An empty paper outcome cannot contain evidence candidates.")


@dataclass(frozen=True)
class SearchPapersPayload:
    paper_results: tuple[PaperSearchOutcome, ...]
    kind: str = field(default="search_papers", init=False)


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    tool_name: ToolName
    payload: SearchPapersPayload
    metadata: dict[str, object] = field(default_factory=dict)
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ToolStatus(self.status))
        object.__setattr__(self, "tool_name", ToolName(self.tool_name))
        if self.tool_name.value != self.payload.kind:
            raise ValueError(
                f"tool_name {self.tool_name.value!r} does not match payload kind "
                f"{self.payload.kind!r}."
            )
        object.__setattr__(self, "metadata", dict(self.metadata))


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
