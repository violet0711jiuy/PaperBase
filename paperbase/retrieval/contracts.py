"""Agentic retrieval 的稳定请求契约。

本模块只描述检索意图，不包含 Agent、Coverage 或最终回答语义。底层适配器可以
来自正式知识库或 Temporary Workspace，但必须消费相同的 ``PaperSearchSpec``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Iterable, TypeAlias


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
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    SEARCH_REFERENCES = "search_references"
    GET_SECTION_CONTEXT = "get_section_context"
    GET_PAPER_OVERVIEW = "get_paper_overview"


class EvidenceSourceType(str, Enum):
    CONTENT = "content"
    BIBLIOGRAPHY = "bibliography"
    OVERVIEW = "overview"
    SECTION_CONTEXT = "section_context"


class ScoreKind(str, Enum):
    DENSE = "dense"
    RERANKER = "reranker"
    RRF = "rrf"
    BM25 = "bm25"
    BM25_FTS5 = "bm25_fts5"
    NON_RANKED = "non_ranked"
    OTHER = "other"
    LEGACY_UNKNOWN = "legacy_unknown"


class ScoreDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class PaperSearchSpec:
    """一篇论文的一次定向检索；不同论文可以拥有不同 Query。"""

    paper_id: str
    query: str
    lexical_terms: tuple[str, ...] = ()
    # Legacy/model-facing retrieval hint, not Registry intent or Search identity.
    evidence_targets: frozenset[str] = frozenset()
    requirement_ids: tuple[str, ...] = ()

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
            "requirement_ids",
            _normalized_text_tuple(self.requirement_ids, "requirement_ids"),
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

    search: PaperSearchSpec
    result: object | None = None
    error_code: str | None = None
    error_message: str | None = None
    execution_started: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.search, PaperSearchSpec):
            raise TypeError("search must be a PaperSearchSpec.")

    @property
    def succeeded(self) -> bool:
        return self.error_code is None


@dataclass(frozen=True)
class RetrievedEvidence:
    """Unnumbered Tool evidence with source-local, uncalibrated retrieval scores.

    raw_score is meaningful only within the producing scoring channel/request.
    retrieval_rank is its 1-based returned position, never global relevance.
    score is deprecated: retained solely for old constructor/checkpoint payloads.
    Legacy score-only data keeps its numeric value with unknown semantics; a
    persisted metadata.rank can migrate as rank without guessing score direction.
    New producers populate the explicit fields and leave score unset.
    """

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
    score: float | None = field(default=None, metadata={"deprecated": True})
    retrieval_metadata: dict[str, object] = field(default_factory=dict)
    raw_score: float | None = None
    score_kind: ScoreKind = ScoreKind.LEGACY_UNKNOWN
    score_direction: ScoreDirection = ScoreDirection.NOT_APPLICABLE
    retrieval_rank: int | None = None

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
        kind = ScoreKind(self.score_kind)
        direction = ScoreDirection(self.score_direction)
        object.__setattr__(self, "score_kind", kind)
        object.__setattr__(self, "score_direction", direction)
        raw = self.raw_score if self.raw_score is not None else self.score
        if raw is not None:
            raw = float(raw)
            if not isfinite(raw):
                raise ValueError("raw_score must be finite when provided.")
            object.__setattr__(self, "raw_score", raw)
        if kind is ScoreKind.LEGACY_UNKNOWN:
            if direction is not ScoreDirection.NOT_APPLICABLE:
                raise ValueError("Legacy unknown scores have no ordering semantics.")
            old_rank = self.retrieval_metadata.get("rank")
            if self.retrieval_rank is None and type(old_rank) is int and old_rank > 0:
                object.__setattr__(self, "retrieval_rank", old_rank)
        expected = {
            ScoreKind.DENSE: ScoreDirection.HIGHER_IS_BETTER,
            ScoreKind.RERANKER: ScoreDirection.HIGHER_IS_BETTER,
            ScoreKind.RRF: ScoreDirection.HIGHER_IS_BETTER,
            ScoreKind.BM25: ScoreDirection.HIGHER_IS_BETTER,
            ScoreKind.BM25_FTS5: ScoreDirection.LOWER_IS_BETTER,
            ScoreKind.NON_RANKED: ScoreDirection.NOT_APPLICABLE,
        }.get(kind)
        if expected is not None and direction is not expected:
            raise ValueError(f"{kind.value} requires score_direction={expected.value}.")
        if kind is ScoreKind.NON_RANKED and (raw is not None or self.retrieval_rank is not None):
            raise ValueError("Non-ranked evidence must not invent a score or retrieval rank.")
        if self.retrieval_rank is not None and (
            type(self.retrieval_rank) is not int or self.retrieval_rank < 1
        ):
            raise ValueError("retrieval_rank must be a positive 1-based integer.")


@dataclass(frozen=True)
class PaperSearchOutcome:
    search: PaperSearchSpec
    status: PaperSearchStatus
    evidence_candidates: tuple[RetrievedEvidence, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PaperSearchStatus(self.status))
        if not isinstance(self.search, PaperSearchSpec):
            raise TypeError("search must be a PaperSearchSpec.")
        if self.status is PaperSearchStatus.OK and not self.evidence_candidates:
            raise ValueError("An ok paper outcome requires evidence candidates.")
        if self.status is PaperSearchStatus.EMPTY and self.evidence_candidates:
            raise ValueError("An empty paper outcome cannot contain evidence candidates.")

    @property
    def paper_id(self) -> str:
        return self.search.paper_id

    @property
    def query(self) -> str:
        return self.search.query

    @property
    def lexical_terms(self) -> tuple[str, ...]:
        return self.search.lexical_terms

    @property
    def evidence_targets(self) -> frozenset[str]:
        return self.search.evidence_targets

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return self.search.requirement_ids


@dataclass(frozen=True)
class SearchPapersPayload:
    paper_results: tuple[PaperSearchOutcome, ...]
    kind: str = field(default="search_papers", init=False)


@dataclass(frozen=True)
class KnowledgeBaseSearchPayload:
    query: str
    evidence_candidates: tuple[RetrievedEvidence, ...] = ()
    kind: str = field(default="search_knowledge_base", init=False)


@dataclass(frozen=True)
class ReferenceSearchPayload:
    paper_id: str
    query: str
    evidence_candidates: tuple[RetrievedEvidence, ...] = ()
    kind: str = field(default="search_references", init=False)


@dataclass(frozen=True)
class SectionContextPayload:
    paper_id: str
    section_id: str
    evidence_candidates: tuple[RetrievedEvidence, ...] = ()
    kind: str = field(default="get_section_context", init=False)


@dataclass(frozen=True)
class PaperOverviewPayload:
    paper_id: str
    evidence_candidates: tuple[RetrievedEvidence, ...] = ()
    kind: str = field(default="get_paper_overview", init=False)


ToolPayload: TypeAlias = (
    SearchPapersPayload
    | KnowledgeBaseSearchPayload
    | ReferenceSearchPayload
    | SectionContextPayload
    | PaperOverviewPayload
)


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    tool_name: ToolName
    payload: ToolPayload
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
