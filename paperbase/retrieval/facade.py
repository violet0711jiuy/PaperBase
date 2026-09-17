"""正式知识库与 Workspace 检索适配器之上的统一入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, TypeVar

from .contracts import (
    PaperSearchSpec,
    QueryOrigin,
    RetrievalSource,
    SearchPapersRequest,
)


SearchResultT = TypeVar("SearchResultT", covariant=True)


class RetrievalAdapter(Protocol[SearchResultT]):
    """Facade 依赖的最窄适配器接口。"""

    def search(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        result_limit: int | None = None,
    ) -> SearchResultT:
        """在适配器自己的数据边界内执行一次论文检索。"""


@dataclass(frozen=True)
class PaperBatchExecution:
    """Batch 内一篇论文的隔离执行结果；状态语义在 Tool 层统一映射。"""

    paper_id: str
    query: str
    lexical_terms: tuple[str, ...]
    result: object | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_code is None


class RetrievalService:
    """按显式数据来源分发请求，不混合正式库与 Workspace 生命周期。"""

    def __init__(
        self,
        *,
        formal_adapter: RetrievalAdapter[object] | None = None,
    ) -> None:
        self._formal_adapter = formal_adapter
        self._workspace_adapters: dict[str, RetrievalAdapter[object]] = {}

    def register_workspace(
        self, workspace_id: str, adapter: RetrievalAdapter[object]
    ) -> None:
        normalized_id = " ".join(workspace_id.split())
        if not normalized_id:
            raise ValueError("workspace_id must be a non-empty string.")
        self._workspace_adapters[normalized_id] = adapter

    def search_paper(
        self,
        search: PaperSearchSpec,
        *,
        query_origin: QueryOrigin,
        source: RetrievalSource,
        workspace_id: str | None = None,
        result_limit: int | None = None,
    ) -> object:
        """执行一篇论文的一次检索；批处理能力在后续提交加入。"""
        adapter = self._resolve_adapter(source=source, workspace_id=workspace_id)
        return adapter.search(
            search,
            query_origin=QueryOrigin(query_origin),
            result_limit=result_limit,
        )

    def search_papers(
        self,
        request: SearchPapersRequest,
        *,
        source: RetrievalSource,
        workspace_ids_by_paper: Mapping[str, str] | None = None,
    ) -> tuple[PaperBatchExecution, ...]:
        """逐篇执行异构 Query；一篇失败不得丢弃其他论文结果。"""
        executions: list[PaperBatchExecution] = []
        for search in request.searches:
            workspace_id = (
                workspace_ids_by_paper.get(search.paper_id)
                if workspace_ids_by_paper is not None
                else None
            )
            try:
                result = self.search_paper(
                    search,
                    query_origin=request.query_origin,
                    source=source,
                    workspace_id=workspace_id,
                    result_limit=request.top_k_per_paper,
                )
            except Exception as error:
                executions.append(
                    PaperBatchExecution(
                        paper_id=search.paper_id,
                        query=search.query,
                        lexical_terms=search.lexical_terms,
                        error_code=type(error).__name__,
                        error_message=str(error),
                    )
                )
                continue
            executions.append(
                PaperBatchExecution(
                    paper_id=search.paper_id,
                    query=search.query,
                    lexical_terms=search.lexical_terms,
                    result=result,
                )
            )
        return tuple(executions)

    def _resolve_adapter(
        self, *, source: RetrievalSource, workspace_id: str | None
    ) -> RetrievalAdapter[object]:
        normalized_source = RetrievalSource(source)
        if normalized_source is RetrievalSource.FORMAL_KNOWLEDGE_BASE:
            if workspace_id is not None:
                raise ValueError("workspace_id is not valid for formal knowledge-base search.")
            if self._formal_adapter is None:
                raise LookupError("Formal knowledge-base adapter is not configured.")
            return self._formal_adapter
        if workspace_id is None or not workspace_id.strip():
            raise ValueError("workspace_id is required for workspace search.")
        try:
            return self._workspace_adapters[" ".join(workspace_id.split())]
        except KeyError as error:
            raise LookupError(f"Workspace adapter is not registered: {workspace_id}") from error
