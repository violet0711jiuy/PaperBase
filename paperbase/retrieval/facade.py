"""正式知识库与 Workspace 检索适配器之上的统一入口。"""

from __future__ import annotations

from typing import Mapping, Protocol, TypeVar
from dataclasses import replace
from .requests import request_trace

from .contracts import (
    PaperBatchExecution,
    PaperSearchSpec,
    QueryOrigin,
    RetrievalSource,
    SearchPapersRequest,
    ToolResult,
)
from .tool_results import build_search_papers_tool_result


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
        canonical_request=None,
    ) -> object:
        """执行一篇论文的一次检索；批处理能力在后续提交加入。"""
        adapter = self._resolve_adapter(source=source, workspace_id=workspace_id)
        return adapter.search(
            search,
            query_origin=QueryOrigin(query_origin),
            result_limit=result_limit,
            **({"canonical_request": canonical_request} if canonical_request is not None else {}),
        )

    def request_parameters(self, *, source, workspace_id=None):
        try:
            adapter = self._resolve_adapter(source=source, workspace_id=workspace_id)
        except (LookupError, ValueError):
            return {}
        return adapter.request_parameters() if hasattr(adapter, "request_parameters") else {}

    def search_papers(
        self,
        request: SearchPapersRequest,
        *,
        source: RetrievalSource,
        workspace_ids_by_paper: Mapping[str, str] | None = None,
        canonical_requests=(),
    ) -> ToolResult:
        """执行异构 Batch 并返回带逐论文状态的类型化 ToolResult。"""
        executions = self.execute_papers(
            request,
            source=source,
            workspace_ids_by_paper=workspace_ids_by_paper,
            canonical_requests=canonical_requests,
        )
        result = build_search_papers_tool_result(
            executions,
            query_origin=request.query_origin,
        )
        if canonical_requests:
            records = [{"request_identity": canonical.identity, "attempted": True,
                        "executed": execution.execution_started,
                        "failed": outcome.status.value in {"error", "unavailable"},
                        "successful": outcome.status.value in {"ok", "empty"}}
                       for canonical, execution, outcome in zip(canonical_requests, executions, result.payload.paper_results)]
            result = replace(result, metadata={**result.metadata,
                             "canonical_effective_requests": request_trace(canonical_requests),
                             "request_executions": records})
        return result

    def execute_papers(
        self,
        request: SearchPapersRequest,
        *,
        source: RetrievalSource,
        workspace_ids_by_paper: Mapping[str, str] | None = None,
        canonical_requests=(),
    ) -> tuple[PaperBatchExecution, ...]:
        """逐篇隔离执行；供 Tool 转换和底层契约测试使用。"""
        executions: list[PaperBatchExecution] = []
        for index, search in enumerate(request.searches):
            canonical = canonical_requests[index] if canonical_requests else None
            execution_started = False
            workspace_id = (
                workspace_ids_by_paper.get(search.paper_id)
                if workspace_ids_by_paper is not None
                else None
            )
            try:
                adapter = self._resolve_adapter(source=source, workspace_id=workspace_id)
                execution_started = True
                result = adapter.search(
                    search, query_origin=request.query_origin, result_limit=request.top_k_per_paper,
                    **({"canonical_request": canonical} if canonical is not None else {}),
                )
            except Exception as error:
                executions.append(
                    PaperBatchExecution(
                        search=search,
                        error_code=type(error).__name__,
                        error_message=str(error),
                        execution_started=execution_started,
                    )
                )
                continue
            executions.append(
                PaperBatchExecution(
                    search=search,
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
