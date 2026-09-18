"""Phase 4 Agentic Research 应用服务与并发、恢复生命周期边界。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import TYPE_CHECKING

from .contracts import ResearchRunResult
from .events import ResearchProgressEvent
from .scope_resolver import TrustedWorkspaceScope

if TYPE_CHECKING:
    from .graph import GraphExecution, LangGraphResearchWorkflow
    from .persistence import SQLiteCheckpointStore


class AgenticResearchBusyError(RuntimeError):
    """本地 Agentic Research 并发额度已用尽。"""


class AgenticResearchService:
    """生产入口只执行 LangGraph workflow；纯领域 Runtime 保留在独立测试中。"""

    def __init__(
        self,
        *,
        workflow: "LangGraphResearchWorkflow",
        checkpoint_store: "SQLiteCheckpointStore | None" = None,
        max_concurrent_runs: int = 1,
    ) -> None:
        if max_concurrent_runs < 1:
            raise ValueError("max_concurrent_runs must be at least 1.")
        self._workflow = workflow
        self._checkpoint_store = checkpoint_store
        self._capacity = BoundedSemaphore(max_concurrent_runs)

    def research(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
        thread_id: str | None = None,
    ) -> ResearchRunResult:
        with self._run_slot():
            result = self._workflow.run(
                query,
                trusted_workspace=trusted_workspace,
                selected_paper_ids=selected_paper_ids,
                knowledge_base_all=knowledge_base_all,
                thread_id=thread_id,
            )
            self._touch_thread(result.thread_id)
            return result

    def stream_research(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
        thread_id: str | None = None,
        interrupt_after: tuple[str, ...] = (),
    ) -> Iterator[ResearchProgressEvent]:
        with self._run_slot():
            last_thread_id: str | None = None
            for event in self._workflow.stream(
                query,
                trusted_workspace=trusted_workspace,
                selected_paper_ids=selected_paper_ids,
                knowledge_base_all=knowledge_base_all,
                thread_id=thread_id,
                interrupt_after=interrupt_after,
            ):
                last_thread_id = event.thread_id
                yield event
            self._touch_thread(last_thread_id)

    def resume(
        self,
        thread_id: str,
        *,
        interrupt_after: tuple[str, ...] = (),
    ) -> "GraphExecution":
        with self._run_slot():
            execution = self._workflow.resume(
                thread_id,
                interrupt_after=interrupt_after,
            )
            self._touch_thread(thread_id)
            return execution

    def resume_stream(
        self,
        thread_id: str,
        *,
        interrupt_after: tuple[str, ...] = (),
    ) -> Iterator[ResearchProgressEvent]:
        with self._run_slot():
            for event in self._workflow.resume_stream(
                thread_id,
                interrupt_after=interrupt_after,
            ):
                yield event
            self._touch_thread(thread_id)

    def inspect(self, thread_id: str) -> "GraphExecution":
        return self._workflow.inspect(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        if self._checkpoint_store is None:
            raise RuntimeError("当前 Service 未配置可管理的持久化 checkpoint。")
        self._checkpoint_store.delete_thread(thread_id)

    def close(self) -> None:
        if self._checkpoint_store is not None:
            self._checkpoint_store.close()

    @contextmanager
    def _run_slot(self):
        if not self._capacity.acquire(blocking=False):
            raise AgenticResearchBusyError(
                "当前已有 Agentic Research 任务运行，请等待其完成后重试。"
            )
        try:
            yield
        finally:
            self._capacity.release()

    def _touch_thread(self, thread_id: str | None) -> None:
        if self._checkpoint_store is None or thread_id is None:
            return
        self._checkpoint_store.touch_thread(thread_id)
