"""Agentic 审计状态在 UI 层的用户提示测试。"""

from __future__ import annotations

from paperbase.agentic.audit import AgenticAuditSnapshot, BudgetAudit
from app.pages.knowledge_base import _audit_status_message


def _snapshot(*, status: str, reason: str) -> AgenticAuditSnapshot:
    return AgenticAuditSnapshot(
        schema_version=1,
        run_status=status,
        stop_reason=reason,
        thread_id="thread-1",
        scope="knowledge_base",
        coverage_complete=False,
        candidates=(),
        papers=(),
        steps=(),
        evidence=(),
        budget=BudgetAudit(),
        llm_calls=(),
    )


def test_completed_audit_uses_success_message() -> None:
    message, level = _audit_status_message(
        _snapshot(status="completed", reason="agent_finished")
    )

    assert level == "success"
    assert "已完成" in message
    assert "证据收集" in message


def test_budget_exhausted_audit_explains_partial_result() -> None:
    message, level = _audit_status_message(
        _snapshot(status="partial", reason="budget_exhausted")
    )

    assert level == "warning"
    assert "预算" in message
    assert "部分回答" in message


def test_fatal_error_audit_preserves_completed_work_message() -> None:
    message, level = _audit_status_message(
        _snapshot(status="failed", reason="fatal_error")
    )

    assert level == "error"
    assert "保留" in message
