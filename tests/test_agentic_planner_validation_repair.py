"""Planner scope, safe diagnostics, and semantic-preserving fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paperbase.agentic.audit import build_agentic_audit_snapshot
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    EvidenceTarget, Intent, PlanningMode, QueryContext, QueryScope,
    ResearchGoal, ResearchPlan, ScopeSource,
)
from paperbase.agentic.planner import ResearchPlanner


class QueuedClient:
    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def complete_json(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["user_prompt"]))
        return self.payloads.pop(0)


def context(question: str) -> QueryContext:
    return QueryContext(
        original_query=question, resolved_query=question,
        intent=Intent.SYNTHESIZE, scope=QueryScope.KNOWLEDGE_BASE,
        target_paper_ids=(), scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
        retrieval_goal=ResearchGoal.SYNTHESIS_SUPPORT,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def payload(description: str) -> str:
    return json.dumps({
        "goal": "解释所问机制",
        "requirements": [{
            "description": description, "required": True,
            "evidence_targets": ["method"], "projection_type": "fact"}],
    }, ensure_ascii=False)


def audit(result: object) -> object:
    return build_agentic_audit_snapshot(SimpleNamespace(
        status="completed", stop_reason="agent_finished", thread_id="planner-test",
        query_context=None, candidate_selection=None, coverage=None,
        plan=result.plan, requirement_states=(), evidence=(),
        selected_evidence=(), observations=(), budget_usage=None,
        answer=None, llm_diagnostics=result.llm_diagnostics, error_codes=(),
        clarification=None,
    ))


def test_mechanism_implementation_word_is_allowed() -> None:
    question = "该方法如何结合损失约束和预测区间？"
    result = ResearchPlanner(QueuedClient(payload("解释两者结合的具体实现方式"))).plan(
        context(question)
    )
    assert not result.used_fallback
    assert result.plan.planning_mode is PlanningMode.MODEL
    assert result.plan.requirements[0].description == "解释两者结合的具体实现方式"


def test_unasked_code_implementation_is_rejected_with_auditable_diagnostic() -> None:
    question = "该方法如何结合损失约束和预测区间？"
    result = ResearchPlanner(
        QueuedClient(payload("解释代码实现")), max_attempts=1
    ).plan(context(question))
    assert result.used_fallback
    diagnostic = result.llm_diagnostics[0].validation_diagnostics[0]
    assert diagnostic.error_code == "PLANNER_UNASKED_IMPLEMENTATION"
    assert diagnostic.field_path == "requirements[0].description"
    assert diagnostic.requirement_id == "R1"
    snapshot = audit(result)
    assert snapshot.validation_errors[0].stage == "domain_validation"
    assert snapshot.validation_errors[0].rule_code == "PLANNER_UNASKED_IMPLEMENTATION"
    assert snapshot.validation_errors[0].field_path == "requirements[0].description"
    assert snapshot.validation_errors[0].requirement_id == "R1"
    assert snapshot.validation_errors[0].validator_name == "planner_requirement_scope"
    assert snapshot.validation_errors[0].safe_reason
    assert snapshot.planning_mode == "literal_fallback"
    assert "解释代码实现" not in snapshot.to_json()


def test_explicit_code_implementation_is_allowed() -> None:
    result = ResearchPlanner(QueuedClient(payload("解释代码实现"))).plan(
        context("请解释该方法的代码实现")
    )
    assert not result.used_fallback


@pytest.mark.parametrize("description", (
    "Explain the implementation mechanism",
    "解释输出机制",
    "说明工作机制",
))
def test_mechanism_language_does_not_trigger_engineering_rejection(description: str) -> None:
    result = ResearchPlanner(QueuedClient(payload(description))).plan(
        context("该方法如何结合约束与预测区间？")
    )
    assert not result.used_fallback


@pytest.mark.parametrize("description", (
    "给出源码", "说明部署步骤", "列出 API 接口", "提供工程复现细节",
))
def test_explicit_unasked_engineering_dimensions_are_rejected(description: str) -> None:
    result = ResearchPlanner(QueuedClient(payload(description)), max_attempts=1).plan(
        context("该方法如何结合约束与预测区间？")
    )
    assert result.used_fallback
    assert result.llm_diagnostics[0].validation_diagnostics[0].error_code == "PLANNER_UNASKED_IMPLEMENTATION"


def test_two_domain_failures_preserve_entire_resolved_query_in_literal_fallback() -> None:
    question = "LSTM-EMVE 如何结合 MLE/L_QV、MPIWcapt/C_i 和 μ/σ 分支，得到区间上下界？"
    client = QueuedClient(payload("给出代码实现"), payload("给出部署代码实现"))
    result = ResearchPlanner(client).plan(context(question))
    assert result.used_fallback
    assert result.llm_calls == 2
    assert result.plan.planning_mode is PlanningMode.LITERAL_FALLBACK
    assert result.plan.goal == question
    assert len(result.plan.requirements) == 1
    fallback = result.plan.requirements[0]
    assert fallback.requirement_id == "R1"
    assert fallback.description == question
    assert fallback.required is True
    assert fallback.evidence_targets == frozenset({EvidenceTarget.METHOD})
    assert "核心结论" not in fallback.description
    assert result.llm_diagnostics[0].recovery_action == "repair_domain"
    assert json.loads(client.prompts[1])["validation_repair"]["issues"][0]["rule_code"] == "PLANNER_UNASKED_IMPLEMENTATION"
    assert "给出代码实现" not in client.prompts[1]
    assert load_contract(dump_contract(result.plan, ResearchPlan), ResearchPlan).planning_mode is PlanningMode.LITERAL_FALLBACK


def test_schema_failure_then_domain_failure_remain_distinct_in_audit() -> None:
    invalid_schema = json.dumps({"goal": "x", "requirements": [{"description": "x"}]})
    result = ResearchPlanner(QueuedClient(
        invalid_schema, payload("给出代码实现"),
    )).plan(context("如何结合约束与预测区间？"))
    snapshot = audit(result)
    assert [call.error_stage for call in result.llm_diagnostics] == [
        "schema_validation", "domain_validation",
    ]
    assert list(dict.fromkeys(item.stage for item in snapshot.validation_errors)) == [
        "schema_validation", "domain_validation",
    ]
    assert snapshot.planning_mode == "literal_fallback"
