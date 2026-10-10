from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paperbase.agentic.contracts import (
    BudgetUsage,
    EvidenceTarget,
    Intent,
    PlanningMode,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
    RequirementState,
    RequirementStatus,
    SearchReferencesAction,
    ScopeSource,
    SearchPapersAction,
)
from paperbase.agentic.decision import DecisionState, LLMDecisionAgent
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.agentic.planner import ResearchPlanner
from paperbase.agentic.schemas import AGENT_DECISION_ADAPTER
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import PaperSearchSpec


def _context(*, exhaustive: bool = False) -> QueryContext:
    return QueryContext(
        original_query="目前知识库中能检索到哪些风速预测的方法",
        resolved_query="目前知识库中能检索到哪些风速预测的方法",
        intent=Intent.SYNTHESIZE,
        scope=QueryScope.KNOWLEDGE_BASE,
        target_paper_ids=(),
        scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
        retrieval_goal=(
            ResearchGoal.EXHAUSTIVE_ENUMERATION
            if exhaustive
            else ResearchGoal.SYNTHESIS_SUPPORT
        ),
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="识别风速预测方法",
        requirements=(
            ResearchRequirement(
                "R1",
                "识别论文自身提出或实际采用的风速预测核心方法，不包含 baseline 或 prior work",
                required=True,
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
            ),
            ResearchRequirement(
                "R2",
                "识别用于实验比较的 baseline/comparison methods",
                required=True,
                evidence_targets=frozenset({EvidenceTarget.METHOD}),
            ),
            ResearchRequirement(
                "R3",
                "识别正文明确讨论的 prior work 方法或方法族",
                required=True,
                evidence_targets=frozenset({EvidenceTarget.PRIOR_WORK}),
            ),
        ),
    )


class _QueuedClient:
    def __init__(self, payloads: tuple[str, ...]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.payloads.pop(0)


def test_enumeration_planner_does_not_expand_to_unasked_research_dimensions() -> None:
    plan = ResearchPlanner().plan(_context(exhaustive=True)).plan

    assert not hasattr(plan, "subtasks")
    assert [item.requirement_id for item in plan.requirements] == ["R1"]
    assert plan.requirements[0].description == _context(exhaustive=True).resolved_query
    assert plan.planning_mode is PlanningMode.LITERAL_FALLBACK
    descriptions = " ".join(item.description.casefold() for item in plan.requirements)
    for forbidden in (
        "dataset",
        "数据集",
        "metric",
        "指标",
        "预处理",
        "超参数",
        "代码",
        "去重",
        "归并",
        "整理",
        "provenance",
    ):
        assert forbidden not in descriptions


def test_exhaustive_model_plan_keeps_only_user_requested_obligations():
    raw = {"goal":"只列出核心方法", "requirements":[{
        "description":"只识别用户要求的核心方法，不包含 baseline 或 prior work",
        "required":True, "evidence_targets":["method"], "projection_type":"enumeration"}]}
    result = ResearchPlanner(_QueuedClient((json.dumps(raw),))).plan(_context(exhaustive=True))
    assert not result.used_fallback
    assert result.plan.planning_mode is PlanningMode.MODEL
    assert len(result.plan.requirements) == 1
    requirement = result.plan.requirements[0]
    assert requirement.description == raw['requirements'][0]['description']
    assert requirement.required and requirement.projection_type.value == 'enumeration'
    assert requirement.evidence_targets == frozenset({EvidenceTarget.METHOD})


def test_exhaustive_scope_validation_is_not_bypassed_by_a_template():
    raw = {"goal":"列出方法", "requirements":[{
        "description":"列出每个数据集的完整预处理和超参数",
        "required":True, "evidence_targets":["method"], "projection_type":"enumeration"}]}
    result = ResearchPlanner(_QueuedClient((json.dumps(raw),)), max_attempts=1).plan(_context(exhaustive=True))
    assert result.used_fallback
    assert result.llm_diagnostics[0].validation_diagnostics[0].error_code == 'PLANNER_UNASKED_DATASET'
    assert len(result.plan.requirements) == 1
    assert result.plan.requirements[0].description == _context(exhaustive=True).resolved_query


def test_exhaustive_optional_only_model_plan_cannot_be_promoted_by_a_template():
    raw = {"goal":"列出方法", "requirements":[{
        "description":"识别核心方法", "required":False,
        "evidence_targets":["method"], "projection_type":"enumeration"}]}
    result = ResearchPlanner(_QueuedClient((json.dumps(raw),)), max_attempts=1).plan(_context(exhaustive=True))
    assert result.used_fallback
    assert result.llm_diagnostics[0].validation_diagnostics[0].error_code == 'PLANNER_NO_REQUIRED_REQUIREMENT'
    assert result.plan.requirements[0].description == _context(exhaustive=True).resolved_query


def test_planner_output_is_requirement_first_and_rejects_scope_expansion() -> None:
    valid = json.dumps(
        {
            "goal": "总结风速预测方法",
            "requirements": [
                {
                    "description": "识别论文提出或采用的风速预测方法",
                    "required": True,
                    "evidence_targets": ["method"], "projection_type": "fact"}
            ],
        }
    )
    plan = ResearchPlanner(_QueuedClient((valid,))).plan(_context()).plan
    assert not hasattr(plan, "subtasks")
    assert [(item.requirement_id, item.required) for item in plan.requirements] == [("R1", True)]

    overbroad = json.dumps(
        {
            "goal": "总结风速预测方法",
            "requirements": [
                {
                    "description": "列出数据集和超参数",
                    "required": True,
                    "evidence_targets": ["method"], "projection_type": "fact"}
            ],
        }
    )
    fallback = ResearchPlanner(_QueuedClient((overbroad,)), max_attempts=1).plan(_context())
    assert fallback.used_fallback is True
    assert fallback.plan.requirements[0].description != "列出数据集和超参数"

    execution_step = json.dumps(
        {
            "goal": "总结风速预测方法",
            "requirements": [
                {
                    "description": "先搜索论文，再识别风速预测方法",
                    "required": True,
                    "evidence_targets": ["method"], "projection_type": "fact"}
            ],
        }
    )
    fallback = ResearchPlanner(_QueuedClient((execution_step,)), max_attempts=1).plan(_context())
    assert fallback.used_fallback is True


def test_planner_rejects_plan_without_required_answer_obligation() -> None:
    optional_only = json.dumps({
        "goal": "总结风速预测方法",
        "requirements": [{
            "description": "识别风速预测方法",
            "required": False,
            "evidence_targets": ["method"], "projection_type": "fact"}],
    })
    result = ResearchPlanner(_QueuedClient((optional_only,)), max_attempts=1).plan(_context())
    assert result.used_fallback
    assert any(item.required for item in result.plan.requirements)


@pytest.mark.parametrize(
    "decision",
    (
        {"action": "search_papers", "searches": [{"paper_id": "A", "query": "Find wind speed forecasting methods used by this paper", "lexical_terms": ["wind speed forecasting"], "requirement_ids": ["R1"]}]},
        {"action": "search_knowledge_base", "query": "Find papers that describe wind speed forecasting methods", "lexical_terms": ["wind speed forecasting"], "requirement_ids": ["R1"]},
        {"action": "search_references", "paper_id": "A", "query": "Find cited prior work on wind speed forecasting methods", "lexical_terms": ["prior work", "wind speed forecasting"], "requirement_ids": ["R3"]},
        {"action": "get_section_context", "paper_id": "A", "section_id": "methods", "requirement_ids": ["R1"]},
        {"action": "get_paper_overview", "paper_id": "A", "requirement_ids": ["R1"]},
        {"action": "finish", "reason": "No further evidence is needed."},
    ),
)
def test_agent_action_discriminated_union_accepts_each_tool_shape(
    decision: dict[str, object],
) -> None:
    parsed = AGENT_DECISION_ADAPTER.validate_json(json.dumps({"decision": decision}))
    assert parsed.decision.action == decision["action"]


def test_agent_action_rejects_cross_tool_fields_and_unbound_search() -> None:
    with pytest.raises(ValidationError):
        AGENT_DECISION_ADAPTER.validate_json(
            json.dumps(
                {
                    "decision": {
                        "action": "get_paper_overview",
                        "paper_id": "A",
                        "query": "not allowed for overview",
                        "requirement_ids": ["R1"],
                    }
                }
            )
        )
    with pytest.raises(ValidationError):
        AGENT_DECISION_ADAPTER.validate_json(
            json.dumps(
                {
                    "decision": {
                        "action": "search_papers",
                        "searches": [
                            {
                                "paper_id": "A",
                                "query": "Find methods",
                                "lexical_terms": [],
                                "evidence_targets": ["method"],
                                "requirement_ids": [],
                            }
                        ],
                    }
                }
            )
        )


def test_runtime_guard_requires_unresolved_requirement_and_rejects_satisfied_one() -> None:
    action = SearchPapersAction(
        (PaperSearchSpec("A", "Find wind speed forecasting methods", requirement_ids=("R1",)),)
    )
    tracker = BudgetAndGuardTracker(AgenticResearchSettings())
    context = QueryContext(
        original_query="方法",
        resolved_query="方法",
        intent=Intent.RETRIEVE,
        scope=QueryScope.SINGLE_PAPER,
        target_paper_ids=("A",),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    assert tracker.validate_action(
        action, context=context, plan=_plan(), snapshot=None
    ).accepted

    rejected = BudgetAndGuardTracker(AgenticResearchSettings()).validate_action(
        action,
        context=context,
        plan=_plan(),
        snapshot=None,
        requirement_states=(RequirementState("R1", RequirementStatus.SATISFIED, ("E1",)),),
    )
    assert rejected.accepted  # support_found does not close further research


def test_decision_binds_baseline_and_prior_work_to_their_requirements() -> None:
    response = json.dumps(
        {
            "decision": {
                "action": "search_papers",
                "searches": [
                    {
                        "paper_id": "A",
                        "query": "Identify the wind speed forecasting method proposed by paper A",
                        "lexical_terms": ["wind speed forecasting", "proposed method"],
                        "evidence_targets": ["method"],
                        "requirement_ids": ["R1"],
                    },
                    {
                        "paper_id": "B",
                        "query": "Identify baseline wind speed forecasting methods compared in paper B",
                        "lexical_terms": ["baseline", "comparison"],
                        "evidence_targets": ["method"],
                        "requirement_ids": ["R2"],
                    },
                ],
            }
        }
    )
    context = QueryContext(
        original_query="方法",
        resolved_query="方法",
        intent=Intent.RETRIEVE,
        scope=QueryScope.SELECTED_PAPERS,
        target_paper_ids=("A", "B"),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    action = LLMDecisionAgent(_QueuedClient((response,))).decide(
        DecisionState(context, _plan(), (), (), BudgetUsage())
    ).action
    assert isinstance(action, SearchPapersAction)
    assert [(item.paper_id, item.lexical_terms) for item in action.searches] == [
        ("A", ("wind speed forecasting", "proposed method")),
        ("B", ("baseline", "comparison")),
    ]
    assert [item.requirement_ids for item in action.searches] == [("R1",), ("R2",)]
    assert action.searches[0].query != action.searches[1].query

    prior_work_response = json.dumps(
        {
            "decision": {
                "action": "search_references",
                "paper_id": "A",
                "query": "Find prior work methods discussed by paper A for wind speed forecasting",
                "lexical_terms": ["prior work", "wind speed forecasting"],
                "requirement_ids": ["R3"],
            }
        }
    )
    prior_work_action = LLMDecisionAgent(_QueuedClient((prior_work_response,))).decide(
        DecisionState(context, _plan(), (), (), BudgetUsage())
    ).action
    assert isinstance(prior_work_action, SearchReferencesAction)
    assert prior_work_action.requirement_ids == ("R3",)
    assert prior_work_action.lexical_terms == ("prior work", "wind speed forecasting")
