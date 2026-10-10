"""把 QueryContext 转成回答义务（Requirements），不规划执行步骤。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCompletion,
    LLMRequestError,
    ValidationDiagnostic,
    complete_json_with_diagnostics,
    structured_client_view,
)
from paperbase.prompts.agentic_planner import (
    RESEARCH_PLANNER_SYSTEM_PROMPT,
    build_research_planner_user_prompt,
)

from .contracts import (
    LLMCallRecord,
    PlanningMode,
    QueryContext,
    EvidenceProjectionType,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
)
from paperbase.llm.diagnostics import (
    classify_llm_error, failure_record, recovery_action_for,
    safe_validation_summary, success_record,
)
from .schemas import ResearchPlanDraft
from .projection_shapes import _legacy_projection_type


@dataclass(frozen=True)
class PlannerResult:
    plan: ResearchPlan
    used_fallback: bool
    llm_calls: int
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()


class PlannerDomainValidationError(ValueError):
    """A deterministic scope violation with no model text in diagnostics."""

    def __init__(
        self, *, code: str, field_path: str, requirement_id: str | None,
        safe_reason: str, expected_constraint: str, actual_summary: str,
    ) -> None:
        super().__init__(safe_reason)
        self.code = code
        self.field_path = field_path
        self.requirement_id = requirement_id
        self.validator_name = "planner_requirement_scope"
        self.safe_reason = safe_reason
        self.expected_constraint = expected_constraint
        self.actual_summary = actual_summary


class ResearchPlanner:
    def __init__(
        self, client: ChatCompletionClient | None = None, *, max_attempts: int = 2,
        temperature: float = 0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        self._client = structured_client_view(client, temperature=temperature) if client is not None else None
        self._max_attempts = max_attempts

    def plan(self, context: QueryContext) -> PlannerResult:
        if self._client is None:
            return PlannerResult(_fallback_plan(context), True, 0)
        user_prompt = build_research_planner_user_prompt(
            question=context.resolved_query,
            intent=context.intent.value,
            scope=context.scope.value,
            trusted_paper_ids=context.target_paper_ids,
            evidence_targets=sorted(
                target.value for target in context.default_evidence_targets
            ),
            retrieval_goal=context.retrieval_goal.value,
        )
        diagnostics: list[LLMCallRecord] = []
        for attempt in range(1, self._max_attempts + 1):
            completion: LLMCompletion | None = None
            try:
                completion = complete_json_with_diagnostics(
                    self._client,
                    system_prompt=RESEARCH_PLANNER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_schema=ResearchPlanDraft.model_json_schema(),
                    schema_name="paperbase_research_plan",
                )
                draft = ResearchPlanDraft.model_validate_json(completion.content)
                plan = _validate_and_build(draft, context)
                diagnostics.append(success_record(
                    node="build_plan", attempt=attempt, completion=completion,
                ))
                return PlannerResult(plan, False, attempt, tuple(diagnostics))
            except (LLMRequestError, ValidationError, ValueError, TypeError) as error:
                can_retry = (
                    attempt < self._max_attempts
                    and not isinstance(error, LLMRequestError)
                )
                validation_diagnostics = _validation_diagnostics(
                    error, attempt=attempt, can_retry=can_retry
                )
                diagnostics.append(failure_record(
                    node="build_plan", attempt=attempt, error=error,
                    completion=completion, can_retry=can_retry,
                    validation_diagnostics=validation_diagnostics,
                ))
                if not can_retry:
                    break
                user_prompt = _repair_prompt(user_prompt, validation_diagnostics, error)
        return PlannerResult(
            _literal_fallback_plan(context), True, len(diagnostics),
            tuple(diagnostics),
        )


def _validate_and_build(draft: ResearchPlanDraft, context: QueryContext) -> ResearchPlan:
    requirements = tuple(
        ResearchRequirement(
            requirement_id=f"R{index}",
            description=item.description,
            required=item.required,
            evidence_targets=frozenset(item.evidence_targets),
            projection_type=item.projection_type,
        )
        for index, item in enumerate(draft.requirements, start=1)
    )
    _validate_requirement_scope(requirements, context)
    if any(item.projection_type is None for item in requirements):
        raise ValueError("New Planner Requirements must declare projection_type.")
    if not any(item.required for item in requirements):
        raise PlannerDomainValidationError(
            code="PLANNER_NO_REQUIRED_REQUIREMENT",
            field_path="requirements", requirement_id=None,
            safe_reason="研究计划至少需要一项必需的回答义务。",
            expected_constraint="At least one Requirement must be required.",
            actual_summary="required_count=0",
        )
    return ResearchPlan(
        goal=draft.goal, requirements=requirements,
        planning_mode=PlanningMode.MODEL,
    )


def _fallback_plan(context: QueryContext) -> ResearchPlan:
    return _literal_fallback_plan(context)


def _literal_fallback_plan(context: QueryContext) -> ResearchPlan:
    return ResearchPlan(
        goal=context.resolved_query,
        requirements=(ResearchRequirement(
            "R1", context.resolved_query, required=True,
            evidence_targets=context.default_evidence_targets,
            # Literal fallback has no reliable entity-domain judgment. Fact can
            # preserve any requested collection without inventing entity types;
            # model-authored method/model enumeration remains unchanged.
            projection_type=(EvidenceProjectionType.FACT
                if context.retrieval_goal is ResearchGoal.EXHAUSTIVE_ENUMERATION
                else _legacy_projection_type(context)),
        ),),
        planning_mode=PlanningMode.LITERAL_FALLBACK,
    )


def _validate_requirement_scope(
    requirements: tuple[ResearchRequirement, ...], context: QueryContext
) -> None:
    """Keep Planner output to answer obligations rather than execution plans."""
    question = context.resolved_query.casefold()
    forbidden = {
        "dataset": ("dataset", "数据集"),
        "metric": ("metric", "指标"),
        "preprocess": ("preprocess", "预处理"),
        "hyperparameter": ("hyperparameter", "超参数"),
    }
    for index, requirement in enumerate(requirements):
        description = requirement.description.casefold()
        if any(
            marker in description
            for marker in ("调用 tool", "调用工具", "先检索", "先搜索", "search papers")
        ):
            raise PlannerDomainValidationError(
                code="PLANNER_EXECUTION_STEP",
                field_path=f"requirements[{index}].description",
                requirement_id=requirement.requirement_id,
                safe_reason="Requirement 必须描述回答义务，不能描述 Tool 调用步骤。",
                expected_constraint="Answer obligation, not a Tool step.",
                actual_summary="execution_step_marker_present",
            )
        unasked_detail = next((
            label for label, pattern in _ENGINEERING_DETAIL_PATTERNS.items()
            if pattern.search(description) and not pattern.search(question)
        ), None)
        if unasked_detail is not None:
            raise PlannerDomainValidationError(
                code="PLANNER_UNASKED_IMPLEMENTATION",
                field_path=f"requirements[{index}].description",
                requirement_id=requirement.requirement_id,
                safe_reason="Requirement 不得扩张为用户未要求的代码或工程实现义务。",
                expected_constraint="Engineering details must be explicitly requested.",
                actual_summary=f"unasked_{unasked_detail}_detail_present",
            )
        for label, terms in forbidden.items():
            if any(term in description for term in terms) and not any(
                term in question for term in terms
            ):
                raise PlannerDomainValidationError(
                    code=f"PLANNER_UNASKED_{label.upper()}",
                    field_path=f"requirements[{index}].description",
                    requirement_id=requirement.requirement_id,
                    safe_reason=f"Requirement 不得在用户未要求时扩张为 {label} 义务。",
                    expected_constraint="Only user-requested answer dimensions are allowed.",
                    actual_summary=f"unasked_{label}_dimension_present",
                )


_ENGINEERING_DETAIL_PATTERNS = {
    "code": re.compile(r"代码|源码|源代码|\b(?:source code|code implementation)\b"),
    "deployment": re.compile(r"部署|\b(?:deployment|deploy)\b"),
    "api": re.compile(r"\bapi\b"),
    "engineering_reproduction": re.compile(r"工程复现|工程实现|\bengineering implementation\b"),
}


def _validation_diagnostics(
    error: Exception, *, attempt: int, can_retry: bool,
) -> tuple[ValidationDiagnostic, ...]:
    if isinstance(error, PlannerDomainValidationError):
        return (ValidationDiagnostic(
            node="build_plan", attempt=attempt, error_stage="domain_validation",
            error_code=error.code, field_path=error.field_path,
            validator_name=error.validator_name, safe_reason=error.safe_reason,
            expected_constraint=error.expected_constraint,
            actual_summary=error.actual_summary,
            recovery_action=recovery_action_for(error, can_retry=can_retry),
            requirement_id=error.requirement_id,
        ),)
    if isinstance(error, ValidationError):
        items = error.errors()
        return tuple(ValidationDiagnostic(
            node="build_plan", attempt=attempt,
            error_stage=classify_llm_error(error),
            error_code=str(item.get("type", "schema_validation")),
            field_path=".".join(str(part) for part in item.get("loc", ())) or "root",
            validator_name="ResearchPlanDraft",
            safe_reason="Planner 输出未通过 Schema 校验。",
            expected_constraint="Follow ResearchPlanDraft JSON Schema.",
            actual_summary=str(item.get("type", "validation_error")),
            recovery_action=recovery_action_for(error, can_retry=can_retry),
        ) for item in items[:3])
    return ()


def _repair_prompt(
    original_prompt: str,
    diagnostics: tuple[ValidationDiagnostic, ...],
    error: Exception,
) -> str:
    payload = json.loads(original_prompt)
    payload["validation_repair"] = {
        "instruction": "上一份计划未通过校验。保留用户问题的全部限定条件，只修复违例字段，重新输出完整 JSON。",
        "error_stage": classify_llm_error(error),
        "validation_summary": safe_validation_summary(error),
        "issues": [
            {
                "rule_code": item.error_code,
                "field_path": item.field_path,
                "requirement_id": item.requirement_id,
                "safe_reason": item.safe_reason,
            }
            for item in diagnostics
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
