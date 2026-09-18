"""Decision Agent：只选择下一步动作，不执行 Tool，也不生成最终答案。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from paperbase.llm.client import ChatCompletionClient
from paperbase.prompts.agentic_decision import (
    AGENT_DECISION_SYSTEM_PROMPT,
    build_agent_decision_user_prompt,
)
from paperbase.retrieval.contracts import PaperSearchSpec

from .contracts import (
    AgentAction,
    BudgetUsage,
    CoverageState,
    EvidenceQualification,
    EvidenceRecord,
    FinishAction,
    GetPaperOverviewAction,
    GetSectionContextAction,
    QueryContext,
    ResearchObservation,
    ResearchPlan,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)
from .schemas import (
    AGENT_DECISION_ADAPTER,
    ActionDraft,
    GetPaperOverviewDraft,
    GetSectionContextDraft,
    SearchKnowledgeBaseDraft,
    SearchPapersDraft,
    SearchReferencesDraft,
)


class DecisionError(RuntimeError):
    def __init__(self, message: str, *, llm_calls: int) -> None:
        super().__init__(message)
        self.llm_calls = llm_calls


@dataclass(frozen=True)
class DecisionState:
    context: QueryContext
    plan: ResearchPlan
    evidence: tuple[EvidenceRecord, ...]
    observations: tuple[ResearchObservation, ...]
    budget_usage: BudgetUsage
    suppressed_paper_ids: tuple[str, ...] = ()
    coverage: CoverageState | None = None


@dataclass(frozen=True)
class DecisionResult:
    action: AgentAction
    llm_calls: int
    qualifications: tuple[EvidenceQualification, ...] = ()


class DecisionAgent(Protocol):
    def decide(self, state: DecisionState) -> DecisionResult: ...


class LLMDecisionAgent:
    def __init__(self, client: ChatCompletionClient, *, max_attempts: int = 2) -> None:
        self._client = client
        self._max_attempts = max_attempts

    def decide(self, state: DecisionState) -> DecisionResult:
        errors: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._client.complete_json(
                    system_prompt=AGENT_DECISION_SYSTEM_PROMPT,
                    user_prompt=_state_prompt(state, errors),
                    json_schema=AGENT_DECISION_ADAPTER.json_schema(),
                    schema_name="paperbase_agent_action",
                )
                draft = AGENT_DECISION_ADAPTER.validate_json(raw)
                action = _to_action(draft.decision)
                _validate_action_references(action, state)
                qualifications = tuple(
                    EvidenceQualification(
                        evidence_id=item.evidence_id,
                        relation=item.relation,
                        ownership=item.ownership,
                        rationale=item.rationale,
                    )
                    for item in draft.qualifications
                )
                _validate_qualifications(qualifications, state)
                return DecisionResult(action, attempt, qualifications)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        raise DecisionError("Decision output remained invalid after repair.", llm_calls=self._max_attempts)


class ScriptedDecisionAgent:
    """确定性测试替身；每次决定仍计为一次语义调用。"""

    def __init__(self, actions: tuple[AgentAction, ...]) -> None:
        self._actions = list(actions)

    def decide(self, state: DecisionState) -> DecisionResult:
        if not self._actions:
            return DecisionResult(FinishAction("script exhausted"), 1)
        return DecisionResult(self._actions.pop(0), 1)


def _to_action(draft: ActionDraft) -> AgentAction:
    if isinstance(draft, SearchPapersDraft):
        return SearchPapersAction(
            searches=tuple(
                PaperSearchSpec(
                    paper_id=item.paper_id,
                    query=item.query,
                    lexical_terms=tuple(item.lexical_terms),
                    evidence_targets=frozenset(
                        target.value for target in item.evidence_targets
                    ),
                    missing_information=tuple(item.missing_information),
                    target_subtask_id=draft.target_subtask_id,
                )
                for item in draft.searches
            ),
            target_subtask_id=draft.target_subtask_id,
        )
    if isinstance(draft, SearchKnowledgeBaseDraft):
        return SearchKnowledgeBaseAction(
            draft.query, tuple(draft.lexical_terms), draft.target_subtask_id
        )
    if isinstance(draft, SearchReferencesDraft):
        return SearchReferencesAction(
            draft.paper_id,
            draft.query,
            tuple(draft.lexical_terms),
            draft.target_subtask_id,
        )
    if isinstance(draft, GetSectionContextDraft):
        return GetSectionContextAction(
            draft.paper_id, draft.section_id, draft.target_subtask_id
        )
    if isinstance(draft, GetPaperOverviewDraft):
        return GetPaperOverviewAction(draft.paper_id, draft.target_subtask_id)
    return FinishAction(draft.reason)


def _state_prompt(state: DecisionState, errors: list[str]) -> str:
    # Decision 只需要近期、有界的证据摘要来形成下一条动态 Query；完整原文继续保留在
    # Registry，不把无界检索上下文塞进每次模型调用。
    visible_evidence = state.evidence[-12:]
    summary = {
        "question": state.context.resolved_query,
        "scope": state.context.scope.value,
        "trusted_paper_ids": state.context.target_paper_ids,
        "subtasks": [
            {
                "id": task.subtask_id,
                "goal": task.goal,
                "paper_ids": task.target_paper_ids,
                "status": task.status.value,
            }
            for task in state.plan.subtasks
        ],
        "evidence": [
            {
                "id": record.evidence_id,
                "paper_id": record.evidence.paper_id,
                "source_type": record.evidence.source_type.value,
                "section_id": record.evidence.section_id,
                "section": record.evidence.section,
                "excerpt": record.evidence.content[:800],
                "supporting_queries": record.supporting_queries[-3:],
                "target_subtask_ids": record.target_subtask_ids,
                "qualification": (
                    {
                        "relation": record.qualification.relation.value,
                        "ownership": record.qualification.ownership.value,
                        "rationale": record.qualification.rationale,
                    }
                    if record.qualification is not None
                    else None
                ),
            }
            for record in visible_evidence
        ],
        "evidence_visibility": {
            "shown": len(visible_evidence),
            "total": len(state.evidence),
        },
        "recent_observations": [
            {
                "action": item.action_name,
                "status": item.tool_status.value,
                "added": item.added_evidence_ids,
                "errors": item.error_codes,
            }
            for item in state.observations[-4:]
        ],
        "suppressed_paper_ids": state.suppressed_paper_ids,
        "budget_usage": state.budget_usage.__dict__,
        "coverage": (
            {
                "strategy": state.coverage.strategy.value,
                "complete": state.coverage.coverage_complete,
                "required_paper_ids": state.coverage.required_paper_ids,
                "remaining_paper_ids": state.coverage.remaining_paper_ids,
                "assessments": [
                    {
                        "paper_id": item.paper_id,
                        "status": item.status.value,
                        "queries_tried": item.queries_tried,
                        "rationale": item.rationale,
                    }
                    for item in state.coverage.assessments
                ],
            }
            if state.coverage is not None
            else None
        ),
        "previous_validation_errors": errors[-1:],
    }
    return build_agent_decision_user_prompt(summary)


def _validate_action_references(action: AgentAction, state: DecisionState) -> None:
    """拒绝无法从可信 DecisionState 中定位来源的模型引用。"""
    if not isinstance(action, GetSectionContextAction):
        return
    known_sections = {
        (record.evidence.paper_id, record.evidence.section_id)
        for record in state.evidence
        if record.evidence.section_id
    }
    if (action.paper_id, action.section_id) not in known_sections:
        raise ValueError(
            "get_section_context 的 paper_id 和 section_id 必须已经出现在 Evidence 中。"
        )


def _validate_qualifications(
    qualifications: tuple[EvidenceQualification, ...], state: DecisionState
) -> None:
    known_ids = {record.evidence_id for record in state.evidence}
    ids = [item.evidence_id for item in qualifications]
    if len(ids) != len(set(ids)):
        raise ValueError("qualifications 不能重复引用同一 Evidence ID。")
    unknown_ids = tuple(evidence_id for evidence_id in ids if evidence_id not in known_ids)
    if unknown_ids:
        raise ValueError("qualifications 引用了不可见 Evidence：" + ", ".join(unknown_ids))
