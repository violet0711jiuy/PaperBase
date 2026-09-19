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
    CandidateSelection,
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
    candidate_selection: CandidateSelection | None = None


@dataclass(frozen=True)
class DecisionResult:
    action: AgentAction
    llm_calls: int
    # 兼容旧的 ScriptedDecisionAgent / ResearchRuntime；生产 LLM Decision 不再生成此字段。
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
                return DecisionResult(action, attempt)
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
        paper_ids = [item.paper_id for item in draft.searches]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError(
                "searches 内不能重复 paper_id；同一论文的不同 Query 必须分轮执行。"
            )
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
    # Decision 只读取 Evidence Analysis 产生的短摘要与 Claim；完整原文保留在 Registry。
    visible_evidence = _visible_evidence(state.evidence)
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
                "analysis_status": record.analysis_status.value,
                "digest": record.digest,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "statement": claim.statement,
                        "relation": claim.relation.value,
                        "ownership": claim.ownership.value,
                    }
                    for claim in record.claims
                ],
                "supporting_queries": record.supporting_queries[-3:],
                "target_subtask_ids": record.target_subtask_ids,
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
                "screened_paper_ids": state.coverage.screened_paper_ids,
                "candidate_paper_ids": state.coverage.candidate_paper_ids,
                "uncertain_paper_ids": state.coverage.uncertain_paper_ids,
                "excluded_paper_ids": state.coverage.excluded_paper_ids,
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
        "candidate_selection": (
            [
                {
                    "paper_id": item.paper_id,
                    "disposition": item.disposition.value,
                    "reason": item.reason,
                }
                for item in state.candidate_selection.assessments
            ]
            if state.candidate_selection is not None
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


def _visible_evidence(
    evidence: tuple[EvidenceRecord, ...], *, limit: int = 12
) -> tuple[EvidenceRecord, ...]:
    """优先给 Decision 展示已分析、可支持结论且跨论文分布的短摘要。"""
    ranked = sorted(
        evidence,
        key=lambda record: (
            bool(record.claims),
            any(
                claim.relation.value in {"supports", "contradicts"}
                for claim in record.claims
            ),
            record.evidence.score or 0.0,
        ),
        reverse=True,
    )
    selected: list[EvidenceRecord] = []
    counts: dict[str, int] = {}
    for record in ranked:
        paper_id = record.evidence.paper_id
        if counts.get(paper_id, 0) >= 3:
            continue
        selected.append(record)
        counts[paper_id] = counts.get(paper_id, 0) + 1
        if len(selected) >= limit:
            break
    return tuple(selected)
