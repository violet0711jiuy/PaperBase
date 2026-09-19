"""Decision Agent：只选择下一步动作，不执行 Tool，也不生成最终答案。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCompletion,
    complete_json_with_diagnostics,
)
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
    LLMCallRecord,
    QueryContext,
    ResearchObservation,
    ResearchPlan,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)
from paperbase.llm.diagnostics import failure_record, repair_hint, success_record
from .schemas import (
    AGENT_DECISION_ADAPTER,
    ActionDraft,
    GetPaperOverviewDraft,
    GetSectionContextDraft,
    SearchKnowledgeBaseDraft,
    SearchPapersDraft,
    SearchReferencesDraft,
)
from .research_profile import build_research_profiles


class DecisionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        llm_calls: int,
        llm_diagnostics: tuple[LLMCallRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.llm_calls = llm_calls
        self.llm_diagnostics = llm_diagnostics


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
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()


class DecisionAgent(Protocol):
    def decide(self, state: DecisionState) -> DecisionResult: ...


class LLMDecisionAgent:
    def __init__(self, client: ChatCompletionClient, *, max_attempts: int = 2) -> None:
        self._client = client
        self._max_attempts = max_attempts

    def decide(self, state: DecisionState) -> DecisionResult:
        errors: list[str] = []
        diagnostics: list[LLMCallRecord] = []
        evidence_limit = 12
        for attempt in range(1, self._max_attempts + 1):
            completion: LLMCompletion | None = None
            try:
                completion = complete_json_with_diagnostics(
                    self._client,
                    system_prompt=AGENT_DECISION_SYSTEM_PROMPT,
                    user_prompt=_state_prompt(
                        state, errors, evidence_limit=evidence_limit
                    ),
                    json_schema=AGENT_DECISION_ADAPTER.json_schema(),
                    schema_name="paperbase_agent_action",
                )
                draft = AGENT_DECISION_ADAPTER.validate_json(completion.content)
                action = _to_action(draft.decision)
                _validate_action_references(action, state)
                diagnostics.append(
                    success_record(
                        node="decide_action", attempt=attempt, completion=completion
                    )
                )
                return DecisionResult(
                    action, attempt, llm_diagnostics=tuple(diagnostics)
                )
            except Exception as error:
                record = failure_record(
                    node="decide_action",
                    attempt=attempt,
                    error=error,
                    completion=completion,
                    can_retry=attempt < self._max_attempts,
                )
                diagnostics.append(record)
                errors.append(repair_hint(error))
                if record.recovery_action == "compact_context":
                    evidence_limit = 6
        raise DecisionError(
            "Decision output remained invalid after repair.",
            llm_calls=self._max_attempts,
            llm_diagnostics=tuple(diagnostics),
        )


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


def _state_prompt(
    state: DecisionState, errors: list[str], *, evidence_limit: int = 12
) -> str:
    # Decision 只读取 Evidence Analysis 产生的短摘要与 Claim；完整原文保留在 Registry。
    visible_evidence = _visible_evidence(state, limit=evidence_limit)
    profile_paper_ids = _profile_paper_ids(state)
    research_profiles = build_research_profiles(
        paper_ids=profile_paper_ids,
        plan=state.plan,
        evidence=state.evidence,
        observations=state.observations,
        default_targets=state.context.default_evidence_targets,
    )
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
        "research_profiles": [
            {
                "paper_id": profile.paper_id,
                "queries_tried": profile.queries_tried,
                "requested_targets": sorted(
                    target.value for target in profile.requested_targets
                ),
                "covered_targets": sorted(
                    target.value for target in profile.covered_targets
                ),
                "remaining_targets": sorted(
                    target.value for target in profile.remaining_targets
                ),
                "requested_information": profile.requested_information,
                "addressed_information": profile.addressed_information,
                "remaining_information": profile.remaining_information,
                "sections_seen": profile.sections_seen,
                "qualified_digests": profile.qualified_digests,
                "qualified_claims": profile.qualified_claims,
                "empty_search_count": profile.empty_search_count,
            }
            for profile in research_profiles
        ],
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


def _profile_paper_ids(state: DecisionState) -> tuple[str, ...]:
    if state.coverage is not None:
        return state.coverage.required_paper_ids
    if state.candidate_selection is not None:
        return tuple(
            dict.fromkeys(
                (
                    *state.candidate_selection.candidate_paper_ids,
                    *state.candidate_selection.uncertain_paper_ids,
                )
            )
        )
    return tuple(
        dict.fromkeys(
            (
                *state.context.target_paper_ids,
                *(record.evidence.paper_id for record in state.evidence),
            )
        )
    )


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
    state: DecisionState, *, limit: int = 12
) -> tuple[EvidenceRecord, ...]:
    """按当前问题相关性排序，并先为每篇论文保留一个可见位置。"""
    evidence = state.evidence
    query_terms = _query_terms(state.context)
    remaining_ids = set(state.coverage.remaining_paper_ids if state.coverage else ())

    def rank(record: EvidenceRecord) -> tuple[float, ...]:
        searchable = " ".join(
            (
                record.digest or "",
                *(claim.statement for claim in record.claims),
            )
        ).casefold()
        relevance = float(sum(term in searchable for term in query_terms))
        return (
            float(record.evidence.paper_id in remaining_ids),
            relevance,
            float(bool(record.claims)),
            float(
                any(
                    claim.relation.value in {"supports", "contradicts"}
                    for claim in record.claims
                )
            ),
            record.evidence.score or 0.0,
        )

    ranked = sorted(
        evidence,
        key=rank,
        reverse=True,
    )
    selected: list[EvidenceRecord] = []
    counts: dict[str, int] = {}
    # 第一轮每篇只取最高相关的一条，避免 limit 较小时被单篇高分 Evidence 吞没。
    for record in ranked:
        paper_id = record.evidence.paper_id
        if paper_id in counts:
            continue
        selected.append(record)
        counts[paper_id] = 1
        if len(selected) >= limit:
            return tuple(selected)
    # 第二轮再按全局相关性补齐，同时保留单篇上限。
    for record in ranked:
        paper_id = record.evidence.paper_id
        if record in selected:
            continue
        if counts.get(paper_id, 0) >= 3:
            continue
        selected.append(record)
        counts[paper_id] = counts.get(paper_id, 0) + 1
        if len(selected) >= limit:
            break
    return tuple(selected)


def _query_terms(context: QueryContext) -> tuple[str, ...]:
    """提取轻量、确定性的中英文相关性词项，不额外调用模型。"""
    values = [
        context.resolved_query,
        context.semantic_query_en or "",
        *context.lexical_keywords_en,
    ]
    terms: list[str] = []
    for value in values:
        folded = value.casefold()
        terms.extend(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", folded))
        for sequence in re.findall(r"[\u3400-\u9fff]+", folded):
            terms.extend(
                sequence[index : index + 2]
                for index in range(max(0, len(sequence) - 1))
            )
    return tuple(dict.fromkeys(term for term in terms if len(term) >= 2))
