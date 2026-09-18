"""Agent Loop 的预算、重复动作、Scope 与无新增证据保护。"""

from __future__ import annotations

from dataclasses import dataclass
import json

from paperbase.config import AgenticResearchSettings

from .contracts import (
    AgentAction,
    BudgetUsage,
    CorpusScopeSnapshot,
    FinishAction,
    GuardState,
    GetPaperOverviewAction,
    GetSectionContextAction,
    QueryContext,
    QueryScope,
    ResearchPlan,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    code: str
    detail: str
    budget_exhausted: bool = False


class BudgetAndGuardTracker:
    def __init__(
        self,
        settings: AgenticResearchSettings,
        *,
        initial_llm_calls: int = 0,
        state: GuardState | None = None,
    ) -> None:
        self.settings = settings
        snapshot = state or GuardState(
            usage=BudgetUsage(llm_calls=initial_llm_calls)
        )
        self.iterations = snapshot.usage.agent_iterations
        self.tool_calls = snapshot.usage.tool_calls
        self.retrieval_queries = snapshot.usage.retrieval_queries
        self.llm_calls = snapshot.usage.llm_calls
        self.invalid_actions = snapshot.usage.invalid_actions
        self.queries_per_paper = dict(snapshot.usage.queries_per_paper)
        self.no_new_per_paper = dict(snapshot.no_new_per_paper)
        self.suppressed_papers = set(snapshot.suppressed_paper_ids)
        self._fingerprints = set(snapshot.executed_action_fingerprints)

    def can_decide(self, *, reserved_llm_calls: int = 0) -> GuardDecision:
        if reserved_llm_calls < 0:
            raise ValueError("reserved_llm_calls must not be negative.")
        if self.iterations >= self.settings.max_agent_iterations:
            return GuardDecision(False, "IterationBudget", "Agent iteration budget exhausted.", True)
        if self.llm_calls + reserved_llm_calls >= self.settings.max_llm_calls:
            return GuardDecision(False, "LLMBudget", "LLM call budget exhausted.", True)
        return GuardDecision(True, "OK", "")

    def record_decision(
        self, llm_calls: int, *, reserved_llm_calls: int = 0
    ) -> GuardDecision:
        if reserved_llm_calls < 0:
            raise ValueError("reserved_llm_calls must not be negative.")
        self.iterations += 1
        self.llm_calls += llm_calls
        if self.llm_calls + reserved_llm_calls > self.settings.max_llm_calls:
            return GuardDecision(False, "LLMBudget", "Decision exceeded LLM call budget.", True)
        return GuardDecision(True, "OK", "")

    def record_preparation_llm_calls(self, llm_calls: int) -> None:
        """记录 Query Understanding / Planning 调用，不增加 Agent iteration。"""
        if llm_calls < 0:
            raise ValueError("llm_calls must not be negative.")
        self.llm_calls += llm_calls

    def validate_action(
        self,
        action: AgentAction,
        *,
        context: QueryContext,
        plan: ResearchPlan,
        snapshot: CorpusScopeSnapshot | None,
    ) -> GuardDecision:
        if isinstance(action, FinishAction):
            return GuardDecision(True, "OK", "")
        known_subtasks = {task.subtask_id for task in plan.subtasks}
        for subtask_id in action_subtask_ids(action):
            if subtask_id not in known_subtasks:
                return GuardDecision(False, "UnknownSubtask", f"Unknown subtask: {subtask_id}")
        target_ids = affected_paper_ids(action)
        allowed = (
            set(snapshot.paper_ids)
            if context.scope is QueryScope.KNOWLEDGE_BASE and snapshot is not None
            else set(context.target_paper_ids)
        )
        if isinstance(action, SearchKnowledgeBaseAction):
            if context.scope is not QueryScope.KNOWLEDGE_BASE or snapshot is None:
                return GuardDecision(False, "InvalidScope", "Global search requires a corpus snapshot.")
        elif not set(target_ids).issubset(allowed):
            return GuardDecision(False, "InvalidScope", "Action targets papers outside trusted scope.")
        if any(paper_id in self.suppressed_papers for paper_id in target_ids):
            return GuardDecision(False, "NoNewEvidenceSuppressed", "Paper has reached the no-new-evidence limit.")
        fingerprint = action_fingerprint(action)
        if fingerprint in self._fingerprints:
            return GuardDecision(False, "DuplicateAction", "The same action was already executed.")
        queries = retrieval_query_count(action)
        if self.tool_calls + 1 > self.settings.max_tool_calls:
            return GuardDecision(False, "ToolBudget", "Tool call budget exhausted.", True)
        if self.retrieval_queries + queries > self.settings.max_retrieval_queries:
            return GuardDecision(False, "RetrievalBudget", "Retrieval query budget exhausted.", True)
        for paper_id in target_ids:
            if self.queries_per_paper.get(paper_id, 0) + queries_for_paper(action, paper_id) > self.settings.max_queries_per_paper:
                return GuardDecision(False, "PerPaperBudget", f"Query budget exhausted for {paper_id}.", True)
        self._fingerprints.add(fingerprint)
        return GuardDecision(True, "OK", "")

    def record_tool_call(self, action: AgentAction) -> None:
        self.tool_calls += 1
        self.retrieval_queries += retrieval_query_count(action)
        for paper_id in affected_paper_ids(action):
            self.queries_per_paper[paper_id] = (
                self.queries_per_paper.get(paper_id, 0) + queries_for_paper(action, paper_id)
            )

    def record_invalid_action(self) -> bool:
        self.invalid_actions += 1
        return self.invalid_actions > self.settings.max_invalid_actions

    def record_evidence_delta(
        self, action: AgentAction, *, added_paper_ids: set[str]
    ) -> None:
        affected = affected_paper_ids(action)
        for paper_id in affected or ("__knowledge_base__",):
            added = (
                bool(added_paper_ids)
                if paper_id == "__knowledge_base__"
                else paper_id in added_paper_ids
            )
            if added:
                self.no_new_per_paper[paper_id] = 0
            else:
                count = self.no_new_per_paper.get(paper_id, 0) + 1
                self.no_new_per_paper[paper_id] = count
                if count >= self.settings.max_no_new_evidence_per_paper:
                    self.suppressed_papers.add(paper_id)

    def no_available_actions(self, context: QueryContext) -> bool:
        required = (
            {"__knowledge_base__"}
            if context.scope is QueryScope.KNOWLEDGE_BASE
            else set(context.target_paper_ids)
        )
        return bool(required) and required.issubset(self.suppressed_papers)

    def usage(self) -> BudgetUsage:
        return BudgetUsage(
            agent_iterations=self.iterations,
            tool_calls=self.tool_calls,
            retrieval_queries=self.retrieval_queries,
            llm_calls=self.llm_calls,
            invalid_actions=self.invalid_actions,
            queries_per_paper=tuple(sorted(self.queries_per_paper.items())),
        )

    def remaining(self) -> dict[str, int]:
        return {
            "agent_iterations": max(0, self.settings.max_agent_iterations - self.iterations),
            "tool_calls": max(0, self.settings.max_tool_calls - self.tool_calls),
            "retrieval_queries": max(0, self.settings.max_retrieval_queries - self.retrieval_queries),
            "llm_calls": max(0, self.settings.max_llm_calls - self.llm_calls),
        }

    def snapshot(self) -> GuardState:
        return GuardState(
            usage=self.usage(),
            no_new_per_paper=tuple(sorted(self.no_new_per_paper.items())),
            suppressed_paper_ids=tuple(sorted(self.suppressed_papers)),
            executed_action_fingerprints=tuple(sorted(self._fingerprints)),
        )


def affected_paper_ids(action: AgentAction) -> tuple[str, ...]:
    if isinstance(action, SearchPapersAction):
        return tuple(search.paper_id for search in action.searches)
    if isinstance(
        action, (SearchReferencesAction, GetSectionContextAction, GetPaperOverviewAction)
    ):
        return (action.paper_id,)
    return ()


def action_subtask_ids(action: AgentAction) -> tuple[str, ...]:
    values: list[str] = []
    action_value = getattr(action, "target_subtask_id", None)
    if action_value:
        values.append(action_value)
    if isinstance(action, SearchPapersAction):
        values.extend(
            search.target_subtask_id
            for search in action.searches
            if search.target_subtask_id is not None
        )
    return tuple(dict.fromkeys(values))


def retrieval_query_count(action: AgentAction) -> int:
    if isinstance(action, SearchPapersAction):
        return len(action.searches)
    if isinstance(action, (SearchKnowledgeBaseAction, SearchReferencesAction)):
        return 1
    return 0


def queries_for_paper(action: AgentAction, paper_id: str) -> int:
    if isinstance(action, SearchPapersAction):
        return sum(search.paper_id == paper_id for search in action.searches)
    if isinstance(action, SearchReferencesAction):
        return int(action.paper_id == paper_id)
    return 0


def action_fingerprint(action: AgentAction) -> str:
    if isinstance(action, SearchPapersAction):
        value = [
            (item.paper_id, item.query.casefold(), tuple(term.casefold() for term in item.lexical_terms))
            for item in action.searches
        ]
    elif isinstance(action, SearchKnowledgeBaseAction):
        value = [action.action_name, action.query.casefold(), tuple(term.casefold() for term in action.lexical_terms)]
    elif isinstance(action, SearchReferencesAction):
        value = [action.action_name, action.paper_id, action.query.casefold(), tuple(term.casefold() for term in action.lexical_terms)]
    elif isinstance(action, GetSectionContextAction):
        value = [action.action_name, action.paper_id, action.section_id]
    elif isinstance(action, GetPaperOverviewAction):
        value = [action.action_name, action.paper_id]
    else:
        value = [action.action_name, action.reason.casefold()]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
