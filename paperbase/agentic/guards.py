"""Agent Loop 的预算、重复动作、Scope 与无新增证据保护。"""

from __future__ import annotations

from dataclasses import dataclass
import json

from paperbase.config import AgenticResearchSettings

from .search_requests import compile_action_requests, canonical_action_identity
from paperbase.retrieval.requests import validate_effective_request
from .contracts import (
    AgentAction,
    CandidateSelection,
    BudgetUsage,
    CorpusScopeSnapshot,
    FinishAction,
    GuardState,
    GetPaperOverviewAction,
    GetSectionContextAction,
    QueryContext,
    QueryScope,
    ResearchPlan,
    RequirementState,
    RequirementStatus,
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
        self.no_goal_progress_per_paper = dict(
            snapshot.no_goal_progress_per_paper
        )
        self.suppressed_papers = set(snapshot.suppressed_paper_ids)
        # Search fingerprints from old checkpoints described wire strings, not
        # executed retrieval semantics. Preserve non-search keys; never guess.
        legacy_search = tuple(value for value in snapshot.executed_action_fingerprints
                              if _legacy_search_fingerprint(value))
        self._fingerprints = set(snapshot.executed_action_fingerprints) - set(legacy_search)
        self.request_executions = list(snapshot.request_executions)
        self.legacy_request_notes = list(snapshot.legacy_request_notes)
        if legacy_search or any(not value.startswith("search-v1:") for value in snapshot.executed_paper_query_fingerprints):
            self.legacy_request_notes.append("legacy search identity is not reliably migratable; fail-open, usage retained")
        self._paper_query_fingerprints = set(
            value for value in snapshot.executed_paper_query_fingerprints if value.startswith("search-v1:")
        )

    def can_decide(self) -> GuardDecision:
        if self.iterations >= self.settings.max_agent_iterations:
            return GuardDecision(False, "IterationBudget", "Agent iteration budget exhausted.", True)
        return GuardDecision(True, "OK", "")

    def record_decision(self, llm_calls: int) -> GuardDecision:
        self.iterations += 1
        self.llm_calls += llm_calls
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
        candidate_selection: CandidateSelection | None = None,
        requirement_states: tuple[RequirementState, ...] = (),
        known_section_pairs: frozenset[tuple[str, str]] = frozenset(),
        canonical_requests=None,
    ) -> GuardDecision:
        if isinstance(action, FinishAction):
            return GuardDecision(True, "OK", "")
        requests = (compile_action_requests(action, context=context, snapshot=snapshot,
                    top_k=self.settings.top_k_per_paper) if canonical_requests is None else canonical_requests)
        self.request_executions.extend({"request_identity": request.identity,
            "iteration": self.iterations, "state": "attempted", "attempted": True,
            "executed": False, "failed": False, "successful": False} for request in requests)
        for request in requests:
            rejection = validate_effective_request(request)
            if rejection is not None:
                return GuardDecision(False, rejection.code, rejection.detail)
        if not _validate_requirement_bindings(action, plan, requirement_states):
            return GuardDecision(
                False,
                "InvalidRequirementBinding",
                "Action must bind existing active Requirements; support_found can still be researched.",
            )
        if isinstance(action, GetSectionContextAction):
            if (action.paper_id, action.section_id) not in known_section_pairs:
                return GuardDecision(
                    False,
                    "UnknownSection",
                    "Section context must reference a section already exposed by evidence for the same paper.",
                )
        target_ids = (tuple(paper_id for request in requests
                            if request.kind in {"search_papers", "search_references"}
                            for paper_id in request.paper_ids) if requests else affected_paper_ids(action))
        if isinstance(action, SearchPapersAction) and len(target_ids) != len(
            set(target_ids)
        ):
            return GuardDecision(
                False,
                "DuplicatePaperSearch",
                "A search_papers batch may contain each paper_id only once.",
            )
        allowed = (
            set(snapshot.paper_ids)
            if context.scope is QueryScope.KNOWLEDGE_BASE and snapshot is not None
            else set(context.target_paper_ids)
        )
        if isinstance(action, SearchKnowledgeBaseAction):
            if candidate_selection is not None:
                return GuardDecision(
                    False,
                    "CandidateScopeLocked",
                    "Candidate Selection 完成后必须按候选论文执行 paper-scoped 检索。",
                )
            if context.scope is not QueryScope.KNOWLEDGE_BASE or snapshot is None:
                return GuardDecision(False, "InvalidScope", "Global search requires a corpus snapshot.")
            if any(not set(request.paper_ids).issubset(allowed) for request in requests):
                return GuardDecision(False, "InvalidScope", "Canonical KB scope exceeds trusted corpus snapshot.")
        else:
            if candidate_selection is not None:
                allowed = set(
                    (*candidate_selection.candidate_paper_ids, *candidate_selection.uncertain_paper_ids)
                )
            if not set(target_ids).issubset(allowed):
                return GuardDecision(False, "InvalidScope", "Action targets papers outside trusted candidate scope.")
        fingerprint = canonical_action_identity(requests) if requests else action_fingerprint(action)
        if fingerprint in self._fingerprints:
            return GuardDecision(False, "DuplicateAction", "The same action was already executed.")
        paper_query_fingerprints = tuple(request.identity for request in requests
                                         if request.kind in {"search_papers", "search_references"})
        repeated_queries = tuple(
            value
            for value in paper_query_fingerprints
            if value in self._paper_query_fingerprints
        )
        if repeated_queries:
            return GuardDecision(
                False,
                "DuplicatePaperQuery",
                "The same effective search request for this paper has already completed. "
                "Revise a retrieval-relevant input according to the remaining evidence gap "
                "(for example semantic query or lexical focus); changing audit-only fields does not create a new search.",
            )
        for paper_id in target_ids:
            if self.queries_per_paper.get(paper_id, 0) + queries_for_paper(action, paper_id) > self.settings.max_queries_per_paper:
                return GuardDecision(False, "PerPaperBudget", f"Query budget exhausted for {paper_id}.", True)
        return GuardDecision(True, "OK", "")

    def record_tool_call(self, action: AgentAction, *, canonical_requests=None, result=None) -> None:
        requests = compile_action_requests(action, top_k=self.settings.top_k_per_paper) if canonical_requests is None else canonical_requests
        records = result.metadata.get("request_executions") if result is not None else None
        if records is None:
            records = [{"request_identity": request.identity, "attempted": True,
                        "executed": True, "failed": result is not None and result.status.value in {"error", "unavailable"},
                        "successful": result is None or result.status.value in {"ok", "empty", "partial"}}
                       for request in requests]
        self.request_executions.extend({**record, "iteration": self.iterations,
            "state": "successful" if record["successful"] else "failed"} for record in records)
        successful = {record["request_identity"] for record in records
                      if record["executed"] and record["successful"]}
        self._paper_query_fingerprints.update(request.identity for request in requests
            if request.identity in successful and request.kind in {"search_papers", "search_references"})
        if requests and all(request.identity in successful for request in requests):
            self._fingerprints.add(canonical_action_identity(requests))
        elif not requests and (result is None or result.status.value in {"ok", "empty", "partial"}):
            self._fingerprints.add(action_fingerprint(action))
        self.tool_calls += 1
        self.retrieval_queries += sum(bool(record["executed"]) for record in records) if requests else retrieval_query_count(action)
        for paper_id in affected_paper_ids(action):
            count = (sum(bool(record["executed"]) for request, record in zip(requests, records)
                         if request.kind in {"search_papers", "search_references"} and paper_id in request.paper_ids)
                     if requests else queries_for_paper(action, paper_id))
            self.queries_per_paper[paper_id] = self.queries_per_paper.get(paper_id, 0) + count

    def record_invalid_action(self) -> bool:
        self.invalid_actions += 1
        return self.invalid_actions >= self.settings.node_max_attempts

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

    def record_goal_progress(
        self,
        paper_ids: tuple[str, ...],
        *,
        progressed: bool,
    ) -> None:
        """只把新增用户目标信息视为进展，防止无关新 Evidence 掩盖空转。"""
        for paper_id in dict.fromkeys(paper_ids):
            if progressed:
                self.no_goal_progress_per_paper[paper_id] = 0
                continue
            count = self.no_goal_progress_per_paper.get(paper_id, 0) + 1
            self.no_goal_progress_per_paper[paper_id] = count

    def no_available_actions(
        self,
        context: QueryContext,
        *,
        candidate_selection: CandidateSelection | None = None,
    ) -> bool:
        # Kept for checkpoint/API compatibility.  A lack of new evidence is
        # an observation for the decision agent, never a runtime stop rule.
        return False

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
        }

    def snapshot(self) -> GuardState:
        return GuardState(
            usage=self.usage(),
            request_executions=tuple(self.request_executions),
            legacy_request_notes=tuple(dict.fromkeys(self.legacy_request_notes)),
            no_new_per_paper=tuple(sorted(self.no_new_per_paper.items())),
            no_goal_progress_per_paper=tuple(
                sorted(self.no_goal_progress_per_paper.items())
            ),
            suppressed_paper_ids=tuple(sorted(self.suppressed_papers)),
            executed_action_fingerprints=tuple(sorted(self._fingerprints)),
            executed_paper_query_fingerprints=tuple(
                sorted(self._paper_query_fingerprints)
            ),
        )


def affected_paper_ids(action: AgentAction) -> tuple[str, ...]:
    if isinstance(action, SearchPapersAction):
        return tuple(search.paper_id for search in action.searches)
    if isinstance(
        action, (SearchReferencesAction, GetSectionContextAction, GetPaperOverviewAction)
    ):
        return (action.paper_id,)
    return ()


def action_requirement_ids(action: AgentAction) -> tuple[str, ...]:
    """Return Requirement bindings without exposing legacy Subtask IDs."""
    if isinstance(action, SearchPapersAction):
        return tuple(
            dict.fromkeys(
                requirement_id
                for search in action.searches
                for requirement_id in search.requirement_ids
            )
        )
    return tuple(getattr(action, "requirement_ids", ()))


def _validate_requirement_bindings(
    action: AgentAction,
    plan: ResearchPlan,
    requirement_states: tuple[RequirementState, ...],
) -> bool:
    """Validate action bindings against the current Requirement plan."""
    known = {item.requirement_id for item in plan.requirements}
    if not known:
        return True
    if isinstance(action, SearchPapersAction):
        if any(not search.requirement_ids for search in action.searches):
            return False
    elif not action_requirement_ids(action):
        return False
    requested = set(action_requirement_ids(action))
    if not requested.issubset(known):
        return False
    state_by_id = {item.requirement_id: item.status for item in requirement_states}
    # Missing state means an old checkpoint: retain its previous behaviour.
    # Legacy satisfied means support found; more research remains legal.
    return not any(
        state_by_id.get(requirement_id)
        in {
            RequirementStatus.NO_SUPPORT_FOUND,
            RequirementStatus.UNCERTAIN,
        }
        for requirement_id in requested
    )


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
    requests = compile_action_requests(action)
    if requests:
        return canonical_action_identity(requests)
    if isinstance(action, GetSectionContextAction):
        value = [action.action_name, action.paper_id, action.section_id]
    elif isinstance(action, GetPaperOverviewAction):
        value = [action.action_name, action.paper_id]
    else:
        value = [action.action_name, action.reason.casefold()]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def paper_query_fingerprints_for_action(action: AgentAction) -> tuple[str, ...]:
    return tuple(request.identity for request in compile_action_requests(action)
                 if request.kind in {"search_papers", "search_references"})


def _legacy_search_fingerprint(value: str) -> bool:
    if value.startswith("search-action-v1:"):
        return False
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return False
    return bool(isinstance(decoded, list) and decoded and
                (isinstance(decoded[0], list) or decoded[0] in {"search_papers", "search_references", "search_knowledge_base"}))
