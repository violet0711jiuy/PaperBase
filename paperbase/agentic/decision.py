"""Decision Agent：只选择下一步动作，不执行 Tool，也不生成最终答案。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import json
from typing import Callable, Protocol

from pydantic import ValidationError

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCompletion,
    complete_json_with_diagnostics,
)
from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT
from paperbase.retrieval.contracts import PaperSearchSpec, ToolStatus

from .contracts import (
    AgentAction,
    BudgetUsage,
    CandidateSelection,
    CoverageState,
    EvidenceRecord,
    GroundedAnswerAttempt,
    TryAnswerAction,
    StopResearchAction,
    GetPaperOverviewAction,
    GetSectionContextAction,
    LLMCallRecord,
    QueryContext,
    QueryScope,
    ResearchObservation,
    ResearchPlan,
    RequirementState,
    ActionRejectionContext,
    RequirementStatus,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
    ActionCanonicalization,
    RuntimeActionCapabilities,
    RuntimeUnavailableAction,
)
from paperbase.llm.client import ValidationDiagnostic
from paperbase.llm.diagnostics import failure_record, success_record
from .schemas import (
    AGENT_DECISION_ADAPTER,
    ActionDraft,
    GetPaperOverviewDraft,
    GetSectionContextDraft,
    SearchKnowledgeBaseDraft,
    SearchPapersDraft,
    SearchReferencesDraft,
    TryAnswerDraft,
    StopResearchDraft,
)
from .requirement_state import derive_requirement_states
from .action_control import ActionCanonicalizer
from .guards import GuardDecision
from paperbase.retrieval.requests import CanonicalSearchRequest
from .decision_view import CompactDecisionLimits, build_compact_decision_view
from .requirement_targets import targets_for_requirement_ids
from .requirement_terms import foreign_requirement_terms, term_present


class DecisionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        llm_calls: int,
        llm_diagnostics: tuple[LLMCallRecord, ...] = (),
        recovery_rejections: tuple[ActionRejectionContext, ...] = (),
    ) -> None:
        super().__init__(message)
        self.llm_calls = llm_calls
        self.llm_diagnostics = llm_diagnostics
        self.recovery_rejections = recovery_rejections


class ActionDomainValidationError(ValueError):
    """A domain rejection with a stable code and a safe repair location."""

    def __init__(
        self,
        *,
        code: str,
        field_path: str,
        validator_name: str,
        safe_reason: str,
        expected_constraint: str,
        actual_summary: str,
    ) -> None:
        super().__init__(safe_reason)
        self.code = code
        self.field_path = field_path
        self.validator_name = validator_name
        self.safe_reason = safe_reason
        self.expected_constraint = expected_constraint
        self.actual_summary = actual_summary


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
    requirement_states: tuple[RequirementState, ...] = ()
    grounded_answer: GroundedAnswerAttempt | None = None
    # Complete Registry snapshot used by the last Grounded Answer attempt.
    # GroundedAnswerAttempt.evidence_ids names cited Evidence only, so it cannot
    # determine whether fresh uncited Evidence exists.
    answer_evidence_ids: tuple[str, ...] = ()
    answer_attempt_count: int = 0
    # Ephemeral observability for tool choice; RequirementState remains the
    # only completion-state contract.
    overview_status_by_paper: tuple[tuple[str, str], ...] = ()
    last_action_rejection: ActionRejectionContext | None = None
    action_rejections: tuple[ActionRejectionContext, ...] = ()
    per_paper_query_limit: int | None = None
    max_agent_iterations: int | None = None
    # Ephemeral Graph-owned callback: assesses a disposable Guard snapshot.
    # Never checkpointed; the formal Graph Guard still admits execution.
    guard_preflight: Callable[[AgentAction], tuple[GuardDecision, tuple[CanonicalSearchRequest, ...]]] | None = None


@dataclass(frozen=True)
class DecisionResult:
    action: AgentAction
    llm_calls: int
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()
    action_canonicalization: ActionCanonicalization | None = None
    recovery_rejections: tuple[ActionRejectionContext, ...] = ()


class DecisionAgent(Protocol):
    def decide(self, state: DecisionState) -> DecisionResult: ...


class LLMDecisionAgent:
    def __init__(
        self, client: ChatCompletionClient, *, max_attempts: int = 2,
        temperature: float = 0,
        thinking_enabled: bool = False,
        view_limits: CompactDecisionLimits | None = None,
    ) -> None:
        # Reuse the provider's immutable views, omitting application output caps.
        # Other nodes continue to use their independently configured clients.
        for method, value in (("with_max_tokens", None), ("with_temperature", temperature),
                              ("with_thinking_enabled", thinking_enabled),
                              ("with_structured_output_mode", "json_schema"),
                              ("with_schema_prompt_injection", False)):
            configure = getattr(client, method, None)
            if callable(configure):
                client = configure(value)
            elif method == "with_thinking_enabled" and thinking_enabled:
                raise ValueError("Decision client cannot enable thinking mode.")
        self._client = client
        self._max_attempts = max_attempts
        self._canonicalizer = ActionCanonicalizer()
        self._view_limits = view_limits or CompactDecisionLimits()

    def decide(self, state: DecisionState) -> DecisionResult:
        errors: list[ValidationDiagnostic] = []
        diagnostics: list[LLMCallRecord] = []
        rejections: list[ActionRejectionContext] = []
        current_state = state
        recovery_deadline = self._max_attempts
        evidence_limit = 12
        for attempt in range(1, self._max_attempts + 1):
            if attempt > recovery_deadline:
                break
            completion: LLMCompletion | None = None
            limits = self._view_limits
            if evidence_limit < 12:
                # Keep the established provider-length recovery, now shrinking
                # only view detail. All Requirement IDs and hard legality remain.
                limits = replace(limits, **{
                    name: max(1, getattr(limits, name) // 2)
                    for name in ("support_entries_per_requirement", "recent_observations",
                                 "recent_queries", "recent_actions", "sections", "entity_hints",
                                 "research_leads",
                                 "unresolved_aspect_chars")
                })
            view = build_compact_decision_view(current_state, errors, limits=limits)
            compact_prompt = view.to_json()
            metrics = dict(view.metrics)
            try:
                completion = complete_json_with_diagnostics(
                    _repair_client_if_needed(
                        self._client,
                        has_validation_repair=bool(errors),
                        has_runtime_rejection=(
                            current_state.last_action_rejection is not None
                        ),
                    ),
                    system_prompt=AGENT_DECISION_SYSTEM_PROMPT,
                    user_prompt=compact_prompt,
                    json_schema=_decision_json_schema(current_state),
                    schema_name="paperbase_agent_action",
                )
                raw_payload = json.loads(completion.content)
                canonical_payload, canonicalization = self._canonicalizer.canonicalize(
                    raw_payload
                )
                # Validate in JSON mode after canonicalization so strict Enum
                # fields accept the same JSON representation sent to the LLM.
                draft = AGENT_DECISION_ADAPTER.validate_json(
                    json.dumps(canonical_payload, ensure_ascii=False)
                )
                action = _to_action(draft.decision)
                _validate_action_references(action, current_state)
                action = _bind_plan_targets(action, current_state.plan)
                if state.guard_preflight is not None:
                    assessment, requests = state.guard_preflight(action)
                    if not assessment.accepted:
                        # At most one immediate replacement candidate, including
                        # when the configured validation retry pool is larger.
                        recovery_deadline = min(recovery_deadline, attempt + 1)
                        from .guards import action_fingerprint
                        from .search_requests import canonical_action_identity
                        identity = canonical_action_identity(requests) if requests else action_fingerprint(action)
                        rejected = ActionRejectionContext(
                            action_name=action.action_name, action_arguments=asdict(action),
                            requirement_ids=_action_requirement_ids(action),
                            rejection_code=assessment.code, rejection_reason=assessment.detail,
                            guard_name=assessment.code, iteration=state.budget_usage.agent_iterations + 1,
                            rejection_stage="guard_preflight", validator_name="runtime_guard_preflight",
                        )
                        rejections.append(rejected)
                        current_state = replace(state, last_action_rejection=rejected,
                            action_rejections=(*state.action_rejections, *rejections))
                        raise ActionDomainValidationError(
                            code=assessment.code, field_path="decision", validator_name="runtime_guard_preflight",
                            safe_reason=assessment.detail,
                            expected_constraint="Choose a different effective request, another available tool, or Finish; formal Guard revalidates.",
                            actual_summary=json.dumps(dict(
                                rejected_tool=action.action_name, rejected_action_identity=identity,
                                affected_requirement_ids=rejected.requirement_ids,
                                canonical_effective_requests=[dict(
                                    request_identity=request.identity, kind=request.kind,
                                    paper_ids=request.paper_ids, semantic_query=getattr(request, "semantic_query", None),
                                    lexical_terms=request.lexical_terms, lexical_query=request.lexical_query,
                                    top_k=request.top_k,
                                ) for request in requests],
                                original_decision_iteration=rejected.iteration,
                            ), ensure_ascii=False),
                        )
                diagnostics.append(
                    replace(success_record(
                        node="decide_action", attempt=attempt, completion=completion
                    ), decision_iteration=state.budget_usage.agent_iterations + 1,
                        decision_view_metrics=metrics)
                )
                return DecisionResult(
                    action,
                    attempt,
                    llm_diagnostics=tuple(diagnostics),
                    action_canonicalization=canonicalization,
                    recovery_rejections=tuple(rejections),
                )
            except Exception as error:
                record = failure_record(
                    node="decide_action",
                    attempt=attempt,
                    error=error,
                    completion=completion,
                    can_retry=attempt < recovery_deadline,
                )
                diagnostic = _decision_validation_diagnostic(
                    error,
                    attempt=attempt,
                    recovery_action=record.recovery_action,
                )
                if diagnostic is not None:
                    record = replace(
                        record,
                        validation_diagnostics=(diagnostic,),
                    )
                    if diagnostic.validator_name == "runtime_guard_preflight":
                        record = replace(record, error_stage="guard_preflight_rejection",
                            recovery_action="replan_action" if attempt < recovery_deadline else None)
                        diagnostic = replace(diagnostic, error_stage="guard_preflight_rejection",
                            recovery_action=record.recovery_action)
                        record = replace(record, validation_diagnostics=(diagnostic,))
                record = replace(record, decision_iteration=state.budget_usage.agent_iterations + 1,
                                 decision_view_metrics=metrics)
                diagnostics.append(record)
                if diagnostic is not None:
                    errors.append(diagnostic)
                if record.recovery_action == "compact_context" or record.error_stage == "timeout":
                    # A timeout is also a context-pressure signal for Decision.
                    # Retry with a genuinely smaller view instead of replaying
                    # the exact same large request.
                    evidence_limit = 6
        raise DecisionError(
            "Decision output remained invalid after repair.",
            llm_calls=len(diagnostics),
            llm_diagnostics=tuple(diagnostics),
            recovery_rejections=tuple(rejections),
        )


class ScriptedDecisionAgent:
    """确定性测试替身；每次决定仍计为一次语义调用。"""

    def __init__(self, actions: tuple[AgentAction, ...]) -> None:
        self._actions = list(actions)

    def decide(self, state: DecisionState) -> DecisionResult:
        if not self._actions:
            return DecisionResult(StopResearchAction("script exhausted"), 1)
        return DecisionResult(self._actions.pop(0), 1)


def _repair_client_if_needed(
    client: ChatCompletionClient,
    *,
    has_validation_repair: bool,
    has_runtime_rejection: bool,
) -> ChatCompletionClient:
    """Keep the configured immutable structured-output view on repair attempts.

    Effective sampling is supplied by the active Agentic sampling profile.
    """
    return client


def _decision_json_schema(state: DecisionState) -> dict[str, object]:
    """Project one Runtime structural domain; never reproduce Search identity."""
    capabilities = _runtime_action_capabilities(state)
    schema = deepcopy(AGENT_DECISION_ADAPTER.json_schema())
    definitions = schema["$defs"]
    decision = schema["properties"]["decision"]
    mapping = decision["discriminator"]["mapping"]
    available = set(capabilities.available_tools)
    allowed_refs = {ref for tool, ref in mapping.items() if tool in available}
    decision["oneOf"] = [item for item in decision["oneOf"]
                         if item["$ref"] in allowed_refs]
    decision["discriminator"]["mapping"] = {
        tool: ref for tool, ref in mapping.items() if tool in available
    }
    for name in ("SearchPapersDraft", "SearchKnowledgeBaseDraft", "SearchReferencesDraft",
                 "GetSectionContextDraft", "GetPaperOverviewDraft"):
        # With no active binding only answer/stop actions are reachable.
        if capabilities.valid_requirement_ids:
            definitions[name]["properties"]["requirement_id"]["enum"] = list(
                capabilities.valid_requirement_ids
            )
    for name, paper_ids in (
        ("SearchSpecDraft", capabilities.search_paper_ids),
        ("SearchReferencesDraft", capabilities.reference_paper_ids),
        ("GetPaperOverviewDraft", capabilities.overview_paper_ids),
    ):
        if paper_ids:
            definitions[name]["properties"]["paper_id"]["enum"] = list(paper_ids)
        # Empty domains have no root branch; do not expose arbitrary strings.
    if capabilities.search_paper_ids:
        searches = definitions["SearchPapersDraft"]["properties"]["searches"]
        searches["maxItems"] = min(searches.get("maxItems", 12),
                                  len(capabilities.search_paper_ids))
    pairs = capabilities.available_section_ids_by_paper
    if pairs:
        definitions["GetSectionContextDraft"]["oneOf"] = [
            {"properties": {"paper_id": {"const": paper_id},
                            "section_id": {"enum": list(section_ids)}},
             "required": ["paper_id", "section_id"]}
            for paper_id, section_ids in pairs
        ]
    return schema


def _to_action(draft: ActionDraft) -> AgentAction:
    if isinstance(draft, SearchPapersDraft):
        paper_ids = [item.paper_id for item in draft.searches]
        if len(paper_ids) != len(set(paper_ids)):
            seen: set[str] = set()
            duplicate_index = 0
            for index, paper_id in enumerate(paper_ids):
                if paper_id in seen:
                    duplicate_index = index
                    break
                seen.add(paper_id)
            raise ActionDomainValidationError(
                code="DuplicatePaperSearch",
                field_path=f"searches[{duplicate_index}].paper_id",
                validator_name="search_papers_batch_validator",
                safe_reason="A search_papers batch cannot repeat the same paper ID.",
                expected_constraint="each paper_id appears at most once per batch",
                actual_summary=f"duplicate_count={len(paper_ids) - len(set(paper_ids))}",
            )
        return SearchPapersAction(
            searches=tuple(
                PaperSearchSpec(
                    paper_id=item.paper_id,
                    query=item.semantic_query,
                    lexical_terms=tuple(item.lexical_terms),
                    evidence_targets=frozenset(),
                    requirement_ids=(draft.requirement_id,),
                )
                for item in draft.searches
            ),
        )
    if isinstance(draft, SearchKnowledgeBaseDraft):
        return SearchKnowledgeBaseAction(
            query=draft.semantic_query,
            lexical_terms=tuple(draft.lexical_terms),
            requirement_ids=(draft.requirement_id,),
        )
    if isinstance(draft, SearchReferencesDraft):
        return SearchReferencesAction(
            paper_id=draft.paper_id,
            query=draft.semantic_query,
            lexical_terms=tuple(draft.lexical_terms),
            requirement_ids=(draft.requirement_id,),
        )
    if isinstance(draft, GetSectionContextDraft):
        return GetSectionContextAction(
            paper_id=draft.paper_id,
            section_id=draft.section_id,
            requirement_ids=(draft.requirement_id,),
        )
    if isinstance(draft, GetPaperOverviewDraft):
        return GetPaperOverviewAction(
            paper_id=draft.paper_id,
            requirement_ids=(draft.requirement_id,),
        )
    if isinstance(draft, TryAnswerDraft):
        return TryAnswerAction(draft.reason)
    if isinstance(draft, StopResearchDraft):
        return StopResearchAction(draft.reason)
    raise TypeError(f"Unsupported Decision draft: {type(draft).__name__}")


def _bind_plan_targets(action: AgentAction, plan: ResearchPlan) -> AgentAction:
    """Compile Plan-owned routing hints after structural/domain validation."""
    if not isinstance(action, SearchPapersAction):
        return action
    return SearchPapersAction(searches=tuple(
        replace(
            search,
            evidence_targets=targets_for_requirement_ids(
                plan, search.requirement_ids
            ),
        )
        for search in action.searches
    ))


def _state_prompt(
    state: DecisionState,
    errors: list[ValidationDiagnostic],
    *,
    evidence_limit: int = 12,
) -> str:
    """Production input; the full-state baseline is a diagnostic helper only."""
    return build_compact_decision_view(
        state, errors, limits=CompactDecisionLimits(support_entries_per_requirement=min(6, evidence_limit))
    ).to_json()


def _action_recovery_constraint(
    state: DecisionState,
) -> dict[str, object] | None:
    """Return the immediate, non-prescriptive repair boundary after a rejection.

    This is deliberately a constraint rather than an action recommendation: the
    Decision Agent must still choose *which* materially different legal action
    has the best information value.  The same action arguments are retained so
    the model can distinguish a true replan from whitespace-only variation.
    """

    rejection = state.last_action_rejection
    # Historical rejections remain auditable telemetry, but are not a pending
    # recovery. Runtime clears this pointer after successful replacement execution.
    if rejection is None:
        return None
    return {
        "required": True,
        "rejection_code": rejection.rejection_code,
        "rejection_stage": rejection.rejection_stage,
        "blocked_action": {
            "action_name": rejection.action_name,
            "action_arguments": rejection.action_arguments,
            "requirement_ids": rejection.requirement_ids,
        },
        "constraint": (
            "Return a materially different legal action. The previous effective request "
            "was rejected; canonical request identity and Runtime Guard decide "
            "Search duplication, not raw paper/query history."
        ),
    }


def _observation_requirement_ids(observation: ResearchObservation) -> tuple[str, ...]:
    direct = observation.action_arguments.get("requirement_ids", ())
    values = list(direct) if isinstance(direct, (list, tuple)) else []
    for search in observation.action_arguments.get("searches", ()):
        if isinstance(search, dict):
            values.extend(search.get("requirement_ids", ()))
    return tuple(dict.fromkeys(value for value in values if isinstance(value, str)))


def _last_search_outcome(state: DecisionState) -> dict[str, object] | None:
    """Expose the latest deterministic retrieval delta without semantic scoring."""
    for observation in reversed(state.observations):
        if not observation.search_attempts and not observation.action_name.startswith("search_"):
            continue
        attempts = observation.search_attempts
        paper_id = attempts[-1].paper_id if attempts else ""
        query = attempts[-1].query if attempts else ""
        return {
            "paper_id": paper_id,
            "query": query,
            "requirement_ids": _observation_requirement_ids(observation),
            "added_evidence_ids": observation.added_evidence_ids,
            "reused_evidence_ids": observation.reused_evidence_ids,
            "newly_exposed_section_ids": observation.newly_exposed_section_ids,
        }
    return None


def _tool_budget_state(state: DecisionState) -> dict[str, object]:
    limit = state.per_paper_query_limit
    by_paper = {
        paper_id: used
        for paper_id, used in state.budget_usage.queries_per_paper
    }
    if limit is None:
        return {"per_paper": [], "per_paper_limit": None}
    paper_ids = set(by_paper)
    paper_ids.update(record.evidence.paper_id for record in state.evidence)
    if state.coverage is not None:
        paper_ids.update(state.coverage.required_paper_ids)
        paper_ids.update(state.coverage.candidate_paper_ids)
    return {
        "per_paper_limit": limit,
        "per_paper": [
            {
                "paper_id": paper_id,
                "used": by_paper.get(paper_id, 0),
                "remaining": max(limit - by_paper.get(paper_id, 0), 0),
            }
            for paper_id in sorted(paper_ids)
        ],
    }


def _validate_action_references(action: AgentAction, state: DecisionState) -> None:
    """拒绝无法从可信 DecisionState 中定位来源的模型引用。"""
    capabilities = _runtime_action_capabilities(state)
    known_requirement_ids = {item.requirement_id for item in state.plan.requirements}
    requirement_ids = _action_requirement_ids(action)
    if known_requirement_ids and not isinstance(action, (TryAnswerAction, StopResearchAction)):
        if not requirement_ids:
            raise ActionDomainValidationError(
                code="InvalidRequirementBinding",
                field_path="requirement_id",
                validator_name="requirement_binding_validator",
                safe_reason="A retrieval Action must bind at least one Requirement.",
                expected_constraint="at least one active requirement_id",
                actual_summary="empty=true",
            )
        if len(requirement_ids) != 1:
            raise ActionDomainValidationError(
                code="InvalidRequirementBinding",
                field_path="requirement_id",
                validator_name="requirement_binding_validator",
                safe_reason="Each retrieval Action must bind exactly one Requirement.",
                expected_constraint="exactly one active requirement_id",
                actual_summary=f"count={len(requirement_ids)}",
            )
        unknown = set(requirement_ids) - known_requirement_ids
        if unknown:
            unknown_id = sorted(unknown)[0]
            raise ActionDomainValidationError(
                code="InvalidRequirementId",
                field_path=_requirement_field_path(action, unknown_id),
                validator_name="requirement_binding_validator",
                safe_reason="Action references a Requirement ID not created by the Planner.",
                expected_constraint="one of the active Planner requirement IDs",
                actual_summary=f"unknown_id={unknown_id}",
            )
        terminal = known_requirement_ids - set(capabilities.valid_requirement_ids)
        if set(requirement_ids).intersection(terminal):
            terminal_id = sorted(set(requirement_ids).intersection(terminal))[0]
            raise ActionDomainValidationError(
                code="TerminalRequirementBinding",
                field_path=_requirement_field_path(action, terminal_id),
                validator_name="requirement_binding_validator",
                safe_reason="Action cannot reopen a terminal Requirement.",
                expected_constraint="an active requirement_id",
                actual_summary=f"terminal_id={terminal_id}",
            )
        _validate_search_scope(action, state.plan)
    if isinstance(action, GetSectionContextAction):
        known_sections = {
            (paper_id, section_id)
            for paper_id, section_ids in capabilities.known_section_ids_by_paper
            for section_id in section_ids
        }
        if (action.paper_id, action.section_id) not in known_sections:
            raise ActionDomainValidationError(
                code="InvalidSectionId",
                field_path="section_id",
                validator_name="known_section_validator",
                safe_reason="Section context must reference a section exposed by evidence for the same paper.",
                expected_constraint="a known section_id for the selected paper",
                actual_summary=f"unknown_id={action.section_id}",
            )

    if isinstance(action, (TryAnswerAction, StopResearchAction)):
        return
    tool = action.action_name
    paper_domains = {
        "search_papers": capabilities.search_paper_ids,
        "search_references": capabilities.reference_paper_ids,
        "get_paper_overview": capabilities.overview_paper_ids,
    }
    paper_ids = (tuple(item.paper_id for item in action.searches)
                 if isinstance(action, SearchPapersAction)
                 else (getattr(action, "paper_id", None),))
    invalid_paper = tool in paper_domains and any(
        paper_id not in paper_domains[tool] for paper_id in paper_ids)
    invalid_pair = isinstance(action, GetSectionContextAction) and (
        action.section_id not in dict(capabilities.available_section_ids_by_paper).get(
            action.paper_id, ()))
    if tool not in capabilities.available_tools or invalid_paper or invalid_pair:
        raise ActionDomainValidationError(
            code="UnavailableActionDomain",
            field_path="action" if tool not in capabilities.available_tools else
                       "section_id" if invalid_pair else "paper_id",
            validator_name="runtime_structural_domain_validator",
            safe_reason="Action is outside the current Runtime structural feasible domain.",
            expected_constraint="runtime_action_capabilities tool-specific domain",
            actual_summary=f"tool={tool}",
        )


def _action_requirement_ids(action: AgentAction) -> tuple[str, ...]:
    if isinstance(action, SearchPapersAction):
        return tuple(
            dict.fromkeys(
                requirement_id
                for search in action.searches
                for requirement_id in search.requirement_ids
            )
        )
    return tuple(getattr(action, "requirement_ids", ()))


def _requirement_field_path(action: AgentAction, requirement_id: str) -> str:
    del action, requirement_id
    return "requirement_id"


def _validate_search_scope(action: AgentAction, plan: ResearchPlan) -> None:
    """Reject only explicit identifiers owned by another Requirement.

    The check is intentionally lexical and narrow.  It prevents an Action
    bound to R1 from copying a named R2 symbol into its retrieval request,
    without pretending to judge broader semantic overlap.
    """
    requirement_ids = _action_requirement_ids(action)
    if len(requirement_ids) != 1:
        return
    primary_id = requirement_ids[0]
    foreign = foreign_requirement_terms(
        (
            (requirement.requirement_id, requirement.description)
            for requirement in plan.requirements
        ),
        primary_id,
    )
    if not foreign:
        return
    searches: tuple[tuple[int | None, str, tuple[str, ...]], ...]
    if isinstance(action, SearchPapersAction):
        searches = tuple(
            (index, search.query, search.lexical_terms)
            for index, search in enumerate(action.searches)
        )
    elif isinstance(action, (SearchKnowledgeBaseAction, SearchReferencesAction)):
        searches = ((None, action.query, action.lexical_terms),)
    else:
        return
    for index, semantic_query, lexical_terms in searches:
        query_hits = tuple(term for term in foreign if term_present(term, semantic_query))
        lexical_text = "\n".join(lexical_terms)
        lexical_hits = tuple(term for term in foreign if term_present(term, lexical_text))
        if not query_hits and not lexical_hits:
            continue
        prefix = f"searches[{index}]." if index is not None else ""
        field = "semantic_query" if query_hits else "lexical_terms"
        raise ActionDomainValidationError(
            code="CrossRequirementSearchScope",
            field_path=f"{prefix}{field}",
            validator_name="requirement_search_scope_validator",
            safe_reason=(
                "Search text contains an explicit identifier owned only by "
                "another Requirement. Keep this Action scoped to requirement_id."
            ),
            expected_constraint=(
                "semantic_query and lexical_terms contain no explicit identifiers "
                "owned only by another Requirement"
            ),
            actual_summary=json.dumps(
                {
                    "requirement_id": primary_id,
                    "foreign_query_terms": query_hits,
                    "foreign_lexical_terms": lexical_hits,
                },
                ensure_ascii=False,
            ),
        )


def _decision_validation_diagnostic(
    error: Exception,
    *,
    attempt: int,
    recovery_action: str | None,
) -> ValidationDiagnostic | None:
    """Map Decision parse/domain failures to stable, safe repair facts."""
    if isinstance(error, json.JSONDecodeError):
        return ValidationDiagnostic(
            node="decide_action",
            attempt=attempt,
            error_stage="invalid_json",
            error_code="InvalidJson",
            field_path="root",
            validator_name="json_decoder",
            safe_reason="Response could not be parsed as a JSON object.",
            expected_constraint="valid JSON object",
            actual_summary="type=str",
            recovery_action=recovery_action,
        )
    if isinstance(error, ValidationError):
        detail = error.errors()[0] if error.errors() else {}
        location = tuple(detail.get("loc", ()))
        kind = str(detail.get("type", "validation_error"))
        return ValidationDiagnostic(
            node="decide_action",
            attempt=attempt,
            error_stage="schema_validation",
            error_code=_schema_error_code(kind, location),
            field_path=_field_path(location),
            validator_name=f"pydantic.{kind}",
            safe_reason=_schema_safe_reason(kind, location),
            expected_constraint=_schema_expected_constraint(kind, location, detail),
            actual_summary=_safe_actual_summary(detail.get("input")),
            recovery_action=recovery_action,
        )
    if isinstance(error, ActionDomainValidationError):
        return ValidationDiagnostic(
            node="decide_action",
            attempt=attempt,
            error_stage="domain_validation",
            error_code=error.code,
            field_path=error.field_path,
            validator_name=error.validator_name,
            safe_reason=error.safe_reason,
            expected_constraint=error.expected_constraint,
            actual_summary=error.actual_summary,
            recovery_action=recovery_action,
        )
    if isinstance(error, ValueError):
        return ValidationDiagnostic(
            node="decide_action",
            attempt=attempt,
            error_stage="domain_validation",
            error_code="InvalidActionArguments",
            field_path="decision",
            validator_name="action_contract_validator",
            safe_reason="Action arguments violate the runtime Action contract.",
            expected_constraint="a valid tool-specific Action shape",
            actual_summary="contract_rejected=true",
            recovery_action=recovery_action,
        )
    return None


def _field_path(location: tuple[object, ...]) -> str:
    values = list(location)
    if values and values[0] == "decision":
        values = values[1:]
    if values and values[0] in {
        "search_papers",
        "search_knowledge_base",
        "search_references",
        "get_section_context",
        "get_paper_overview",
        "try_answer",
        "stop_research",
    }:
        values = values[1:]
    if not values:
        return "root"
    parts: list[str] = []
    for value in values:
        if isinstance(value, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{value}]"
            else:
                parts.append(f"[{value}]")
        else:
            parts.append(str(value))
    return ".".join(parts)


def _schema_error_code(kind: str, location: tuple[object, ...]) -> str:
    field_path = _field_path(location)
    if kind == "missing":
        return "MissingRequiredField"
    if kind == "extra_forbidden":
        return "UnknownField"
    if kind in {"union_tag_invalid", "union_tag_not_found"} or field_path == "action":
        return "InvalidActionType"
    if "requirement_id" in field_path and kind in {
        "too_short",
        "list_too_short",
        "string_too_short",
    }:
        return "InvalidRequirementBinding"
    if field_path.endswith("query") and kind in {"string_too_short", "string_type"}:
        return "EmptyQuery"
    if "lexical_terms" in field_path and kind in {"too_long", "list_too_long"}:
        return "TooManyLexicalTerms"
    return "InvalidActionArguments"


def _schema_safe_reason(kind: str, location: tuple[object, ...]) -> str:
    code = _schema_error_code(kind, location)
    reasons = {
        "MissingRequiredField": "A required Action field is missing.",
        "UnknownField": "Action contains a field that is not declared by the schema.",
        "InvalidActionType": "Action name must select one declared tool-specific schema.",
        "InvalidRequirementBinding": "Requirement bindings do not meet the Action schema.",
        "EmptyQuery": "Query must be a non-empty string.",
        "TooManyLexicalTerms": "Lexical terms exceed the declared maximum.",
    }
    return reasons.get(code, "Action field does not meet the declared schema.")


def _schema_expected_constraint(
    kind: str, location: tuple[object, ...], detail: dict[str, object]
) -> str:
    field_path = _field_path(location)
    if kind == "missing":
        return "field is required"
    if kind == "extra_forbidden":
        return "only declared schema fields are allowed"
    if kind in {"union_tag_invalid", "union_tag_not_found"} or field_path == "action":
        return "one declared Action name"
    if field_path.endswith("query"):
        return "non-empty string"
    if "lexical_terms" in field_path and kind in {"too_long", "list_too_long"}:
        return f"at most {detail['ctx']['max_length']} lexical terms"
    if "requirement_id" in field_path and kind in {"too_short", "string_too_short"}:
        return "one active requirement_id"
    if kind.endswith("_type"):
        return "declared field type"
    return "declared field constraint"


def _safe_actual_summary(value: object) -> str:
    if value is None:
        return "type=null"
    if isinstance(value, str):
        return f"type=str;empty={str(not value.strip()).lower()};chars={len(value)}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"type=list;count={len(value)}"
    if isinstance(value, dict):
        return f"type=object;keys={len(value)}"
    if isinstance(value, bool):
        return "type=bool"
    if isinstance(value, (int, float)):
        return f"type={type(value).__name__}"
    return f"type={type(value).__name__}"


def _repair_payload(diagnostic: ValidationDiagnostic) -> dict[str, object]:
    payload = {
        "error_stage": diagnostic.error_stage,
        "error_code": diagnostic.error_code,
        "field_path": diagnostic.field_path,
        "expected_constraint": diagnostic.expected_constraint,
        "safe_reason": diagnostic.safe_reason,
    }
    if diagnostic.validator_name == "runtime_guard_preflight":
        # Only our structured Guard feedback is eligible. Do not forward arbitrary
        # diagnostic/model text; keep the same actual compiler identity visible.
        try:
            summary = json.loads(diagnostic.actual_summary)
        except (TypeError, ValueError):
            return payload
        if isinstance(summary, dict):
            keys = ("rejected_tool", "rejected_action_identity", "affected_requirement_ids",
                    "original_decision_iteration")
            request_keys = ("request_identity", "kind", "paper_ids", "semantic_query",
                            "lexical_terms", "lexical_query", "top_k")
            payload["rejected_effective_request"] = {
                **{key: summary[key] for key in keys if key in summary},
                "canonical_effective_requests": [
                    {key: request[key] for key in request_keys if key in request}
                    for request in summary.get("canonical_effective_requests", ())
                    if isinstance(request, dict)
                ],
            }
    return payload


def _runtime_action_capabilities(state: DecisionState) -> RuntimeActionCapabilities:
    requirement_states = state.requirement_states or derive_requirement_states(
        plan=state.plan,
        evidence=state.evidence,
        observations=state.observations,
        context=state.context,
        coverage=state.coverage,
    )
    terminal = {
        RequirementStatus.NO_SUPPORT_FOUND,
        RequirementStatus.UNCERTAIN,
    }
    states_by_id = {item.requirement_id: item.status for item in requirement_states}
    valid_requirement_ids = tuple(
        item.requirement_id
        for item in state.plan.requirements
        if states_by_id.get(item.requirement_id, RequirementStatus.PENDING) not in terminal
    )
    valid_paper_ids = _valid_paper_ids(state)
    reasons: list[str] = []
    usage = dict(state.budget_usage.queries_per_paper)
    search_paper_ids = tuple(paper_id for paper_id in valid_paper_ids
        if state.per_paper_query_limit is None
        or usage.get(paper_id, 0) < state.per_paper_query_limit)
    for paper_id in valid_paper_ids:
        if paper_id not in search_paper_ids:
            reasons.append(f"search:{paper_id}:per_paper_budget_exhausted")
    statuses = dict(state.overview_status_by_paper)
    overview_paper_ids = tuple(paper_id for paper_id in valid_paper_ids
                              if statuses.get(paper_id, "unknown") != "unavailable")
    for paper_id in valid_paper_ids:
        if paper_id not in overview_paper_ids:
            reasons.append(f"get_paper_overview:{paper_id}:unavailable")
    sections: dict[str, list[str]] = {}
    for record in state.evidence:
        section_id = record.evidence.section_id
        if section_id and record.evidence.paper_id in valid_paper_ids:
            sections.setdefault(record.evidence.paper_id, []).append(section_id)
    executed_sections = _executed_section_pairs(state)
    available_sections = tuple(
        (paper_id, available)
        for paper_id, section_ids in sorted(sections.items())
        if (available := tuple(section_id for section_id in dict.fromkeys(section_ids)
                               if (paper_id, section_id) not in executed_sections))
    )
    current_evidence_ids = tuple(record.evidence_id for record in state.evidence)
    answer_is_fresh = (
        state.grounded_answer is not None
        and state.answer_evidence_ids == current_evidence_ids
    )
    available_tools = ["stop_research"]
    can_retry_same_evidence = (
        answer_is_fresh
        and state.grounded_answer is not None
        and not state.grounded_answer.complete
        and state.answer_attempt_count < 2
    )
    # Fresh Registry snapshots route directly to Grounded Answer in the Graph.
    # Decision may request try_answer only for the one bounded reread of an
    # already assessed snapshot.
    if can_retry_same_evidence:
        available_tools.insert(0, "try_answer")
    else:
        reasons.append("try_answer:no_focused_reread_available")
    domains = {
        "search_papers": search_paper_ids,
        "search_references": search_paper_ids,
        "get_paper_overview": overview_paper_ids,
        "get_section_context": available_sections,
        "search_knowledge_base": (
            state.context.scope is QueryScope.KNOWLEDGE_BASE
            and state.candidate_selection is None),
    }
    for tool, domain in domains.items():
        if not valid_requirement_ids:
            reasons.append(f"{tool}:no_active_requirements")
        elif domain:
            available_tools.append(tool)
        else:
            reasons.append(f"{tool}:empty_or_locked_domain")
    budget_state = _tool_budget_state(state)
    budget_state["available_tools"] = tuple(available_tools)  # compatibility mirror
    return RuntimeActionCapabilities(
        valid_requirement_ids=valid_requirement_ids,
        valid_paper_ids=valid_paper_ids,  # compatibility/scope view only
        known_section_ids_by_paper=tuple(
            (paper_id, tuple(dict.fromkeys(section_ids)))
            for paper_id, section_ids in sorted(sections.items())
        ),
        available_section_ids_by_paper=available_sections,
        tool_budget_state=budget_state,
        unavailable_capabilities=tuple(reasons),  # compatibility mirror
        recent_unavailable_actions=_recent_unavailable_actions(state),
        available_tools=tuple(available_tools),
        search_paper_ids=search_paper_ids,
        reference_paper_ids=search_paper_ids,
        overview_paper_ids=overview_paper_ids,
        capability_reasons=tuple(reasons),
    )


def _executed_section_pairs(state: DecisionState) -> set[tuple[str, str]]:
    """Recover only successfully executed section-context pairs from audit state."""

    pairs: set[tuple[str, str]] = set()
    for observation in state.observations:
        if (observation.action_name != "get_section_context"
                or observation.tool_status not in {ToolStatus.OK, ToolStatus.EMPTY}):
            continue
        arguments = observation.action_arguments or {}
        paper_id = arguments.get("paper_id")
        section_id = arguments.get("section_id")
        if isinstance(paper_id, str) and isinstance(section_id, str):
            pairs.add((paper_id, section_id))
    return pairs


def _valid_paper_ids(state: DecisionState) -> tuple[str, ...]:
    if state.candidate_selection is not None:
        return tuple(
            dict.fromkeys(
                (
                    *state.candidate_selection.candidate_paper_ids,
                    *state.candidate_selection.uncertain_paper_ids,
                )
            )
        )
    if state.context.scope is QueryScope.KNOWLEDGE_BASE:
        return ()
    return state.context.target_paper_ids


def _recent_unavailable_actions(
    state: DecisionState, *, limit: int = 8
) -> tuple[RuntimeUnavailableAction, ...]:
    summaries: list[RuntimeUnavailableAction] = []
    for observation in state.observations[-limit:]:
        arguments = dict(observation.action_arguments or {})
        if arguments:
            summaries.append(
                RuntimeUnavailableAction(
                    action_name=observation.action_name,
                    action_arguments=arguments,
                    requirement_ids=_observation_requirement_ids(observation),
                    reason_code="ExecutedAction",
                )
            )
    for rejection in state.action_rejections[-limit:]:
        if rejection.rejection_code not in {
            "DuplicateAction",
            "DuplicatePaperQuery",
            "PerPaperBudget",
        }:
            continue
        summaries.append(
            RuntimeUnavailableAction(
                action_name=rejection.action_name,
                action_arguments=dict(rejection.action_arguments),
                requirement_ids=rejection.requirement_ids,
                reason_code=rejection.rejection_code,
            )
        )
    seen: set[str] = set()
    unique: list[RuntimeUnavailableAction] = []
    for item in reversed(summaries):
        key = json.dumps(
            [item.action_name, item.action_arguments, item.reason_code],
            ensure_ascii=False,
            sort_keys=True,
            default=list,
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(reversed(unique[:limit]))
