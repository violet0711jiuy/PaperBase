"""Agentic Research 的稳定领域契约，不包含具体 LLM 或 Tool 实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import TypeAlias

from paperbase.llm.client import LLMCallRecord
from paperbase.generation.answer_coverage import AnswerRequirementCoverage
from paperbase.retrieval.contracts import PaperSearchSpec, RetrievedEvidence, ToolStatus, ScoreKind, ScoreDirection
from paperbase.retrieval.evidence_fingerprint import build_evidence_fingerprint


class Intent(str, Enum):
    RETRIEVE = "retrieve"
    EXPLAIN = "explain"
    COMPARE = "compare"
    SYNTHESIZE = "synthesize"


class QueryScope(str, Enum):
    SINGLE_PAPER = "single_paper"
    SELECTED_PAPERS = "selected_papers"
    KNOWLEDGE_BASE = "knowledge_base"


class ScopeSource(str, Enum):
    TRUSTED_WORKSPACE = "trusted_workspace"
    UI_SELECTION = "ui_selection"
    RESOLVED_FROM_QUERY = "resolved_from_query"
    KNOWLEDGE_BASE_ALL = "knowledge_base_all"


class ResearchGoal(str, Enum):
    FIND_EVIDENCE = "find_evidence"
    FIND_EXAMPLES = "find_examples"
    EXHAUSTIVE_ENUMERATION = "exhaustive_enumeration"
    COMPARISON_SUPPORT = "comparison_support"
    SYNTHESIS_SUPPORT = "synthesis_support"


class PlanningMode(str, Enum):
    UNSPECIFIED = "unspecified"
    MODEL = "model"
    LITERAL_FALLBACK = "literal_fallback"
    DETERMINISTIC_ENUMERATION = "deterministic_enumeration"


class EvidenceTarget(str, Enum):
    """Retrieval/routing hints, never completion, verified support or a hard filter."""

    PAPER_OWN_FACT = "paper_own_fact"
    METHOD = "method"
    RESULT = "result"
    CLAIM = "claim"
    PRIOR_WORK = "prior_work"
    FIELD_OVERVIEW = "field_overview"
    REFERENCE = "reference"
    GENERAL = "general"


class PaperAssessmentStatus(str, Enum):
    UNVISITED = "unvisited"
    CANDIDATE = "candidate"
    POSITIVE = "positive"
    NO_SUPPORT_FOUND = "no_support_found"
    UNCERTAIN = "uncertain"


class CoverageStrategy(str, Enum):
    EXHAUSTIVE = "exhaustive"
    CANDIDATE_VERIFICATION = "candidate_verification"


class ScopeResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


class CandidateDisposition(str, Enum):
    """元数据筛选后论文进入正文检索的处置结果。"""

    CANDIDATE = "candidate"
    UNCERTAIN = "uncertain"
    EXCLUDED = "excluded"


class RequirementStatus(str, Enum):
    """Research progress; legacy ``satisfied`` means support_found, not complete."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SATISFIED = "satisfied"
    NO_SUPPORT_FOUND = "no_support_found"
    UNCERTAIN = "uncertain"


class ResearchRunStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class StopReason(str, Enum):
    AGENT_FINISHED = "agent_finished"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PER_PAPER_QUERY_LIMIT = "per_paper_query_limit"
    SCOPE_UNRESOLVED = "scope_unresolved"
    NO_AVAILABLE_ACTIONS = "no_available_actions"
    INVALID_ACTION_LIMIT = "invalid_action_limit"
    FATAL_ERROR = "fatal_error"


@dataclass(frozen=True)
class ActionRejectionContext:
    """Unified runtime action-validation feedback for the next Agent replan."""

    action_name: str
    action_arguments: dict[str, object]
    requirement_ids: tuple[str, ...]
    rejection_code: str
    rejection_reason: str
    guard_name: str
    iteration: int
    recovery_action: str = "replan_action"
    rejection_stage: str = "guard"
    # Structured when the runtime can name a concrete blocker (for example a
    # per-paper query budget).  `object` intentionally keeps old checkpoints
    # with string blockers readable.
    blockers: tuple[object, ...] = ()
    validator_name: str = "runtime_guard"
    action_canonicalization: "ActionCanonicalization | None" = None


# Checkpoints produced during Phase A.1 used this public name.  Keep the
# import alias only; new state and payloads use ActionRejectionContext.
GuardRejectionContext = ActionRejectionContext


@dataclass(frozen=True)
class CanonicalizationOperation:
    """One deterministic, non-semantic normalization made to an Action."""

    field_path: str
    operation: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_path", _text(self.field_path, "field_path"))
        object.__setattr__(self, "operation", _text(self.operation, "operation"))
        if self.count < 1:
            raise ValueError("Canonicalization operation count must be positive.")


@dataclass(frozen=True)
class ActionCanonicalization:
    """Safe audit summary for an Action canonicalized before domain validation."""

    changed: bool = False
    changed_fields: tuple[str, ...] = ()
    operations: tuple[CanonicalizationOperation, ...] = ()

    def __post_init__(self) -> None:
        fields = _unique_texts(self.changed_fields)
        operations = tuple(self.operations)
        if self.changed != bool(operations):
            raise ValueError("Canonicalization changed must match its operations.")
        object.__setattr__(self, "changed_fields", fields)
        object.__setattr__(self, "operations", operations)


@dataclass(frozen=True)
class RuntimeUnavailableAction:
    """Historical Action telemetry, not an effective Search identity or prohibition."""

    action_name: str
    action_arguments: dict[str, object]
    requirement_ids: tuple[str, ...]
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_name", _text(self.action_name, "action_name"))
        object.__setattr__(self, "action_arguments", dict(self.action_arguments))
        object.__setattr__(self, "requirement_ids", _unique_texts(self.requirement_ids))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))


@dataclass(frozen=True)
class RuntimeUnavailablePaperQuery:
    """Deprecated checkpoint/API telemetry; never a Search dedup authority."""

    paper_id: str
    query: str
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))


@dataclass(frozen=True)
class RuntimeActionCapabilities:
    """Single structural feasible-domain contract, projected from Graph State.

    valid_paper_ids, tool_budget_state.available_tools and unavailable_paper_queries
    are compatibility/view fields only. Search identity belongs to canonical
    requests and Guard, never to raw Action history.
    """

    valid_requirement_ids: tuple[str, ...] = ()
    valid_paper_ids: tuple[str, ...] = ()
    known_section_ids_by_paper: tuple[tuple[str, tuple[str, ...]], ...] = ()
    available_section_ids_by_paper: tuple[tuple[str, tuple[str, ...]], ...] = ()
    tool_budget_state: dict[str, object] = field(default_factory=dict)
    unavailable_capabilities: tuple[str, ...] = ()
    recent_unavailable_actions: tuple[RuntimeUnavailableAction, ...] = ()
    unavailable_paper_queries: tuple[RuntimeUnavailablePaperQuery, ...] = ()
    available_tools: tuple[str, ...] = ()
    search_paper_ids: tuple[str, ...] = ()
    reference_paper_ids: tuple[str, ...] = ()
    overview_paper_ids: tuple[str, ...] = ()
    capability_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid_requirement_ids", _unique_texts(self.valid_requirement_ids))
        object.__setattr__(self, "valid_paper_ids", _unique_texts(self.valid_paper_ids))
        sections: list[tuple[str, tuple[str, ...]]] = []
        for paper_id, section_ids in self.known_section_ids_by_paper:
            sections.append((_text(paper_id, "paper_id"), _unique_texts(section_ids)))
        object.__setattr__(self, "known_section_ids_by_paper", tuple(sections))
        available_sections: list[tuple[str, tuple[str, ...]]] = []
        for paper_id, section_ids in self.available_section_ids_by_paper:
            available_sections.append(
                (_text(paper_id, "paper_id"), _unique_texts(section_ids))
            )
        object.__setattr__(
            self, "available_section_ids_by_paper", tuple(available_sections)
        )
        object.__setattr__(self, "tool_budget_state", dict(self.tool_budget_state))
        object.__setattr__(self, "unavailable_capabilities", _unique_texts(self.unavailable_capabilities))
        object.__setattr__(self, "recent_unavailable_actions", tuple(self.recent_unavailable_actions))
        object.__setattr__(self, "unavailable_paper_queries", tuple(self.unavailable_paper_queries))
        for name in ("available_tools", "search_paper_ids", "reference_paper_ids",
                     "overview_paper_ids", "capability_reasons"):
            object.__setattr__(self, name, _unique_texts(getattr(self, name)))


@dataclass(frozen=True)
class CorpusScopeSnapshot:
    snapshot_id: str
    paper_ids: tuple[str, ...]
    created_at: str

    @classmethod
    def create(cls, paper_ids: tuple[str, ...]) -> "CorpusScopeSnapshot":
        normalized = tuple(sorted(_unique_texts(paper_ids)))
        digest = sha256("\n".join(sorted(normalized)).encode("utf-8")).hexdigest()[:16]
        return cls(
            snapshot_id=f"corpus_{digest}",
            paper_ids=normalized,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class ScopeResolution:
    status: ScopeResolutionStatus
    scope: QueryScope | None = None
    scope_source: ScopeSource | None = None
    target_paper_ids: tuple[str, ...] = ()
    candidate_paper_ids: tuple[str, ...] = ()
    clarification: str | None = None
    corpus_snapshot: CorpusScopeSnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ScopeResolutionStatus(self.status))
        object.__setattr__(self, "target_paper_ids", _unique_texts(self.target_paper_ids))
        object.__setattr__(self, "candidate_paper_ids", _unique_texts(self.candidate_paper_ids))
        if self.status is ScopeResolutionStatus.RESOLVED:
            if self.scope is None or self.scope_source is None:
                raise ValueError("A resolved scope requires scope and scope_source.")
            object.__setattr__(self, "scope", QueryScope(self.scope))
            object.__setattr__(self, "scope_source", ScopeSource(self.scope_source))


@dataclass(frozen=True)
class CandidateAssessment:
    paper_id: str
    disposition: CandidateDisposition
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "disposition", CandidateDisposition(self.disposition))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))


@dataclass(frozen=True)
class CandidateSelection:
    """冻结 Scope 内完整、互斥且可审计的论文元数据筛选结果。"""

    scope_paper_ids: tuple[str, ...]
    assessments: tuple[CandidateAssessment, ...]

    def __post_init__(self) -> None:
        scope_ids = _unique_texts(self.scope_paper_ids)
        assessments = tuple(self.assessments)
        assessment_ids = tuple(item.paper_id for item in assessments)
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("Candidate Selection 不能重复处理同一 paper_id。")
        if set(assessment_ids) != set(scope_ids):
            raise ValueError("Candidate Selection 必须完整覆盖冻结 Scope。")
        object.__setattr__(self, "scope_paper_ids", scope_ids)
        object.__setattr__(self, "assessments", assessments)

    def paper_ids_for(self, disposition: CandidateDisposition) -> tuple[str, ...]:
        return tuple(
            item.paper_id
            for item in self.assessments
            if item.disposition is disposition
        )

    @property
    def screened_paper_ids(self) -> tuple[str, ...]:
        return tuple(item.paper_id for item in self.assessments)

    @property
    def candidate_paper_ids(self) -> tuple[str, ...]:
        return self.paper_ids_for(CandidateDisposition.CANDIDATE)

    @property
    def uncertain_paper_ids(self) -> tuple[str, ...]:
        return self.paper_ids_for(CandidateDisposition.UNCERTAIN)

    @property
    def excluded_paper_ids(self) -> tuple[str, ...]:
        return self.paper_ids_for(CandidateDisposition.EXCLUDED)


@dataclass(frozen=True)
class QueryContext:
    original_query: str
    resolved_query: str
    intent: Intent
    scope: QueryScope
    target_paper_ids: tuple[str, ...]
    scope_source: ScopeSource
    retrieval_goal: ResearchGoal
    default_evidence_targets: frozenset[EvidenceTarget]

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_query", _text(self.original_query, "original_query"))
        object.__setattr__(self, "resolved_query", _text(self.resolved_query, "resolved_query"))
        object.__setattr__(self, "intent", Intent(self.intent))
        object.__setattr__(self, "scope", QueryScope(self.scope))
        object.__setattr__(self, "scope_source", ScopeSource(self.scope_source))
        object.__setattr__(self, "retrieval_goal", ResearchGoal(self.retrieval_goal))
        ids = _unique_texts(self.target_paper_ids)
        object.__setattr__(self, "target_paper_ids", ids)
        targets = frozenset(EvidenceTarget(value) for value in self.default_evidence_targets)
        if not targets:
            raise ValueError("default_evidence_targets must not be empty.")
        object.__setattr__(self, "default_evidence_targets", targets)
        if self.scope is QueryScope.SINGLE_PAPER and len(ids) != 1:
            raise ValueError("single_paper scope requires exactly one paper_id.")
        if self.scope is QueryScope.SELECTED_PAPERS and len(ids) < 2:
            raise ValueError("selected_papers scope requires at least two paper_ids.")
        if self.scope is QueryScope.KNOWLEDGE_BASE and ids:
            raise ValueError("knowledge_base scope stores membership in a corpus snapshot.")

    @property
    def requires_complete_coverage(self) -> bool:
        return self.retrieval_goal is ResearchGoal.EXHAUSTIVE_ENUMERATION

    @property
    def search_bibliography(self) -> bool:
        return EvidenceTarget.REFERENCE in self.default_evidence_targets


@dataclass(frozen=True)
class ResearchRequirement:
    """A stable answer obligation.

    evidence_targets are this Requirement's authoritative retrieval/routing
    intent. They do not assert completion or semantic support.
    """

    requirement_id: str
    description: str
    evidence_targets: frozenset[EvidenceTarget] = frozenset()
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        object.__setattr__(self, "description", _text(self.description, "description"))
        if not isinstance(self.required, bool):
            raise ValueError("ResearchRequirement.required must be boolean.")
        object.__setattr__(
            self,
            "evidence_targets",
            frozenset(EvidenceTarget(value) for value in self.evidence_targets),
        )
        if not self.evidence_targets:
            raise ValueError("ResearchRequirement 至少需要一个 evidence target。")


@dataclass(frozen=True)
class RequirementState:
    """Research progress for one Requirement, never final answer completeness."""

    requirement_id: str
    status: RequirementStatus = RequirementStatus.PENDING
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _text(self.requirement_id, "requirement_id"))
        object.__setattr__(self, "status", RequirementStatus(self.status))
        object.__setattr__(self, "evidence_ids", _unique_texts(self.evidence_ids))


@dataclass(frozen=True)
class ResearchPlan:
    """The answer goal and its stable requirements."""

    goal: str
    requirements: tuple[ResearchRequirement, ...] = ()
    planning_mode: PlanningMode = PlanningMode.UNSPECIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        requirements = tuple(self.requirements)
        requirement_ids = [item.requirement_id for item in requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Research requirement IDs must be unique.")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "planning_mode", PlanningMode(self.planning_mode))


@dataclass(frozen=True)
class SearchPapersAction:
    searches: tuple[PaperSearchSpec, ...]
    action_name: str = field(default="search_papers", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "searches", tuple(self.searches))
        if not self.searches:
            raise ValueError("search_papers requires at least one search.")


@dataclass(frozen=True)
class SearchKnowledgeBaseAction:
    query: str
    lexical_terms: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    action_name: str = field(default="search_knowledge_base", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "lexical_terms", _unique_texts(self.lexical_terms))
        object.__setattr__(self, "requirement_ids", _unique_texts(self.requirement_ids))


@dataclass(frozen=True)
class SearchReferencesAction:
    paper_id: str
    query: str
    lexical_terms: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    action_name: str = field(default="search_references", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "lexical_terms", _unique_texts(self.lexical_terms))
        object.__setattr__(self, "requirement_ids", _unique_texts(self.requirement_ids))


@dataclass(frozen=True)
class GetSectionContextAction:
    paper_id: str
    section_id: str
    requirement_ids: tuple[str, ...] = ()
    action_name: str = field(default="get_section_context", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "section_id", _text(self.section_id, "section_id"))
        object.__setattr__(self, "requirement_ids", _unique_texts(self.requirement_ids))


@dataclass(frozen=True)
class GetPaperOverviewAction:
    paper_id: str
    requirement_ids: tuple[str, ...] = ()
    action_name: str = field(default="get_paper_overview", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "requirement_ids", _unique_texts(self.requirement_ids))


@dataclass(frozen=True)
class TryAnswerAction:
    reason: str
    action_name: str = field(default="try_answer", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _text(self.reason, "reason"))


@dataclass(frozen=True)
class StopResearchAction:
    reason: str
    action_name: str = field(default="stop_research", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _text(self.reason, "reason"))


AgentAction: TypeAlias = (
    SearchPapersAction
    | SearchKnowledgeBaseAction
    | SearchReferencesAction
    | GetSectionContextAction
    | GetPaperOverviewAction
    | TryAnswerAction
    | StopResearchAction
)


@dataclass(frozen=True)
class GroundedCitation:
    """A provider-selected source range restored from one raw Evidence item."""

    evidence_id: str
    source_span_ids: tuple[str, ...]
    source_excerpt: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "source_span_ids", _unique_texts(self.source_span_ids))
        if not self.source_span_ids:
            raise ValueError("GroundedCitation requires at least one source span ID.")
        object.__setattr__(self, "source_excerpt", _text(self.source_excerpt, "source_excerpt"))


@dataclass(frozen=True)
class RequirementAnswer:
    """One Requirement's answer state; the cited claim and its gap stay together."""

    requirement_id: str
    answer: str | None
    citations: tuple[GroundedCitation, ...] = ()
    remaining_gap: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_id", _text(self.requirement_id, "requirement_id"))
        answer = self.answer.strip() if isinstance(self.answer, str) else None
        gap = self.remaining_gap.strip() if isinstance(self.remaining_gap, str) else None
        object.__setattr__(self, "answer", answer or None)
        object.__setattr__(self, "remaining_gap", gap or None)
        object.__setattr__(self, "citations", tuple(self.citations))
        if self.answer is None:
            if self.citations or self.remaining_gap is None:
                raise ValueError("An unanswered Requirement needs an explicit remaining_gap and no citations.")
        elif not self.citations:
            raise ValueError("A Requirement answer must cite at least one validated source span.")

    @property
    def complete(self) -> bool:
        return self.answer is not None and self.remaining_gap is None


@dataclass(frozen=True)
class GroundedAnswerAttempt:
    """The single semantic answer authority for one try_answer operation."""

    requirements: tuple[RequirementAnswer, ...]
    # Evidence actually cited by Requirement answers, not the whole input pack.
    evidence_ids: tuple[str, ...]
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()

    def __post_init__(self) -> None:
        requirements = tuple(self.requirements)
        ids = tuple(item.requirement_id for item in requirements)
        if len(ids) != len(set(ids)):
            raise ValueError("Grounded answer Requirement IDs must be unique.")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "evidence_ids", _unique_texts(self.evidence_ids))

    @property
    def complete(self) -> bool:
        return bool(self.requirements) and all(item.complete for item in self.requirements)

    @property
    def has_grounded_answer(self) -> bool:
        return any(item.answer is not None for item in self.requirements)

    @property
    def open_gaps(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.requirement_id, item.remaining_gap)
            for item in self.requirements
            if item.remaining_gap is not None
        )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    evidence: RetrievedEvidence
    supporting_queries: tuple[str, ...]
    first_seen_iteration: int
    last_seen_iteration: int
    originating_tools: tuple[str, ...] = ()
    fingerprint: str | None = None
    # Derived Plan intent for audit/compatibility; not a support authority.
    requested_evidence_targets: frozenset[EvidenceTarget] = frozenset()
    requested_requirement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fingerprint = self.fingerprint or build_evidence_fingerprint(self.evidence)
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint.casefold()
        ):
            raise ValueError("fingerprint 必须是 64 位十六进制 SHA-256。")
        object.__setattr__(self, "fingerprint", fingerprint.casefold())
        object.__setattr__(
            self,
            "requested_evidence_targets",
            frozenset(EvidenceTarget(value) for value in self.requested_evidence_targets),
        )
        object.__setattr__(
            self,
            "requested_requirement_ids",
            _unique_texts(self.requested_requirement_ids),
        )


@dataclass(frozen=True)
class RegistryDelta:
    added_ids: tuple[str, ...] = ()
    reused_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegistryState:
    """可进入 LangGraph checkpoint 的 Evidence Registry 完整快照。"""

    records: tuple[EvidenceRecord, ...] = ()
    content_counter: int = 0
    reference_counter: int = 0


@dataclass(frozen=True)
class EvidenceFilterDecision:
    """Evidence Admission Filter 对单条候选的无正文审计记录。"""

    fingerprint: str
    paper_id: str
    source_chunk_ids: tuple[str, ...]
    decision: str
    reasons: tuple[str, ...]
    # Deprecated compatibility field for old Observation/checkpoint payloads.
    score: float | None = None
    raw_score: float | None = None
    score_kind: ScoreKind = ScoreKind.LEGACY_UNKNOWN
    score_direction: ScoreDirection = ScoreDirection.NOT_APPLICABLE
    retrieval_rank: int | None = None


@dataclass(frozen=True)
class EvidenceFilterReport:
    total_candidates: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    deprioritized_candidates: int = 0
    decisions: tuple[EvidenceFilterDecision, ...] = ()


@dataclass(frozen=True)
class PaperSearchAttempt:
    """一次逐论文搜索的完整意图与结果摘要。"""

    paper_id: str
    query: str
    lexical_terms: tuple[str, ...] = ()
    evidence_targets: frozenset[EvidenceTarget] = frozenset()
    requirement_ids: tuple[str, ...] = ()
    status: str = ""
    error_code: str | None = None


@dataclass(frozen=True)
class ResearchObservation:
    iteration: int
    action_name: str
    tool_status: ToolStatus
    added_evidence_ids: tuple[str, ...]
    reused_evidence_ids: tuple[str, ...]
    queries_by_paper: tuple[tuple[str, str], ...] = ()
    error_codes: tuple[str, ...] = ()
    goal_progress: bool = False
    new_goal_items: tuple[str, ...] = ()
    # Tool-specific, JSON-safe arguments for audit.  It deliberately excludes
    # legacy Subtask fields and any retrieved document content.
    action_arguments: dict[str, object] = field(default_factory=dict)
    search_attempts: tuple[PaperSearchAttempt, ...] = ()
    evidence_filter_report: EvidenceFilterReport | None = None
    requirement_progress_before: tuple[tuple[str, str], ...] = ()
    requirement_progress_after: tuple[tuple[str, str], ...] = ()
    newly_exposed_section_ids: tuple[str, ...] = ()
    action_canonicalization: ActionCanonicalization | None = None
    raw_action_arguments: dict[str, object] = field(default_factory=dict)
    canonical_effective_requests: tuple[dict[str, object], ...] = ()
    request_identities: tuple[str, ...] = ()
    request_executions: tuple[dict[str, object], ...] = ()
    request_contract_version: str = "legacy"
    # Plan-derived audit rows only; search_index/paper_id retain batch attribution.
    requirement_target_bindings: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class BudgetUsage:
    agent_iterations: int = 0
    tool_calls: int = 0
    retrieval_queries: int = 0
    llm_calls: int = 0
    invalid_actions: int = 0
    queries_per_paper: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class GuardState:
    """预算、去重与逐论文抑制的可恢复状态，不包含运行依赖。"""

    usage: BudgetUsage = field(default_factory=BudgetUsage)
    no_new_per_paper: tuple[tuple[str, int], ...] = ()
    no_goal_progress_per_paper: tuple[tuple[str, int], ...] = ()
    suppressed_paper_ids: tuple[str, ...] = ()
    executed_action_fingerprints: tuple[str, ...] = ()
    executed_paper_query_fingerprints: tuple[str, ...] = ()
    request_executions: tuple[dict[str, object], ...] = ()
    legacy_request_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionTrace:
    iteration: int
    action_name: str
    accepted: bool
    detail: str


@dataclass(frozen=True)
class ResearchAnswer:
    """Agentic Research 的最终用户可见回答及引用状态。"""

    status: str
    answer: str
    citations: tuple[str, ...] = ()
    insufficient_evidence: bool = False
    partial_answer: bool = False
    coverage_note: str | None = None
    requirement_coverage: tuple[AnswerRequirementCoverage, ...] = ()
    grounded_requirements: tuple[RequirementAnswer, ...] = ()
    overall_status: str | None = None
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _text(self.status, "status"))
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string.")
        # 回答正文可能包含 Markdown 标题和段落，不能像标识符一样折叠全部空白。
        object.__setattr__(self, "answer", self.answer.strip())
        object.__setattr__(self, "citations", _unique_texts(self.citations))
        object.__setattr__(self, "grounded_requirements", tuple(self.grounded_requirements))


@dataclass(frozen=True)
class PaperAssessment:
    paper_id: str
    status: PaperAssessmentStatus
    queries_tried: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "status", PaperAssessmentStatus(self.status))
        object.__setattr__(self, "queries_tried", _unique_texts(self.queries_tried))
        object.__setattr__(self, "evidence_ids", _unique_texts(self.evidence_ids))


@dataclass(frozen=True)
class CoverageState:
    """由 Scope、Candidate Selection 与正文验证历史派生的覆盖状态。"""

    scope_paper_ids: tuple[str, ...]
    assessments: tuple[PaperAssessment, ...]
    strategy: CoverageStrategy
    candidate_paper_ids: tuple[str, ...] = ()
    uncertain_paper_ids: tuple[str, ...] = ()
    excluded_paper_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scope_ids = _unique_texts(self.scope_paper_ids)
        candidate_ids = _unique_texts(self.candidate_paper_ids)
        uncertain_ids = _unique_texts(self.uncertain_paper_ids)
        excluded_ids = _unique_texts(self.excluded_paper_ids)
        partitions = (set(candidate_ids), set(uncertain_ids), set(excluded_ids))
        if any(not values.issubset(set(scope_ids)) for values in partitions):
            raise ValueError("Coverage screening partitions must stay inside Scope.")
        if any(partitions[i].intersection(partitions[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Coverage screening partitions must be mutually exclusive.")
        assessment_ids = tuple(item.paper_id for item in self.assessments)
        if len(set(assessment_ids)) != len(assessment_ids):
            raise ValueError("Coverage assessments must not contain duplicate paper IDs.")
        if not set(assessment_ids).issubset(set(scope_ids)):
            raise ValueError("Coverage assessments must stay inside scope_paper_ids.")
        object.__setattr__(self, "scope_paper_ids", scope_ids)
        object.__setattr__(self, "candidate_paper_ids", candidate_ids)
        object.__setattr__(self, "uncertain_paper_ids", uncertain_ids)
        object.__setattr__(self, "excluded_paper_ids", excluded_ids)
        object.__setattr__(self, "strategy", CoverageStrategy(self.strategy))

    @property
    def checked_paper_ids(self) -> tuple[str, ...]:
        terminal = {
            PaperAssessmentStatus.POSITIVE,
            PaperAssessmentStatus.NO_SUPPORT_FOUND,
            PaperAssessmentStatus.UNCERTAIN,
        }
        return tuple(
            item.paper_id for item in self.assessments if item.status in terminal
        )

    @property
    def screened_paper_ids(self) -> tuple[str, ...]:
        screened = set(self.candidate_paper_ids).union(
            self.uncertain_paper_ids, self.excluded_paper_ids
        )
        return tuple(paper_id for paper_id in self.scope_paper_ids if paper_id in screened)

    @property
    def positive_paper_ids(self) -> tuple[str, ...]:
        return tuple(
            item.paper_id
            for item in self.assessments
            if item.status is PaperAssessmentStatus.POSITIVE
        )

    @property
    def required_paper_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.candidate_paper_ids, *self.uncertain_paper_ids)))

    @property
    def verified_paper_ids(self) -> tuple[str, ...]:
        return self.checked_paper_ids

    @property
    def remaining_paper_ids(self) -> tuple[str, ...]:
        checked = set(self.checked_paper_ids)
        return tuple(
            paper_id for paper_id in self.required_paper_ids if paper_id not in checked
        )

    @property
    def coverage_complete(self) -> bool:
        if set(self.screened_paper_ids) != set(self.scope_paper_ids):
            return False
        return not self.remaining_paper_ids


@dataclass(frozen=True)
class ResearchRunResult:
    status: ResearchRunStatus
    stop_reason: StopReason
    query_context: QueryContext | None
    plan: ResearchPlan | None
    evidence: tuple[EvidenceRecord, ...]
    observations: tuple[ResearchObservation, ...]
    traces: tuple[ActionTrace, ...]
    budget_usage: BudgetUsage
    coverage: CoverageState | None = None
    candidate_selection: CandidateSelection | None = None
    selected_evidence: tuple[EvidenceRecord, ...] = ()
    answer: ResearchAnswer | None = None
    thread_id: str | None = None
    error_codes: tuple[str, ...] = ()
    clarification: str | None = None
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()
    # Phase 1 canonical completion-state projection.  Empty is valid for a
    # historical checkpoint/result that predates RequirementState.
    requirement_states: tuple[RequirementState, ...] = ()
    action_rejections: tuple[ActionRejectionContext, ...] = ()
    # Complete Registry snapshot supplied to the most recent Grounded Answer
    # attempt.  This is distinct from selected_evidence, which is citation-only.
    answer_input_evidence_ids: tuple[str, ...] = ()
    # Read-only compatibility for results persisted before the unified field.
    guard_rejections: tuple[ActionRejectionContext, ...] = ()

    def __post_init__(self) -> None:
        states = tuple(self.requirement_states)
        ids = [item.requirement_id for item in states]
        if len(ids) != len(set(ids)):
            raise ValueError("Requirement states must not contain duplicate requirement IDs.")
        object.__setattr__(self, "requirement_states", states)
        object.__setattr__(
            self, "answer_input_evidence_ids", _unique_texts(self.answer_input_evidence_ids)
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return " ".join(value.split())


def _unique_texts(values: tuple[str, ...] | object) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:  # type: ignore[union-attr]
        normalized = _text(value, "value")
        if normalized not in result:
            result.append(normalized)
    return tuple(result)
