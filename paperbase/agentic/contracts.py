"""Agentic Research 的稳定领域契约，不包含具体 LLM 或 Tool 实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import TypeAlias

from paperbase.retrieval.contracts import PaperSearchSpec, RetrievedEvidence, ToolStatus


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


class EvidenceTarget(str, Enum):
    PAPER_OWN_FACT = "paper_own_fact"
    METHOD = "method"
    RESULT = "result"
    CLAIM = "claim"
    PRIOR_WORK = "prior_work"
    FIELD_OVERVIEW = "field_overview"
    REFERENCE = "reference"
    GENERAL = "general"


class EvidenceRelation(str, Enum):
    """证据与当前研究目标之间的关系。"""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    BACKGROUND_ONLY = "background_only"
    UNCERTAIN = "uncertain"
    IRRELEVANT = "irrelevant"


class EvidenceOwnership(str, Enum):
    """证据描述的是本文工作、既有工作还是一般背景。"""

    THIS_PAPER = "this_paper"
    PRIOR_WORK = "prior_work"
    GENERAL_BACKGROUND = "general_background"
    UNKNOWN = "unknown"


class EvidenceAnalysisStatus(str, Enum):
    """原始 Evidence 在自动证据分析流水线中的状态。"""

    PENDING = "pending"
    QUALIFIED = "qualified"
    IRRELEVANT = "irrelevant"
    FAILED = "failed"


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


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ResearchRunStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class StopReason(str, Enum):
    AGENT_FINISHED = "agent_finished"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SCOPE_UNRESOLVED = "scope_unresolved"
    NO_AVAILABLE_ACTIONS = "no_available_actions"
    INVALID_ACTION_LIMIT = "invalid_action_limit"
    FATAL_ERROR = "fatal_error"


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
    semantic_query_en: str | None = None
    lexical_keywords_en: tuple[str, ...] = ()
    requires_complete_coverage: bool = False
    search_bibliography: bool = False

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
        object.__setattr__(self, "lexical_keywords_en", _unique_texts(self.lexical_keywords_en))
        if self.scope is QueryScope.SINGLE_PAPER and len(ids) != 1:
            raise ValueError("single_paper scope requires exactly one paper_id.")
        if self.scope is QueryScope.SELECTED_PAPERS and len(ids) < 2:
            raise ValueError("selected_papers scope requires at least two paper_ids.")
        if self.scope is QueryScope.KNOWLEDGE_BASE and ids:
            raise ValueError("knowledge_base scope stores membership in a corpus snapshot.")
        expected_complete = self.retrieval_goal is ResearchGoal.EXHAUSTIVE_ENUMERATION
        if self.requires_complete_coverage != expected_complete:
            raise ValueError("requires_complete_coverage is derived from retrieval_goal.")
        expected_bibliography = EvidenceTarget.REFERENCE in targets
        if self.search_bibliography != expected_bibliography:
            raise ValueError("search_bibliography is derived from evidence targets.")


@dataclass(frozen=True)
class ResearchSubtask:
    subtask_id: str
    goal: str
    target_paper_ids: tuple[str, ...]
    evidence_targets: frozenset[EvidenceTarget]
    success_criteria: str
    status: SubtaskStatus = SubtaskStatus.PENDING

    def __post_init__(self) -> None:
        object.__setattr__(self, "subtask_id", _text(self.subtask_id, "subtask_id"))
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        object.__setattr__(self, "success_criteria", _text(self.success_criteria, "success_criteria"))
        object.__setattr__(self, "target_paper_ids", _unique_texts(self.target_paper_ids))
        object.__setattr__(self, "evidence_targets", frozenset(EvidenceTarget(v) for v in self.evidence_targets))
        object.__setattr__(self, "status", SubtaskStatus(self.status))


@dataclass(frozen=True)
class ResearchPlan:
    goal: str
    subtasks: tuple[ResearchSubtask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        object.__setattr__(self, "subtasks", tuple(self.subtasks))
        if not self.subtasks:
            raise ValueError("A research plan requires at least one subtask.")
        ids = [task.subtask_id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Research subtask IDs must be unique.")


@dataclass(frozen=True)
class SearchPapersAction:
    searches: tuple[PaperSearchSpec, ...]
    target_subtask_id: str | None = None
    action_name: str = field(default="search_papers", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "searches", tuple(self.searches))
        if not self.searches:
            raise ValueError("search_papers requires at least one search.")


@dataclass(frozen=True)
class SearchKnowledgeBaseAction:
    query: str
    lexical_terms: tuple[str, ...] = ()
    target_subtask_id: str | None = None
    action_name: str = field(default="search_knowledge_base", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "lexical_terms", _unique_texts(self.lexical_terms))


@dataclass(frozen=True)
class SearchReferencesAction:
    paper_id: str
    query: str
    lexical_terms: tuple[str, ...] = ()
    target_subtask_id: str | None = None
    action_name: str = field(default="search_references", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "lexical_terms", _unique_texts(self.lexical_terms))


@dataclass(frozen=True)
class GetSectionContextAction:
    paper_id: str
    section_id: str
    target_subtask_id: str | None = None
    action_name: str = field(default="get_section_context", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        object.__setattr__(self, "section_id", _text(self.section_id, "section_id"))


@dataclass(frozen=True)
class GetPaperOverviewAction:
    paper_id: str
    target_subtask_id: str | None = None
    action_name: str = field(default="get_paper_overview", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _text(self.paper_id, "paper_id"))
        if self.target_subtask_id is not None:
            object.__setattr__(
                self,
                "target_subtask_id",
                _text(self.target_subtask_id, "target_subtask_id"),
            )


@dataclass(frozen=True)
class FinishAction:
    reason: str
    action_name: str = field(default="finish", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _text(self.reason, "reason"))


AgentAction: TypeAlias = (
    SearchPapersAction
    | SearchKnowledgeBaseAction
    | SearchReferencesAction
    | GetSectionContextAction
    | GetPaperOverviewAction
    | FinishAction
)


@dataclass(frozen=True)
class EvidenceQualification:
    """Agent 对一条已登记 Evidence 的受控标注。"""

    evidence_id: str
    relation: EvidenceRelation
    ownership: EvidenceOwnership
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "relation", EvidenceRelation(self.relation))
        object.__setattr__(self, "ownership", EvidenceOwnership(self.ownership))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))


@dataclass(frozen=True)
class EvidenceClaim:
    """从一条 Evidence 中提取的原子事实及其关系、归属与原文锚点。"""

    claim_id: str
    statement: str
    relation: EvidenceRelation
    ownership: EvidenceOwnership
    rationale: str
    source_excerpt: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(self, "relation", EvidenceRelation(self.relation))
        object.__setattr__(self, "ownership", EvidenceOwnership(self.ownership))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))
        object.__setattr__(
            self, "source_excerpt", _text(self.source_excerpt, "source_excerpt")
        )


@dataclass(frozen=True)
class EvidenceAnalysis:
    """Evidence Analysis 节点对一条已登记 Evidence 的完整处理结果。"""

    evidence_id: str
    status: EvidenceAnalysisStatus
    digest: str | None = None
    claims: tuple[EvidenceClaim, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        status = EvidenceAnalysisStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "claims", tuple(self.claims))
        if self.digest is not None:
            object.__setattr__(self, "digest", _text(self.digest, "digest"))
        if self.error_code is not None:
            object.__setattr__(
                self, "error_code", _text(self.error_code, "error_code")
            )
        if status is EvidenceAnalysisStatus.QUALIFIED:
            if self.digest is None or not self.claims:
                raise ValueError("qualified evidence requires digest and claims.")
            if self.error_code is not None:
                raise ValueError("qualified evidence cannot contain error_code.")
        elif status is EvidenceAnalysisStatus.IRRELEVANT:
            if self.digest is None or self.claims:
                raise ValueError("irrelevant evidence requires digest and no claims.")
            if self.error_code is not None:
                raise ValueError("irrelevant evidence cannot contain error_code.")
        elif status is EvidenceAnalysisStatus.FAILED:
            if self.error_code is None or self.claims:
                raise ValueError("failed evidence requires error_code and no claims.")
        elif self.digest is not None or self.claims or self.error_code is not None:
            raise ValueError("pending evidence cannot contain analysis output.")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    evidence: RetrievedEvidence
    supporting_queries: tuple[str, ...]
    first_seen_iteration: int
    last_seen_iteration: int
    originating_tools: tuple[str, ...] = ()
    target_subtask_ids: tuple[str, ...] = ()
    analysis_status: EvidenceAnalysisStatus = EvidenceAnalysisStatus.PENDING
    digest: str | None = None
    claims: tuple[EvidenceClaim, ...] = ()
    analysis_error: str | None = None
    # Phase 3 旧 checkpoint 的兼容字段；新链路只写入 Claim 级分析结果。
    qualification: EvidenceQualification | None = None

    def __post_init__(self) -> None:
        status = EvidenceAnalysisStatus(self.analysis_status)
        claims = tuple(self.claims)
        digest = self.digest
        error = self.analysis_error
        # 旧测试与 checkpoint 可能只有 Evidence 级 Qualification。加载时把它提升为
        # 单 Claim，避免历史数据在新 fail-closed Selector 中被静默丢弃。
        if self.qualification is not None and status is EvidenceAnalysisStatus.PENDING:
            legacy = self.qualification
            status = (
                EvidenceAnalysisStatus.IRRELEVANT
                if legacy.relation is EvidenceRelation.IRRELEVANT
                else EvidenceAnalysisStatus.QUALIFIED
            )
            digest = self.evidence.content[:500]
            if status is EvidenceAnalysisStatus.QUALIFIED:
                claims = (
                    EvidenceClaim(
                        claim_id=f"{self.evidence_id}:C1",
                        statement=self.evidence.content[:500],
                        relation=legacy.relation,
                        ownership=legacy.ownership,
                        rationale=legacy.rationale,
                        source_excerpt=self.evidence.content[:300],
                    ),
                )
        object.__setattr__(self, "analysis_status", status)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "analysis_error", error)
        EvidenceAnalysis(
            evidence_id=self.evidence_id,
            status=status,
            digest=digest,
            claims=claims,
            error_code=error,
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
class ResearchObservation:
    iteration: int
    action_name: str
    tool_status: ToolStatus
    added_evidence_ids: tuple[str, ...]
    reused_evidence_ids: tuple[str, ...]
    paper_statuses: tuple[tuple[str, str], ...] = ()
    queries_by_paper: tuple[tuple[str, str], ...] = ()
    error_codes: tuple[str, ...] = ()
    no_new_evidence: bool = False
    remaining_budget: dict[str, int] = field(default_factory=dict)


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
    suppressed_paper_ids: tuple[str, ...] = ()
    executed_action_fingerprints: tuple[str, ...] = ()


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _text(self.status, "status"))
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string.")
        # 回答正文可能包含 Markdown 标题和段落，不能像标识符一样折叠全部空白。
        object.__setattr__(self, "answer", self.answer.strip())
        object.__setattr__(self, "citations", _unique_texts(self.citations))


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
    screened_paper_ids: tuple[str, ...] = ()
    uncertain_paper_ids: tuple[str, ...] = ()
    excluded_paper_ids: tuple[str, ...] = ()
    discovery_completed: bool = False

    def __post_init__(self) -> None:
        scope_ids = _unique_texts(self.scope_paper_ids)
        candidate_ids = _unique_texts(self.candidate_paper_ids)
        screened_ids = _unique_texts(self.screened_paper_ids)
        uncertain_ids = _unique_texts(self.uncertain_paper_ids)
        excluded_ids = _unique_texts(self.excluded_paper_ids)
        partitions = (set(candidate_ids), set(uncertain_ids), set(excluded_ids))
        if any(not values.issubset(set(scope_ids)) for values in partitions):
            raise ValueError("Coverage screening partitions must stay inside Scope.")
        if any(partitions[i].intersection(partitions[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Coverage screening partitions must be mutually exclusive.")
        if set(screened_ids) != set().union(*partitions):
            raise ValueError("screened_paper_ids must equal all screening partitions.")
        assessment_ids = tuple(item.paper_id for item in self.assessments)
        if len(set(assessment_ids)) != len(assessment_ids):
            raise ValueError("Coverage assessments must not contain duplicate paper IDs.")
        if not set(assessment_ids).issubset(set(scope_ids)):
            raise ValueError("Coverage assessments must stay inside scope_paper_ids.")
        object.__setattr__(self, "scope_paper_ids", scope_ids)
        object.__setattr__(self, "candidate_paper_ids", candidate_ids)
        object.__setattr__(self, "screened_paper_ids", screened_ids)
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
