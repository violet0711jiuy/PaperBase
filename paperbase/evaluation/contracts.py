"""Typed deterministic evaluation contracts for the Grounded Answer architecture."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol

EVALUATION_SCHEMA_VERSION = "agentic-grounded-answer-eval-v5"


def json_safe(value):
    if is_dataclass(value):
        return {f.name: json_safe(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum): return value.value
    if isinstance(value, Path): return str(value)
    if isinstance(value, Mapping): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        values = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [json_safe(v) for v in values]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError(f"Unsupported evaluation value: {type(value).__name__}")


@dataclass(frozen=True)
class DatasetIssue:
    case_id: str | None
    code: str
    detail: str


@dataclass(frozen=True)
class GoldenValidation:
    case_count: int
    issues: tuple[DatasetIssue, ...]
    actual_fields: tuple[str, ...]
    database_checked: bool
    @property
    def valid(self): return not self.issues


@dataclass(frozen=True)
class EvidenceTrace:
    evidence_id: str
    paper_id: str | None
    source_chunk_ids: tuple[str, ...] | None
    seed_chunk_ids: tuple[str, ...] = ()
    presented_to_answerer: bool | None = None
    cited_requirement_ids: tuple[str, ...] | None = None
    selected: bool | None = None
    cited: bool | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    paper_id: str | None
    source_chunk_ids: tuple[str, ...]
    seed_chunk_ids: tuple[str, ...] = ()
    retrieval_rank: int | None = None


@dataclass(frozen=True)
class ResearchStep:
    iteration: int
    tool: str
    executed: bool | None
    retrieval_executed: bool | None
    executed_queries: int | None
    candidates: tuple[RetrievalCandidate, ...] | None
    new_evidence_count: int | None


@dataclass(frozen=True)
class RequirementTrace:
    requirement_id: str
    description: str
    required: bool
    status: str | None
    answer_present: bool | None = None
    citation_evidence_ids: tuple[str, ...] | None = None
    remaining_gap: str | None = None


@dataclass(frozen=True)
class AnswerTrace:
    has_answer: bool | None
    insufficient_evidence: bool | None
    partial_answer: bool | None
    citations: tuple[str, ...] | None
    status: str | None


@dataclass(frozen=True)
class CaseTrace:
    run_status: str | None
    stop_reason: str | None
    evidence: tuple[EvidenceTrace, ...] | None
    steps: tuple[ResearchStep, ...] | None
    requirements: tuple[RequirementTrace, ...] | None
    selected_evidence_ids: tuple[str, ...] | None
    selected_source_chunk_ids: tuple[str, ...] | None
    answer: AnswerTrace | None
    counters: tuple[tuple[str, int], ...] = ()
    guard_rejection_codes: tuple[str, ...] | None = None
    validation_error_count: int | None = None
    llm_nodes: tuple[tuple[str, bool], ...] | None = None
    legacy_ranked_chunk_ids: tuple[str, ...] | None = None
    planning_mode: str | None = None
    setup_failure_stage: str | None = None
    observability_gaps: tuple[str, ...] = ()
    decision_calls: int | None = None
    steps_complete: bool = True


@dataclass(frozen=True)
class RetrievalEvaluation:
    hit_at_5: float | None
    recall_at_5: float | None
    initial_search_hit: bool | None
    initial_search_gold_recall: float | None
    cumulative_gold_retrieval_recall: float | None
    queries_until_first_gold: int | None
    retrieval_queries: int | None
    initial_search_evidence_path_coverage: float | None
    cumulative_evidence_path_coverage: float | None


@dataclass(frozen=True)
class EvidenceFunnelEvaluation:
    gold_chunk_ids: tuple[str, ...]
    gold_retrieved_chunk_ids: tuple[str, ...] | None
    gold_registered_chunk_ids: tuple[str, ...] | None
    gold_answer_input_chunk_ids: tuple[str, ...] | None
    gold_cited_chunk_ids: tuple[str, ...] | None
    observed_gold_retrieved_chunk_ids: tuple[str, ...]
    path_covered_gold_retrieved_chunk_ids: tuple[str, ...] | None
    path_covered_gold_registered_chunk_ids: tuple[str, ...] | None
    path_covered_gold_answer_input_chunk_ids: tuple[str, ...] | None
    path_covered_gold_cited_chunk_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class RequirementEvaluation:
    requirement_id: str
    description: str
    status: str | None
    required: bool
    answer_present: bool | None
    citation_evidence_ids: tuple[str, ...] | None
    remaining_gap: str | None


@dataclass(frozen=True)
class ResearchEvaluation:
    planned_requirement_count: int | None
    required_requirement_count: int | None
    answered_requirement_count: int | None
    open_gap_requirement_count: int | None
    requirement_answer_rate: float | None
    requirements: tuple[RequirementEvaluation, ...]


@dataclass(frozen=True)
class AnswerEvaluation:
    has_answer: bool | None
    insufficient_evidence: bool | None
    partial_answer: bool | None
    citation_count: int | None
    citation_validity: float | None
    invalid_citation_ids: tuple[str, ...] | None
    answer_input_citation_coverage: float | None
    gold_citation_coverage: float | None
    gold_citation_path_coverage: float | None
    correct_abstention: bool | None
    unsafe_non_abstention: bool | None
    false_abstention: bool | None


@dataclass(frozen=True)
class EfficiencyEvaluation:
    tool_calls: int | None
    retrieval_queries: int | None
    decision_calls: int | None
    guard_rejections: int | None
    duplicate_rejections: int | None
    validation_errors: int | None
    iterations: int | None
    llm_calls: int | None
    executed_actions: int | None
    zero_new_evidence_actions: int | None
    actions_until_first_new_evidence: int | None
    consecutive_no_progress_peak: int | None
    decision_llm_calls: int | None


@dataclass(frozen=True)
class AgenticCaseEvaluation:
    case_id: str
    primary_type: str
    answerable: bool
    run_status: str | None
    stop_reason: str | None
    retrieval: RetrievalEvaluation
    research: ResearchEvaluation
    evidence_funnel: EvidenceFunnelEvaluation
    answer: AnswerEvaluation
    efficiency: EfficiencyEvaluation
    primary_failure_stage: str | None
    failure_labels: tuple[str, ...]
    observability_gaps: tuple[str, ...]
    dataset_issues: tuple[DatasetIssue, ...] = ()
    evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION
    @property
    def observability_gap(self): return bool(self.observability_gaps)
    def to_dict(self): return {**json_safe(self), "observability_gap": self.observability_gap}


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    evaluation_schema_version: str
    dataset_path: str
    dataset_sha256: str
    case_count: int
    timestamp: str
    mode: str
    model_provider_config: dict[str, object]
    agent_config: dict[str, object]
    production_semantic_authority_version: str | None
    code_snapshot_identifier: str | None
    source_semantic_authority_version: str | None = None
    source_artifact_hashes: tuple[tuple[str, str], ...] = ()
    observability_gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticJudgeResult:
    correctness: float | None = None
    completeness: float | None = None
    faithfulness: float | None = None
    reference_answer_consistency: float | None = None
    abstention_appropriateness: float | None = None


class SemanticAnswerJudge(Protocol):
    """Optional E2 extension point. The deterministic E1 evaluator never invokes it."""
    def judge(self, case, answer, evidence) -> SemanticJudgeResult: ...
