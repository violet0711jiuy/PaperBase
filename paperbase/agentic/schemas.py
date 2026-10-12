"""Strict untrusted-input schemas for the Agentic research loop."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from .contracts import CandidateDisposition, EvidenceTarget, Intent, ResearchGoal


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QueryUnderstandingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    resolved_query: NonEmptyText
    intent: Intent
    retrieval_goal: ResearchGoal
    evidence_targets: list[EvidenceTarget] = Field(min_length=1, max_length=8)


class ResearchRequirementDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    description: NonEmptyText = Field(max_length=300)
    required: bool
    evidence_targets: list[EvidenceTarget] = Field(min_length=1, max_length=8)


class ResearchPlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    goal: NonEmptyText
    requirements: list[ResearchRequirementDraft] = Field(min_length=1, max_length=8)


class CandidateAssessmentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paper_id: NonEmptyText
    disposition: CandidateDisposition
    reason: NonEmptyText = Field(max_length=300)


class CandidateSelectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    assessments: list[CandidateAssessmentDraft] = Field(min_length=1)


class SearchSpecDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paper_id: NonEmptyText
    semantic_query: NonEmptyText
    lexical_terms: list[NonEmptyText] = Field(min_length=1, max_length=5)


class SearchPapersDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["search_papers"]
    requirement_id: NonEmptyText
    searches: list[SearchSpecDraft] = Field(min_length=1, max_length=12)


class SearchKnowledgeBaseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["search_knowledge_base"]
    requirement_id: NonEmptyText
    semantic_query: NonEmptyText
    lexical_terms: list[NonEmptyText] = Field(min_length=1, max_length=5)


class SearchReferencesDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["search_references"]
    requirement_id: NonEmptyText
    paper_id: NonEmptyText
    semantic_query: NonEmptyText
    lexical_terms: list[NonEmptyText] = Field(min_length=1, max_length=5)


class GetSectionContextDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["get_section_context"]
    requirement_id: NonEmptyText
    paper_id: NonEmptyText
    section_id: NonEmptyText


class GetPaperOverviewDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["get_paper_overview"]
    requirement_id: NonEmptyText
    paper_id: NonEmptyText


class TryAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["try_answer"]
    reason: NonEmptyText


class StopResearchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["stop_research"]
    reason: NonEmptyText


ActionDraft: TypeAlias = Annotated[
    SearchPapersDraft
    | SearchKnowledgeBaseDraft
    | SearchReferencesDraft
    | GetSectionContextDraft
    | GetPaperOverviewDraft
    | TryAnswerDraft
    | StopResearchDraft,
    Field(discriminator="action"),
]


class AgentDecisionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: ActionDraft


AGENT_DECISION_ADAPTER = TypeAdapter(AgentDecisionDraft)


class GroundedCitationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: NonEmptyText
    source_span_ids: list[NonEmptyText] = Field(min_length=1, max_length=16)


class GroundedRequirementAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requirement_id: NonEmptyText
    answer: NonEmptyText | None
    citations: list[GroundedCitationDraft] = Field(max_length=8)
    remaining_gap: NonEmptyText | None


class GroundedAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answers: list[GroundedRequirementAnswerDraft] = Field(min_length=1, max_length=8)
