"""Agentic RAG 面向外部 LLM 的严格 Schema。\n\n这些 DTO 是不可信输入边界：JSON Schema 先发送给模型供应商，原始响应还必须在本地\n再次校验，之后才能转换成领域契约。\n"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

from .contracts import (
    CandidateDisposition,
    EvidenceOwnership,
    EvidenceRelation,
    EvidenceTarget,
    Intent,
    ResearchGoal,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QueryUnderstandingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    resolved_query: NonEmptyText
    # 这是“必填但可为 null”的字段：严格结构化输出要求所有属性进入 required，
    # null 仍然表示“没有可用的语义改写”。
    semantic_query_en: NonEmptyText | None
    lexical_keywords_en: list[NonEmptyText] = Field(max_length=8)
    intent: Intent
    retrieval_goal: ResearchGoal
    evidence_targets: list[EvidenceTarget] = Field(min_length=1, max_length=8)


class ResearchSubtaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: NonEmptyText
    target_paper_ids: list[NonEmptyText] = Field(max_length=32)
    evidence_targets: list[EvidenceTarget] = Field(min_length=1, max_length=8)
    success_criteria: NonEmptyText


class ResearchPlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: NonEmptyText
    subtasks: list[ResearchSubtaskDraft] = Field(min_length=1, max_length=4)


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
    query: NonEmptyText
    lexical_terms: list[NonEmptyText] = Field(max_length=8)
    evidence_targets: list[EvidenceTarget] = Field(max_length=8)
    missing_information: list[NonEmptyText] = Field(max_length=8)


class SearchPapersDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search_papers"]
    searches: list[SearchSpecDraft] = Field(min_length=1, max_length=12)
    target_subtask_id: NonEmptyText | None


class SearchKnowledgeBaseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search_knowledge_base"]
    query: NonEmptyText
    lexical_terms: list[NonEmptyText] = Field(max_length=8)
    target_subtask_id: NonEmptyText | None


class SearchReferencesDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search_references"]
    paper_id: NonEmptyText
    query: NonEmptyText
    lexical_terms: list[NonEmptyText] = Field(max_length=8)
    target_subtask_id: NonEmptyText | None


class GetSectionContextDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["get_section_context"]
    paper_id: NonEmptyText
    section_id: NonEmptyText
    target_subtask_id: NonEmptyText | None


class GetPaperOverviewDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["get_paper_overview"]
    paper_id: NonEmptyText
    target_subtask_id: NonEmptyText | None


class FinishDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["finish"]
    reason: NonEmptyText


ActionDraft: TypeAlias = Annotated[
    SearchPapersDraft
    | SearchKnowledgeBaseDraft
    | SearchReferencesDraft
    | GetSectionContextDraft
    | GetPaperOverviewDraft
    | FinishDraft,
    Field(discriminator="action"),
]


class EvidenceQualificationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: NonEmptyText
    relation: EvidenceRelation
    ownership: EvidenceOwnership
    rationale: NonEmptyText


class AgentDecisionDraft(BaseModel):
    """严格结构化输出接口要求的根对象。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    decision: ActionDraft


AGENT_DECISION_ADAPTER = TypeAdapter(AgentDecisionDraft)


class EvidenceClaimDraft(BaseModel):
    """模型提取的 Claim；claim_id 由程序按 Evidence ID 稳定分配。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    statement: NonEmptyText = Field(max_length=500)
    relation: EvidenceRelation
    ownership: EvidenceOwnership
    rationale: NonEmptyText = Field(max_length=300)
    source_excerpt: NonEmptyText = Field(max_length=300)


class EvidenceAnalysisItemDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_id: NonEmptyText
    status: Literal["qualified", "irrelevant"]
    digest: NonEmptyText = Field(max_length=500)
    claims: list[EvidenceClaimDraft] = Field(max_length=6)
    covered_targets: list[EvidenceTarget] = Field(max_length=8)
    addressed_information: list[NonEmptyText] = Field(max_length=12)


class EvidenceAnalysisBatchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analyses: list[EvidenceAnalysisItemDraft] = Field(min_length=1, max_length=4)
