"""冻结 Scope 内基于元数据的 Candidate Selection。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from paperbase.llm.client import ChatCompletionClient
from paperbase.overview.repository import FormalOverviewRepository
from paperbase.prompts.candidate_selection import (
    CANDIDATE_SELECTION_SYSTEM_PROMPT,
    build_candidate_selection_user_prompt,
)

from .contracts import (
    CandidateAssessment,
    CandidateDisposition,
    CandidateSelection,
    CorpusScopeSnapshot,
)
from .schemas import CandidateSelectionDraft


@dataclass(frozen=True)
class PaperMetadata:
    paper_id: str
    title: str
    abstract: str | None = None
    keywords: str | None = None
    overview: str | None = None


@dataclass(frozen=True)
class CandidateSelectionResult:
    selection: CandidateSelection
    llm_calls: int
    used_fallback: bool = False


class PaperMetadataSource(Protocol):
    def get_many(self, paper_ids: tuple[str, ...]) -> tuple[PaperMetadata, ...]: ...


class CandidateSelector(Protocol):
    def select(
        self, *, question: str, snapshot: CorpusScopeSnapshot
    ) -> CandidateSelectionResult: ...


class FormalPaperMetadataRepository:
    """优先组合标题、摘要、关键词和已持久化 Overview，不触发任何生成。"""

    def __init__(
        self,
        *,
        database: object,
        overview_repository: FormalOverviewRepository | None = None,
        max_field_chars: int = 1_200,
    ) -> None:
        self._database = database
        self._overview_repository = overview_repository
        self._max_field_chars = max_field_chars

    def get_many(self, paper_ids: tuple[str, ...]) -> tuple[PaperMetadata, ...]:
        return tuple(self._get(paper_id) for paper_id in paper_ids)

    def _get(self, paper_id: str) -> PaperMetadata:
        document = self._database.get_document(paper_id)
        if document is None:
            raise ValueError(f"Candidate Selection 找不到冻结论文：{paper_id}")
        front_matter = {"abstract": [], "keywords": []}
        list_front_matter = getattr(self._database, "list_front_matter_chunks", None)
        if callable(list_front_matter):
            for row in list_front_matter(paper_id):
                kind = str(row["front_matter_type"] or "").casefold()
                if kind in front_matter:
                    front_matter[kind].append(str(row["raw_text"]))
        overview_text = None
        if self._overview_repository is not None:
            artifact = self._overview_repository.get(paper_id)
            if artifact is not None:
                overview_text = "\n".join(
                    (
                        artifact.overview.research_problem.content,
                        artifact.overview.main_method.content,
                        artifact.overview.contributions.content,
                    )
                )
        return PaperMetadata(
            paper_id=paper_id,
            title=str(document["paper_title"] or paper_id)[: self._max_field_chars],
            abstract=_bounded_join(front_matter["abstract"], self._max_field_chars),
            keywords=_bounded_join(front_matter["keywords"], self._max_field_chars),
            overview=(overview_text[: self._max_field_chars] if overview_text else None),
        )


class LLMCandidateSelector:
    """严格校验完整分区；失败时保守回退为 uncertain，避免标题假阴性。"""

    def __init__(
        self,
        client: ChatCompletionClient,
        metadata_source: PaperMetadataSource,
        *,
        max_attempts: int = 2,
    ) -> None:
        self._client = client
        self._metadata_source = metadata_source
        self._max_attempts = max_attempts

    def select(
        self, *, question: str, snapshot: CorpusScopeSnapshot
    ) -> CandidateSelectionResult:
        metadata = self._metadata_source.get_many(snapshot.paper_ids)
        errors: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._client.complete_json(
                    system_prompt=CANDIDATE_SELECTION_SYSTEM_PROMPT,
                    user_prompt=build_candidate_selection_user_prompt(
                        {
                            "question": question,
                            "scope_snapshot_id": snapshot.snapshot_id,
                            "papers": [item.__dict__ for item in metadata],
                            "previous_validation_errors": errors[-1:],
                        }
                    ),
                    json_schema=CandidateSelectionDraft.model_json_schema(),
                    schema_name="paperbase_candidate_selection",
                )
                draft = CandidateSelectionDraft.model_validate_json(raw)
                selection = _validate_and_build(snapshot, draft)
                return CandidateSelectionResult(selection, attempt)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        fallback = CandidateSelection(
            scope_paper_ids=snapshot.paper_ids,
            assessments=tuple(
                CandidateAssessment(
                    paper_id,
                    CandidateDisposition.UNCERTAIN,
                    "元数据筛选失败，保守进入一次浅层正文验证。",
                )
                for paper_id in snapshot.paper_ids
            ),
        )
        return CandidateSelectionResult(fallback, self._max_attempts, True)


def _validate_and_build(
    snapshot: CorpusScopeSnapshot, draft: CandidateSelectionDraft
) -> CandidateSelection:
    return CandidateSelection(
        scope_paper_ids=snapshot.paper_ids,
        assessments=tuple(
            CandidateAssessment(item.paper_id, item.disposition, item.reason)
            for item in draft.assessments
        ),
    )


def _bounded_join(values: list[str], limit: int) -> str | None:
    joined = "\n".join(value.strip() for value in values if value.strip())
    return joined[:limit] or None
