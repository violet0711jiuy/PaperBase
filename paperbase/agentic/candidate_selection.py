"""冻结 Scope 内基于元数据的 Candidate Selection。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCompletion,
    complete_json_with_diagnostics,
)
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
    LLMCallRecord,
)
from paperbase.llm.diagnostics import failure_record, repair_hint, success_record, request_is_nonretryable
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
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()


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
        batch_size: int = 12,
    ) -> None:
        if max_attempts < 1 or batch_size < 1:
            raise ValueError("Candidate Selection limits must be positive.")
        self._client = client
        self._metadata_source = metadata_source
        self._max_attempts = max_attempts
        self._batch_size = batch_size

    def select(
        self, *, question: str, snapshot: CorpusScopeSnapshot
    ) -> CandidateSelectionResult:
        metadata = self._metadata_source.get_many(snapshot.paper_ids)
        if not metadata:
            return CandidateSelectionResult(CandidateSelection((), ()), 0)
        assessments: list[CandidateAssessment] = []
        llm_calls = 0
        used_fallback = False
        diagnostics: list[LLMCallRecord] = []
        for offset in range(0, len(metadata), self._batch_size):
            batch = metadata[offset : offset + self._batch_size]
            expected_ids = tuple(item.paper_id for item in batch)
            errors: list[str] = []
            accepted: CandidateSelection | None = None
            compact_context = False
            rejected_request = False
            for attempt in range(1, self._max_attempts + 1):
                llm_calls += 1
                completion: LLMCompletion | None = None
                try:
                    completion = complete_json_with_diagnostics(
                        self._client,
                        system_prompt=CANDIDATE_SELECTION_SYSTEM_PROMPT,
                        user_prompt=build_candidate_selection_user_prompt(
                            {
                                "question": question,
                                "scope_snapshot_id": snapshot.snapshot_id,
                                "papers": [
                                    _metadata_payload(item, compact=compact_context)
                                    for item in batch
                                ],
                                "previous_validation_errors": errors[-1:],
                            }
                        ),
                        json_schema=CandidateSelectionDraft.model_json_schema(),
                        schema_name="paperbase_candidate_selection",
                    )
                    draft = CandidateSelectionDraft.model_validate_json(
                        completion.content
                    )
                    accepted = _validate_and_build(expected_ids, draft)
                    diagnostics.append(
                        success_record(
                            node="select_candidates",
                            attempt=attempt,
                            completion=completion,
                        )
                    )
                    break
                except Exception as error:
                    record = failure_record(
                        node="select_candidates",
                        attempt=attempt,
                        error=error,
                        completion=completion,
                        can_retry=attempt < self._max_attempts,
                    )
                    diagnostics.append(record)
                    if request_is_nonretryable(error):
                        rejected_request = True
                        break
                    errors.append(repair_hint(error))
                    compact_context = record.recovery_action == "compact_context"
            if accepted is not None:
                assessments.extend(accepted.assessments)
                continue
            used_fallback = True
            assessments.extend(
                CandidateAssessment(
                    paper_id,
                    CandidateDisposition.UNCERTAIN,
                    "元数据筛选失败，保守进入一次浅层正文验证。",
                )
                for paper_id in expected_ids
            )
            if rejected_request:
                # An invalid provider request is not repaired by another batch.
                # Preserve the scope partition conservatively without more API calls.
                assessments.extend(CandidateAssessment(
                    item.paper_id, CandidateDisposition.UNCERTAIN,
                    "供应商拒绝筛选请求，尚未完成元数据评估。",
                ) for item in metadata[offset + len(batch):])
                break
        selection = CandidateSelection(
            scope_paper_ids=snapshot.paper_ids,
            assessments=tuple(assessments),
        )
        return CandidateSelectionResult(
            selection, llm_calls, used_fallback, tuple(diagnostics)
        )


def _metadata_payload(item: PaperMetadata, *, compact: bool) -> dict[str, object]:
    """length 截断后缩短可选字段，但始终保留 paper_id 与完整标题。"""
    if not compact:
        return item.__dict__
    return {
        "paper_id": item.paper_id,
        "title": item.title,
        "abstract": item.abstract[:240] if item.abstract else None,
        "keywords": item.keywords[:160] if item.keywords else None,
        "overview": item.overview[:240] if item.overview else None,
    }


def _validate_and_build(
    expected_ids: tuple[str, ...], draft: CandidateSelectionDraft
) -> CandidateSelection:
    draft_ids = [item.paper_id for item in draft.assessments]
    if len(draft_ids) != len(set(draft_ids)) or set(draft_ids) != set(expected_ids):
        raise ValueError("Candidate Selection 必须完整且不重复地返回输入 paper_id。")
    by_id = {item.paper_id: item for item in draft.assessments}
    return CandidateSelection(
        scope_paper_ids=expected_ids,
        assessments=tuple(
            CandidateAssessment(
                paper_id,
                by_id[paper_id].disposition,
                by_id[paper_id].reason,
            )
            for paper_id in expected_ids
        ),
    )


def _bounded_join(values: list[str], limit: int) -> str | None:
    joined = "\n".join(value.strip() for value in values if value.strip())
    return joined[:limit] or None
