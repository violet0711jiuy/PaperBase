"""独立的 Claim 级 Evidence Analysis，不再把证据审核塞进 Decision。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from paperbase.llm.client import ChatCompletionClient
from paperbase.prompts.evidence_analysis import (
    EVIDENCE_ANALYSIS_SYSTEM_PROMPT,
    build_evidence_analysis_user_prompt,
)

from .contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceRecord,
)
from .schemas import EvidenceAnalysisBatchDraft


@dataclass(frozen=True)
class EvidenceAnalysisResult:
    analyses: tuple[EvidenceAnalysis, ...]
    llm_calls: int


class EvidenceAnalyzer(Protocol):
    def analyze(
        self, *, question: str, evidence: tuple[EvidenceRecord, ...]
    ) -> EvidenceAnalysisResult: ...


class LLMEvidenceAnalyzer:
    """按有界 Batch 分析新 Evidence；失败项保持 fail-closed。"""

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        batch_size: int = 4,
        max_content_chars: int = 1_200,
        max_attempts: int = 2,
    ) -> None:
        if batch_size < 1 or max_content_chars < 100 or max_attempts < 1:
            raise ValueError("Evidence Analysis limits must be positive.")
        self._client = client
        self._batch_size = batch_size
        self._max_content_chars = max_content_chars
        self._max_attempts = max_attempts

    def analyze(
        self, *, question: str, evidence: tuple[EvidenceRecord, ...]
    ) -> EvidenceAnalysisResult:
        pending = tuple(
            item
            for item in evidence
            if item.analysis_status is EvidenceAnalysisStatus.PENDING
        )
        analyses: list[EvidenceAnalysis] = []
        llm_calls = 0
        for offset in range(0, len(pending), self._batch_size):
            batch = pending[offset : offset + self._batch_size]
            errors: list[str] = []
            accepted: tuple[EvidenceAnalysis, ...] | None = None
            for _attempt in range(self._max_attempts):
                llm_calls += 1
                try:
                    raw = self._client.complete_json(
                        system_prompt=EVIDENCE_ANALYSIS_SYSTEM_PROMPT,
                        user_prompt=build_evidence_analysis_user_prompt(
                            {
                                "question": question,
                                "evidence": [_payload(item, self._max_content_chars) for item in batch],
                                "previous_validation_errors": errors[-1:],
                            }
                        ),
                        json_schema=EvidenceAnalysisBatchDraft.model_json_schema(),
                        schema_name="paperbase_evidence_analysis",
                    )
                    draft = EvidenceAnalysisBatchDraft.model_validate_json(raw)
                    accepted = _convert_and_validate(batch, draft)
                    break
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
            if accepted is not None:
                analyses.extend(accepted)
            else:
                error_code = errors[-1][:300] if errors else "EvidenceAnalysisFailed"
                analyses.extend(
                    EvidenceAnalysis(
                        evidence_id=item.evidence_id,
                        status=EvidenceAnalysisStatus.FAILED,
                        error_code=error_code,
                    )
                    for item in batch
                )
        return EvidenceAnalysisResult(tuple(analyses), llm_calls)


def _payload(record: EvidenceRecord, max_content_chars: int) -> dict[str, object]:
    source = record.evidence
    return {
        "evidence_id": record.evidence_id,
        "paper_id": source.paper_id,
        "paper_title": source.paper_title,
        "section": source.section,
        "pages": [source.page_start, source.page_end],
        "content": source.content[:max_content_chars],
    }


def _convert_and_validate(
    records: tuple[EvidenceRecord, ...], draft: EvidenceAnalysisBatchDraft
) -> tuple[EvidenceAnalysis, ...]:
    expected = [item.evidence_id for item in records]
    actual = [item.evidence_id for item in draft.analyses]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Evidence Analysis 必须完整且不重复地返回输入 evidence_id。")
    by_id = {item.evidence_id: item for item in records}
    converted: list[EvidenceAnalysis] = []
    for item in draft.analyses:
        record = by_id[item.evidence_id]
        if item.status == "irrelevant":
            if item.claims:
                raise ValueError("irrelevant Evidence 的 claims 必须为空。")
            converted.append(
                EvidenceAnalysis(
                    evidence_id=item.evidence_id,
                    status=EvidenceAnalysisStatus.IRRELEVANT,
                    digest=item.digest,
                )
            )
            continue
        if not item.claims:
            raise ValueError("qualified Evidence 至少需要一条 Claim。")
        claims: list[EvidenceClaim] = []
        for index, claim in enumerate(item.claims, start=1):
            if _fold(claim.source_excerpt) not in _fold(record.evidence.content):
                raise ValueError(
                    f"{item.evidence_id} 的 source_excerpt 不是原文连续片段。"
                )
            claims.append(
                EvidenceClaim(
                    claim_id=f"{item.evidence_id}:C{index}",
                    statement=claim.statement,
                    relation=claim.relation,
                    ownership=claim.ownership,
                    rationale=claim.rationale,
                    source_excerpt=claim.source_excerpt,
                )
            )
        converted.append(
            EvidenceAnalysis(
                evidence_id=item.evidence_id,
                status=EvidenceAnalysisStatus.QUALIFIED,
                digest=item.digest,
                claims=tuple(claims),
            )
        )
    return tuple(converted)


def _fold(value: str) -> str:
    return " ".join(value.split()).casefold()
