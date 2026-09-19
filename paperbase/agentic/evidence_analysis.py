"""独立的 Claim 级 Evidence Analysis，不再把证据审核塞进 Decision。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCompletion,
    complete_json_with_diagnostics,
)
from paperbase.prompts.evidence_analysis import (
    EVIDENCE_ANALYSIS_SYSTEM_PROMPT,
    build_evidence_analysis_user_prompt,
)

from .contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceRecord,
    LLMCallRecord,
)
from paperbase.llm.diagnostics import failure_record, repair_hint, success_record
from .schemas import EvidenceAnalysisBatchDraft


@dataclass(frozen=True)
class EvidenceAnalysisResult:
    analyses: tuple[EvidenceAnalysis, ...]
    llm_calls: int
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()


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
        diagnostics: list[LLMCallRecord] = []
        for offset in range(0, len(pending), self._batch_size):
            batch = pending[offset : offset + self._batch_size]
            errors: list[str] = []
            accepted: tuple[EvidenceAnalysis, ...] | None = None
            content_limit = self._max_content_chars
            for attempt in range(1, self._max_attempts + 1):
                llm_calls += 1
                completion: LLMCompletion | None = None
                try:
                    completion = complete_json_with_diagnostics(
                        self._client,
                        system_prompt=EVIDENCE_ANALYSIS_SYSTEM_PROMPT,
                        user_prompt=build_evidence_analysis_user_prompt(
                            {
                                "question": question,
                                "evidence": [
                                    _payload(item, content_limit) for item in batch
                                ],
                                "previous_validation_errors": errors[-1:],
                            }
                        ),
                        json_schema=EvidenceAnalysisBatchDraft.model_json_schema(),
                        schema_name="paperbase_evidence_analysis",
                    )
                    draft = EvidenceAnalysisBatchDraft.model_validate_json(
                        completion.content
                    )
                    accepted = _convert_and_validate(batch, draft)
                    diagnostics.append(
                        success_record(
                            node="analyze_evidence",
                            attempt=attempt,
                            completion=completion,
                        )
                    )
                    break
                except Exception as error:
                    record = failure_record(
                        node="analyze_evidence",
                        attempt=attempt,
                        error=error,
                        completion=completion,
                        can_retry=attempt < self._max_attempts,
                    )
                    diagnostics.append(record)
                    errors.append(repair_hint(error))
                    if record.recovery_action == "compact_context":
                        content_limit = max(100, content_limit // 2)
            if accepted is not None:
                analyses.extend(accepted)
            else:
                error_code = (
                    diagnostics[-1].error_stage if diagnostics else None
                ) or "EvidenceAnalysisFailed"
                analyses.extend(
                    EvidenceAnalysis(
                        evidence_id=item.evidence_id,
                        status=EvidenceAnalysisStatus.FAILED,
                        error_code=error_code,
                    )
                    for item in batch
                )
        return EvidenceAnalysisResult(
            tuple(analyses), llm_calls, tuple(diagnostics)
        )


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
