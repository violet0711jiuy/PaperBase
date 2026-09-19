"""Agentic Research 专用的自动 Evidence 准入与审计策略。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    KnowledgeBaseSearchPayload,
    PaperOverviewPayload,
    PaperSearchOutcome,
    PaperSearchStatus,
    ReferenceSearchPayload,
    RetrievedEvidence,
    SearchPapersPayload,
    SectionContextPayload,
    ToolResult,
    ToolStatus,
)
from paperbase.retrieval.evidence_fingerprint import build_evidence_fingerprint

from .contracts import (
    EvidenceFilterDecision,
    EvidenceFilterReport,
    EvidenceTarget,
)


@dataclass(frozen=True)
class EvidenceAdmissionResult:
    tool_result: ToolResult
    report: EvidenceFilterReport


class EvidenceAdmissionFilter:
    """在 Registry 前过滤确定性低质项，并保守保留结果型表格证据。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_rerank_score: float = 0.05,
        source_overlap_threshold: float = 0.8,
    ) -> None:
        if not 0.0 <= min_rerank_score <= 1.0:
            raise ValueError("min_rerank_score 必须位于 0 到 1。")
        if not 0.5 <= source_overlap_threshold <= 1.0:
            raise ValueError("source_overlap_threshold 必须位于 0.5 到 1。")
        self._enabled = enabled
        self._min_rerank_score = min_rerank_score
        self._source_overlap_threshold = source_overlap_threshold

    def apply(self, result: ToolResult) -> EvidenceAdmissionResult:
        if not self._enabled:
            candidates = _all_candidates(result)
            decisions = tuple(
                _decision(item, "accepted", ("filter_disabled",))
                for item in candidates
            )
            return EvidenceAdmissionResult(
                result,
                _report(decisions, len(candidates)),
            )

        decisions: list[EvidenceFilterDecision] = []
        payload = result.payload
        if isinstance(payload, SearchPapersPayload):
            outcomes: list[PaperSearchOutcome] = []
            for outcome in payload.paper_results:
                filtered, group_decisions = self._filter_group(
                    outcome.evidence_candidates,
                    frozenset(EvidenceTarget(value) for value in outcome.evidence_targets),
                )
                decisions.extend(group_decisions)
                status = outcome.status
                if status is PaperSearchStatus.OK and not filtered:
                    status = PaperSearchStatus.EMPTY
                outcomes.append(
                    replace(
                        outcome,
                        status=status,
                        evidence_candidates=filtered,
                    )
                )
            filtered_payload = replace(payload, paper_results=tuple(outcomes))
            filtered_result = replace(
                result,
                payload=filtered_payload,
                status=_aggregate_search_status(tuple(outcomes)),
            )
        else:
            candidates = tuple(getattr(payload, "evidence_candidates", ()))
            filtered, group_decisions = self._filter_group(candidates, frozenset())
            decisions.extend(group_decisions)
            if isinstance(
                payload,
                (
                    KnowledgeBaseSearchPayload,
                    ReferenceSearchPayload,
                    SectionContextPayload,
                    PaperOverviewPayload,
                ),
            ):
                filtered_payload = replace(payload, evidence_candidates=filtered)
                status = result.status
                if status is ToolStatus.OK and not filtered:
                    status = ToolStatus.EMPTY
                filtered_result = replace(result, payload=filtered_payload, status=status)
            else:
                filtered_result = result
        total = len(_all_candidates(result))
        return EvidenceAdmissionResult(
            filtered_result,
            _report(tuple(decisions), total),
        )

    def _filter_group(
        self,
        candidates: tuple[RetrievedEvidence, ...],
        targets: frozenset[EvidenceTarget],
    ) -> tuple[tuple[RetrievedEvidence, ...], tuple[EvidenceFilterDecision, ...]]:
        rejected: dict[int, tuple[str, ...]] = {}
        by_fingerprint: dict[str, RetrievedEvidence] = {}
        for candidate in sorted(candidates, key=_score, reverse=True):
            fingerprint = build_evidence_fingerprint(candidate)
            if fingerprint in by_fingerprint:
                rejected[id(candidate)] = ("duplicate_content",)
                continue
            low_score = (
                candidate.retrieval_metadata.get("reranking_status") == "success"
                and candidate.score is not None
                and candidate.score < self._min_rerank_score
            )
            if low_score:
                rejected[id(candidate)] = ("below_rerank_threshold",)
                continue
            overlapping = next(
                (
                    kept
                    for kept in by_fingerprint.values()
                    if _source_overlap(candidate, kept)
                    >= self._source_overlap_threshold
                ),
                None,
            )
            if overlapping is not None:
                rejected[id(candidate)] = ("source_overlap",)
                continue
            by_fingerprint[fingerprint] = candidate

        # RESULT 检索可能只召回低文本密度的表格行；若全部因低分被拒，保留最强表格项，
        # 让后续 Claim 校验决定是否可用，而不是在准入阶段丢失唯一结果证据。
        rescued: RetrievedEvidence | None = None
        if (
            not by_fingerprint
            and EvidenceTarget.RESULT in targets
            and candidates
        ):
            table_candidates = tuple(item for item in candidates if _table_like(item.content))
            if table_candidates:
                rescued = max(table_candidates, key=_score)
                fingerprint = build_evidence_fingerprint(rescued)
                by_fingerprint[fingerprint] = rescued
                rejected.pop(id(rescued), None)

        accepted_ids = {id(item) for item in by_fingerprint.values()}
        filtered = tuple(item for item in candidates if id(item) in accepted_ids)
        decisions: list[EvidenceFilterDecision] = []
        emitted_fingerprints: set[str] = set()
        for candidate in candidates:
            fingerprint = build_evidence_fingerprint(candidate)
            if id(candidate) in accepted_ids:
                reasons: tuple[str, ...] = ()
                decision = "accepted"
                if candidate is rescued:
                    reasons = ("result_table_rescued",)
                elif _table_like(candidate.content) and EvidenceTarget.RESULT not in targets:
                    decision = "deprioritized"
                    reasons = ("table_like_low_prose",)
                decisions.append(_decision(candidate, decision, reasons))
                emitted_fingerprints.add(fingerprint)
                continue
            reasons = rejected.get(id(candidate))
            if reasons is None and fingerprint in emitted_fingerprints:
                reasons = ("duplicate_content",)
            decisions.append(
                _decision(candidate, "rejected", reasons or ("source_overlap",))
            )
        return filtered, tuple(decisions)


def fail_open_admission(
    result: ToolResult, *, error_code: str
) -> EvidenceAdmissionResult:
    """过滤服务异常时保留全部候选，并只记录类型化错误，不复制正文。"""
    candidates = _all_candidates(result)
    decisions = tuple(
        _decision(item, "accepted", (f"filter_error:{error_code}",))
        for item in candidates
    )
    return EvidenceAdmissionResult(result, _report(decisions, len(candidates)))


def _decision(
    evidence: RetrievedEvidence, decision: str, reasons: tuple[str, ...]
) -> EvidenceFilterDecision:
    return EvidenceFilterDecision(
        fingerprint=build_evidence_fingerprint(evidence),
        paper_id=evidence.paper_id,
        source_chunk_ids=evidence.source_chunk_ids,
        decision=decision,
        reasons=reasons,
        score=evidence.score,
    )


def _score(evidence: RetrievedEvidence) -> float:
    return evidence.score if evidence.score is not None else float("-inf")


def _source_overlap(left: RetrievedEvidence, right: RetrievedEvidence) -> float:
    if left.paper_id != right.paper_id or left.source_type is not right.source_type:
        return 0.0
    left_ids = set(left.source_chunk_ids)
    right_ids = set(right.source_chunk_ids)
    union = left_ids | right_ids
    return len(left_ids & right_ids) / len(union) if union else 0.0


def _table_like(content: str) -> bool:
    tokens = re.findall(r"\S+", content)
    if not tokens:
        return False
    numeric = sum(bool(re.search(r"\d", token)) for token in tokens)
    prose_words = len(re.findall(r"[A-Za-z]{4,}|[\u3400-\u9fff]{2,}", content))
    separators = len(re.findall(r"[|=,:;]", content))
    return numeric / len(tokens) >= 0.3 and (separators >= 2 or prose_words <= 4)


def _all_candidates(result: ToolResult) -> tuple[RetrievedEvidence, ...]:
    payload = result.payload
    if isinstance(payload, SearchPapersPayload):
        return tuple(
            evidence
            for outcome in payload.paper_results
            for evidence in outcome.evidence_candidates
        )
    return tuple(getattr(payload, "evidence_candidates", ()))


def _report(
    decisions: tuple[EvidenceFilterDecision, ...], total: int
) -> EvidenceFilterReport:
    accepted = sum(item.decision in {"accepted", "deprioritized"} for item in decisions)
    return EvidenceFilterReport(
        total_candidates=total,
        accepted_candidates=accepted,
        rejected_candidates=sum(item.decision == "rejected" for item in decisions),
        deprioritized_candidates=sum(
            item.decision == "deprioritized" for item in decisions
        ),
        decisions=decisions,
    )


def _aggregate_search_status(
    outcomes: tuple[PaperSearchOutcome, ...]
) -> ToolStatus:
    statuses = tuple(item.status for item in outcomes)
    if statuses and all(item is PaperSearchStatus.EMPTY for item in statuses):
        return ToolStatus.EMPTY
    if statuses and all(item is PaperSearchStatus.UNAVAILABLE for item in statuses):
        return ToolStatus.UNAVAILABLE
    if statuses and all(item is PaperSearchStatus.ERROR for item in statuses):
        return ToolStatus.ERROR
    failed = any(
        item in {PaperSearchStatus.ERROR, PaperSearchStatus.UNAVAILABLE}
        for item in statuses
    )
    succeeded = any(
        item in {PaperSearchStatus.OK, PaperSearchStatus.EMPTY} for item in statuses
    )
    if failed and succeeded:
        return ToolStatus.PARTIAL
    if failed:
        return ToolStatus.ERROR
    return ToolStatus.OK
