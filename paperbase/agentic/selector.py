"""最终 Evidence Selector：平衡论文覆盖并保留完整 provenance。"""

from __future__ import annotations

from .contracts import (
    CoverageState,
    EvidenceAnalysisStatus,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    PaperAssessmentStatus,
    ResearchPlan,
)


class FinalEvidenceSelector:
    def __init__(self, *, max_units: int = 8, max_per_paper: int = 4) -> None:
        if max_units < 1 or max_per_paper < 1:
            raise ValueError("Evidence Selector 上限必须大于零。")
        self._max_units = max_units
        self._max_per_paper = max_per_paper

    def select(
        self,
        *,
        plan: ResearchPlan,
        coverage: CoverageState,
        evidence: tuple[EvidenceRecord, ...],
    ) -> tuple[EvidenceRecord, ...]:
        eligible = tuple(record for record in evidence if self._eligible(record))
        ranked = sorted(eligible, key=lambda record: self._rank(record, plan), reverse=True)
        selected: list[EvidenceRecord] = []
        counts: dict[str, int] = {}
        seen_sections: dict[str, set[tuple[object, ...]]] = {}

        # 先为每篇 positive 论文保留一条核心证据，避免多论文比较被高分单篇吞没。
        positive_ids = tuple(
            item.paper_id
            for item in coverage.assessments
            if item.status is PaperAssessmentStatus.POSITIVE
        )
        for paper_id in positive_ids:
            candidate = next(
                (record for record in ranked if record.evidence.paper_id == paper_id), None
            )
            if candidate is not None:
                selected.append(candidate)
                counts[paper_id] = 1
                seen_sections.setdefault(paper_id, set()).add(
                    _section_identity(candidate)
                )
        # 第二轮优先补齐每篇论文的不同章节或来源窗口，避免最终上下文被同一段吞没。
        for record in ranked:
            if len(selected) >= self._max_units:
                break
            if record in selected:
                continue
            paper_id = record.evidence.paper_id
            if counts.get(paper_id, 0) >= self._max_per_paper:
                continue
            section = _section_identity(record)
            if section in seen_sections.get(paper_id, set()):
                continue
            selected.append(record)
            counts[paper_id] = counts.get(paper_id, 0) + 1
            seen_sections.setdefault(paper_id, set()).add(section)
        # 最后一轮才允许同章节的次级证据按相关性补位。
        for record in ranked:
            if len(selected) >= self._max_units:
                break
            if record in selected:
                continue
            paper_id = record.evidence.paper_id
            if counts.get(paper_id, 0) >= self._max_per_paper:
                continue
            selected.append(record)
            counts[paper_id] = counts.get(paper_id, 0) + 1
        return tuple(selected[: self._max_units])

    @staticmethod
    def _eligible(record: EvidenceRecord) -> bool:
        if record.analysis_status is not EvidenceAnalysisStatus.QUALIFIED:
            return False
        return any(
            claim.relation in {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}
            and claim.ownership
            not in {EvidenceOwnership.GENERAL_BACKGROUND, EvidenceOwnership.UNKNOWN}
            for claim in record.claims
        )

    @staticmethod
    def _rank(record: EvidenceRecord, plan: ResearchPlan) -> tuple[float, ...]:
        direct = float(
            any(
                claim.ownership is EvidenceOwnership.THIS_PAPER
                and claim.relation
                in {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}
                for claim in record.claims
            )
        )
        relation = float(
            any(
                claim.relation
                in {EvidenceRelation.SUPPORTS, EvidenceRelation.CONTRADICTS}
                for claim in record.claims
            )
        )
        subtask_match = float(
            bool(set(record.target_subtask_ids).intersection(
                task.subtask_id for task in plan.subtasks
            ))
        )
        source_priority = float(
            record.evidence.source_type.value not in {"bibliography", "overview"}
        )
        score = record.evidence.score if record.evidence.score is not None else 0.0
        return (direct, relation, subtask_match, source_priority, float(score))


def _section_identity(record: EvidenceRecord) -> tuple[object, ...]:
    evidence = record.evidence
    if evidence.section_id:
        return (evidence.source_type.value, "section_id", evidence.section_id)
    if evidence.section:
        return (
            evidence.source_type.value,
            "section",
            " ".join(evidence.section.casefold().split()),
        )
    return (
        evidence.source_type.value,
        "chunks",
        tuple(sorted(evidence.source_chunk_ids)),
    )
