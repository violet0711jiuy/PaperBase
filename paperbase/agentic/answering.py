"""把 Agentic Evidence 转换为现有 Grounded Answer Generator 的稳定桥接层。"""

from __future__ import annotations

from typing import Protocol, Sequence

from paperbase.generation.answer_generator import AnswerGenerationOutcome
from paperbase.generation.section_expander import EvidenceUnit

from .contracts import EvidenceRecord, ResearchAnswer


class AgenticAnswerGenerator(Protocol):
    def generate(
        self,
        *,
        query: str,
        evidence: Sequence[EvidenceUnit],
        conversation_context: Sequence[str] | None = None,
    ) -> AnswerGenerationOutcome: ...


def generate_agentic_answer(
    *,
    generator: AgenticAnswerGenerator,
    query: str,
    evidence: tuple[EvidenceRecord, ...],
) -> ResearchAnswer:
    """复用现有引用校验能力生成答案，不重新编号或改写 Evidence provenance。"""
    outcome = generator.generate(
        query=query,
        evidence=tuple(_to_evidence_unit(record) for record in evidence),
        conversation_context=None,
    )
    return ResearchAnswer(
        status=outcome.status,
        answer=outcome.answer,
        citations=outcome.citations,
        insufficient_evidence=outcome.insufficient_evidence,
        partial_answer=outcome.partial_answer,
        coverage_note=outcome.coverage_note,
    )


def _to_evidence_unit(record: EvidenceRecord) -> EvidenceUnit:
    source = record.evidence
    return EvidenceUnit(
        evidence_id=record.evidence_id,
        kind=source.source_type.value,
        paper_id=source.paper_id,
        paper_title=source.paper_title or source.paper_id,
        section=source.section or "未标注章节",
        page_start=source.page_start,
        page_end=source.page_end,
        seed_chunk_ids=source.seed_chunk_ids,
        chunk_ids=source.source_chunk_ids,
        text=source.content,
        token_count=max(1, (len(source.content) + 3) // 4),
        section_id=source.section_id,
    )
