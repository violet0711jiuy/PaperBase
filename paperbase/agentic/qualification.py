"""Evidence Qualification 的受控写入边界。"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EvidenceQualification
from .evidence_registry import EvidenceRegistry


@dataclass(frozen=True)
class QualificationApplyResult:
    applied_evidence_ids: tuple[str, ...]


class EvidenceQualificationManager:
    """校验 Agent 标注引用，并以不可变快照方式更新 Registry。"""

    def apply(
        self,
        registry: EvidenceRegistry,
        qualifications: tuple[EvidenceQualification, ...],
    ) -> QualificationApplyResult:
        return QualificationApplyResult(
            applied_evidence_ids=registry.apply_qualifications(qualifications)
        )
