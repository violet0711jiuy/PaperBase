"""Step 8：将检索证据扩展为可审计的论文问答上下文，并生成受约束回答。"""

from .answer_generator import AnswerGenerationResult, GroundedAnswerGenerator
from .section_expander import EvidenceUnit, ExpansionResult, SectionAwareNeighborExpander
from .service import AnswerService, AnswerServiceResult, create_answer_service

__all__ = [
    "AnswerGenerationResult",
    "AnswerService",
    "AnswerServiceResult",
    "EvidenceUnit",
    "ExpansionResult",
    "GroundedAnswerGenerator",
    "SectionAwareNeighborExpander",
    "create_answer_service",
]
