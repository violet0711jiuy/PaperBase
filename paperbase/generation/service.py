"""Step 8 端到端服务：检索、同节邻居扩展、受证据约束的回答生成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from paperbase.config import load_settings
from paperbase.database import MetadataDatabase
from paperbase.retrieval.hybrid_retriever import RetrievalResult
from paperbase.retrieval.service import create_hybrid_retriever

from .answer_generator import AnswerGenerationOutcome, GroundedAnswerGenerator, create_answer_generator
from .relevance import NO_RELEVANT_EVIDENCE_MESSAGE, has_insufficient_retrieval_relevance
from .section_expander import ExpansionResult, SectionAwareNeighborExpander


@dataclass(frozen=True)
class AnswerServiceResult:
    """Step 8 的完整只读结果：保留检索、扩展和最终答案，便于审计每一层。"""

    retrieval: RetrievalResult
    expansion: ExpansionResult
    answer: AnswerGenerationOutcome


class AnswerService:
    """组合已有 Retriever 与 Step 8 新增组件，不在这里重复实现任何召回逻辑。"""

    def __init__(
        self,
        *,
        retriever: object,
        expander: SectionAwareNeighborExpander,
        generator: GroundedAnswerGenerator,
        min_rerank_score: float = 0.05,
    ) -> None:
        self._retriever = retriever
        self._expander = expander
        self._generator = generator
        self._min_rerank_score = min_rerank_score

    def answer_query(
        self,
        query: str,
        *,
        conversation_context: Sequence[str] | None = None,
        retrieval_result_limit: int | None = None,
    ) -> AnswerServiceResult:
        """执行 Step 6/7 既有检索，再以 Step 8 生成答案；不写入 SQLite 或索引。"""
        retrieval = self._retriever.retrieve(
            query,
            conversation_context=conversation_context,
            result_limit=retrieval_result_limit,
        )
        expansion = self._expander.expand(retrieval)
        if retrieval.rewrite_plan.resolution_status == "unresolved":
            answer = AnswerGenerationOutcome(
                status="needs_clarification",
                answer=retrieval.rewrite_plan.clarification_message,
                citations=(),
                insufficient_evidence=True,
            )
        elif has_insufficient_retrieval_relevance(
            retrieval,
            min_rerank_score=self._min_rerank_score,
        ):
            # Top-K 总会返回候选；低相关性保护防止这些候选被错误地送入 LLM 后产生无依据回答。
            answer = AnswerGenerationOutcome(
                status="insufficient_evidence",
                answer=NO_RELEVANT_EVIDENCE_MESSAGE,
                citations=(),
                insufficient_evidence=True,
            )
        else:
            answer = self._generator.generate(
                # 回答同样使用已消解的问题，避免模型把“它”重新解释成另一个对象。
                query=retrieval.rewrite_plan.resolved_query or retrieval.query,
                evidence=expansion.evidence,
                # 检索与回答复用同一小段会话上下文：前者用于改写检索查询，后者只用于
                # 消解“本文/它”等指代，不能越过 E#/R# 证据边界产生事实结论。
                conversation_context=conversation_context,
            )
        return AnswerServiceResult(
            retrieval=retrieval,
            expansion=expansion,
            answer=answer,
        )


def create_answer_service(*, config_path: Path | str | None = None) -> AnswerService:
    """从统一配置装配 Step 8；LLM 凭证仍只从项目根目录 .env 读取。"""
    settings = load_settings(config_path)
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    database.initialize()
    return AnswerService(
        retriever=create_hybrid_retriever(config_path=config_path),
        expander=SectionAwareNeighborExpander(
            database=database,
            settings=settings.context_expansion,
        ),
        generator=create_answer_generator(
            settings=settings.answer_generation,
            env_path=settings.config_path.parent / ".env",
        ),
        min_rerank_score=settings.answer_generation.min_rerank_score,
    )
