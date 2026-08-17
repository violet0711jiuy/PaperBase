"""Step 8 端到端服务：检索、同节邻居扩展、受证据约束的回答生成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from paperbase.config import load_settings
from paperbase.conversations import ConversationStore, format_turns_as_context
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
    conversation_id: str | None


class AnswerService:
    """组合已有 Retriever 与 Step 8 新增组件，不在这里重复实现任何召回逻辑。"""

    def __init__(
        self,
        *,
        retriever: object,
        expander: SectionAwareNeighborExpander,
        generator: GroundedAnswerGenerator,
        conversation_store: ConversationStore | None = None,
        conversation_max_context_turns: int = 0,
        min_rerank_score: float = 0.05,
    ) -> None:
        self._retriever = retriever
        self._expander = expander
        self._generator = generator
        self._conversation_store = conversation_store
        self._conversation_max_context_turns = conversation_max_context_turns
        self._min_rerank_score = min_rerank_score

    def answer_query(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        conversation_context: Sequence[str] | None = None,
        retrieval_result_limit: int | None = None,
    ) -> AnswerServiceResult:
        """执行 KB 问答，并将最终用户可见回合写入独立 Conversation Store。"""
        active_conversation_id, stored_context = self._load_conversation_context(
            conversation_id=conversation_id
        )
        # --context 仍可用于调试；自动历史位于前方，显式输入作为额外最近上下文。
        effective_context = (*stored_context, *(conversation_context or ()))
        retrieval = self._retriever.retrieve(
            query,
            conversation_context=effective_context,
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
                # 历史已经只用于 Resolution；Answer LLM 只能使用本轮重新检索出的 Evidence。
                conversation_context=None,
            )
        if active_conversation_id is not None:
            self._conversation_store.append_turn(  # type: ignore[union-attr]
                active_conversation_id,
                user_query=query,
                assistant_answer=answer.answer,
                resolved_query=retrieval.rewrite_plan.resolved_query,
                resolution_status=retrieval.rewrite_plan.resolution_status,
            )
        return AnswerServiceResult(
            retrieval=retrieval,
            expansion=expansion,
            answer=answer,
            conversation_id=active_conversation_id,
        )

    def _load_conversation_context(
        self, *, conversation_id: str | None
    ) -> tuple[str | None, tuple[str, ...]]:
        """获取或创建 KB scope 会话，并把最近完整回合交给 Query Resolution。"""
        if self._conversation_store is None:
            if conversation_id is not None:
                raise ValueError("Conversation Store is not configured for this AnswerService.")
            return None, ()
        if conversation_id is None:
            record = self._conversation_store.create_conversation("knowledge_base", "default")
        else:
            record = self._conversation_store.get_conversation(
                conversation_id,
                expected_scope_type="knowledge_base",
                expected_scope_id="default",
            )
        turns = self._conversation_store.get_recent_turns(
            record.conversation_id, self._conversation_max_context_turns
        )
        return record.conversation_id, format_turns_as_context(turns)


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
        conversation_store=ConversationStore(
            settings.conversation.path,
            busy_timeout_ms=settings.conversation.busy_timeout_ms,
        ),
        conversation_max_context_turns=settings.conversation.max_context_turns,
        min_rerank_score=settings.answer_generation.min_rerank_score,
    )
