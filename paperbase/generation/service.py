"""Step 8 端到端服务：检索、同节邻居扩展、受证据约束的回答生成。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

from paperbase.config import load_settings
from paperbase.conversations import (
    FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
    ConversationStore,
    format_turns_as_context,
)
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
        conversation_scope_id: str = "default",
        min_rerank_score: float = 0.05,
    ) -> None:
        self._retriever = retriever
        self._expander = expander
        self._generator = generator
        self._conversation_store = conversation_store
        self._conversation_max_context_turns = conversation_max_context_turns
        # 直接实例化 AnswerService 时保留旧的 default 行为；统一 factory 会使用正式 KB
        # scope。这样旧的底层单元测试/调用仍可运行，同时产品入口不会再混用 scope。
        self._conversation_scope_id = conversation_scope_id
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
                evidence_json=_evidence_snapshot_json(
                    expansion,
                    answer.citations,
                    answer_status=answer.status,
                ),
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
            record = self._conversation_store.create_conversation(
                "knowledge_base", self._conversation_scope_id
            )
        else:
            record = self._conversation_store.get_conversation(
                conversation_id,
                expected_scope_type="knowledge_base",
                expected_scope_id=self._conversation_scope_id,
            )
        turns = self._conversation_store.get_recent_turns(
            record.conversation_id, self._conversation_max_context_turns
        )
        return record.conversation_id, format_turns_as_context(turns)


def _evidence_snapshot_json(
    expansion: ExpansionResult,
    citations: Sequence[str] = (),
    *,
    answer_status: str | None = None,
) -> str | None:
    """把本轮最终可展示 Evidence 序列化为最小快照。

    这里故意不保存 chunk、向量、候选集或分数，只保留前端需要恢复的论文定位
    和原文；因此不会把 UI 持久化扩展成检索审计数据。
    """
    # 澄清和证据不足分支没有可展示的正常回答 Evidence；不要把检索候选误当成
    # 这类 turn 的证据入口。disabled 分支则保留原文，提示用户可以直接阅读证据。
    if answer_status in {"needs_clarification", "insufficient_evidence"}:
        return None
    citation_set = set(citations)
    # 正常生成答案时只落正文实际引用的 E#/R#；无引用的降级答案仍保留检索到的
    # 用户可读证据，方便“回答生成未启用”时查看原文。
    selected_evidence = (
        tuple(item for item in expansion.evidence if item.evidence_id in citation_set)
        if citation_set
        else expansion.evidence
    )
    evidence = []
    for item in selected_evidence:
        evidence.append(
            {
                "evidence_id": item.evidence_id,
                "type": item.kind,
                "paper_title": item.paper_title,
                "section": item.section or None,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "raw_text": item.text,
            }
        )
    if not evidence:
        return None
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def create_answer_service(
    *,
    config_path: Path | str | None = None,
    conversation_scope_id: str = FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
) -> AnswerService:
    """从统一配置装配 Step 8，默认连接正式 Knowledge Base scope。"""
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
        conversation_scope_id=conversation_scope_id,
        min_rerank_score=settings.answer_generation.min_rerank_score,
    )
