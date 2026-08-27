"""Ask This Paper 的端到端服务与 staging 结果工件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence
from uuid import uuid4

from paperbase.config import AppSettings, default_config_path, load_settings
from paperbase.conversations import ConversationStore, format_turns_as_context
from paperbase.embedding import QueryEmbedder, create_document_embedder
from paperbase.generation.answer_generator import (
    AnswerGenerationOutcome,
    GroundedAnswerGenerator,
    create_answer_generator,
)
from paperbase.generation.section_expander import ExpansionResult
from paperbase.generation.relevance import NO_RELEVANT_EVIDENCE_MESSAGE, has_insufficient_retrieval_relevance
from paperbase.reranking import create_reranker
from paperbase.retrieval.query_rewriter import TrustedPaperScope, create_query_planner
from paperbase.staging.bm25 import WorkspaceBM25IndexCache
from paperbase.staging.sections import WorkspaceSectionRepository, WorkspaceSectionSnapshot

from .evidence import WorkspaceSectionEvidenceExpander
from .retriever import WorkspaceHybridRetriever


class AskThisPaperError(RuntimeError):
    """Ask This Paper 的工作区装配、检索或工件保存失败。"""


@dataclass(frozen=True)
class AskThisPaperResult:
    """一次单论文问答的完整审计结果。"""

    workspace_id: str
    retrieval: object
    expansion: ExpansionResult
    answer: AnswerGenerationOutcome
    result_path: Path
    conversation_id: str | None


class AskThisPaperService:
    """只装配当前 Temporary Workspace 的问答链；不创建 MetadataDatabase。"""

    def __init__(
        self,
        *,
        snapshot: WorkspaceSectionSnapshot,
        retriever: WorkspaceHybridRetriever,
        expander: WorkspaceSectionEvidenceExpander,
        generator: GroundedAnswerGenerator,
        conversation_store: ConversationStore | None = None,
        conversation_max_context_turns: int = 0,
        min_rerank_score: float = 0.05,
    ) -> None:
        self._snapshot = snapshot
        self._retriever = retriever
        self._expander = expander
        self._generator = generator
        self._conversation_store = conversation_store
        self._conversation_max_context_turns = conversation_max_context_turns
        self._min_rerank_score = min_rerank_score

    def ask(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        conversation_context: Sequence[str] | None = None,
        retrieval_result_limit: int | None = None,
    ) -> AskThisPaperResult:
        """执行当前论文独占的检索、证据扩展和一次 LLM 回答，并保存可审计结果。"""
        active_conversation_id, stored_context = self._load_conversation_context(
            conversation_id=conversation_id
        )
        effective_context = (*stored_context, *(conversation_context or ()))
        # workspace 论文范围是可信程序状态，不能伪装成一条 conversation_context 文本。
        trusted_scope = TrustedPaperScope(
            paper_id=self._snapshot.paper_id,
            paper_title=self._snapshot.paper_title,
        )
        retrieval = self._retriever.retrieve(
            query,
            trusted_scope=trusted_scope,
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
            answer = AnswerGenerationOutcome(
                status="insufficient_evidence",
                answer=NO_RELEVANT_EVIDENCE_MESSAGE,
                citations=(),
                insufficient_evidence=True,
            )
        else:
            answer = self._generator.generate(
                query=retrieval.rewrite_plan.resolved_query or retrieval.query,
                evidence=expansion.evidence,
                # 只使用 resolved_query 与本轮 Evidence，历史回答不能充当论文事实。
                conversation_context=None,
            )
        result_path = _write_result_artifact(
        snapshot=self._snapshot,
        retrieval=retrieval,
        expansion=expansion,
        answer=answer,
        conversation_id=active_conversation_id,
        )
        if active_conversation_id is not None:
            self._conversation_store.append_turn(  # type: ignore[union-attr]
                active_conversation_id,
                user_query=query,
                assistant_answer=answer.answer,
                resolved_query=retrieval.rewrite_plan.resolved_query,
                resolution_status=retrieval.rewrite_plan.resolution_status,
                audit_path=str(result_path),
            )
        return AskThisPaperResult(
            workspace_id=self._snapshot.workspace_id,
            retrieval=retrieval,
            expansion=expansion,
            answer=answer,
            result_path=result_path,
            conversation_id=active_conversation_id,
        )

    def _load_conversation_context(
        self, *, conversation_id: str | None
    ) -> tuple[str | None, tuple[str, ...]]:
        """读取当前 workspace 的独占会话，禁止跨 workspace 复用历史。"""
        if self._conversation_store is None:
            if conversation_id is not None:
                raise ValueError("Conversation Store is not configured for this AskThisPaperService.")
            return None, ()
        if conversation_id is None:
            record = self._conversation_store.create_conversation(
                "workspace", self._snapshot.workspace_id
            )
        else:
            record = self._conversation_store.get_conversation(
                conversation_id,
                expected_scope_type="workspace",
                expected_scope_id=self._snapshot.workspace_id,
            )
        turns = self._conversation_store.get_recent_turns(
            record.conversation_id, self._conversation_max_context_turns
        )
        return record.conversation_id, format_turns_as_context(turns)


def create_ask_this_paper_service(
    *, workspace_id: str, config_path: Path | str | None = None
) -> AskThisPaperService:
    """从配置和 workspace 工件装配服务；所有路径均以 staging 根目录为边界。"""
    settings = load_settings(config_path)
    snapshot = WorkspaceSectionRepository(settings.storage.staging_dir).load(workspace_id)
    embedder = create_document_embedder(settings.embedding)
    if not isinstance(embedder, QueryEmbedder):
        raise AskThisPaperError("Configured embedding backend does not implement query embeddings.")
    cache = WorkspaceBM25IndexCache()
    retriever = WorkspaceHybridRetriever.from_workspace(
        snapshot=snapshot,
        query_embedder=embedder,
        settings=settings.retrieval,
        query_planner=create_query_planner(
            settings=settings.retrieval.query_rewrite,
            env_path=settings.config_path.parent / ".env",
        ),
        reranker=create_reranker(settings.reranking) if settings.reranking.enabled else None,
        reranking_settings=settings.reranking,
        bm25_cache=cache,
    )
    return AskThisPaperService(
        snapshot=snapshot,
        retriever=retriever,
        expander=WorkspaceSectionEvidenceExpander(
            snapshot=snapshot, settings=settings.context_expansion
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


def main(argv: Sequence[str] | None = None) -> None:
    """运行 ``python -m paperbase.ask_paper <workspace_id> <query>``。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import argparse

    parser = argparse.ArgumentParser(description="Ask only the selected temporary paper.")
    parser.add_argument("workspace_id", help="storage/staging 下的 staging_<uuid> 工作区。")
    parser.add_argument("query", help="针对当前论文的自然语言问题。")
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="复用此前返回的当前 workspace conversation_id；省略时自动创建新会话。",
    )
    parser.add_argument("--context", action="append", default=[], metavar="TEXT")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive.")
    result = create_ask_this_paper_service(
        workspace_id=args.workspace_id, config_path=args.config
    ).ask(
        args.query,
        conversation_id=args.conversation_id,
        conversation_context=args.context,
        retrieval_result_limit=args.top_k,
    )
    print(json.dumps(_result_to_json(result), ensure_ascii=False, indent=2))


def _write_result_artifact(
    *,
    snapshot: WorkspaceSectionSnapshot,
    retrieval: object,
    expansion: ExpansionResult,
    answer: AnswerGenerationOutcome,
    conversation_id: str | None,
) -> Path:
    """把一次问答保存到 workspace 内，方便后续前端读取；不修改 chunks、索引或 PDF。"""
    directory = snapshot.root_dir / "ask_paper"
    directory.mkdir(exist_ok=True)
    operation_id = uuid4().hex[:16]
    path = directory / f"{operation_id}.json"
    payload = _result_to_json(
        AskThisPaperResult(
            workspace_id=snapshot.workspace_id,
            retrieval=retrieval,
            expansion=expansion,
            answer=answer,
            result_path=path,
            conversation_id=conversation_id,
        )
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _result_to_json(result: AskThisPaperResult) -> dict[str, object]:
    """定义 CLI 与 workspace 工件共同使用的、可追溯 JSON 输出。"""
    retrieval = result.retrieval
    return {
        "workspace_id": result.workspace_id,
        "conversation_id": result.conversation_id,
        "query": retrieval.query,
        "rewrite_plan": {
            "resolution_status": retrieval.rewrite_plan.resolution_status,
            "resolved_query": retrieval.rewrite_plan.resolved_query,
            "semantic_query_en": retrieval.rewrite_plan.semantic_query_en,
            "lexical_keywords_en": list(retrieval.rewrite_plan.lexical_keywords_en),
            "semantic_status": retrieval.rewrite_plan.semantic_status,
            "lexical_status": retrieval.rewrite_plan.lexical_status,
            "validation_diagnostics": list(retrieval.rewrite_plan.validation_diagnostics),
            "rewrite_status": retrieval.rewrite_plan.rewrite_status,
            "search_bibliography": retrieval.rewrite_plan.search_bibliography,
        },
        "reranking_status": retrieval.reranking_status,
        "answer_status": result.answer.status,
        "answer": result.answer.answer,
        "citations": list(result.answer.citations),
        "insufficient_evidence": result.answer.insufficient_evidence,
        "partial_answer": result.answer.partial_answer,
        "coverage_note": result.answer.coverage_note,
        "retrieved_chunks": [
            {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "paper_title": chunk.paper_title,
                "section": chunk.section,
                "section_type": chunk.section_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "pre_rerank_rank": chunk.pre_rerank_rank,
                "rerank_score": chunk.rerank_score,
                "fused_score": chunk.fused_score,
                "source_matches": [asdict(source) for source in chunk.source_matches],
            }
            for chunk in retrieval.chunks
        ],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind,
                "paper_id": item.paper_id,
                "paper_title": item.paper_title,
                "section": item.section,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "seed_chunk_ids": list(item.seed_chunk_ids),
                "chunk_ids": list(item.chunk_ids),
                "token_count": item.token_count,
                "text": item.text,
            }
            for item in result.expansion.evidence
        ],
        "result_path": str(result.result_path),
    }
