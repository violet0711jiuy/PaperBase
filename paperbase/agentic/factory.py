"""Phase 2.5 Agentic RAG Workflow 的生产装配入口。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from paperbase.config import load_settings
from paperbase.database import MetadataDatabase
from paperbase.generation.section_expander import SectionAwareNeighborExpander
from paperbase.generation.answer_generator import GroundedAnswerGenerator
from paperbase.llm.client import (
    ChatCompletionClient,
    OpenAICompatibleChatClient,
    load_llm_runtime_settings,
)
from paperbase.overview.repository import FormalOverviewRepository
from paperbase.retrieval.adapters import FormalKnowledgeBaseAdapter
from paperbase.retrieval.facade import RetrievalService
from paperbase.retrieval.hybrid_retriever import HybridRetriever
from paperbase.retrieval.service import create_hybrid_retriever

from .decision import LLMDecisionAgent
from .candidate_selection import FormalPaperMetadataRepository, LLMCandidateSelector
from .evidence_analysis import LLMEvidenceAnalyzer
from .graph import LangGraphResearchWorkflow
from .persistence import SQLiteCheckpointStore
from .planner import ResearchPlanner
from .query_understanding import QueryUnderstandingService
from .scope_resolver import ScopeResolver
from .service import AgenticResearchService
from .tools import ResearchToolExecutor

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


class AgenticResearchConfigurationError(RuntimeError):
    """功能关闭时请求 Agentic 生产装配。"""


def create_agentic_research_service(
    *,
    config_path: Path | str | None = None,
    client: ChatCompletionClient | None = None,
    database: MetadataDatabase | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    checkpointer: "BaseCheckpointSaver | None" = None,
) -> AgenticResearchService:
    """用一个共享的真实 LLM Client 装配正式知识库 LangGraph Service。

    正常应用调用不传 ``client``，工厂会从项目 ``.env`` 加载并创建
    :class:`OpenAICompatibleChatClient`。显式依赖注入只用于契约测试或替换供应商。
    Workspace Adapter 的生命周期属于单个临时 Workspace，将由后续 UI/Router 层装配，
    不应在正式知识库工厂中常驻。
    """
    settings = load_settings(config_path)
    if not settings.agentic_research.enabled:
        raise AgenticResearchConfigurationError(
            "agentic_research.enabled 已关闭，不能创建 Agentic Research Service。"
        )

    active_database = database or MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    active_database.initialize()
    active_client = client or OpenAICompatibleChatClient(
        load_llm_runtime_settings(settings.config_path.parent / ".env")
    )
    active_retriever = hybrid_retriever or create_hybrid_retriever(
        config_path=settings.config_path,
        database=active_database,
        llm_client=active_client,
    )
    answer_client = active_client
    with_max_tokens = getattr(active_client, "with_max_tokens", None)
    if callable(with_max_tokens):
        answer_client = with_max_tokens(settings.answer_generation.max_output_tokens)
    evidence_analysis_client = active_client
    if callable(with_max_tokens):
        evidence_analysis_client = with_max_tokens(
            settings.agentic_research.evidence_analysis_max_output_tokens
        )

    expander = SectionAwareNeighborExpander(
        database=active_database,
        settings=settings.context_expansion,
    )
    retrieval_service = RetrievalService(
        formal_adapter=FormalKnowledgeBaseAdapter(
            database=active_database,
            retriever=active_retriever,
            expander=expander,
        )
    )
    overview_repository = FormalOverviewRepository(
        settings.storage.papers_dir.parent / "paper_overviews"
    )
    tool_executor = ResearchToolExecutor(
        retrieval_service=retrieval_service,
        database=active_database,
        global_retriever=active_retriever,
        top_k_per_paper=settings.agentic_research.top_k_per_paper,
        formal_overview_repository=overview_repository,
    )
    checkpoint_store = None
    active_checkpointer = checkpointer
    if active_checkpointer is None:
        checkpoint_store = SQLiteCheckpointStore.open(
            settings.agentic_research.checkpoint_path
        )
        checkpoint_store.prune_expired_threads(
            retention_days=settings.agentic_research.checkpoint_retention_days
        )
        active_checkpointer = checkpoint_store.saver

    workflow = LangGraphResearchWorkflow(
        scope_resolver=ScopeResolver(active_database),
        query_understanding=QueryUnderstandingService(active_client),
        planner=ResearchPlanner(active_client),
        decision_agent=LLMDecisionAgent(active_client),
        candidate_selector=LLMCandidateSelector(
            active_client,
            FormalPaperMetadataRepository(
                database=active_database,
                overview_repository=overview_repository,
            ),
        ),
        evidence_analyzer=LLMEvidenceAnalyzer(evidence_analysis_client),
        tool_executor=tool_executor,
        settings=settings.agentic_research,
        answer_generator=(
            GroundedAnswerGenerator(
                settings=settings.answer_generation,
                client=answer_client,
            )
            if settings.answer_generation.enabled
            else None
        ),
        checkpointer=active_checkpointer,
    )
    return AgenticResearchService(
        workflow=workflow,
        checkpoint_store=checkpoint_store,
        max_concurrent_runs=settings.agentic_research.max_concurrent_runs,
    )
