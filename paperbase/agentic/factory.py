"""当前 LangGraph Agentic Research 的生产装配入口。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from paperbase.config import load_settings
from paperbase.database import MetadataDatabase
from paperbase.generation.section_expander import SectionAwareNeighborExpander
from paperbase.llm.client import (
    ChatCompletionClient,
    create_chat_completion_client,
    load_llm_runtime_settings,
    structured_client_view,
)
from paperbase.overview.repository import FormalOverviewRepository
from paperbase.retrieval.adapters import FormalKnowledgeBaseAdapter
from paperbase.retrieval.facade import RetrievalService
from paperbase.retrieval.hybrid_retriever import HybridRetriever
from paperbase.retrieval.service import create_hybrid_retriever

from .decision import LLMDecisionAgent
from .decision_view import CompactDecisionLimits
from .candidate_selection import FormalPaperMetadataRepository, LLMCandidateSelector
from .grounded_answer import LLMGroundedAnswerer
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
    由运行时 ``LLM_PROVIDER`` 选择的 Client。显式依赖注入只用于契约测试或替换供应商。
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
    active_client = client or create_chat_completion_client(
        load_llm_runtime_settings(settings.config_path.parent / ".env")
    )
    active_retriever = hybrid_retriever or create_hybrid_retriever(
        config_path=settings.config_path,
        database=active_database,
        llm_client=active_client,
    )
    with_max_tokens = getattr(active_client, "with_max_tokens", None)
    # Agentic nodes omit application output caps; retain their temperature/mode.
    agentic_client = with_max_tokens(None) if callable(with_max_tokens) else active_client
    qwen_profile = settings.agentic_research.llm_sampling_profile == "qwen38_recommended"

    def temperature_for(node: str, thinking: bool = False) -> float:
        return (1.0 if thinking else 0.7) if qwen_profile else getattr(
            settings.agentic_research, f"{node}_temperature")

    def sampling_view(node_client, *, thinking: bool = False):
        if not qwen_profile:
            return node_client
        configure = getattr(node_client, "with_sampling_parameters", None)
        think = getattr(node_client, "with_thinking_enabled", None)
        if not callable(configure) or not callable(think):
            raise AgenticResearchConfigurationError("Client cannot apply qwen38_recommended sampling.")
        return configure(top_p=0.95 if thinking else 0.8,
                         presence_penalty=0.0 if thinking else 1.5,
                         top_k=20, min_p=0.0, repetition_penalty=1.0).with_thinking_enabled(
                             thinking).with_temperature(1.0 if thinking else 0.7)

    agentic_client = sampling_view(agentic_client)
    decision_client = sampling_view(agentic_client, thinking=settings.agentic_research.decision_thinking_enabled)
    decision_timeout = getattr(decision_client, "with_timeout_seconds", None)
    if callable(decision_timeout):
        decision_client = decision_timeout(settings.agentic_research.decision_timeout_seconds)
    decision_retries = getattr(decision_client, "with_transport_max_retries", None)
    if callable(decision_retries):
        decision_client = decision_retries(settings.agentic_research.decision_transport_max_retries)
    critical_mode = settings.agentic_research.schema_critical_output_mode
    planner_client = sampling_view(structured_client_view(
        active_client, temperature=temperature_for("planner"),
        structured_output_mode=critical_mode,
    ))
    answer_client = sampling_view(structured_client_view(
        active_client, temperature=temperature_for("answer"),
        structured_output_mode=critical_mode,
    ))
    disable_thinking = getattr(answer_client, "with_thinking_disabled", None)
    if callable(disable_thinking):
        answer_client = disable_thinking()
    with_answer_timeout = getattr(answer_client, "with_timeout_seconds", None)
    if callable(with_answer_timeout):
        answer_client = with_answer_timeout(
            settings.answer_generation.timeout_seconds
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
        query_understanding=QueryUnderstandingService(agentic_client),
        planner=ResearchPlanner(planner_client, temperature=temperature_for("planner")),
        decision_agent=LLMDecisionAgent(
            decision_client,
            temperature=temperature_for("decision", settings.agentic_research.decision_thinking_enabled),
            thinking_enabled=settings.agentic_research.decision_thinking_enabled,
            max_attempts=settings.agentic_research.node_max_attempts,
            view_limits=CompactDecisionLimits(
                support_entries_per_requirement=settings.agentic_research.decision_view_support_limit,
                recent_observations=settings.agentic_research.decision_view_recent_observations,
                recent_queries=settings.agentic_research.decision_view_recent_queries,
                recent_actions=settings.agentic_research.decision_view_recent_actions,
                sections=settings.agentic_research.decision_view_sections,
                unresolved_aspects=settings.agentic_research.decision_view_unresolved_limit,
                unresolved_aspect_chars=settings.agentic_research.decision_view_unresolved_chars,
                entity_hints=settings.agentic_research.decision_view_entity_limit,
            ),
        ),
        candidate_selector=LLMCandidateSelector(
            agentic_client,
            FormalPaperMetadataRepository(
                database=active_database,
                overview_repository=overview_repository,
            ),
            max_attempts=settings.agentic_research.node_max_attempts,
        ),
        tool_executor=tool_executor,
        settings=settings.agentic_research,
        grounded_answerer=(
            LLMGroundedAnswerer(
                answer_client,
                max_attempts=settings.agentic_research.node_max_attempts,
                temperature=temperature_for("answer"),
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
