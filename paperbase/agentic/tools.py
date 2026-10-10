"""Phase 2 Research Tools：参数由 Agent 给出，上限与 Scope 由运行时控制。"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence
from dataclasses import asdict, replace
from paperbase.retrieval.requests import CanonicalSearchRequest, request_trace, retriever_parameters, validate_effective_request
from .search_requests import compile_action_requests

from paperbase.database import MetadataDatabase
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    KnowledgeBaseSearchPayload,
    PaperOverviewPayload,
    PaperSearchSpec,
    QueryOrigin,
    ReferenceSearchPayload,
    RetrievalSource,
    RetrievedEvidence,
    ScoreKind,
    ScoreDirection,
    SearchPapersRequest,
    SectionContextPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)
from paperbase.retrieval.facade import RetrievalService
from paperbase.retrieval.evidence_scores import chunk_score_fields
from paperbase.retrieval.hybrid_retriever import RetrievalResult
from paperbase.overview.repository import FormalOverviewRepository

from .contracts import (
    AgentAction,
    CorpusScopeSnapshot,
    GetPaperOverviewAction,
    GetSectionContextAction,
    QueryContext,
    QueryScope,
    ScopeSource,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
)


class GlobalRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        paper_ids: Sequence[str] | None = None,
        result_limit: int | None = None,
        query_origin: QueryOrigin = QueryOrigin.INITIAL_UNDERSTANDING,
        lexical_terms: Sequence[str] = (),
        canonical_request: CanonicalSearchRequest | None = None,
    ) -> RetrievalResult: ...


class ToolExecutor(Protocol):
    """Execution boundary used by the LangGraph workflow."""

    def execute(
        self,
        action: AgentAction,
        *,
        context: QueryContext,
        corpus_snapshot: CorpusScopeSnapshot | None = None,
        canonical_requests: tuple[CanonicalSearchRequest, ...] | None = None,
    ) -> ToolResult: ...


class ResearchToolExecutor:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        database: MetadataDatabase,
        global_retriever: GlobalRetriever | None = None,
        top_k_per_paper: int = 5,
        workspace_ids_by_paper: Mapping[str, str] | None = None,
        formal_overview_repository: FormalOverviewRepository | None = None,
    ) -> None:
        if top_k_per_paper < 1:
            raise ValueError("top_k_per_paper must be positive.")
        self._retrieval_service = retrieval_service
        self._database = database
        self._global_retriever = global_retriever
        self._top_k = top_k_per_paper
        self._workspace_ids = dict(workspace_ids_by_paper or {})
        self._formal_overviews = formal_overview_repository

    def compile_requests(self, action, *, context, corpus_snapshot=None):
        source = RetrievalSource.WORKSPACE if context.scope_source is ScopeSource.TRUSTED_WORKSPACE else RetrievalSource.FORMAL_KNOWLEDGE_BASE
        parameters = {search.paper_id: self._retrieval_service.request_parameters(
            source=source, workspace_id=self._workspace_ids.get(search.paper_id))
            if hasattr(self._retrieval_service, "request_parameters") else {}
            for search in action.searches} if isinstance(action, SearchPapersAction) else {}
        return compile_action_requests(action, context=context, snapshot=corpus_snapshot,
            top_k=self._top_k, workspace_ids=self._workspace_ids, parameters=parameters,
            kb_parameters=retriever_parameters(self._global_retriever) if self._global_retriever else {})

    def execute(self, action, *, context, corpus_snapshot=None, canonical_requests=None):
        requests = (self.compile_requests(action, context=context, corpus_snapshot=corpus_snapshot)
                    if canonical_requests is None else canonical_requests)
        flags = [False] * len(requests)
        rejection = next((rejection for request in requests
                          if (rejection := validate_effective_request(request)) is not None), None)
        if rejection is not None:
            result = ToolResult(ToolStatus.ERROR, ToolName.SEARCH_REFERENCES,
                ReferenceSearchPayload(action.paper_id, action.query),
                metadata={"request_rejection_detail": rejection.detail}, error_code=rejection.code)
        else:
            result = self._execute(action, context=context, corpus_snapshot=corpus_snapshot,
                                   canonical_requests=requests, execution_flags=flags)
        if not requests:
            return result
        executions = result.metadata.get("request_executions", [
            {"request_identity": request.identity, "attempted": True, "executed": flags[index],
             "failed": result.status in {ToolStatus.ERROR, ToolStatus.UNAVAILABLE},
             "successful": result.status in {ToolStatus.OK, ToolStatus.EMPTY, ToolStatus.PARTIAL}}
            for index, request in enumerate(requests)])
        return replace(result, metadata={**result.metadata, "request_contract_version": "search-v1",
                       "raw_action_arguments": asdict(action),
                       "canonical_effective_requests": request_trace(requests),
                       "request_executions": executions})

    def _execute(
        self,
        action: AgentAction,
        *,
        context: QueryContext,
        corpus_snapshot: CorpusScopeSnapshot | None = None,
        canonical_requests=(),
        execution_flags=None,
    ) -> ToolResult:
        if isinstance(action, SearchPapersAction):
            self._validate_paper_ids(
                tuple(search.paper_id for search in action.searches), context, corpus_snapshot
            )
            source = (
                RetrievalSource.WORKSPACE
                if context.scope_source is ScopeSource.TRUSTED_WORKSPACE
                else RetrievalSource.FORMAL_KNOWLEDGE_BASE
            )
            return self._retrieval_service.search_papers(
                SearchPapersRequest(
                    query_origin=QueryOrigin.AGENT_SUPPLIED,
                    searches=action.searches,
                    top_k_per_paper=self._top_k,
                ),
                source=source,
                workspace_ids_by_paper=self._workspace_ids,
                canonical_requests=canonical_requests,
            )
        if isinstance(action, SearchKnowledgeBaseAction):
            return self._search_knowledge_base(action, context, corpus_snapshot, canonical_requests[0], execution_flags)
        if isinstance(action, SearchReferencesAction):
            self._validate_paper_ids((action.paper_id,), context, corpus_snapshot)
            try:
                return self._search_references(action, canonical_requests[0], execution_flags)
            except Exception as error:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool_name=ToolName.SEARCH_REFERENCES,
                    payload=ReferenceSearchPayload(action.paper_id, action.query),
                    error_code=type(error).__name__,
                )
        if isinstance(action, GetSectionContextAction):
            self._validate_paper_ids((action.paper_id,), context, corpus_snapshot)
            try:
                return self._get_section_context(action)
            except Exception as error:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool_name=ToolName.GET_SECTION_CONTEXT,
                    payload=SectionContextPayload(action.paper_id, action.section_id),
                    error_code=type(error).__name__,
                )
        if isinstance(action, GetPaperOverviewAction):
            self._validate_paper_ids((action.paper_id,), context, corpus_snapshot)
            return self._get_paper_overview(action, context)
        raise TypeError(f"Unsupported executable action: {type(action).__name__}")

    def overview_statuses(
        self, paper_ids: Sequence[str], *, context: QueryContext
    ) -> tuple[tuple[str, str], ...]:
        """Expose only availability metadata so Decision can choose a tool.

        This is intentionally an ephemeral tool-capability view, not a new
        research-state or evidence source.
        """
        unique_ids = tuple(dict.fromkeys(paper_ids))
        if context.scope_source is ScopeSource.TRUSTED_WORKSPACE:
            return tuple((paper_id, "unavailable") for paper_id in unique_ids)
        if self._formal_overviews is None:
            return tuple((paper_id, "unavailable") for paper_id in unique_ids)
        statuses: list[tuple[str, str]] = []
        for paper_id in unique_ids:
            try:
                statuses.append(
                    (paper_id, "available" if self._formal_overviews.get(paper_id) else "unavailable")
                )
            except Exception:
                statuses.append((paper_id, "unknown"))
        return tuple(statuses)

    def _get_paper_overview(
        self, action: GetPaperOverviewAction, context: QueryContext
    ) -> ToolResult:
        if context.scope_source is ScopeSource.TRUSTED_WORKSPACE:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                tool_name=ToolName.GET_PAPER_OVERVIEW,
                payload=PaperOverviewPayload(action.paper_id),
                error_code="FormalOverviewNotAvailableForWorkspace",
            )
        if self._formal_overviews is None:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                tool_name=ToolName.GET_PAPER_OVERVIEW,
                payload=PaperOverviewPayload(action.paper_id),
                error_code="FormalOverviewRepositoryUnavailable",
            )
        try:
            artifact = self._formal_overviews.get(action.paper_id)
            if artifact is None:
                return ToolResult(
                    status=ToolStatus.UNAVAILABLE,
                    tool_name=ToolName.GET_PAPER_OVERVIEW,
                    payload=PaperOverviewPayload(action.paper_id),
                    error_code="FormalOverviewUnavailable",
                )
            evidence = _overview_to_evidence(action.paper_id, artifact.overview)
            return ToolResult(
                status=ToolStatus.OK if evidence else ToolStatus.EMPTY,
                tool_name=ToolName.GET_PAPER_OVERVIEW,
                payload=PaperOverviewPayload(action.paper_id, evidence),
                metadata={
                    "overview_path": str(artifact.overview_path),
                    "context_path": str(artifact.context_path),
                },
            )
        except Exception as error:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool_name=ToolName.GET_PAPER_OVERVIEW,
                payload=PaperOverviewPayload(action.paper_id),
                error_code=type(error).__name__,
            )

    def _search_knowledge_base(
        self,
        action: SearchKnowledgeBaseAction,
        context: QueryContext,
        snapshot: CorpusScopeSnapshot | None,
        request, execution_flags,
    ) -> ToolResult:
        if context.scope is not QueryScope.KNOWLEDGE_BASE or snapshot is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                payload=KnowledgeBaseSearchPayload(query=action.query),
                error_code="KnowledgeBaseScopeRequired",
            )
        if self._global_retriever is None:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                payload=KnowledgeBaseSearchPayload(query=action.query),
                error_code="GlobalRetrieverUnavailable",
            )
        try:
            execution_flags[0] = True
            result = self._global_retriever.retrieve(
                request.semantic_query,
                paper_ids=request.paper_ids,
                result_limit=request.top_k,
                query_origin=QueryOrigin.AGENT_SUPPLIED,
                lexical_terms=request.lexical_terms,
                canonical_request=request,
            )
            evidence = tuple(_retrieved_chunk_to_evidence(chunk) for chunk in result.chunks)
            return ToolResult(
                status=ToolStatus.OK if evidence else ToolStatus.EMPTY,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                payload=KnowledgeBaseSearchPayload(
                    query=action.query, evidence_candidates=evidence
                ),
                metadata={"corpus_snapshot_id": snapshot.snapshot_id},
            )
        except Exception as error:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                payload=KnowledgeBaseSearchPayload(query=action.query),
                error_code=type(error).__name__,
            )

    def _search_references(self, action: SearchReferencesAction, request, execution_flags) -> ToolResult:
        if self._database.get_document(action.paper_id) is None:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                tool_name=ToolName.SEARCH_REFERENCES,
                payload=ReferenceSearchPayload(action.paper_id, action.query),
                error_code="UnknownPaper",
            )
        execution_flags[0] = True
        rows = self._database.search_compiled_lexical(
            request.lexical_query, paper_ids=request.paper_ids,
            top_k=request.top_k, bibliography=True,
        )
        # Database contract already orders FTS5 BM25 ascending (lower is better).
        evidence = tuple(_row_to_evidence(row, EvidenceSourceType.BIBLIOGRAPHY, rank=rank)
                         for rank, row in enumerate(rows, start=1))
        return ToolResult(
            status=ToolStatus.OK if evidence else ToolStatus.EMPTY,
            tool_name=ToolName.SEARCH_REFERENCES,
            payload=ReferenceSearchPayload(
                paper_id=action.paper_id,
                query=action.query,
                evidence_candidates=evidence,
            ),
            metadata={"lexical_terms": request.lexical_terms},
        )

    def _get_section_context(self, action: GetSectionContextAction) -> ToolResult:
        section = self._database.get_section(action.section_id)
        if section is None or str(section["paper_id"]) != action.paper_id:
            return ToolResult(
                status=ToolStatus.UNAVAILABLE,
                tool_name=ToolName.GET_SECTION_CONTEXT,
                payload=SectionContextPayload(action.paper_id, action.section_id),
                error_code="UnknownSection",
            )
        rows = self._database.list_content_chunks_in_section_id(
            paper_id=action.paper_id, section_id=action.section_id
        )
        evidence: tuple[RetrievedEvidence, ...] = ()
        if rows:
            evidence = (
                RetrievedEvidence(
                    source_type=EvidenceSourceType.SECTION_CONTEXT,
                    paper_id=action.paper_id,
                    paper_title=str(rows[0]["paper_title"]),
                    seed_chunk_ids=tuple(str(row["chunk_id"]) for row in rows),
                    source_chunk_ids=tuple(str(row["chunk_id"]) for row in rows),
                    section_id=action.section_id,
                    section=str(section["section_title"]),
                    page_start=min(
                        (int(row["page_start"]) for row in rows if row["page_start"] is not None),
                        default=None,
                    ),
                    page_end=max(
                        (int(row["page_end"]) for row in rows if row["page_end"] is not None),
                        default=None,
                    ),
                    content="\n\n".join(str(row["raw_text"]) for row in rows),
                    score_kind=ScoreKind.NON_RANKED,
                    retrieval_metadata={"chunk_count": len(rows)},
                ),
            )
        return ToolResult(
            status=ToolStatus.OK if evidence else ToolStatus.EMPTY,
            tool_name=ToolName.GET_SECTION_CONTEXT,
            payload=SectionContextPayload(
                paper_id=action.paper_id,
                section_id=action.section_id,
                evidence_candidates=evidence,
            ),
        )

    @staticmethod
    def _validate_paper_ids(
        paper_ids: tuple[str, ...],
        context: QueryContext,
        snapshot: CorpusScopeSnapshot | None,
    ) -> None:
        allowed = (
            set(snapshot.paper_ids)
            if context.scope is QueryScope.KNOWLEDGE_BASE and snapshot is not None
            else set(context.target_paper_ids)
        )
        invalid = tuple(paper_id for paper_id in paper_ids if paper_id not in allowed)
        if invalid:
            raise ValueError("Action targets paper IDs outside the trusted scope: " + ", ".join(invalid))


def _row_to_evidence(row: object, source_type: EvidenceSourceType, *, rank: int) -> RetrievedEvidence:
    return RetrievedEvidence(
        source_type=source_type,
        paper_id=str(row["paper_id"]),  # type: ignore[index]
        paper_title=str(row["paper_title"]),  # type: ignore[index]
        seed_chunk_ids=(str(row["chunk_id"]),),  # type: ignore[index]
        source_chunk_ids=(str(row["chunk_id"]),),  # type: ignore[index]
        section_id=row["section_id"],  # type: ignore[index]
        section=str(row["section"]),  # type: ignore[index]
        page_start=row["page_start"],  # type: ignore[index]
        page_end=row["page_end"],  # type: ignore[index]
        content=str(row["raw_text"]),  # type: ignore[index]
        raw_score=float(row["bm25_score"]),  # type: ignore[index]
        score_kind=ScoreKind.BM25_FTS5,
        score_direction=ScoreDirection.LOWER_IS_BETTER,
        retrieval_rank=rank,
        retrieval_metadata={"route": "bibliography_fts"},
    )


def _overview_to_evidence(paper_id: str, overview: object) -> tuple[RetrievedEvidence, ...]:
    """把字段级 Overview 转成可独立标注且保留原始 chunk provenance 的证据。"""
    evidence: list[RetrievedEvidence] = []
    for field_name in (
        "research_problem",
        "main_method",
        "contributions",
        "datasets",
        "experimental_setup",
        "main_results",
        "limitations",
    ):
        field = getattr(overview, field_name)
        source_ids = tuple(field.source_chunk_ids)
        if not source_ids:
            continue
        evidence.append(
            RetrievedEvidence(
                source_type=EvidenceSourceType.OVERVIEW,
                paper_id=paper_id,
                paper_title=overview.paper_title,
                seed_chunk_ids=source_ids,
                source_chunk_ids=source_ids,
                section=" / ".join(field.source_sections) or "Paper Overview",
                content=f"[{field_name}] {field.content}",
                score_kind=ScoreKind.NON_RANKED,
                retrieval_metadata={"overview_field": field_name},
            )
        )
    return tuple(evidence)


def _retrieved_chunk_to_evidence(chunk: object) -> RetrievedEvidence:
    return RetrievedEvidence(
        source_type=EvidenceSourceType(str(chunk.section_type)),
        paper_id=str(chunk.paper_id),
        paper_title=str(chunk.paper_title),
        seed_chunk_ids=(str(chunk.chunk_id),),
        source_chunk_ids=(str(chunk.chunk_id),),
        section_id=chunk.section_id,
        section=str(chunk.section),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        content=str(chunk.raw_text),
        **chunk_score_fields(chunk),
        retrieval_metadata={"rank": chunk.rank, "expanded": False,
                            "rerank_score_normalized": getattr(chunk, "rerank_score_normalized", None)},
    )
