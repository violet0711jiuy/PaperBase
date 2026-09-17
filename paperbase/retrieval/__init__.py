"""Step 6 混合检索模块：经典检索与 Agentic Retrieval Facade。"""

from .adapters import FormalKnowledgeBaseAdapter
from .contracts import PaperSearchSpec, QueryOrigin, RetrievalSource, SearchPapersRequest
from .facade import RetrievalAdapter, RetrievalService
from .hybrid_retriever import HybridRetriever, RetrievedChunk, RetrievalResult
from .query_rewriter import (
    LLMQueryPlanner,
    QueryRewritePlan,
    TrustedPaperScope,
    create_query_planner,
)

__all__ = [
    "FormalKnowledgeBaseAdapter",
    "HybridRetriever",
    "LLMQueryPlanner",
    "PaperSearchSpec",
    "QueryOrigin",
    "QueryRewritePlan",
    "RetrievedChunk",
    "RetrievalAdapter",
    "RetrievalResult",
    "RetrievalService",
    "RetrievalSource",
    "SearchPapersRequest",
    "TrustedPaperScope",
    "create_query_planner",
]
