"""Step 6 混合检索模块：Query 改写、稠密召回、BM25 与 RRF 融合。"""

from .hybrid_retriever import HybridRetriever, RetrievedChunk, RetrievalResult
from .query_rewriter import LLMQueryPlanner, QueryRewritePlan, TrustedPaperScope, create_query_planner

__all__ = [
    "HybridRetriever",
    "LLMQueryPlanner",
    "QueryRewritePlan",
    "RetrievedChunk",
    "RetrievalResult",
    "TrustedPaperScope",
    "create_query_planner",
]
from .contracts import PaperSearchSpec, QueryOrigin, RetrievalSource, SearchPapersRequest
from .facade import RetrievalAdapter, RetrievalService

__all__ = [
    "PaperSearchSpec",
    "QueryOrigin",
    "RetrievalAdapter",
    "RetrievalService",
    "RetrievalSource",
    "SearchPapersRequest",
]
