"""PaperBase Step 5 全局向量索引模块。"""

from .faiss_store import FaissIndexStore, IndexBuildSummary

__all__ = ["FaissIndexStore", "IndexBuildSummary"]
