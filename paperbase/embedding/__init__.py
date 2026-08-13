"""PaperBase Step 4 embedding 模块。"""

from .base import DocumentEmbedder, EmbeddingInput, QueryEmbedder
from .factory import create_document_embedder

__all__ = [
    "DocumentEmbedder",
    "EmbeddingInput",
    "QueryEmbedder",
    "create_document_embedder",
]
