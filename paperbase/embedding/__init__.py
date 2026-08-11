"""PaperBase Step 4 embedding 模块。"""

from .base import DocumentEmbedder, EmbeddingInput
from .factory import create_document_embedder

__all__ = ["DocumentEmbedder", "EmbeddingInput", "create_document_embedder"]
