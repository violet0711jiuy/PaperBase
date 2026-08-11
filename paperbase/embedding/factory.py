"""根据配置创建 Step 4 embedding 适配器。"""

from __future__ import annotations

from paperbase.config import EmbeddingSettings

from .base import DocumentEmbedder
from .qwen_embedding import QwenSentenceTransformerEmbedder


def create_document_embedder(settings: EmbeddingSettings) -> DocumentEmbedder:
    """将配置中的 backend 字符串映射为实现，避免调用方了解具体模型类。"""
    if settings.backend == "qwen_sentence_transformers":
        return QwenSentenceTransformerEmbedder(settings)
    raise ValueError(f"Unsupported embedding backend: {settings.backend!r}")
