"""根据集中配置创建可替换的论文分块器。"""

from __future__ import annotations

from paperbase.config import AppSettings

from .base import PaperChunker
from .docling_hybrid_chunker import ChunkerSettings, DoclingHybridPaperChunker


class UnsupportedChunkingBackendError(ValueError):
    """配置选择了尚未接入的分块器时抛出的明确错误。"""


def create_chunker(settings: AppSettings) -> PaperChunker:
    """从 ``config.yaml`` 的 ``chunking.backend`` 创建当前分块器。"""
    backend = settings.chunking.backend.casefold()
    if backend == "docling_hybrid":
        chunking = settings.chunking
        return DoclingHybridPaperChunker(
            ChunkerSettings(
                tokenizer_path=chunking.tokenizer_path,
                max_tokens=chunking.max_tokens,
                embedding_metadata_reserve_tokens=(
                    chunking.embedding_metadata_reserve_tokens
                ),
                repeat_table_header=chunking.repeat_table_header,
                merge_peers=chunking.merge_peers,
            )
        )
    raise UnsupportedChunkingBackendError(
        f"Unsupported chunking backend: {settings.chunking.backend!r}. "
        "Currently registered backends: docling_hybrid."
    )
