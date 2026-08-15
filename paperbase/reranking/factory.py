"""根据配置创建 Step 7 重排序器。"""

from __future__ import annotations

from paperbase.config import RerankingSettings

from .base import Reranker
from .bge_cross_encoder import BGECrossEncoderReranker


def create_reranker(settings: RerankingSettings) -> Reranker:
    """将配置中的 backend 标识映射为具体适配器，隔离业务检索代码与模型库。"""
    if settings.backend == "bge_cross_encoder":
        return BGECrossEncoderReranker(settings)
    raise ValueError(f"Unsupported reranking backend: {settings.backend!r}")
