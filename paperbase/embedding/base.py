"""Step 4 embedding 输入、输出与适配器的统一契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EmbeddingInput:
    """一条需要生成文档向量的 SQLite chunk 快照。

    ``embedding_text`` 是 Step 2 已构造好的检索文本，包含论文标题与章节上下文；这里绝不
    读取 ``raw_text`` 重新拼接，也不为文档侧附加 query instruction。这样同一 chunk 的向量
    输入可以被稳定审计，并与未来的中文 query 编码策略明确分离。
    """

    chunk_id: str
    paper_id: str
    chunk_index: int
    embedding_text: str


@runtime_checkable
class DocumentEmbedder(Protocol):
    """可替换的文档 embedding 适配器接口。

    Step 4 只调用文档侧编码。未来更换 Qwen、改用远程服务或加入其他本地模型时，只需实现
    这一小接口；SQLite 读取、产物清单与后续 FAISS 建索引都不依赖具体模型库。
    """

    backend_id: str
    model_id: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """返回与输入顺序完全一致的二维 ``float32`` 向量矩阵。"""


@runtime_checkable
class QueryEmbedder(Protocol):
    """Query 侧 embedding 接口；与文档侧分开以避免误把 instruction 加入论文文本。"""

    def embed_queries(self, texts: list[str], *, instruction: str) -> np.ndarray:
        """返回与输入顺序一致、适合余弦检索的单位化 Query 向量矩阵。"""
