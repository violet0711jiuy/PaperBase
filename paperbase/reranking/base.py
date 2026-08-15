"""Step 7 重排序输入、输出与可替换适配器契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RerankScore:
    """一个正文候选被 Cross-Encoder 打分后的结果。"""

    # 输入候选在正文 RRF 队列中的序号，从 0 开始，仅用于将分数映射回原候选。
    input_index: int
    # 重排序相关性分数；默认经过 sigmoid，范围为 0 到 1。
    score: float


@runtime_checkable
class Reranker(Protocol):
    """可替换的“问题 + 候选正文”Cross-Encoder 重排序接口。"""

    backend_id: str
    model_id: str

    def rerank(self, query: str, passages: list[str]) -> tuple[RerankScore, ...]:
        """按输入顺序对所有非空候选打分，返回每个候选的稳定 input_index 与分数。"""
