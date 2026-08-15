"""BAAI/bge-reranker-v2-m3 的本地 Transformers Cross-Encoder 适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from paperbase.config import RerankingSettings

from .base import RerankScore


class RerankerModelError(RuntimeError):
    """本地 reranker 模型缺失、无法加载或输出不符合契约时抛出。"""


class BGECrossEncoderReranker:
    """使用本地 BGE Cross-Encoder 对 query 与正文 chunk 成对计算相关性。"""

    backend_id = "bge_cross_encoder"

    def __init__(self, settings: RerankingSettings) -> None:
        self._settings = settings
        self.model_id = settings.model_id
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def rerank(self, query: str, passages: list[str]) -> tuple[RerankScore, ...]:
        """分批计算每个 query-passage 配对的相关性，不改变调用方候选文本。"""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise RerankerModelError("Reranker query must not be empty.")
        if not passages:
            return ()
        if any(not passage.strip() for passage in passages):
            raise RerankerModelError("Reranker passage must not be empty.")

        tokenizer, model, torch = self._load_model()
        scores: list[float] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(passages), self._settings.batch_size):
                batch = passages[start : start + self._settings.batch_size]
                encoded = tokenizer(
                    [normalized_query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._settings.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    name: value.to(self._settings.device)
                    for name, value in encoded.items()
                }
                logits = model(**encoded).logits.reshape(-1)
                if self._settings.normalize_scores:
                    logits = torch.sigmoid(logits)
                scores.extend(float(value) for value in logits.detach().float().cpu().tolist())

        array = np.asarray(scores, dtype=np.float32)
        if array.shape != (len(passages),) or not np.all(np.isfinite(array)):
            raise RerankerModelError("Reranker returned invalid relevance scores.")
        return tuple(RerankScore(input_index=index, score=float(score)) for index, score in enumerate(array))

    def _load_model(self) -> tuple[Any, Any, Any]:
        """延迟加载本地权重，确保导入模块和单元测试时不占用 GPU，也绝不隐式联网。"""
        if self._tokenizer is not None and self._model is not None:
            import torch

            return self._tokenizer, self._model, torch
        if not self._settings.model_path.is_dir():
            raise FileNotFoundError(
                f"Reranker model directory not found: {self._settings.model_path}"
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RerankerModelError(
                "torch and transformers are required for the BGE Cross-Encoder reranker."
            ) from error

        model_path = str(Path(self._settings.model_path))
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self._settings.device)
        return self._tokenizer, self._model, torch
