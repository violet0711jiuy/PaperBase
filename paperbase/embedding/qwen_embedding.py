"""Qwen3-Embedding 本地 SentenceTransformer 适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from paperbase.config import EmbeddingSettings


class EmbeddingModelError(RuntimeError):
    """本地 embedding 模型缺失、无法加载或输出不满足统一契约时抛出。"""


class QwenSentenceTransformerEmbedder:
    """只从本地目录加载 Qwen3-Embedding 的文档侧编码器。

    Qwen 官方模型卡说明：检索文档不应附加 query instruction；模型自身的 ``document``
    prompt 当前为空。因此编码时显式使用该 prompt 名称，以便模型配置将来调整文档提示时，
    行为仍由模型工件而非业务代码决定。
    """

    backend_id = "qwen_sentence_transformers"

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self.model_id = settings.model_id
        self._model: Any | None = None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """批量生成保持输入顺序的单位化 ``float32`` 文档向量。"""
        if not texts:
            raise EmbeddingModelError("Cannot embed an empty document batch.")
        if any(not text.strip() for text in texts):
            raise EmbeddingModelError("Embedding input contains an empty text.")

        model = self._load_model()
        vectors = model.encode(
            texts,
            # 文档侧不添加检索问题 instruction；Qwen 模型工件中该 prompt 定义为空字符串。
            prompt_name="document",
            batch_size=self._settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._settings.normalize_embeddings,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] < 1:
            raise EmbeddingModelError(
                "Embedding model returned an invalid matrix shape: "
                f"{matrix.shape!r} for {len(texts)} inputs."
            )

        # GPU 半精度计算会让库内的单位化结果有极小误差。这里在 float32 上再归一化一次，
        # 确保 Step 5 的 IndexFlatIP 可以严格等价于余弦相似度，而不依赖某个模型库的细节。
        if self._settings.normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(~np.isfinite(norms)) or np.any(norms == 0):
                raise EmbeddingModelError("Embedding model returned a non-finite or zero vector.")
            matrix = matrix / norms
        if not np.all(np.isfinite(matrix)):
            raise EmbeddingModelError("Embedding model returned non-finite vector values.")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _load_model(self) -> Any:
        """延迟加载 0.6B 权重，避免导入模块或运行单元测试时占用 GPU 显存。"""
        if self._model is not None:
            return self._model
        if not self._settings.model_path.is_dir():
            raise FileNotFoundError(
                "Embedding model directory not found: "
                f"{self._settings.model_path}"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingModelError(
                "sentence-transformers is required for the Qwen embedding backend."
            ) from error

        # ``local_files_only=True`` 保证不会因缓存缺失而下载到 C 盘或隐式访问网络。
        self._model = SentenceTransformer(
            str(Path(self._settings.model_path)),
            device=self._settings.device,
            local_files_only=True,
        )
        return self._model
