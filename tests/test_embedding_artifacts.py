"""Step 4 向量工件与数据库读取编排的无模型回归测试。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from paperbase.embedding.artifacts import (
    EmbeddingArtifactError,
    load_embedding_artifact,
)
from paperbase.embedding.service import generate_database_embeddings


class _FakeDatabase:
    """只实现 Step 4 所需的最小数据库读取接口，避免单元测试加载真实 Qwen 权重。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def list_embedding_inputs(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "chunk_id": "paper_a_chunk_0000",
                "paper_id": "paper_a",
                "chunk_index": 0,
                "embedding_text": "Document one.",
            },
            {
                "chunk_id": "paper_a_chunk_0001",
                "paper_id": "paper_a",
                "chunk_index": 1,
                "embedding_text": "Document two.",
            },
        )


class _FakeEmbedder:
    """返回确定性单位向量，专门验证工件顺序与完整性，不测试第三方模型。"""

    backend_id = "fake"
    model_id = "fake/model"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if texts != ["Document one.", "Document two."]:
            raise AssertionError("Embedding service changed SQLite input order or text.")
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class EmbeddingArtifactTests(unittest.TestCase):
    """验证 Step 4 不依赖 GPU 也能保证向量—chunk 映射完整。"""

    def test_service_writes_and_loads_a_valid_artifact(self) -> None:
        """向量行、records 映射和 manifest 应构成可被 Step 5 安全读取的一组工件。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = generate_database_embeddings(
                database=_FakeDatabase(root / "paperbase.sqlite3"),  # type: ignore[arg-type]
                embedder=_FakeEmbedder(),  # type: ignore[arg-type]
                output_dir=root / "embeddings",
                normalized=True,
            )

            artifact = load_embedding_artifact(summary.artifact_paths.output_dir)

            self.assertEqual(summary.chunk_count, 2)
            self.assertEqual(summary.dimension, 2)
            self.assertEqual(artifact.vectors.dtype, np.float32)
            self.assertEqual(artifact.vectors.shape, (2, 2))
            self.assertEqual(
                [record.chunk_id for record in artifact.records],
                ["paper_a_chunk_0000", "paper_a_chunk_0001"],
            )
            self.assertEqual(artifact.manifest["record_count"], 2)
            self.assertTrue(artifact.manifest["normalized"])

    def test_loader_rejects_file_that_no_longer_matches_manifest(self) -> None:
        """文件被手动篡改或中断替换时，不能把混合批次静默交给未来 FAISS。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = generate_database_embeddings(
                database=_FakeDatabase(root / "paperbase.sqlite3"),  # type: ignore[arg-type]
                embedder=_FakeEmbedder(),  # type: ignore[arg-type]
                output_dir=root / "embeddings",
                normalized=True,
            )
            summary.artifact_paths.records_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(EmbeddingArtifactError):
                load_embedding_artifact(summary.artifact_paths.output_dir)


if __name__ == "__main__":
    unittest.main()
