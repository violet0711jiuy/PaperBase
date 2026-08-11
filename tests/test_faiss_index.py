"""Step 5 全局 FAISS 与 SQLite vector_id 映射的回归测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from paperbase.config import IndexingSettings
from paperbase.database import MetadataDatabase
from paperbase.embedding.artifacts import (
    load_embedding_artifact,
    write_embedding_artifact,
)
from paperbase.embedding.base import EmbeddingInput
from paperbase.indexing.faiss_store import (
    FaissIndexError,
    FaissIndexStore,
    VectorAssignment,
    _build_manifest,
    _write_faiss_index,
    _write_json_atomic,
)


class FaissIndexStoreTests(unittest.TestCase):
    """使用微型 SQLite 和确定性二维单位向量，验证不依赖 GPU 的正式索引契约。"""

    def test_initial_build_assigns_ids_and_matches_search_results(self) -> None:
        """每个向量必须以同一个 vector_id 同时进入 FAISS 与 SQLite。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database, output_dir, store = _prepare_database_artifact_and_store(root)

            summary = store.build_initial_index(
                database=database,
                embedding_output_dir=output_dir,
            )

            self.assertEqual(summary.vector_count, 2)
            self.assertEqual((summary.vector_id_min, summary.vector_id_max), (1, 2))
            self.assertTrue(store.index_path.is_file())
            self.assertTrue(store.manifest_path.is_file())
            self.assertEqual(
                [(row["vector_id"], row["chunk_id"]) for row in database.vector_id_mapping()],
                [(1, "paper_test_chunk_0000"), (2, "paper_test_chunk_0001")],
            )
            verified = store.verify_against_sqlite(
                database=database,
                embedding_output_dir=output_dir,
            )
            self.assertEqual(verified.vector_count, 2)

            import faiss

            index = faiss.deserialize_index(
                np.frombuffer(store.index_path.read_bytes(), dtype=np.uint8)
            )
            scores, ids = index.search(
                np.asarray([[1.0, 0.0]], dtype=np.float32), 1
            )
            self.assertEqual(int(ids[0, 0]), 1)
            self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=6)

    def test_build_rejects_artifact_when_sqlite_embedding_text_changed(self) -> None:
        """生成向量后文本改变时，不能把旧 vectors.npy 建入新的知识库。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database, output_dir, store = _prepare_database_artifact_and_store(root)
            connection = sqlite3.connect(database.path)
            try:
                connection.execute(
                    "UPDATE chunks SET embedding_text = 'Changed after embedding.' "
                    "WHERE chunk_id = 'paper_test_chunk_0000'"
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaises(FaissIndexError):
                store.build_initial_index(
                    database=database,
                    embedding_output_dir=output_dir,
                )
            self.assertEqual(database.vector_index_state().vectorized_chunk_count, 0)
            self.assertFalse(store.index_path.exists())

    def test_recovery_publishes_candidate_after_sqlite_commit(self) -> None:
        """模拟 SQLite 已提交、FAISS 尚未发布的中断，恢复后两侧映射必须一致。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database, output_dir, store = _prepare_database_artifact_and_store(root)
            artifact = load_embedding_artifact(output_dir)
            assignments = tuple(
                VectorAssignment(
                    chunk_id=record.chunk_id,
                    vector_id=index + 1,
                    embedding_text_sha256=record.embedding_text_sha256,
                )
                for index, record in enumerate(artifact.records)
            )
            candidate_path = store._candidate_index_path("recovery-test")
            _write_faiss_index(
                path=candidate_path,
                vectors=artifact.vectors,
                vector_ids=np.asarray([1, 2], dtype=np.int64),
            )
            manifest = _build_manifest(
                artifact=artifact,
                assignments=assignments,
                index_sha256=_sha256_file(candidate_path),
            )
            _write_json_atomic(
                store._journal_path,
                {
                    "journal_schema_version": 1,
                    "operation_id": "recovery-test",
                    "phase": "database_committed",
                    "index_path": str(store.index_path),
                    "manifest_path": str(store.manifest_path),
                    "candidate_index_path": str(candidate_path),
                    "assignments": [asdict(assignment) for assignment in assignments],
                    "manifest": manifest,
                },
            )
            database.assign_vector_ids(
                [
                    (
                        assignment.chunk_id,
                        assignment.vector_id,
                        assignment.embedding_text_sha256,
                    )
                    for assignment in assignments
                ]
            )

            self.assertTrue(store.recover_pending_publish(database))
            self.assertFalse(store._journal_path.exists())
            self.assertTrue(store.index_path.is_file())
            self.assertTrue(store.manifest_path.is_file())
            self.assertEqual(
                store.verify_against_sqlite(
                    database=database,
                    embedding_output_dir=output_dir,
                ).vector_count,
                2,
            )


def _prepare_database_artifact_and_store(
    root: Path,
) -> tuple[MetadataDatabase, Path, FaissIndexStore]:
    """构造两条 chunk、二维单位向量与空正式索引，用于每个测试的独立环境。"""
    database = MetadataDatabase(root / "paperbase.sqlite3")
    database.initialize()
    connection = sqlite3.connect(database.path)
    try:
        connection.execute(
            """
            INSERT INTO documents (
                paper_id, source_path, source_filename, source_sha256, paper_title,
                title_source, parser_id, chunker_id, parse_diagnostics_json,
                chunking_diagnostics_json, ingested_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "paper_test",
                str(root / "source.pdf"),
                "source.pdf",
                "a" * 64,
                "Test paper",
                "fixture",
                "fixture_parser",
                "fixture_chunker",
                "{}",
                "{}",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, vector_id, paper_id, chunk_index, raw_text, embedding_text,
                section, content_kind, front_matter_type, page_start, page_end,
                raw_token_count, embedding_token_count, prev_chunk_id, next_chunk_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "paper_test_chunk_0000",
                    None,
                    "paper_test",
                    0,
                    "First raw text.",
                    "First embedding text.",
                    "1. Introduction",
                    "body",
                    None,
                    1,
                    1,
                    3,
                    4,
                    None,
                    "paper_test_chunk_0001",
                ),
                (
                    "paper_test_chunk_0001",
                    None,
                    "paper_test",
                    1,
                    "Second raw text.",
                    "Second embedding text.",
                    "1. Introduction",
                    "body",
                    None,
                    1,
                    1,
                    3,
                    4,
                    "paper_test_chunk_0000",
                    None,
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    inputs = (
        EmbeddingInput(
            chunk_id="paper_test_chunk_0000",
            paper_id="paper_test",
            chunk_index=0,
            embedding_text="First embedding text.",
        ),
        EmbeddingInput(
            chunk_id="paper_test_chunk_0001",
            paper_id="paper_test",
            chunk_index=1,
            embedding_text="Second embedding text.",
        ),
    )
    output_dir = root / "embeddings"
    write_embedding_artifact(
        output_dir=output_dir,
        inputs=inputs,
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        backend_id="fixture",
        model_id="fixture/model",
        normalized=True,
        database_path=database.path,
    )
    store = FaissIndexStore(
        IndexingSettings(
            backend="faiss_flat_ip",
            index_path=root / "paperbase.faiss",
            manifest_path=root / "paperbase.faiss.manifest.json",
        )
    )
    return database, output_dir, store


def _sha256_file(path: Path) -> str:
    """测试中为手工 journal 生成与正式实现相同的索引文件哈希。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
