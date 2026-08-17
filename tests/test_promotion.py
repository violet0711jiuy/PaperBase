"""Temporary Workspace 增量提升至正式 Knowledge Base 的回归测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from paperbase.config import default_config_path, load_settings
from paperbase.database import MetadataDatabase
from paperbase.embedding.artifacts import write_embedding_artifact
from paperbase.embedding.base import EmbeddingInput
from paperbase.promotion.service import PromotionError, promote_workspace
from paperbase.indexing.faiss_store import FaissIndexStore


class PromotionTests(unittest.TestCase):
    """以小型完整 staging 工件验证 Add，不加载 Docling 或 embedding 模型。"""

    def test_promote_reuses_workspace_artifacts_and_second_run_is_noop(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings, workspace_id, paper_id = _make_settings_and_workspace(root)

            promoted = promote_workspace(settings=settings, workspace_id=workspace_id)

            self.assertEqual(promoted.status, "promoted")
            self.assertEqual((promoted.document_count_before, promoted.document_count_after), (0, 1))
            self.assertEqual((promoted.section_count_before, promoted.section_count_after), (0, 2))
            self.assertEqual((promoted.chunk_count_before, promoted.chunk_count_after), (0, 2))
            self.assertEqual((promoted.faiss_ntotal_before, promoted.faiss_ntotal_after), (0, 1))
            self.assertEqual((promoted.content_chunk_count, promoted.bibliography_chunk_count), (1, 1))
            self.assertTrue(promoted.formal_pdf_path and promoted.formal_pdf_path.is_file())

            database = MetadataDatabase(settings.database.path)
            self.assertEqual(len(database.list_sections(paper_id)), 2)
            chunks = database.list_chunks(paper_id)
            self.assertEqual([(row["section_type"], row["vector_id"]) for row in chunks], [("content", 1), ("bibliography", None)])
            self.assertEqual(chunks[0]["section_id"], f"{paper_id}_section_0001")
            # FTS 由正式 chunks trigger 自动维护，bibliography 不会混进正文 FTS。
            self.assertEqual(len(database.search_bm25_keyword_group(["fixture"], top_k=5)), 1)
            self.assertEqual(len(database.search_bibliography_keyword_group(["reference"], paper_id=paper_id, top_k=5)), 1)

            repeated = promote_workspace(settings=settings, workspace_id=workspace_id)
            self.assertEqual(repeated.status, "already_exists")
            self.assertEqual(repeated.document_count_after, 1)
            self.assertEqual(repeated.faiss_ntotal_after, 1)

            workspace_manifest = json.loads((settings.storage.staging_dir / workspace_id / "workspace.json").read_text(encoding="utf-8"))
            self.assertTrue(workspace_manifest["added_to_kb"])
            self.assertEqual(workspace_manifest["promotion"]["paper_id"], paper_id)

    def test_preflight_failure_writes_nothing_to_formal_knowledge_base(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings, workspace_id, _ = _make_settings_and_workspace(root)
            # 删除必要 embedding artifact，必须在 SQLite/PDF/FAISS 写入前停止。
            (settings.storage.staging_dir / workspace_id / "embeddings" / "vectors.npy").unlink()

            with self.assertRaises(PromotionError):
                promote_workspace(settings=settings, workspace_id=workspace_id)

            database = MetadataDatabase(settings.database.path)
            self.assertEqual(database.row_counts(), {"documents": 0, "chunks": 0})
            self.assertFalse(settings.indexing.index_path.exists())
            self.assertFalse(any(settings.storage.papers_dir.glob("*.pdf")))

    def test_publish_failure_rolls_back_sqlite_faiss_and_formal_pdf(self) -> None:
        """FAISS 公开阶段失败后，不能留下 SQLite/FTS 或正式 PDF 的半成品。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings, workspace_id, _ = _make_settings_and_workspace(root)
            with patch.object(
                FaissIndexStore,
                "publish_incremental_candidate",
                side_effect=RuntimeError("fixture publish failure"),
            ):
                with self.assertRaises(RuntimeError):
                    promote_workspace(settings=settings, workspace_id=workspace_id)

            database = MetadataDatabase(settings.database.path)
            self.assertEqual(database.row_counts(), {"documents": 0, "chunks": 0})
            self.assertFalse(settings.indexing.index_path.exists())
            self.assertFalse(settings.indexing.manifest_path.exists())
            self.assertFalse(any(settings.storage.papers_dir.glob("*.pdf")))


def _make_settings_and_workspace(root: Path):
    """创建与真实 staging 文件格式相同、但向量为二维确定性夹具的一篇论文。"""
    base = load_settings(default_config_path())
    staging_dir = root / "storage" / "staging"
    papers_dir = root / "storage" / "papers"
    settings = base.model_copy(
        update={
            "storage": base.storage.model_copy(update={"staging_dir": staging_dir, "papers_dir": papers_dir}),
            "database": base.database.model_copy(update={"path": root / "storage" / "paperbase.sqlite3"}),
            "embedding": base.embedding.model_copy(update={"output_dir": root / "storage" / "formal_embeddings"}),
            "indexing": base.indexing.model_copy(update={"index_path": root / "storage" / "paperbase.faiss", "manifest_path": root / "storage" / "paperbase.faiss.manifest.json"}),
        }
    )
    workspace_id = "staging_fixture"
    workspace = staging_dir / workspace_id
    original = workspace / "original" / "paper.pdf"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"%PDF-promotion-fixture")
    source_sha256 = hashlib.sha256(original.read_bytes()).hexdigest()
    paper_id = f"paper_{source_sha256[:16]}"
    section_parent = f"{paper_id}_section_0000"
    section_leaf = f"{paper_id}_section_0001"
    workspace_manifest = {
        "workspace_schema_version": 1, "workspace_id": workspace_id, "paper_id": paper_id,
        "source_filename": "fixture.pdf", "source_sha256": source_sha256,
    }
    (workspace / "parsed").mkdir()
    (workspace / "chunks").mkdir()
    (workspace / "workspace.json").write_text(json.dumps(workspace_manifest), encoding="utf-8")
    parsed = {
        "parser_id": "fixture_parser", "paper_title": "Fixture paper", "title_source": "fixture",
        "title_candidates": ["Fixture paper"], "diagnostics": {"fixture.parse": "ok"},
        "sections": [
            {"section_id": section_parent, "paper_id": paper_id, "section_title": "1 Method", "section_number": "1", "section_level": 1, "parent_section_id": None, "section_index": 0, "page_start": 1, "page_end": 1},
            {"section_id": section_leaf, "paper_id": paper_id, "section_title": "1.1 Detail", "section_number": "1.1", "section_level": 2, "parent_section_id": section_parent, "section_index": 1, "page_start": 1, "page_end": 1},
        ],
    }
    (workspace / "parsed" / "parsed_paper.json").write_text(json.dumps(parsed), encoding="utf-8")
    content_id = f"{paper_id}_chunk_0000"
    bibliography_id = f"{paper_id}_chunk_0001"
    records = [
        {"chunk_id": content_id, "vector_id": None, "paper_id": paper_id, "paper_title": "Fixture paper", "source": str(original), "chunk_index": 0, "raw_text": "The fixture method is described here.", "embedding_text": "Title: Fixture paper\nPassage: fixture method", "section": "1 Method > 1.1 Detail", "page_start": 1, "page_end": 1, "raw_token_count": 6, "embedding_token_count": 10, "prev_chunk_id": None, "next_chunk_id": bibliography_id, "content_kind": "body", "front_matter_type": None, "section_type": "content", "section_id": section_leaf},
        {"chunk_id": bibliography_id, "vector_id": None, "paper_id": paper_id, "paper_title": "Fixture paper", "source": str(original), "chunk_index": 1, "raw_text": "[1] Fixture reference.", "embedding_text": "Title: Fixture paper\nPassage: reference", "section": "References", "page_start": 2, "page_end": 2, "raw_token_count": 4, "embedding_token_count": 8, "prev_chunk_id": content_id, "next_chunk_id": None, "content_kind": "body", "front_matter_type": None, "section_type": "bibliography", "section_id": None},
    ]
    (workspace / "chunks" / "chunks.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    (workspace / "chunks" / "metadata.json").write_text(json.dumps({"chunking_diagnostics": {"chunking.chunker_id": "fixture_chunker"}}), encoding="utf-8")
    write_embedding_artifact(
        output_dir=workspace / "embeddings",
        inputs=(EmbeddingInput(chunk_id=content_id, paper_id=paper_id, chunk_index=0, embedding_text=records[0]["embedding_text"]),),
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32), backend_id="fixture", model_id="fixture/model", normalized=True,
        database_path=workspace / "workspace.json",
    )
    return settings, workspace_id, paper_id


if __name__ == "__main__":
    unittest.main()
