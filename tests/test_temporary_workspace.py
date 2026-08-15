"""v0.2 Temporary Paper Workspace 的隔离与产物回归测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from paperbase.chunking.base import ChunkingResult, PaperChunk
from paperbase.parsing.base import ParsedPaper
from paperbase.staging.service import (
    create_temporary_workspace,
    delete_temporary_workspace,
)


class _FakeParser:
    """生成最小 ParsedPaper，避免此单元测试加载 Docling 与 GPU。"""

    parser_id = "fake_parser"

    def parse(self, source: Path) -> ParsedPaper:
        return ParsedPaper(
            source=source,
            parser_id=self.parser_id,
            markdown="# Test paper\n\nTemporary workspace test.",
            page_furniture=(),
            front_matter=(),
            paper_title="Test paper",
            title_source="fixture",
            title_candidates=("Test paper",),
            diagnostics={"fixture.parse": "ok"},
            native_document=object(),
        )


class _FakeChunker:
    """返回正文和 bibliography 各一块，验证临时索引的过滤边界。"""

    chunker_id = "fake_chunker"

    def chunk(self, parsed_paper: ParsedPaper) -> ChunkingResult:
        paper_id = "paper_fixture"
        content = PaperChunk(
            chunk_id="paper_fixture_chunk_0000",
            vector_id=None,
            paper_id=paper_id,
            paper_title="Test paper",
            source=parsed_paper.source,
            chunk_index=0,
            raw_text="The temporary index contains this body passage.",
            embedding_text="Title: Test paper\nPassage: temporary body",
            section="1 Introduction",
            page_start=1,
            page_end=1,
            raw_token_count=8,
            embedding_token_count=10,
            prev_chunk_id=None,
            next_chunk_id="paper_fixture_chunk_0001",
        )
        bibliography = PaperChunk(
            chunk_id="paper_fixture_chunk_0001",
            vector_id=None,
            paper_id=paper_id,
            paper_title="Test paper",
            source=parsed_paper.source,
            chunk_index=1,
            raw_text="[1] A reference that must not enter the temporary main index.",
            embedding_text="Title: Test paper\nPassage: reference",
            section="References",
            page_start=2,
            page_end=2,
            raw_token_count=10,
            embedding_token_count=12,
            prev_chunk_id="paper_fixture_chunk_0000",
            next_chunk_id=None,
            section_type="bibliography",
        )
        return ChunkingResult(
            parsed_paper=parsed_paper,
            chunks=(content, bibliography),
            diagnostics={"chunking.chunker_id": self.chunker_id},
        )


class _FakeEmbedder:
    """返回确定性单位向量，验证写入真实临时 FAISS 索引的完整链路。"""

    backend_id = "fake"
    model_id = "fake/model"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if texts != ["Title: Test paper\nPassage: temporary body"]:
            raise AssertionError("Bibliography chunk was unexpectedly sent to the embedder.")
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class TemporaryWorkspaceTests(unittest.TestCase):
    """确保 v0.2 新链路只在 staging 内创建和删除数据。"""

    def test_workspace_pipeline_is_isolated_and_deletable(self) -> None:
        """Parse、Chunk、Embedding、临时索引完成后，正式 KB 的任一文件均不变。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_pdf = root / "new-paper.pdf"
            source_pdf.write_bytes(b"%PDF-fixture")
            staging_dir = root / "storage" / "staging"

            # 以字节快照模拟正式 SQLite、FTS5 与 FAISS；工作区服务不接收这些路径。
            formal_files = (
                root / "storage" / "paperbase.sqlite3",
                root / "storage" / "paperbase.sqlite3-wal",
                root / "storage" / "paperbase.faiss",
                root / "storage" / "paperbase.faiss.manifest.json",
            )
            for formal_file in formal_files:
                formal_file.parent.mkdir(parents=True, exist_ok=True)
                formal_file.write_bytes(f"formal:{formal_file.name}".encode("utf-8"))
            before = {path: _sha256(path) for path in formal_files}

            workspace = create_temporary_workspace(
                staging_dir=staging_dir,
                source_pdf=source_pdf,
                parser=_FakeParser(),
                chunker=_FakeChunker(),
                embedder=_FakeEmbedder(),
                normalized_embeddings=True,
            )

            self.assertTrue(workspace.root_dir.is_dir())
            self.assertEqual(workspace.total_chunk_count, 2)
            self.assertEqual(workspace.searchable_chunk_count, 1)
            self.assertEqual(workspace.original_pdf_path.read_bytes(), source_pdf.read_bytes())
            self.assertTrue(workspace.parsed_markdown_path.is_file())
            self.assertTrue(workspace.parsed_data_path.is_file())
            self.assertTrue(workspace.chunks_path.is_file())
            self.assertTrue(workspace.embedding_paths.vectors_path.is_file())
            self.assertTrue(workspace.index_path.is_file())

            # 临时索引只持有正文 chunk；bibliography 仍在 chunks.jsonl 中供未来专用功能使用。
            index_manifest = json.loads(workspace.index_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(index_manifest["included_section_type"], "content")
            self.assertEqual(
                [record["chunk_id"] for record in index_manifest["records"]],
                ["paper_fixture_chunk_0000"],
            )
            self.assertEqual(
                {record["paper_id"] for record in index_manifest["records"]},
                {workspace.paper_id},
            )
            self.assertEqual(len(workspace.chunks_path.read_text(encoding="utf-8").splitlines()), 2)

            import faiss

            index = faiss.deserialize_index(
                np.frombuffer(workspace.index_path.read_bytes(), dtype=np.uint8)
            )
            scores, ids = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 1)
            self.assertEqual(int(ids[0, 0]), 1)
            self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=6)
            self.assertEqual(before, {path: _sha256(path) for path in formal_files})

            delete_temporary_workspace(
                staging_dir=staging_dir, workspace_id=workspace.workspace_id
            )
            self.assertFalse(workspace.root_dir.exists())
            self.assertEqual(before, {path: _sha256(path) for path in formal_files})


def _sha256(path: Path) -> str:
    """为正式知识库文件建立简洁、稳定的不可变快照。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
