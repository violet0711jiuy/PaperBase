"""Step 3 SQLite 元数据层的导入与关联回归测试。"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from paperbase.chunking.base import ChunkingResult, PaperChunk
from paperbase.database import MetadataDatabase, MetadataDatabaseError
from paperbase.parsing.base import FrontMatterBlock, ParsedPaper


class MetadataDatabaseTests(unittest.TestCase):
    def test_bibliography_is_preserved_but_excluded_from_main_fts_and_embedding_inputs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = _sample_result(root)
            bibliography_chunk = replace(
                result.chunks[1],
                raw_text="Graph WaveNet is cited here.",
                embedding_text="Paper title: A test paper\nSection: References\nContent:\nGraph WaveNet is cited here.",
                section="References",
                content_kind="body",
                front_matter_type=None,
                section_type="bibliography",
            )
            result = replace(result, chunks=(result.chunks[0], bibliography_chunk))
            database = MetadataDatabase(root / "paperbase.sqlite3")
            database.import_chunking_result(result)

            stored = database.list_chunks(bibliography_chunk.paper_id)
            self.assertEqual(stored[1]["section_type"], "bibliography")
            self.assertEqual(database.search_bm25("Graph WaveNet", top_k=5), ())
            self.assertEqual(len(database.search_bibliography("Graph WaveNet", top_k=5)), 1)
            self.assertEqual(
                [row["chunk_id"] for row in database.list_embedding_inputs()],
                [result.chunks[0].chunk_id],
            )
    """验证 SQLite 不重复导入、外键关联和失败时原子回滚。"""

    def test_import_persists_document_and_typed_retrievable_chunks(self) -> None:
        """前置元数据与正文都应只写入 chunks，且前者可由类型字段查询。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = _sample_result(root)
            database = MetadataDatabase(root / "metadata.sqlite3")

            summary = database.import_chunking_result(result)

            self.assertFalse(summary.document_replaced)
            self.assertEqual(summary.front_matter_chunk_count, 2)
            self.assertEqual(summary.chunk_count, 2)
            self.assertEqual(
                database.row_counts(),
                {"documents": 1, "chunks": 2},
            )
            document = database.get_document(summary.paper_id)
            self.assertIsNotNone(document)
            self.assertEqual(document["paper_title"], "A test paper")
            self.assertEqual(document["parser_id"], "test_parser")
            self.assertEqual(document["chunker_id"], "test_chunker")
            self.assertNotIn("%PDF", document["source_path"])

            front_matter = database.list_front_matter_chunks(summary.paper_id)
            self.assertEqual(
                [row["front_matter_type"] for row in front_matter],
                ["authors_affiliations", "abstract"],
            )
            chunks = database.list_chunks(summary.paper_id)
            self.assertEqual([row["chunk_index"] for row in chunks], [0, 1])
            self.assertEqual(chunks[0]["next_chunk_id"], chunks[1]["chunk_id"])
            self.assertEqual(chunks[1]["prev_chunk_id"], chunks[0]["chunk_id"])
            self.assertIsNone(chunks[0]["vector_id"])
            self.assertEqual(chunks[0]["content_kind"], "front_matter")
            # FTS5 只是一份可重建倒排索引；标题、section、raw_text 均可参与 BM25。
            bm25_rows = database.search_bm25("First original chunk", top_k=5)
            self.assertEqual([row["chunk_id"] for row in bm25_rows], [chunks[0]["chunk_id"]])
            # 英文关键词组是一条 OR 型 BM25 查询；两个词分别命中不同 chunk 时都应进入候选。
            keyword_group_rows = database.search_bm25_keyword_group(
                ["First", "Second"], top_k=5
            )
            self.assertEqual(
                {row["chunk_id"] for row in keyword_group_rows},
                {chunks[0]["chunk_id"], chunks[1]["chunk_id"]},
            )
            embedding_inputs = database.list_embedding_inputs()
            self.assertEqual(
                [row["chunk_id"] for row in embedding_inputs],
                [chunks[0]["chunk_id"], chunks[1]["chunk_id"]],
            )

            connection = sqlite3.connect(database.path)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()
            self.assertNotIn("front_matter", tables)

    def test_reimport_replaces_one_paper_without_duplicates(self) -> None:
        """同一 PDF 的新解析结果应替换旧记录，而不是追加重复 chunks。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = MetadataDatabase(root / "metadata.sqlite3")
            first_result = _sample_result(root, first_chunk_text="Old parser output.")
            second_result = _sample_result(root, first_chunk_text="Corrected parser output.")

            first_summary = database.import_chunking_result(first_result)
            second_summary = database.import_chunking_result(second_result)

            self.assertEqual(first_summary.paper_id, second_summary.paper_id)
            self.assertTrue(second_summary.document_replaced)
            self.assertEqual(
                database.row_counts(),
                {"documents": 1, "chunks": 2},
            )
            chunks = database.list_chunks(second_summary.paper_id)
            self.assertEqual(chunks[0]["raw_text"], "Corrected parser output.")

    def test_invalid_neighbor_reference_writes_no_partial_document(self) -> None:
        """导入前校验失败时，数据库中不得留下半篇论文。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = _sample_result(root)
            invalid_chunk = PaperChunk(
                **{
                    **result.chunks[0].__dict__,
                    "next_chunk_id": "paper_missing_chunk_0001",
                }
            )
            invalid_result = ChunkingResult(
                parsed_paper=result.parsed_paper,
                chunks=(invalid_chunk, result.chunks[1]),
                diagnostics=result.diagnostics,
            )
            database = MetadataDatabase(root / "metadata.sqlite3")

            with self.assertRaises(MetadataDatabaseError):
                database.import_chunking_result(invalid_result)

            self.assertEqual(
                database.row_counts(),
                {"documents": 0, "chunks": 0},
            )

    def test_reimport_refuses_to_replace_vectorized_chunks(self) -> None:
        """未来已关联 FAISS 的论文不能被本阶段替换入口静默覆盖。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = _sample_result(root)
            database = MetadataDatabase(root / "metadata.sqlite3")
            summary = database.import_chunking_result(result)
            # sqlite3 的连接上下文只负责提交/回滚，并不会在 Windows 上保证立即释放文件
            # 句柄。这里显式 close，避免 TemporaryDirectory 清理时数据库文件仍被锁定。
            connection = sqlite3.connect(database.path)
            try:
                connection.execute(
                    "UPDATE chunks SET vector_id = 101 WHERE paper_id = ? AND chunk_index = 0",
                    (summary.paper_id,),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaises(MetadataDatabaseError):
                database.import_chunking_result(result)
            with self.assertRaises(MetadataDatabaseError):
                database.list_embedding_inputs()

            self.assertEqual(database.row_counts()["chunks"], 2)
            self.assertEqual(
                database.list_chunks(summary.paper_id)[0]["vector_id"],
                101,
            )

    def test_v1_database_migrates_front_matter_to_typed_chunks(self) -> None:
        """升级旧库时应保留 chunk 正文、回填类型，并移除重复的全文表。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "legacy.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE documents (
                        paper_id TEXT PRIMARY KEY,
                        paper_title TEXT
                    );
                    CREATE TABLE front_matter (
                        front_matter_id INTEGER PRIMARY KEY,
                        paper_id TEXT NOT NULL,
                        block_index INTEGER NOT NULL,
                        block_type TEXT NOT NULL,
                        canonical_section TEXT NOT NULL,
                        text TEXT NOT NULL,
                        page_start INTEGER,
                        page_end INTEGER,
                        source_item_count INTEGER NOT NULL,
                        detection_method TEXT NOT NULL,
                        confidence TEXT NOT NULL
                    );
                    CREATE TABLE chunks (
                        chunk_id TEXT PRIMARY KEY,
                        vector_id INTEGER UNIQUE,
                        paper_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        raw_text TEXT NOT NULL,
                        embedding_text TEXT NOT NULL,
                        section TEXT,
                        page_start INTEGER,
                        page_end INTEGER,
                        raw_token_count INTEGER NOT NULL,
                        embedding_token_count INTEGER NOT NULL,
                        prev_chunk_id TEXT,
                        next_chunk_id TEXT
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_info(key, value) VALUES('schema_version', '1')"
                )
                connection.execute(
                    "INSERT INTO documents(paper_id, paper_title) VALUES('paper_legacy', 'Legacy paper')"
                )
                connection.execute(
                    """
                    INSERT INTO front_matter (
                        paper_id, block_index, block_type, canonical_section, text,
                        page_start, page_end, source_item_count, detection_method, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "paper_legacy",
                        0,
                        "abstract",
                        "Front matter > Abstract",
                        "Legacy abstract text.",
                        1,
                        1,
                        1,
                        "explicit_heading",
                        "high",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, vector_id, paper_id, chunk_index, raw_text, embedding_text,
                        section, page_start, page_end, raw_token_count,
                        embedding_token_count, prev_chunk_id, next_chunk_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "paper_legacy_chunk_0000",
                        None,
                        "paper_legacy",
                        0,
                        "Legacy abstract text.",
                        "Paper title: Legacy\nContent:\nLegacy abstract text.",
                        "A B S T R A C T",
                        1,
                        1,
                        4,
                        8,
                        None,
                        None,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            database = MetadataDatabase(database_path)
            database.initialize()

            migrated = database.list_chunks("paper_legacy")
            self.assertEqual(migrated[0]["raw_text"], "Legacy abstract text.")
            self.assertEqual(migrated[0]["content_kind"], "front_matter")
            self.assertEqual(migrated[0]["front_matter_type"], "abstract")
            connection = sqlite3.connect(database_path)
            try:
                version = connection.execute(
                    "SELECT value FROM schema_info WHERE key = 'schema_version'"
                ).fetchone()[0]
                table_names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(version, "4")
            self.assertNotIn("front_matter", table_names)
            self.assertIn("chunks_fts", table_names)


def _sample_result(
    root: Path,
    *,
    first_chunk_text: str = "First original chunk.",
) -> ChunkingResult:
    """构造一套带真实 PDF 字节哈希的最小结构结果，不依赖 Docling 或 GPU。"""
    source = root / "sample.pdf"
    source.write_bytes(b"%PDF-test-paperbase-step-3")
    paper_id = f"paper_{sha256(source.read_bytes()).hexdigest()[:16]}"
    parsed_paper = ParsedPaper(
        source=source,
        parser_id="test_parser",
        markdown="# A test paper",
        page_furniture=(),
        front_matter=(
            FrontMatterBlock(
                block_type="authors_affiliations",
                canonical_section="Front matter > Authors and affiliations",
                text="Alice Example\nExample University",
                page_start=1,
                page_end=1,
                source_item_count=2,
                detection_method="title_anchor_span",
                confidence="high",
            ),
            FrontMatterBlock(
                block_type="abstract",
                canonical_section="Front matter > Abstract",
                text="A compact test abstract.",
                page_start=1,
                page_end=1,
                source_item_count=1,
                detection_method="explicit_heading",
                confidence="high",
            ),
        ),
        paper_title="A test paper",
        title_source="test_fixture",
        title_candidates=("A test paper",),
        diagnostics={"test.parser": "fixture"},
        native_document=object(),
    )
    first_id = f"{paper_id}_chunk_0000"
    second_id = f"{paper_id}_chunk_0001"
    chunks = (
        PaperChunk(
            chunk_id=first_id,
            vector_id=None,
            paper_id=paper_id,
            paper_title=parsed_paper.paper_title,
            source=source,
            chunk_index=0,
            raw_text=first_chunk_text,
            embedding_text=f"Paper title: A test paper\nContent:\n{first_chunk_text}",
            section="Authors and affiliations",
            page_start=1,
            page_end=1,
            raw_token_count=5,
            embedding_token_count=10,
            prev_chunk_id=None,
            next_chunk_id=second_id,
            content_kind="front_matter",
            front_matter_type="authors_affiliations",
        ),
        PaperChunk(
            chunk_id=second_id,
            vector_id=None,
            paper_id=paper_id,
            paper_title=parsed_paper.paper_title,
            source=source,
            chunk_index=1,
            raw_text="Second original chunk.",
            embedding_text="Paper title: A test paper\nContent:\nSecond original chunk.",
            section="Abstract",
            page_start=1,
            page_end=2,
            raw_token_count=5,
            embedding_token_count=10,
            prev_chunk_id=first_id,
            next_chunk_id=None,
            content_kind="front_matter",
            front_matter_type="abstract",
        ),
    )
    return ChunkingResult(
        parsed_paper=parsed_paper,
        chunks=chunks,
        diagnostics={"chunking.chunker_id": "test_chunker"},
    )


if __name__ == "__main__":
    unittest.main()
