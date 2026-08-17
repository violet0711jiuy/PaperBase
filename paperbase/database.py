"""PaperBase 的 SQLite 元数据层与 Step 3 导入命令。

SQLite 只保存可查询、可追溯的结构化元数据：论文记录、解析出的前置元数据块，以及
结构化分块结果。原始 PDF 继续保留在 ``storage/papers``，向量则留给 Step 5 的 FAISS；
因此这里既不保存 PDF 二进制，也不生成 embedding。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from paperbase.chunking.base import ChunkingResult, PaperChunk
from paperbase.chunking.factory import create_chunker
from paperbase.config import default_config_path, load_settings
from paperbase.parsing.factory import create_parser
from paperbase.parsing.base import SectionRecord


# 版本表用于未来的显式迁移。不要只靠 ``CREATE TABLE IF NOT EXISTS`` 猜测 schema
# 是否兼容：一旦结构变化，必须提供可审计迁移，而不是静默让旧库以未知状态继续运行。
_SCHEMA_VERSION = 5

_SCHEMA_INFO_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SCHEMA_V2_SQL = """

CREATE TABLE IF NOT EXISTS documents (
    paper_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    source_sha256 TEXT NOT NULL UNIQUE,
    paper_title TEXT,
    title_source TEXT NOT NULL,
    parser_id TEXT NOT NULL,
    chunker_id TEXT NOT NULL,
    parse_diagnostics_json TEXT NOT NULL,
    chunking_diagnostics_json TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (length(trim(paper_id)) > 0),
    CHECK (length(source_sha256) = 64)
);

CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    section_title TEXT NOT NULL,
    section_number TEXT,
    section_level INTEGER NOT NULL,
    parent_section_id TEXT,
    section_index INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    FOREIGN KEY (paper_id) REFERENCES documents(paper_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_section_id) REFERENCES sections(section_id) ON DELETE CASCADE,
    UNIQUE (paper_id, section_index),
    CHECK (length(trim(section_id)) > 0),
    CHECK (length(trim(section_title)) > 0),
    CHECK (section_level >= 1),
    CHECK (section_index >= 0),
    CHECK (page_start IS NULL OR page_start >= 1),
    CHECK (page_end IS NULL OR page_end >= page_start)
);

CREATE INDEX IF NOT EXISTS idx_sections_paper_index
    ON sections(paper_id, section_index);
CREATE INDEX IF NOT EXISTS idx_sections_parent
    ON sections(parent_section_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    vector_id INTEGER UNIQUE,
    paper_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    section TEXT,
    section_id TEXT,
    content_kind TEXT NOT NULL,
    front_matter_type TEXT,
    section_type TEXT NOT NULL DEFAULT 'content',
    page_start INTEGER,
    page_end INTEGER,
    raw_token_count INTEGER NOT NULL,
    embedding_token_count INTEGER NOT NULL,
    prev_chunk_id TEXT,
    next_chunk_id TEXT,
    FOREIGN KEY (paper_id) REFERENCES documents(paper_id) ON DELETE CASCADE,
    FOREIGN KEY (section_id) REFERENCES sections(section_id) ON DELETE SET NULL,
    UNIQUE (paper_id, chunk_index),
    CHECK (chunk_index >= 0),
    CHECK (length(trim(raw_text)) > 0),
    CHECK (length(trim(embedding_text)) > 0),
    CHECK (content_kind IN ('body', 'front_matter')),
    CHECK (section_type IN ('content', 'bibliography')),
    CHECK (
        (content_kind = 'body' AND front_matter_type IS NULL)
        OR (content_kind = 'front_matter' AND front_matter_type IS NOT NULL)
    ),
    CHECK (raw_token_count >= 0),
    CHECK (embedding_token_count >= raw_token_count),
    CHECK (page_start IS NULL OR page_start >= 1),
    CHECK (page_end IS NULL OR page_end >= page_start)
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper_section
    ON chunks(paper_id, section, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_front_matter
    ON chunks(paper_id, content_kind, front_matter_type, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_vector_id
    ON chunks(vector_id);
"""

# FTS5 是派生检索索引，不是第二份业务真相：正式原文与元数据仍只以 chunks 表为准。
# 索引文本被 SQLite 保存是 FTS5 提供倒排检索的实现需要，和 FAISS 保存向量的性质相同。
_SCHEMA_V3_SQL = _SCHEMA_V2_SQL.replace(
    "    section_type TEXT NOT NULL DEFAULT 'content',\n", ""
) + """

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paper_title,
    section,
    raw_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_after_insert
AFTER INSERT ON chunks
BEGIN
    INSERT INTO chunks_fts (chunk_id, paper_title, section, raw_text)
    VALUES (
        NEW.chunk_id,
        COALESCE((SELECT paper_title FROM documents WHERE paper_id = NEW.paper_id), ''),
        COALESCE(NEW.section, ''),
        NEW.raw_text
    );
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_after_delete
AFTER DELETE ON chunks
BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.chunk_id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_after_searchable_update
AFTER UPDATE OF paper_id, section, raw_text ON chunks
BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.chunk_id;
    INSERT INTO chunks_fts (chunk_id, paper_title, section, raw_text)
    VALUES (
        NEW.chunk_id,
        COALESCE((SELECT paper_title FROM documents WHERE paper_id = NEW.paper_id), ''),
        COALESCE(NEW.section, ''),
        NEW.raw_text
    );
END;
"""


# V4 将正文和参考文献拆成两个派生 FTS5 索引。chunks 仍是唯一业务事实源：
# content 主索引服务普通问答，bibliography 索引只在明确 citation/reference intent 时查询。
_SCHEMA_V4_SQL = _SCHEMA_V2_SQL + """

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paper_title,
    section,
    raw_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS bibliography_fts USING fts5(
    chunk_id UNINDEXED,
    paper_title,
    section,
    raw_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_after_insert
AFTER INSERT ON chunks WHEN NEW.section_type = 'content'
BEGIN
    INSERT INTO chunks_fts (chunk_id, paper_title, section, raw_text)
    VALUES (NEW.chunk_id, COALESCE((SELECT paper_title FROM documents WHERE paper_id = NEW.paper_id), ''), COALESCE(NEW.section, ''), NEW.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS bibliography_fts_after_insert
AFTER INSERT ON chunks WHEN NEW.section_type = 'bibliography'
BEGIN
    INSERT INTO bibliography_fts (chunk_id, paper_title, section, raw_text)
    VALUES (NEW.chunk_id, COALESCE((SELECT paper_title FROM documents WHERE paper_id = NEW.paper_id), ''), COALESCE(NEW.section, ''), NEW.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_after_delete
AFTER DELETE ON chunks
BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.chunk_id;
    DELETE FROM bibliography_fts WHERE chunk_id = OLD.chunk_id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_after_searchable_update
AFTER UPDATE OF paper_id, section, raw_text, section_type ON chunks
BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.chunk_id;
    DELETE FROM bibliography_fts WHERE chunk_id = OLD.chunk_id;
    INSERT INTO chunks_fts (chunk_id, paper_title, section, raw_text)
    SELECT NEW.chunk_id, COALESCE((SELECT paper_title FROM documents WHERE paper_id = NEW.paper_id), ''), COALESCE(NEW.section, ''), NEW.raw_text
    WHERE NEW.section_type = 'content';
    INSERT INTO bibliography_fts (chunk_id, paper_title, section, raw_text)
    SELECT NEW.chunk_id, COALESCE((SELECT paper_title FROM documents WHERE paper_id = NEW.paper_id), ''), COALESCE(NEW.section, ''), NEW.raw_text
    WHERE NEW.section_type = 'bibliography';
END;
"""


class MetadataDatabaseError(RuntimeError):
    """元数据数据库的 schema、导入或一致性检查失败时抛出。"""


@dataclass(frozen=True)
class ImportSummary:
    """一次论文元数据导入的可展示摘要。"""

    paper_id: str
    document_replaced: bool
    front_matter_chunk_count: int
    chunk_count: int


@dataclass(frozen=True)
class VectorIndexState:
    """SQLite 中正式向量映射的只读摘要，供 Step 5 的建库前后校验使用。"""

    total_chunk_count: int
    vectorized_chunk_count: int
    min_vector_id: int | None
    max_vector_id: int | None


class MetadataDatabase:
    """SQLite 的最小数据访问层。

    每次操作都新建一个短连接，避免将 SQLite connection 跨线程传给未来 Streamlit
    会话。``import_chunking_result`` 使用单个事务：若中途任意一行失败，旧数据保持
    原样，不会留下“只导入一半 chunks”的论文。
    """

    def __init__(self, path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        self._path = path.resolve()
        self._busy_timeout_ms = busy_timeout_ms

    @property
    def path(self) -> Path:
        """返回 SQLite 文件的绝对路径，供 CLI 输出和人工检查使用。"""
        return self._path

    def initialize(self) -> None:
        """创建 schema，或验证既有数据库版本与当前代码兼容。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # 先只创建版本表，不能一上来执行 V3 的 ``CREATE TABLE IF NOT EXISTS``：
            # 对于 V1 的 chunks 表，它不会自动补齐新列，反而会掩盖需要迁移的事实。
            connection.executescript(_SCHEMA_INFO_SQL)
            stored_version = connection.execute(
                "SELECT value FROM schema_info WHERE key = 'schema_version'"
            ).fetchone()
            if stored_version is None:
                connection.executescript(_SCHEMA_V4_SQL)
                connection.execute(
                    "INSERT INTO schema_info(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                return
            if stored_version["value"] == "1":
                _migrate_v1_to_v2(connection)
                connection.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    ("2",),
                )
                stored_version = {"value": "2"}
            if stored_version["value"] == "2":
                _migrate_v2_to_v3(connection)
                connection.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    ("3",),
                )
                stored_version = {"value": "3"}
            if stored_version["value"] == "3":
                _migrate_v3_to_v4(connection)
                connection.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    ("4",),
                )
                stored_version = {"value": "4"}
            if stored_version["value"] == "4":
                _migrate_v4_to_v5(connection)
                connection.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    (str(_SCHEMA_VERSION),),
                )
                stored_version = {"value": str(_SCHEMA_VERSION)}
            if stored_version["value"] == str(_SCHEMA_VERSION):
                # 为已是当前版本的数据库补齐可安全重复创建的表、索引与 FTS trigger。
                connection.executescript(_SCHEMA_V4_SQL)
                return
            if stored_version["value"] != str(_SCHEMA_VERSION):
                raise MetadataDatabaseError(
                    "Unsupported SQLite schema version "
                    f"{stored_version['value']!r}; expected {_SCHEMA_VERSION}. "
                    "Add an explicit migration before opening this database."
                )

    def import_chunking_result(self, result: ChunkingResult) -> ImportSummary:
        """原子性写入一篇论文及其全部可检索 chunks。

        同一 PDF 再次导入时，根据稳定的 ``paper_id`` 先删除该论文的旧元数据，再写入
        新的一整套记录。这样解析或分块配置更新后不会残留旧 chunk，也不会产生重复。
        Step 5 写入 vector_id 后禁止走此“替换”路径，避免在无意间使 FAISS 与 SQLite
        脱节；届时将由增量索引服务使用专门接口。
        """
        self.initialize()
        paper_id, source_sha256 = _validate_chunking_result(result)
        parsed_paper = result.parsed_paper
        chunker_id = _required_diagnostic_string(
            result.diagnostics, "chunking.chunker_id"
        )
        now = _utc_timestamp()

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM documents WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            vectorized_count = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks "
                "WHERE paper_id = ? AND vector_id IS NOT NULL",
                (paper_id,),
            ).fetchone()["count"]
            if vectorized_count:
                raise MetadataDatabaseError(
                    f"Cannot replace vectorized document {paper_id}. "
                    "Use the future incremental indexing workflow instead."
                )

            # 删除 documents 会通过外键级联删除其 chunks。随后全部重写，
            # 保证 chunk_index、前后邻居与解析版本始终来自同一次 parse -> chunk 结果。
            connection.execute("DELETE FROM documents WHERE paper_id = ?", (paper_id,))
            connection.execute(
                """
                INSERT INTO documents (
                    paper_id, source_path, source_filename, source_sha256,
                    paper_title, title_source, parser_id, chunker_id,
                    parse_diagnostics_json, chunking_diagnostics_json,
                    ingested_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    str(parsed_paper.source),
                    parsed_paper.source.name,
                    source_sha256,
                    parsed_paper.paper_title,
                    parsed_paper.title_source,
                    parsed_paper.parser_id,
                    chunker_id,
                    _stable_json(parsed_paper.diagnostics),
                    _stable_json(result.diagnostics),
                    now,
                    now,
                ),
            )
            # 父节点即使没有任何直属 chunk，也作为独立记录先写入；随后 chunks 的
            # section_id 外键才能在同一个事务中可靠关联到它。
            connection.executemany(
                """
                INSERT INTO sections (
                    section_id, paper_id, section_title, section_number,
                    section_level, parent_section_id, section_index, page_start, page_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                # 即使调用方传入的 tuple 未排序，也始终先写父节点、再写子节点，满足自引用 FK。
                (
                    _section_row(section)
                    for section in sorted(
                        parsed_paper.sections, key=lambda section: section.section_index
                    )
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, vector_id, paper_id, chunk_index,
                    raw_text, embedding_text, section, content_kind, front_matter_type, section_type,
                    page_start, page_end,
                    raw_token_count, embedding_token_count, prev_chunk_id, next_chunk_id, section_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (_chunk_row(chunk) for chunk in result.chunks),
            )

        return ImportSummary(
            paper_id=paper_id,
            document_replaced=existing is not None,
            front_matter_chunk_count=sum(
                chunk.content_kind == "front_matter" for chunk in result.chunks
            ),
            chunk_count=len(result.chunks),
        )

    def row_counts(self) -> dict[str, int]:
        """返回核心表的行数，专用于 Step 3 验收和后续健康检查。"""
        self.initialize()
        with self._connect() as connection:
            return {
                table: connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table}"
                ).fetchone()["count"]
                for table in ("documents", "chunks")
            }

    def get_document(self, paper_id: str) -> sqlite3.Row | None:
        """按稳定 paper_id 查询一篇论文，不做语义检索。"""
        self.initialize()
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM documents WHERE paper_id = ?", (paper_id,)
            ).fetchone()

    def list_sections(self, paper_id: str) -> tuple[sqlite3.Row, ...]:
        """按原始 heading 阅读顺序返回一篇论文的完整 Section Tree 节点。"""
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    "SELECT * FROM sections WHERE paper_id = ? ORDER BY section_index",
                    (paper_id,),
                ).fetchall()
            )

    def get_section(self, section_id: str) -> sqlite3.Row | None:
        """按稳定 section_id 查询单个 Section，供后续 Explain Section 读取。"""
        self.initialize()
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM sections WHERE section_id = ?", (section_id,)
            ).fetchone()

    def list_front_matter_chunks(self, paper_id: str) -> tuple[sqlite3.Row, ...]:
        """返回前置元数据类型的可检索 chunks，供结构化展示或后续召回回查使用。"""
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    "SELECT * FROM chunks WHERE paper_id = ? "
                    "AND content_kind = 'front_matter' ORDER BY chunk_index",
                    (paper_id,),
                ).fetchall()
            )

    def list_chunks(self, paper_id: str) -> tuple[sqlite3.Row, ...]:
        """按 chunk_index 返回原始 chunk，供后续 Citation 与邻居扩展复用。"""
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    "SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index",
                    (paper_id,),
                ).fetchall()
            )

    def list_content_chunks_in_section(
        self,
        *,
        paper_id: str,
        section: str,
    ) -> tuple[sqlite3.Row, ...]:
        """返回同一论文、同一章节的正文块，供 Step 8 安全扩展相邻上下文。

        该查询从 ``chunks`` 真相表读取，不依赖 FAISS 或 FTS5。``section_type``
        的过滤确保 References/Bibliography 永远不会被当作正文邻居拼入回答证据。
        """
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT chunks.*, documents.paper_title
                    FROM chunks
                    INNER JOIN documents ON documents.paper_id = chunks.paper_id
                    WHERE chunks.paper_id = ?
                      AND chunks.section = ?
                      AND chunks.section_type = 'content'
                    ORDER BY chunks.chunk_index
                    """,
                    (paper_id, section),
                ).fetchall()
            )

    def list_embedding_inputs(self) -> tuple[sqlite3.Row, ...]:
        """按稳定顺序返回 Step 4 所需的文档 embedding 输入快照。

        本方法只读取当前正式知识库的 ``embedding_text``，不重跑 Parse/Chunk，也不写入向量。
        Step 5 一旦分配过 ``vector_id``，说明它已与 FAISS 建立映射；此时拒绝重新生成 staging
        工件，避免使用者误把新向量与旧索引混在一起。
        """
        self.initialize()
        with self._connect() as connection:
            vectorized_count = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE vector_id IS NOT NULL"
            ).fetchone()["count"]
            if vectorized_count:
                raise MetadataDatabaseError(
                    "Cannot regenerate the Step 4 artifact after vector_id assignment. "
                    "Use the future index rebuild workflow instead."
                )
            rows = tuple(
                connection.execute(
                    """
                    SELECT chunk_id, paper_id, chunk_index, embedding_text
                    FROM chunks
                    WHERE section_type = 'content'
                    ORDER BY paper_id, chunk_index
                    """
                ).fetchall()
            )
        if not rows:
            raise MetadataDatabaseError("Cannot generate embeddings because SQLite has no chunks.")
        return rows

    def vector_index_state(self) -> VectorIndexState:
        """读取 vector_id 分配状态，不读取或加载任何 FAISS 文件。"""
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total_chunk_count,
                       SUM(vector_id IS NOT NULL) AS vectorized_chunk_count,
                       MIN(vector_id) AS min_vector_id,
                       MAX(vector_id) AS max_vector_id
                FROM chunks
                WHERE section_type = 'content'
                """
            ).fetchone()
        return VectorIndexState(
            total_chunk_count=int(row["total_chunk_count"]),
            vectorized_chunk_count=int(row["vectorized_chunk_count"] or 0),
            min_vector_id=row["min_vector_id"],
            max_vector_id=row["max_vector_id"],
        )

    def validate_embedding_artifact_records(
        self,
        records: Sequence[tuple[str, str]],
    ) -> None:
        """确认 Step 4 records 仍与当前全部 SQLite chunks 完全一致。

        records 中的元组依次为 ``(chunk_id, embedding_text_sha256)``。此方法既检查 chunk
        是否被增删，也检查生成向量后 embedding_text 是否被修改；因此不能把旧 staging 工件
        误用于新版本的知识库。
        """
        self.initialize()
        with self._connect() as connection:
            _validate_embedding_records_in_connection(connection, records)

    def assign_vector_ids(
        self,
        assignments: Sequence[tuple[str, int, str]],
    ) -> None:
        """在一个 SQLite 事务中，把已验证的全局 vector_id 回写到全部 chunks。

        每个 assignment 为 ``(chunk_id, vector_id, embedding_text_sha256)``。该接口仅服务
        首次全库建索引：要求 records 覆盖当前所有 chunks，且当前没有任何 vector_id。FAISS
        索引文件的发布由 Step 5 日志协调；本方法自身绝不触碰索引文件。
        """
        if not assignments:
            raise MetadataDatabaseError("Cannot assign vector IDs to an empty chunk set.")
        chunk_ids = [chunk_id for chunk_id, _, _ in assignments]
        vector_ids = [vector_id for _, vector_id, _ in assignments]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise MetadataDatabaseError("Vector ID assignments contain duplicate chunk_id values.")
        if len(set(vector_ids)) != len(vector_ids) or min(vector_ids) < 1:
            raise MetadataDatabaseError(
                "Vector ID assignments must contain unique positive integer IDs."
            )

        self.initialize()
        with self._connect() as connection:
            existing_vectorized = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE vector_id IS NOT NULL "
                "AND section_type = 'content'"
            ).fetchone()["count"]
            if existing_vectorized:
                raise MetadataDatabaseError(
                    "Cannot assign initial vector IDs because SQLite already has vectorized chunks."
                )
            _validate_embedding_records_in_connection(
                connection,
                [(chunk_id, text_sha256) for chunk_id, _, text_sha256 in assignments],
            )
            connection.executemany(
                "UPDATE chunks SET vector_id = ? WHERE chunk_id = ?",
                ((vector_id, chunk_id) for chunk_id, vector_id, _ in assignments),
            )
            updated_count = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE vector_id IS NOT NULL "
                "AND section_type = 'content'"
            ).fetchone()["count"]
            if updated_count != len(assignments):
                raise MetadataDatabaseError(
                    "SQLite vector ID update count does not match the requested assignment set."
                )

    def vector_id_mapping(self) -> tuple[sqlite3.Row, ...]:
        """返回 ``vector_id -> chunk_id`` 映射，供索引发布后做双向一致性校验。"""
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT vector_id, chunk_id
                    FROM chunks
                    WHERE vector_id IS NOT NULL AND section_type = 'content'
                    ORDER BY vector_id
                    """
                ).fetchall()
            )

    def vector_id_assignments(self) -> tuple[sqlite3.Row, ...]:
        """返回正式索引验证所需的 ``vector_id、chunk_id、embedding_text`` 快照。"""
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT vector_id, chunk_id, embedding_text
                    FROM chunks
                    WHERE vector_id IS NOT NULL AND section_type = 'content'
                    ORDER BY vector_id
                    """
                ).fetchall()
            )

    def reset_vector_ids_for_rebuild(self) -> int:
        """显式清空派生向量映射，为规则变更后的完整 Step 4/5 重建做准备。

        不删除 documents、chunks、原始 PDF 或任一 FTS5 索引。旧 FAISS 文件会暂时失效，
        调用方必须重新生成正文 embedding 并发布新索引，防止旧 References 向量继续参与召回。
        """
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE vector_id IS NOT NULL"
            ).fetchone()
            connection.execute("UPDATE chunks SET vector_id = NULL WHERE vector_id IS NOT NULL")
        return int(row["count"])

    def search_bm25(self, query: str, *, top_k: int) -> tuple[sqlite3.Row, ...]:
        """使用 SQLite FTS5 对论文标题、章节和 chunk 原文执行词法召回。

        ``query`` 被转成一个字面短语，而非直接拼接到 FTS5 语法中。这样用户问题中的括号、
        连字符、引号与公式符号不会变成 SQLite 查询运算符；中文连续文本也能作为相邻词元
        短语参与匹配。返回的 ``bm25_score`` 数值仅用于调试，跨通道融合必须使用排名而非它。
        """
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise MetadataDatabaseError("BM25 query must not be empty.")
        if top_k < 1:
            raise MetadataDatabaseError("BM25 top_k must be positive.")
        fts_query = _fts5_literal_phrase(normalized_query)
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT
                        chunks.*,
                        documents.paper_title AS paper_title,
                        bm25(chunks_fts, 1.2, 0.8, 1.0) AS bm25_score
                    FROM chunks_fts
                    JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
                    JOIN documents ON documents.paper_id = chunks.paper_id
                    WHERE chunks_fts MATCH ? AND chunks.section_type = 'content'
                    ORDER BY bm25_score ASC, chunks.chunk_id ASC
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                ).fetchall()
            )

    def search_bm25_keyword_group(
        self,
        keywords: Sequence[str],
        *,
        top_k: int,
    ) -> tuple[sqlite3.Row, ...]:
        """用一组英文关键词执行**一条** BM25 查询，并以 OR 保留任一术语的候选。

        关键词组逻辑上是单条 ``rewritten_bm25`` 路径，不能拆成多次检索再在 RRF 中
        重复加分。SQL 中使用的是安全编译后的 FTS5 表达式，例如
        ``"LSTM" OR "wind speed prediction" OR "author"``：多词术语会作为短语匹配，
        任一术语命中即可进入候选，后续由 BM25 排名和 RRF 融合决定优先级。
        """
        normalized_keywords = tuple(
            dict.fromkeys(" ".join(keyword.split()) for keyword in keywords if keyword.strip())
        )
        if not normalized_keywords:
            return ()
        if top_k < 1:
            raise MetadataDatabaseError("BM25 top_k must be positive.")
        fts_query = " OR ".join(_fts5_literal_phrase(keyword) for keyword in normalized_keywords)
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT
                        chunks.*,
                        documents.paper_title AS paper_title,
                        bm25(chunks_fts, 1.2, 0.8, 1.0) AS bm25_score
                    FROM chunks_fts
                    JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
                    JOIN documents ON documents.paper_id = chunks.paper_id
                    WHERE chunks_fts MATCH ? AND chunks.section_type = 'content'
                    ORDER BY bm25_score ASC, chunks.chunk_id ASC
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                ).fetchall()
            )

    def chunks_by_vector_ids(self, vector_ids: Sequence[int]) -> tuple[sqlite3.Row, ...]:
        """按 FAISS 返回的 vector_id 批量回查完整 chunk 证据，不假设 SQLite 返回顺序。"""
        unique_ids = tuple(dict.fromkeys(vector_ids))
        if not unique_ids:
            return ()
        if any(vector_id < 1 for vector_id in unique_ids):
            raise MetadataDatabaseError("FAISS vector IDs must be positive integers.")
        placeholders = ", ".join("?" for _ in unique_ids)
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    f"""
                    SELECT chunks.*, documents.paper_title AS paper_title
                    FROM chunks
                    JOIN documents ON documents.paper_id = chunks.paper_id
                    WHERE chunks.vector_id IN ({placeholders})
                      AND chunks.section_type = 'content'
                    """,
                    unique_ids,
                ).fetchall()
            )

    def search_bibliography(
        self,
        query: str,
        *,
        paper_id: str,
        top_k: int,
    ) -> tuple[sqlite3.Row, ...]:
        """在指定论文内部检索参考文献 FTS5，避免宽泛主题词命中其他论文的引用条目。"""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise MetadataDatabaseError("Bibliography query must not be empty.")
        if top_k < 1:
            raise MetadataDatabaseError("Bibliography top_k must be positive.")
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT chunks.*, documents.paper_title AS paper_title,
                           bm25(bibliography_fts, 1.2, 0.8, 1.0) AS bm25_score
                    FROM bibliography_fts
                    JOIN chunks ON chunks.chunk_id = bibliography_fts.chunk_id
                    JOIN documents ON documents.paper_id = chunks.paper_id
                    WHERE bibliography_fts MATCH ?
                      AND chunks.section_type = 'bibliography'
                      AND chunks.paper_id = ?
                    ORDER BY bm25_score ASC, chunks.chunk_id ASC
                    LIMIT ?
                    """,
                    (_fts5_literal_phrase(normalized_query), paper_id, top_k),
                ).fetchall()
            )

    def search_bibliography_keyword_group(
        self,
        keywords: Sequence[str],
        *,
        paper_id: str,
        top_k: int,
    ) -> tuple[sqlite3.Row, ...]:
        """以英文关键词组查询指定论文的 bibliography FTS5，通常比完整问句更适合引用条目。"""
        normalized = tuple(
            dict.fromkeys(" ".join(keyword.split()) for keyword in keywords if keyword.strip())
        )
        if not normalized:
            return ()
        if top_k < 1:
            raise MetadataDatabaseError("Bibliography top_k must be positive.")
        fts_query = " OR ".join(_fts5_literal_phrase(keyword) for keyword in normalized)
        self.initialize()
        with self._connect() as connection:
            return tuple(
                connection.execute(
                    """
                    SELECT chunks.*, documents.paper_title AS paper_title,
                           bm25(bibliography_fts, 1.2, 0.8, 1.0) AS bm25_score
                    FROM bibliography_fts
                    JOIN chunks ON chunks.chunk_id = bibliography_fts.chunk_id
                    JOIN documents ON documents.paper_id = chunks.paper_id
                    WHERE bibliography_fts MATCH ?
                      AND chunks.section_type = 'bibliography'
                      AND chunks.paper_id = ?
                    ORDER BY bm25_score ASC, chunks.chunk_id ASC
                    LIMIT ?
                    """,
                    (fts_query, paper_id, top_k),
                ).fetchall()
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """建立启用外键和事务语义的短生命周期 SQLite 连接。"""
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _validate_embedding_records_in_connection(
    connection: sqlite3.Connection,
    records: Sequence[tuple[str, str]],
) -> None:
    """在既有 SQLite 连接中校验 Step 4 records 覆盖全库且输入文本未变化。"""
    if not records:
        raise MetadataDatabaseError("Embedding artifact contains no chunk records.")
    record_ids = [chunk_id for chunk_id, _ in records]
    if len(set(record_ids)) != len(record_ids):
        raise MetadataDatabaseError("Embedding artifact contains duplicate chunk_id records.")

    # Step 4 工件只覆盖正文 content；bibliography 仍留在 SQLite/FTS5，但不参与主 FAISS。
    database_rows = connection.execute(
        "SELECT chunk_id, embedding_text FROM chunks WHERE section_type = 'content'"
    ).fetchall()
    database_text_by_chunk_id = {
        row["chunk_id"]: row["embedding_text"] for row in database_rows
    }
    if set(record_ids) != set(database_text_by_chunk_id):
        raise MetadataDatabaseError(
            "Embedding artifact chunk IDs do not exactly match the current SQLite chunk set."
        )
    for chunk_id, expected_text_sha256 in records:
        actual_text_sha256 = hashlib.sha256(
            database_text_by_chunk_id[chunk_id].encode("utf-8")
        ).hexdigest()
        if actual_text_sha256 != expected_text_sha256:
            raise MetadataDatabaseError(
                "Embedding artifact is stale because embedding_text changed for "
                f"{chunk_id}."
            )


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """为既有 chunks 建立 FTS5/BM25 派生索引，并一次性回填当前全部记录。"""
    connection.executescript(_SCHEMA_V3_SQL)
    # 旧 V2 库此前没有 trigger，因此先清空再从 documents/chunks 的正式真相重建。
    # 该函数运行在 initialize 的同一个事务中，任何失败都会保留迁移前的 V2 数据库。
    connection.execute("DELETE FROM chunks_fts")
    connection.execute(
        """
        INSERT INTO chunks_fts (chunk_id, paper_title, section, raw_text)
        SELECT chunks.chunk_id,
               COALESCE(documents.paper_title, ''),
               COALESCE(chunks.section, ''),
               chunks.raw_text
        FROM chunks
        JOIN documents ON documents.paper_id = chunks.paper_id
        ORDER BY chunks.paper_id, chunks.chunk_index
        """
    )


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """把既有 V3 chunks 分类后重建两个派生 FTS5 索引，不删除原始 PDF 或 chunks。"""
    connection.execute(
        "ALTER TABLE chunks ADD COLUMN section_type TEXT NOT NULL DEFAULT 'content' "
        "CHECK (section_type IN ('content', 'bibliography'))"
    )
    # 只依据已存 section 的末级标题分类；不扫描 raw_text，因此正文引用 References 不会误判。
    for row in connection.execute("SELECT chunk_id, section FROM chunks"):
        if _is_bibliography_section(str(row["section"] or "")):
            connection.execute(
                "UPDATE chunks SET section_type = 'bibliography' WHERE chunk_id = ?",
                (row["chunk_id"],),
            )
    # 旧 V3 trigger 只维护 chunks_fts，先删除再按 V4 规则重建两个索引。
    for trigger in (
        "chunks_fts_after_insert",
        "chunks_fts_after_delete",
        "chunks_fts_after_searchable_update",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    connection.executescript(_SCHEMA_V4_SQL)
    connection.execute("DELETE FROM chunks_fts")
    connection.execute("DELETE FROM bibliography_fts")
    connection.execute(
        """
        INSERT INTO chunks_fts (chunk_id, paper_title, section, raw_text)
        SELECT chunks.chunk_id, COALESCE(documents.paper_title, ''), COALESCE(chunks.section, ''), chunks.raw_text
        FROM chunks JOIN documents ON documents.paper_id = chunks.paper_id
        WHERE chunks.section_type = 'content'
        ORDER BY chunks.paper_id, chunks.chunk_index
        """
    )
    connection.execute(
        """
        INSERT INTO bibliography_fts (chunk_id, paper_title, section, raw_text)
        SELECT chunks.chunk_id, COALESCE(documents.paper_title, ''), COALESCE(chunks.section, ''), chunks.raw_text
        FROM chunks JOIN documents ON documents.paper_id = chunks.paper_id
        WHERE chunks.section_type = 'bibliography'
        ORDER BY chunks.paper_id, chunks.chunk_index
        """
    )


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """增加 Section Tree 表及 chunks.section_id，不回填既有论文结构。"""
    # 先创建父表，随后为旧 chunks 增加可空外键列；SQLite 会把已有的 265 条历史
    # chunk 自动保留为 NULL。这里绝不重新解析 PDF，也不触碰 FTS5/FAISS 派生工件。
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sections (
            section_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            section_title TEXT NOT NULL,
            section_number TEXT,
            section_level INTEGER NOT NULL,
            parent_section_id TEXT,
            section_index INTEGER NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            FOREIGN KEY (paper_id) REFERENCES documents(paper_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_section_id) REFERENCES sections(section_id) ON DELETE CASCADE,
            UNIQUE (paper_id, section_index),
            CHECK (length(trim(section_id)) > 0),
            CHECK (length(trim(section_title)) > 0),
            CHECK (section_level >= 1),
            CHECK (section_index >= 0),
            CHECK (page_start IS NULL OR page_start >= 1),
            CHECK (page_end IS NULL OR page_end >= page_start)
        );
        CREATE INDEX IF NOT EXISTS idx_sections_paper_index
            ON sections(paper_id, section_index);
        CREATE INDEX IF NOT EXISTS idx_sections_parent
            ON sections(parent_section_id);
        """
    )
    connection.execute(
        "ALTER TABLE chunks ADD COLUMN section_id TEXT "
        "REFERENCES sections(section_id) ON DELETE SET NULL"
    )


def _is_bibliography_section(section: str) -> bool:
    """为数据库迁移复用切块阶段相同的受控标题规则。"""
    leaf = section.rsplit(">", maxsplit=1)[-1].strip()
    leaf = re.sub(
        r"^(?:\d+(?:\.\d+)*(?:[.)]|\s+)|[IVXLC]+(?:[.)]|\s+))",
        "",
        leaf,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", leaf).strip().casefold().rstrip(":") in {
        "references",
        "bibliography",
        "works cited",
        "literature cited",
    }


def _fts5_literal_phrase(text: str) -> str:
    """将外部输入转为 FTS5 字面短语，避免用户文本意外触发布尔/列过滤语法。"""
    return '"' + text.replace('"', '""') + '"'


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """把 V1 的重复 ``front_matter`` 正文收敛到带类型标记的 ``chunks``。

    V1 已经把同一段前置文本同时写入两个表。迁移先用其标准 section 和页码范围给已有
    chunk 回填类型，再删除冗余表；所以不会丢失任何可供检索、引用或 embedding 的正文。
    调用方在单个 SQLite 事务中执行本函数，任一步失败都会完整回滚。
    """
    # SQLite 的 ADD COLUMN 不能同时添加两个列，因此先添加被引用的可空列，再添加类型列。
    connection.execute("ALTER TABLE chunks ADD COLUMN front_matter_type TEXT")
    connection.execute(
        "ALTER TABLE chunks ADD COLUMN content_kind TEXT NOT NULL "
        "DEFAULT 'body' CHECK (content_kind IN ('body', 'front_matter'))"
    )

    front_matter_by_paper: dict[str, list[sqlite3.Row]] = {}
    for row in connection.execute(
        "SELECT paper_id, block_type, canonical_section, page_start, page_end "
        "FROM front_matter ORDER BY paper_id, block_index"
    ):
        front_matter_by_paper.setdefault(row["paper_id"], []).append(row)

    # V1 的 chunks 正文已经是唯一正确的原文来源；这里只回填查询/检索所需的轻量类型字段。
    for chunk in connection.execute(
        "SELECT chunk_id, paper_id, section, page_start, page_end FROM chunks"
    ):
        block_type = _front_matter_type_for_stored_chunk(
            section=chunk["section"],
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            front_matter_blocks=front_matter_by_paper.get(chunk["paper_id"], []),
        )
        if block_type is not None:
            connection.execute(
                "UPDATE chunks SET content_kind = 'front_matter', "
                "front_matter_type = ? WHERE chunk_id = ?",
                (block_type, chunk["chunk_id"]),
            )

    # text 已完整存在 chunks.raw_text / chunks.embedding_text，移除第二份全文副本。
    connection.execute("DROP TABLE front_matter")
    connection.executescript(_SCHEMA_V2_SQL)


def _front_matter_type_for_stored_chunk(
    *,
    section: str | None,
    page_start: int | None,
    page_end: int | None,
    front_matter_blocks: list[sqlite3.Row],
) -> str | None:
    """用与切块阶段相同的“标题末级 + 页码相交”规则迁移历史数据库。"""
    if section is None:
        return None
    section_leaf = _normalized_section_leaf(section)
    for block in front_matter_blocks:
        if section_leaf != _normalized_section_leaf(block["canonical_section"]):
            continue
        if _page_ranges_overlap(
            page_start,
            page_end,
            block["page_start"],
            block["page_end"],
        ):
            return block["block_type"]
    return None


def _normalized_section_leaf(section: str) -> str:
    """规整 section 的末级标题，兼容 ``A B S T R A C T`` 这种字距化输出。"""
    normalized = re.sub(r"\s+", " ", section.rsplit(">", maxsplit=1)[-1]).strip().casefold()
    pieces = normalized.split()
    if pieces and all(len(piece) == 1 and piece.isalpha() for piece in pieces):
        return "".join(pieces)
    return normalized


def _page_ranges_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> bool:
    """缺失 provenance 时保留已知的语义标题匹配，而不凭空把它判为不相交。"""
    if None in (left_start, left_end, right_start, right_end):
        return True
    return max(left_start, right_start) <= min(left_end, right_end)


def _validate_chunking_result(result: ChunkingResult) -> tuple[str, str]:
    """在开启事务前验证 PaperChunk 批次的身份与链路完整性。"""
    chunks = result.chunks
    if not chunks:
        raise MetadataDatabaseError("Cannot import a paper without chunks.")

    source = result.parsed_paper.source.resolve()
    if not source.is_file():
        raise MetadataDatabaseError(f"Source PDF not found: {source}")
    source_sha256 = _file_sha256(source)
    expected_paper_id = f"paper_{source_sha256[:16]}"
    paper_ids = {chunk.paper_id for chunk in chunks}
    if paper_ids != {expected_paper_id}:
        raise MetadataDatabaseError(
            "Chunk paper_id does not match source PDF hash: "
            f"expected {expected_paper_id}, got {sorted(paper_ids)}."
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_indexes = [chunk.chunk_index for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise MetadataDatabaseError("Chunk IDs must be unique within one import.")
    if sorted(chunk_indexes) != list(range(len(chunks))):
        raise MetadataDatabaseError(
            "Chunk indexes must be a contiguous 0..N-1 sequence before import."
        )

    _validate_sections(result.parsed_paper.sections, paper_id=expected_paper_id)
    known_section_ids = {section.section_id for section in result.parsed_paper.sections}
    known_chunk_ids = set(chunk_ids)
    for chunk in chunks:
        if chunk.content_kind not in {"body", "front_matter"}:
            raise MetadataDatabaseError(
                f"Unsupported content_kind for {chunk.chunk_id}: {chunk.content_kind!r}"
            )
        if (chunk.content_kind == "front_matter") != bool(chunk.front_matter_type):
            raise MetadataDatabaseError(
                "front_matter_type must be present exactly for front_matter chunks: "
                f"{chunk.chunk_id}"
            )
        if chunk.section_type not in {"content", "bibliography"}:
            raise MetadataDatabaseError(
                f"Unsupported section_type for {chunk.chunk_id}: {chunk.section_type!r}"
            )
        if chunk.section_id is not None and chunk.section_id not in known_section_ids:
            raise MetadataDatabaseError(
                f"Unknown section_id for {chunk.chunk_id}: {chunk.section_id}"
            )
        if chunk.prev_chunk_id is not None and chunk.prev_chunk_id not in known_chunk_ids:
            raise MetadataDatabaseError(
                f"Unknown prev_chunk_id for {chunk.chunk_id}: {chunk.prev_chunk_id}"
            )
        if chunk.next_chunk_id is not None and chunk.next_chunk_id not in known_chunk_ids:
            raise MetadataDatabaseError(
                f"Unknown next_chunk_id for {chunk.chunk_id}: {chunk.next_chunk_id}"
            )
    return expected_paper_id, source_sha256


def _validate_sections(
    sections: tuple[SectionRecord, ...], *, paper_id: str
) -> None:
    """在写入前校验 Section Tree 的身份、父子引用与阅读顺序，不猜测或修复数据。"""
    section_ids = [section.section_id for section in sections]
    section_indexes = [section.section_index for section in sections]
    if len(set(section_ids)) != len(section_ids):
        raise MetadataDatabaseError("Section IDs must be unique within one import.")
    if len(set(section_indexes)) != len(section_indexes):
        raise MetadataDatabaseError("Section indexes must be unique within one import.")
    by_id = {section.section_id: section for section in sections}
    for section in sections:
        if section.paper_id != paper_id:
            raise MetadataDatabaseError(
                f"Section paper_id mismatch for {section.section_id}: {section.paper_id!r}"
            )
        if section.section_level < 1:
            raise MetadataDatabaseError(
                f"Section level must be positive for {section.section_id}."
            )
        if section.page_start is not None and section.page_end is not None:
            if section.page_end < section.page_start:
                raise MetadataDatabaseError(
                    f"Section page range is invalid for {section.section_id}."
                )
        parent_id = section.parent_section_id
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            raise MetadataDatabaseError(
                f"Unknown parent_section_id for {section.section_id}: {parent_id}"
            )
        if parent.section_index >= section.section_index:
            raise MetadataDatabaseError(
                f"Parent Section must precede child {section.section_id}."
            )


def _chunk_row(chunk: PaperChunk) -> tuple[object, ...]:
    """将不可变 PaperChunk 转为 SQLite 参数元组，保持字段映射单点维护。"""
    return (
        chunk.chunk_id,
        chunk.vector_id,
        chunk.paper_id,
        chunk.chunk_index,
        chunk.raw_text,
        chunk.embedding_text,
        chunk.section,
        chunk.content_kind,
        chunk.front_matter_type,
        chunk.section_type,
        chunk.page_start,
        chunk.page_end,
        chunk.raw_token_count,
        chunk.embedding_token_count,
        chunk.prev_chunk_id,
        chunk.next_chunk_id,
        chunk.section_id,
    )


def _section_row(section: SectionRecord) -> tuple[object, ...]:
    """将解析器无关的 SectionRecord 转为 SQLite 参数元组，字段顺序只维护一处。"""
    return (
        section.section_id,
        section.paper_id,
        section.section_title,
        section.section_number,
        section.section_level,
        section.parent_section_id,
        section.section_index,
        section.page_start,
        section.page_end,
    )


def _required_diagnostic_string(diagnostics: dict[str, int | str], key: str) -> str:
    """读取必需的运行标识，缺失时拒绝写入无法追溯的文档记录。"""
    value = diagnostics.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MetadataDatabaseError(f"Missing required chunking diagnostic: {key}")
    return value


def _stable_json(value: object) -> str:
    """以稳定键顺序保存诊断，便于不同导入批次的 diff 与人工审计。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    """流式计算 PDF 完整哈希，不把可能较大的论文一次性复制进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    """返回带时区的 UTC 时间戳，避免本机时区变化造成导入顺序歧义。"""
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    """运行 ``PDF -> Parse -> Chunk -> SQLite metadata`` 的 Step 3 导入流程。"""
    argument_parser = argparse.ArgumentParser(
        description="Import PaperBase parsed metadata and chunks into SQLite."
    )
    argument_parser.add_argument("pdfs", nargs="*", type=Path)
    argument_parser.add_argument("--input-dir", type=Path)
    argument_parser.add_argument(
        "--config", type=Path, default=default_config_path()
    )
    args = argument_parser.parse_args()

    settings = load_settings(args.config)
    sources = args.pdfs
    if args.input_dir:
        if args.pdfs:
            argument_parser.error("Use either PDF paths or --input-dir, not both.")
        sources = sorted(args.input_dir.glob("*.pdf"))
    if not sources:
        sources = sorted(settings.storage.papers_dir.glob("*.pdf"))
    if not sources:
        argument_parser.error("No PDF files found to import into SQLite.")

    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    paper_parser = create_parser(settings)
    paper_chunker = create_chunker(settings)
    for source in sources:
        # 与 Step 2 一样直接交接内存中的结构结果，避免从 Markdown 反推标题、页码或表格。
        parsed_paper = paper_parser.parse(source)
        chunking_result = paper_chunker.chunk(parsed_paper)
        summary = database.import_chunking_result(chunking_result)
        action = "replaced" if summary.document_replaced else "inserted"
        print(
            f"{source.name}: {action} {summary.paper_id}, "
            f"front_matter_chunks={summary.front_matter_chunk_count}, "
            f"chunks={summary.chunk_count}, database={database.path}"
        )


if __name__ == "__main__":
    main()
