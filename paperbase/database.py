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


# 版本表用于未来的显式迁移。不要只靠 ``CREATE TABLE IF NOT EXISTS`` 猜测 schema
# 是否兼容：一旦结构变化，必须提供可审计迁移，而不是静默让旧库以未知状态继续运行。
_SCHEMA_VERSION = 2

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

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    vector_id INTEGER UNIQUE,
    paper_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    section TEXT,
    content_kind TEXT NOT NULL,
    front_matter_type TEXT,
    page_start INTEGER,
    page_end INTEGER,
    raw_token_count INTEGER NOT NULL,
    embedding_token_count INTEGER NOT NULL,
    prev_chunk_id TEXT,
    next_chunk_id TEXT,
    FOREIGN KEY (paper_id) REFERENCES documents(paper_id) ON DELETE CASCADE,
    UNIQUE (paper_id, chunk_index),
    CHECK (chunk_index >= 0),
    CHECK (length(trim(raw_text)) > 0),
    CHECK (length(trim(embedding_text)) > 0),
    CHECK (content_kind IN ('body', 'front_matter')),
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
            # 先只创建版本表，不能一上来执行 V2 的 ``CREATE TABLE IF NOT EXISTS``：
            # 对于 V1 的 chunks 表，它不会自动补齐新列，反而会掩盖需要迁移的事实。
            connection.executescript(_SCHEMA_INFO_SQL)
            stored_version = connection.execute(
                "SELECT value FROM schema_info WHERE key = 'schema_version'"
            ).fetchone()
            if stored_version is None:
                connection.executescript(_SCHEMA_V2_SQL)
                connection.execute(
                    "INSERT INTO schema_info(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                return
            if stored_version["value"] == "1":
                _migrate_v1_to_v2(connection)
                connection.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    (str(_SCHEMA_VERSION),),
                )
                return
            if stored_version["value"] == str(_SCHEMA_VERSION):
                # 为已是 V2 的数据库补齐可安全重复创建的索引，便于从早期开发版本恢复。
                connection.executescript(_SCHEMA_V2_SQL)
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
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, vector_id, paper_id, chunk_index,
                    raw_text, embedding_text, section, content_kind, front_matter_type,
                    page_start, page_end,
                    raw_token_count, embedding_token_count, prev_chunk_id, next_chunk_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "SELECT COUNT(*) AS count FROM chunks WHERE vector_id IS NOT NULL"
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
                "SELECT COUNT(*) AS count FROM chunks WHERE vector_id IS NOT NULL"
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
                    WHERE vector_id IS NOT NULL
                    ORDER BY vector_id
                    """
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

    database_rows = connection.execute(
        "SELECT chunk_id, embedding_text FROM chunks"
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
        if chunk.prev_chunk_id is not None and chunk.prev_chunk_id not in known_chunk_ids:
            raise MetadataDatabaseError(
                f"Unknown prev_chunk_id for {chunk.chunk_id}: {chunk.prev_chunk_id}"
            )
        if chunk.next_chunk_id is not None and chunk.next_chunk_id not in known_chunk_ids:
            raise MetadataDatabaseError(
                f"Unknown next_chunk_id for {chunk.chunk_id}: {chunk.next_chunk_id}"
            )
    return expected_paper_id, source_sha256


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
        chunk.page_start,
        chunk.page_end,
        chunk.raw_token_count,
        chunk.embedding_token_count,
        chunk.prev_chunk_id,
        chunk.next_chunk_id,
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
