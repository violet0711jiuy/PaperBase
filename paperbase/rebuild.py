"""从正式 PDF 完整、安全地重建 PaperBase Knowledge Base。

此入口只服务“全量重建”场景：它绝不读取旧 SQLite chunk 或旧 embedding 作为输入，
而是始终从 ``storage.papers_dir`` 中受管理的原始 PDF 开始。所有新产物先写入一个
独立临时目录；只有解析、分块、向量、索引和离线检索校验都成功后，才备份并替换正式
产物。因此任一构建阶段失败时，原正式知识库不会被触碰。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Iterable
from uuid import uuid4

from paperbase.chunking.factory import create_chunker
from paperbase.chunking.inspect import write_chunking_artifacts
from paperbase.config import AppSettings, default_config_path, load_settings
from paperbase.database import MetadataDatabase
from paperbase.embedding import QueryEmbedder, create_document_embedder
from paperbase.embedding.artifacts import artifact_paths, load_embedding_artifact
from paperbase.embedding.service import EmbeddingRunSummary, generate_database_embeddings
from paperbase.indexing import FaissIndexStore, IndexBuildSummary
from paperbase.parsing.factory import create_parser
from paperbase.parsing.inspect import (
    build_inspection_report,
    write_docling_inspection_artifacts,
)
from paperbase.retrieval.hybrid_retriever import HybridRetriever
from paperbase.retrieval.query_rewriter import NoopQueryRewriter


class CleanRebuildError(RuntimeError):
    """全量重建的输入、校验或安全发布不满足契约时抛出。"""


@dataclass(frozen=True)
class RebuildPaths:
    """本次运行的临时路径、备份路径与最终正式路径。

    路径全部由现有 ``config.yaml`` 的正式路径推导，而不是把另一套存储布局写死在代码中。
    ``temporary_root`` 放在既有 staging 目录，``backup_root`` 放在既有 parsed 目录，
    因而不会与用户放入 papers_dir 的原始 PDF 混在一起。
    """

    run_id: str
    temporary_root: Path
    backup_root: Path
    temporary_database: Path
    temporary_embeddings: Path
    temporary_index: Path
    temporary_manifest: Path
    temporary_parsed_artifacts: Path
    temporary_chunk_artifacts: Path
    final_database: Path
    final_embeddings: Path
    final_index: Path
    final_manifest: Path
    final_parsed_artifacts: Path
    final_chunk_artifacts: Path

    @classmethod
    def create(cls, settings: AppSettings) -> "RebuildPaths":
        """以本次唯一 run_id 派生临时与备份路径，避免覆盖历史运行。"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"rebuild_{timestamp}_{uuid4().hex[:8]}"
        temporary_root = settings.storage.staging_dir / "rebuild_tmp" / run_id
        backup_root = settings.storage.parsed_dir / "rebuild_backups" / run_id
        return cls(
            run_id=run_id,
            temporary_root=temporary_root,
            backup_root=backup_root,
            temporary_database=temporary_root / settings.database.path.name,
            temporary_embeddings=temporary_root / "embeddings",
            temporary_index=temporary_root / settings.indexing.index_path.name,
            temporary_manifest=temporary_root / settings.indexing.manifest_path.name,
            temporary_parsed_artifacts=temporary_root / "parsed_artifacts",
            temporary_chunk_artifacts=temporary_root / "chunk_artifacts",
            final_database=settings.database.path,
            final_embeddings=settings.embedding.output_dir,
            final_index=settings.indexing.index_path,
            final_manifest=settings.indexing.manifest_path,
            final_parsed_artifacts=settings.parsing.inspection_output_dir,
            final_chunk_artifacts=settings.chunking.inspection_output_dir,
        )

    @classmethod
    def resume(cls, settings: AppSettings, temporary_root: Path) -> "RebuildPaths":
        """从一次未发布成功的临时运行恢复发布，不重新 Parse 或 Embedding。"""
        temporary_root = temporary_root.resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        publish_run_id = (
            f"{temporary_root.name}_publish_{timestamp}_{uuid4().hex[:8]}"
        )
        return cls(
            run_id=publish_run_id,
            temporary_root=temporary_root,
            backup_root=settings.storage.parsed_dir
            / "rebuild_backups"
            / publish_run_id,
            temporary_database=temporary_root / settings.database.path.name,
            temporary_embeddings=temporary_root / "embeddings",
            temporary_index=temporary_root / settings.indexing.index_path.name,
            temporary_manifest=temporary_root / settings.indexing.manifest_path.name,
            temporary_parsed_artifacts=temporary_root / "parsed_artifacts",
            temporary_chunk_artifacts=temporary_root / "chunk_artifacts",
            final_database=settings.database.path,
            final_embeddings=settings.embedding.output_dir,
            final_index=settings.indexing.index_path,
            final_manifest=settings.indexing.manifest_path,
            final_parsed_artifacts=settings.parsing.inspection_output_dir,
            final_chunk_artifacts=settings.chunking.inspection_output_dir,
        )

    def replacements(self) -> tuple[tuple[str, Path, Path], ...]:
        """返回 ``名称、临时新产物、正式目标``，供统一校验和发布使用。"""
        return (
            ("database", self.temporary_database, self.final_database),
            ("embeddings", self.temporary_embeddings, self.final_embeddings),
            ("faiss_index", self.temporary_index, self.final_index),
            ("faiss_manifest", self.temporary_manifest, self.final_manifest),
            (
                "parsed_artifacts",
                self.temporary_parsed_artifacts,
                self.final_parsed_artifacts,
            ),
            (
                "chunk_artifacts",
                self.temporary_chunk_artifacts,
                self.final_chunk_artifacts,
            ),
        )


@dataclass(frozen=True)
class RebuildValidation:
    """正式替换前后的数据库、FTS5、FAISS 一致性计数。"""

    document_count: int
    section_count: int
    chunk_count: int
    content_chunk_count: int
    bibliography_chunk_count: int
    content_fts_count: int
    bibliography_fts_count: int
    vectorized_content_count: int
    faiss_ntotal: int


@dataclass(frozen=True)
class CleanRebuildSummary:
    """一次成功全量重建的可审计摘要。"""

    source_pdf_count: int
    paper_ids: tuple[str, ...]
    paths: RebuildPaths
    embedding: EmbeddingRunSummary
    index: IndexBuildSummary
    validation: RebuildValidation
    smoke_query: str
    smoke_result_count: int
    published: bool


def run_clean_full_rebuild(settings: AppSettings) -> CleanRebuildSummary:
    """从正式 PDF 构建并验证新库，再安全备份、替换正式产物。"""
    if settings.indexing.backend != "faiss_flat_ip":
        raise CleanRebuildError(
            f"Unsupported indexing backend: {settings.indexing.backend!r}"
        )

    sources = tuple(sorted(settings.storage.papers_dir.glob("*.pdf")))
    if not sources:
        raise CleanRebuildError(
            f"No official PDFs found in {settings.storage.papers_dir}."
        )
    paths = RebuildPaths.create(settings)
    _validate_rebuild_paths(paths=paths, settings=settings, sources=sources)
    paths.temporary_root.mkdir(parents=True, exist_ok=False)

    temporary_database = MetadataDatabase(
        paths.temporary_database,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    # 工厂在一次全量运行中只创建一份实例，确保所有论文使用同一份当前配置。
    parser = create_parser(settings)
    chunker = create_chunker(settings)
    paper_ids: list[str] = []
    try:
        for source in sources:
            # 仅从原始 PDF 开始；不会读取旧 Markdown、旧 chunks 或正式 SQLite。
            parsed_paper = parser.parse(source)
            report = build_inspection_report(parsed_paper)
            write_docling_inspection_artifacts(
                parsed_paper,
                report,
                paths.temporary_parsed_artifacts,
            )
            chunking_result = chunker.chunk(parsed_paper)
            write_chunking_artifacts(chunking_result, paths.temporary_chunk_artifacts)
            imported = temporary_database.import_chunking_result(chunking_result)
            paper_ids.append(imported.paper_id)

        # 相同 PDF 内容不能在正式库中拥有两条不同 documents 记录，提前报错而非静默覆盖。
        if len(set(paper_ids)) != len(paper_ids):
            raise CleanRebuildError(
                "Official PDF directory contains duplicate PDF content / paper_id values."
            )

        embedder = create_document_embedder(settings.embedding)
        embedding_summary = generate_database_embeddings(
            database=temporary_database,
            embedder=embedder,
            output_dir=paths.temporary_embeddings,
            normalized=settings.embedding.normalize_embeddings,
        )
        temporary_index_settings = settings.indexing.model_copy(
            update={
                "index_path": paths.temporary_index,
                "manifest_path": paths.temporary_manifest,
            }
        )
        temporary_index_store = FaissIndexStore(temporary_index_settings)
        index_summary = temporary_index_store.build_initial_index(
            database=temporary_database,
            embedding_output_dir=paths.temporary_embeddings,
        )
        validation = _validate_rebuilt_knowledge_base(
            database=temporary_database,
            index_store=temporary_index_store,
            embedding_output_dir=paths.temporary_embeddings,
            expected_document_count=len(sources),
        )
        smoke_query, smoke_result_count = _run_offline_qa_smoke(
            database=temporary_database,
            index_store=temporary_index_store,
            embedder=embedder,
            settings=settings,
        )
    except Exception:
        # 失败时临时目录保留，便于人工审计错误；正式 KB 尚未进入发布阶段，保持可用。
        raise

    _publish_rebuild(paths)

    # 替换后再次以正式路径验证，排除“临时产物正确但发布路径错误”的可能。
    formal_database = MetadataDatabase(
        paths.final_database,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    formal_index_store = FaissIndexStore(settings.indexing)
    published_validation = _validate_rebuilt_knowledge_base(
        database=formal_database,
        index_store=formal_index_store,
        embedding_output_dir=paths.final_embeddings,
        expected_document_count=len(sources),
    )
    _, published_smoke_result_count = _run_offline_qa_smoke(
        database=formal_database,
        index_store=formal_index_store,
        embedder=embedder,
        settings=settings,
        query=smoke_query,
    )
    _write_publish_report(
        paths=paths,
        validation=published_validation,
        source_pdf_count=len(sources),
        paper_ids=tuple(paper_ids),
        smoke_query=smoke_query,
        smoke_result_count=published_smoke_result_count,
    )
    return CleanRebuildSummary(
        source_pdf_count=len(sources),
        paper_ids=tuple(paper_ids),
        paths=paths,
        embedding=embedding_summary,
        index=index_summary,
        validation=published_validation,
        smoke_query=smoke_query,
        smoke_result_count=published_smoke_result_count,
        published=True,
    )


def publish_existing_rebuild(
    settings: AppSettings, *, temporary_root: Path
) -> CleanRebuildSummary:
    """重新验证既有临时重建产物并尝试发布，专用于此前发布阶段被外部锁阻断的情况。"""
    sources = tuple(sorted(settings.storage.papers_dir.glob("*.pdf")))
    if not sources:
        raise CleanRebuildError(
            f"No official PDFs found in {settings.storage.papers_dir}."
        )
    paths = RebuildPaths.resume(settings, temporary_root)
    _validate_rebuild_paths(paths=paths, settings=settings, sources=sources)
    temporary_database = MetadataDatabase(
        paths.temporary_database,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    temporary_index_settings = settings.indexing.model_copy(
        update={
            "index_path": paths.temporary_index,
            "manifest_path": paths.temporary_manifest,
        }
    )
    temporary_index_store = FaissIndexStore(temporary_index_settings)
    validation = _validate_rebuilt_knowledge_base(
        database=temporary_database,
        index_store=temporary_index_store,
        embedding_output_dir=paths.temporary_embeddings,
        expected_document_count=len(sources),
    )
    embedder = create_document_embedder(settings.embedding)
    smoke_query, smoke_result_count = _run_offline_qa_smoke(
        database=temporary_database,
        index_store=temporary_index_store,
        embedder=embedder,
        settings=settings,
    )
    embedding_summary = _load_embedding_summary(paths.temporary_embeddings)
    index_summary = temporary_index_store.verify_against_sqlite(
        database=temporary_database,
        embedding_output_dir=paths.temporary_embeddings,
    )

    _publish_rebuild(paths)
    formal_database = MetadataDatabase(
        paths.final_database,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    formal_index_store = FaissIndexStore(settings.indexing)
    published_validation = _validate_rebuilt_knowledge_base(
        database=formal_database,
        index_store=formal_index_store,
        embedding_output_dir=paths.final_embeddings,
        expected_document_count=len(sources),
    )
    _, published_smoke_result_count = _run_offline_qa_smoke(
        database=formal_database,
        index_store=formal_index_store,
        embedder=embedder,
        settings=settings,
        query=smoke_query,
    )
    paper_ids = tuple(
        str(row["paper_id"])
        for row in _document_rows(formal_database)
    )
    _write_publish_report(
        paths=paths,
        validation=published_validation,
        source_pdf_count=len(sources),
        paper_ids=paper_ids,
        smoke_query=smoke_query,
        smoke_result_count=published_smoke_result_count,
    )
    return CleanRebuildSummary(
        source_pdf_count=len(sources),
        paper_ids=paper_ids,
        paths=paths,
        embedding=embedding_summary,
        index=index_summary,
        validation=published_validation,
        smoke_query=smoke_query,
        smoke_result_count=published_smoke_result_count,
        published=True,
    )


def _validate_rebuild_paths(
    *, paths: RebuildPaths, settings: AppSettings, sources: Iterable[Path]
) -> None:
    """阻止临时、备份、原始 PDF 与正式目标发生危险重叠。"""
    project_root = settings.config_path.parent.resolve()
    temporary_root = paths.temporary_root.resolve()
    backup_root = paths.backup_root.resolve()
    if temporary_root == backup_root or _is_within(temporary_root, backup_root):
        raise CleanRebuildError("Temporary rebuild path overlaps its backup path.")
    for source in sources:
        resolved_source = source.resolve()
        if not _is_within(resolved_source, settings.storage.papers_dir.resolve()):
            raise CleanRebuildError(f"PDF source escapes papers_dir: {resolved_source}")
    final_targets = [target.resolve() for _, _, target in paths.replacements()]
    if len(set(final_targets)) != len(final_targets):
        raise CleanRebuildError("Configured formal rebuild targets must be distinct.")
    for target in final_targets:
        if not _is_within(target, project_root):
            raise CleanRebuildError(f"Formal rebuild target escapes project root: {target}")
        if _is_within(target, temporary_root) or _is_within(target, backup_root):
            raise CleanRebuildError(f"Formal target overlaps rebuild working paths: {target}")


def _validate_rebuilt_knowledge_base(
    *,
    database: MetadataDatabase,
    index_store: FaissIndexStore,
    embedding_output_dir: Path,
    expected_document_count: int,
) -> RebuildValidation:
    """在 SQLite、FTS5、embedding 工件和 FAISS 之间执行只读交叉校验。"""
    database.initialize()
    # 该校验同时验证 embedding records、SQLite vector_id 映射、manifest 和 FAISS ID 集合。
    index_store.verify_against_sqlite(
        database=database,
        embedding_output_dir=embedding_output_dir,
    )
    with sqlite3.connect(database.path) as connection:
        connection.row_factory = sqlite3.Row
        counts = {
            "documents": _scalar(connection, "SELECT COUNT(*) FROM documents"),
            "sections": _scalar(connection, "SELECT COUNT(*) FROM sections"),
            "chunks": _scalar(connection, "SELECT COUNT(*) FROM chunks"),
            "content": _scalar(
                connection, "SELECT COUNT(*) FROM chunks WHERE section_type = 'content'"
            ),
            "bibliography": _scalar(
                connection,
                "SELECT COUNT(*) FROM chunks WHERE section_type = 'bibliography'",
            ),
            "content_fts": _scalar(connection, "SELECT COUNT(*) FROM chunks_fts"),
            "bibliography_fts": _scalar(
                connection, "SELECT COUNT(*) FROM bibliography_fts"
            ),
            "vectorized_content": _scalar(
                connection,
                "SELECT COUNT(*) FROM chunks WHERE section_type = 'content' AND vector_id IS NOT NULL",
            ),
        }
        violations = {
            "sections_without_document": _scalar(
                connection,
                "SELECT COUNT(*) FROM sections AS s LEFT JOIN documents AS d "
                "ON d.paper_id = s.paper_id WHERE d.paper_id IS NULL",
            ),
            "chunks_without_document": _scalar(
                connection,
                "SELECT COUNT(*) FROM chunks AS c LEFT JOIN documents AS d "
                "ON d.paper_id = c.paper_id WHERE d.paper_id IS NULL",
            ),
            "chunks_with_invalid_section": _scalar(
                connection,
                "SELECT COUNT(*) FROM chunks AS c LEFT JOIN sections AS s "
                "ON s.section_id = c.section_id WHERE c.section_id IS NOT NULL "
                "AND (s.section_id IS NULL OR s.paper_id != c.paper_id)",
            ),
            "orphan_section_parent": _scalar(
                connection,
                "SELECT COUNT(*) FROM sections AS child LEFT JOIN sections AS parent "
                "ON parent.section_id = child.parent_section_id "
                "WHERE child.parent_section_id IS NOT NULL "
                "AND (parent.section_id IS NULL OR parent.paper_id != child.paper_id)",
            ),
            "bibliography_with_vector": _scalar(
                connection,
                "SELECT COUNT(*) FROM chunks WHERE section_type = 'bibliography' "
                "AND vector_id IS NOT NULL",
            ),
            "content_without_vector": _scalar(
                connection,
                "SELECT COUNT(*) FROM chunks WHERE section_type = 'content' "
                "AND vector_id IS NULL",
            ),
            "duplicate_chunk_id": _scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1)",
            ),
            "duplicate_vector_id": _scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT vector_id FROM chunks WHERE vector_id IS NOT NULL "
                "GROUP BY vector_id HAVING COUNT(*) > 1)",
            ),
        }
    index = index_store.load_for_search(database=database)
    faiss_ntotal = int(index.ntotal)
    failures: list[str] = []
    if counts["documents"] != expected_document_count:
        failures.append(
            f"documents={counts['documents']} but official PDFs={expected_document_count}"
        )
    if counts["content"] != counts["content_fts"]:
        failures.append("content FTS5 row count does not match content chunks")
    if counts["bibliography"] != counts["bibliography_fts"]:
        failures.append("bibliography FTS5 row count does not match bibliography chunks")
    if counts["content"] != counts["vectorized_content"]:
        failures.append("not every content chunk has a vector_id")
    if faiss_ntotal != counts["content"]:
        failures.append("FAISS ntotal does not match content chunk count")
    failures.extend(name for name, count in violations.items() if count)
    if failures:
        raise CleanRebuildError("Rebuild consistency validation failed: " + "; ".join(failures))
    return RebuildValidation(
        document_count=counts["documents"],
        section_count=counts["sections"],
        chunk_count=counts["chunks"],
        content_chunk_count=counts["content"],
        bibliography_chunk_count=counts["bibliography"],
        content_fts_count=counts["content_fts"],
        bibliography_fts_count=counts["bibliography_fts"],
        vectorized_content_count=counts["vectorized_content"],
        faiss_ntotal=faiss_ntotal,
    )


def _run_offline_qa_smoke(
    *,
    database: MetadataDatabase,
    index_store: FaissIndexStore,
    embedder: object,
    settings: AppSettings,
    query: str | None = None,
) -> tuple[str, int]:
    """执行一次真实 Dense + BM25 混合检索，但刻意关闭 LLM 改写与 reranker。"""
    if not isinstance(embedder, QueryEmbedder):
        raise CleanRebuildError("Configured document embedder does not support query embeddings.")
    smoke_query = query or _select_smoke_query(database)
    retriever = HybridRetriever(
        database=database,
        query_embedder=embedder,
        index_store=index_store,
        settings=settings.retrieval,
        query_rewriter=NoopQueryRewriter(),
        # 不加载额外 Cross-Encoder；本 smoke test 验证已重建的正式 QA 检索输入链路。
        reranker=None,
        reranking_settings=None,
    )
    result = retriever.retrieve(smoke_query, result_limit=1)
    if not result.chunks:
        raise CleanRebuildError("Offline hybrid retrieval smoke test returned no chunks.")
    return smoke_query, len(result.chunks)


def _select_smoke_query(database: MetadataDatabase) -> str:
    """从新库的真实标题或正文提取一个可被 FTS5 命中的稳定英文词。"""
    with sqlite3.connect(database.path) as connection:
        rows = connection.execute(
            "SELECT paper_title FROM documents UNION ALL "
            "SELECT raw_text FROM chunks WHERE section_type = 'content' ORDER BY 1"
        ).fetchall()
    for (text,) in rows:
        match = re.search(r"[A-Za-z][A-Za-z0-9_-]{3,}", str(text or ""))
        if match:
            return match.group(0)
    raise CleanRebuildError("Cannot find an indexable smoke-test term in rebuilt content.")


def _publish_rebuild(paths: RebuildPaths) -> None:
    """备份现有正式产物，再将已验证临时产物移动到正式配置路径。

    多个文件无法组成操作系统级单一原子操作，因此每一步都写 journal；若同一进程的
    发布失败，会立即用备份回滚。备份被永久保留，便于人工恢复或审计。
    """
    for name, temporary, _ in paths.replacements():
        if not temporary.exists():
            raise CleanRebuildError(f"Validated temporary artifact is missing: {name}={temporary}")
    _assert_formal_database_is_replaceable(paths.final_database)
    paths.backup_root.mkdir(parents=True, exist_ok=False)
    journal_path = paths.backup_root / "publish_journal.json"
    journal: dict[str, object] = {
        "run_id": paths.run_id,
        "phase": "prepared",
        "items": [
            {
                "name": name,
                "temporary": str(temporary),
                "formal": str(formal),
                "backup": str(paths.backup_root / name),
                "backed_up": False,
                "published": False,
            }
            for name, temporary, formal in paths.replacements()
        ],
    }
    _write_json(journal_path, journal)
    items = journal["items"]
    assert isinstance(items, list)
    try:
        # 第一阶段只移动旧正式产物到独立备份，不删除其中任何数据。
        for item in items:
            assert isinstance(item, dict)
            formal = Path(str(item["formal"]))
            backup = Path(str(item["backup"]))
            if formal.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                _move_path(formal, backup)
                item["backed_up"] = True
                _write_json(journal_path, journal)
        journal["phase"] = "old_artifacts_backed_up"
        _write_json(journal_path, journal)
        # 第二阶段发布已经完整验证过的临时产物；同卷 move 保留 rename 的原子性。
        for item in items:
            assert isinstance(item, dict)
            temporary = Path(str(item["temporary"]))
            formal = Path(str(item["formal"]))
            formal.parent.mkdir(parents=True, exist_ok=True)
            _move_path(temporary, formal)
            item["published"] = True
            _write_json(journal_path, journal)
        journal["phase"] = "published"
        _write_json(journal_path, journal)
    except Exception as error:
        _rollback_publish(items)
        journal["phase"] = "rolled_back"
        journal["error"] = f"{type(error).__name__}: {error}"
        _write_json(journal_path, journal)
        raise CleanRebuildError(
            "Formal replacement failed and the in-process rollback was attempted; "
            f"backup journal: {journal_path}"
        ) from error


def _rollback_publish(items: list[object]) -> None:
    """尽力恢复已备份旧产物；失败项由 journal 留给人工处理。"""
    for raw_item in reversed(items):
        if not isinstance(raw_item, dict):
            continue
        formal = Path(str(raw_item["formal"]))
        backup = Path(str(raw_item["backup"]))
        if raw_item.get("published") and formal.exists():
            # 新产物不删除：移动回临时原位置，保留失败现场供核查。
            temporary = Path(str(raw_item["temporary"]))
            temporary.parent.mkdir(parents=True, exist_ok=True)
            _move_path(formal, temporary)
        if raw_item.get("backed_up") and backup.exists() and not formal.exists():
            formal.parent.mkdir(parents=True, exist_ok=True)
            _move_path(backup, formal)


def _write_publish_report(
    *,
    paths: RebuildPaths,
    validation: RebuildValidation,
    source_pdf_count: int,
    paper_ids: tuple[str, ...],
    smoke_query: str,
    smoke_result_count: int,
) -> None:
    """把成功结果写进备份目录，使新库和被替换旧库拥有同一份审计记录。"""
    _write_json(
        paths.backup_root / "rebuild_report.json",
        {
            "run_id": paths.run_id,
            "source_pdf_count": source_pdf_count,
            "paper_ids": list(paper_ids),
            "validation": asdict(validation),
            "smoke_query": smoke_query,
            "smoke_result_count": smoke_result_count,
            "published": True,
        },
    )


def _load_embedding_summary(output_dir: Path) -> EmbeddingRunSummary:
    """从已验证 embedding 工件恢复展示摘要，不重新编码任何文本。"""
    artifact = load_embedding_artifact(output_dir)
    return EmbeddingRunSummary(
        chunk_count=len(artifact.records),
        dimension=int(artifact.vectors.shape[1]),
        backend_id=str(artifact.manifest["backend_id"]),
        model_id=str(artifact.manifest["model_id"]),
        artifact_paths=artifact_paths(output_dir),
    )


def _document_rows(database: MetadataDatabase) -> tuple[sqlite3.Row, ...]:
    """按稳定 paper_id 读取正式 document 身份，供恢复发布后的摘要使用。"""
    with sqlite3.connect(database.path) as connection:
        connection.row_factory = sqlite3.Row
        return tuple(
            connection.execute("SELECT paper_id FROM documents ORDER BY paper_id").fetchall()
        )


def _assert_formal_database_is_replaceable(database_path: Path) -> None:
    """发布前抢占一次 SQLite EXCLUSIVE 锁，尽早发现仍在运行的旧 KB 服务。"""
    if not database_path.exists():
        return
    connection = sqlite3.connect(database_path, timeout=0)
    try:
        connection.execute("PRAGMA busy_timeout = 0")
        connection.execute("BEGIN EXCLUSIVE")
        connection.rollback()
    except sqlite3.OperationalError as error:
        raise CleanRebuildError(
            "Official SQLite is currently in use. Stop the PaperBase process or any SQLite "
            f"viewer before publishing: {database_path}"
        ) from error
    finally:
        connection.close()


def _move_path(source: Path, destination: Path) -> None:
    """移动文件或目录，且拒绝覆盖意外已存在的目的地。"""
    if destination.exists():
        raise CleanRebuildError(f"Refusing to overwrite existing path during publish: {destination}")
    # os.replace 在同一磁盘内是 rename；路径跨卷时 shutil.move 仍保留完整内容。
    try:
        os.replace(source, destination)
    except OSError:
        shutil.move(str(source), str(destination))


def _write_json(path: Path, data: object) -> None:
    """通过同目录临时文件发布 JSON，避免中断时留下半个 journal/report。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    candidate.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(candidate, path)


def _scalar(connection: sqlite3.Connection, sql: str) -> int:
    """读取单个 COUNT 值，统一转换为 int。"""
    row = connection.execute(sql).fetchone()
    return int(row[0])


def _is_within(path: Path, parent: Path) -> bool:
    """返回 path 是否等于或位于 parent 内，兼容较低 Python 版本。"""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> None:
    """提供 ``python -m paperbase.rebuild`` 的受控全量重建入口。"""
    argument_parser = argparse.ArgumentParser(
        description="Cleanly rebuild the official PaperBase KB from managed original PDFs."
    )
    argument_parser.add_argument(
        "--config", type=Path, default=default_config_path(), help="config.yaml path"
    )
    argument_parser.add_argument(
        "--publish-existing",
        type=Path,
        help="Only revalidate and publish a prior rebuild_tmp run; do not parse or embed again.",
    )
    args = argument_parser.parse_args()
    settings = load_settings(args.config)
    summary = (
        publish_existing_rebuild(settings, temporary_root=args.publish_existing)
        if args.publish_existing
        else run_clean_full_rebuild(settings)
    )
    print(
        json.dumps(
            {
                "source_pdf_count": summary.source_pdf_count,
                "paper_ids": list(summary.paper_ids),
                "validation": asdict(summary.validation),
                "embedding_chunks": summary.embedding.chunk_count,
                "embedding_dimension": summary.embedding.dimension,
                "smoke_query": summary.smoke_query,
                "smoke_result_count": summary.smoke_result_count,
                "backup_root": str(summary.paths.backup_root),
                "published": summary.published,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
