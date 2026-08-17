"""复用 staging 产物，把一篇临时论文安全提升到正式知识库。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from paperbase.chunking.base import ChunkingResult, PaperChunk
from paperbase.config import AppSettings
from paperbase.database import (
    DocumentAlreadyExistsError,
    MetadataDatabase,
    PromotionImportSummary,
)
from paperbase.embedding.artifacts import (
    EmbeddingArtifactError,
    EmbeddingArtifactPaths,
    LoadedEmbeddingArtifact,
    artifact_paths,
    load_embedding_artifact,
    write_embedding_artifact,
)
from paperbase.embedding.base import EmbeddingInput
from paperbase.indexing.faiss_store import FaissIndexStore, IncrementalIndexCandidate, VectorAssignment
from paperbase.parsing.base import ParsedPaper, SectionRecord


class PromotionError(RuntimeError):
    """staging 工件不完整或 promotion 发布失败时抛出。"""


@dataclass(frozen=True)
class PromotionResult:
    """一次 Add to Knowledge Base 的可展示结果。"""

    status: str
    workspace_id: str
    paper_id: str
    formal_pdf_path: Path | None
    document_count_before: int
    document_count_after: int
    section_count_before: int
    section_count_after: int
    chunk_count_before: int
    chunk_count_after: int
    faiss_ntotal_before: int
    faiss_ntotal_after: int
    content_chunk_count: int = 0
    bibliography_chunk_count: int = 0


@dataclass(frozen=True)
class _PromotionInput:
    """完成 preflight 后的不可变 workspace 快照。"""

    workspace_id: str
    root: Path
    paper_id: str
    original_pdf: Path
    source_filename: str
    result: ChunkingResult
    artifact: LoadedEmbeddingArtifact


@dataclass(frozen=True)
class _EmbeddingCandidate:
    """尚未替换正式 artifact 的完整向量工件组。"""

    paths: EmbeddingArtifactPaths


class PromotionService:
    """协调 SQLite、FTS5、FAISS、正式 PDF 与 artifact 的单论文提升。"""

    def __init__(self, *, settings: AppSettings) -> None:
        self._settings = settings
        self._database = MetadataDatabase(
            settings.database.path, busy_timeout_ms=settings.database.busy_timeout_ms
        )
        self._index_store = FaissIndexStore(settings.indexing)

    def promote(self, workspace_id: str) -> PromotionResult:
        """提升一个完整 staging workspace；重复 paper_id 安全返回 no-op。"""
        prepared = _load_workspace_for_promotion(
            staging_dir=self._settings.storage.staging_dir, workspace_id=workspace_id
        )
        self._database.initialize()
        before = _formal_counts(self._database, self._index_store)
        if self._database.get_document(prepared.paper_id) is not None:
            return _already_exists_result(prepared=prepared, counts=before)

        # 正式 ID 独立于 temporary FAISS 的 1..N，永远接在当前主索引末尾。
        assignments = _formal_vector_assignments(
            artifact=prepared.artifact, start_after=before.faiss_ntotal
        )
        operation_root = prepared.root / ".promotion" / uuid4().hex
        operation_root.mkdir(parents=True, exist_ok=False)
        formal_pdf, pdf_created = _copy_formal_pdf(
            source=prepared.original_pdf,
            papers_dir=self._settings.storage.papers_dir,
            source_filename=prepared.source_filename,
            paper_id=prepared.paper_id,
        )
        result = _with_formal_source(prepared.result, formal_pdf)
        database_written = False
        backups: list[tuple[Path, Path | None]] = []
        candidate: IncrementalIndexCandidate | None = None
        try:
            candidate = self._index_store.prepare_incremental_append(
                database=self._database,
                artifact=prepared.artifact,
                additions=assignments,
            )
            embedding_candidate = _prepare_formal_embedding_candidate(
                database=self._database,
                formal_output_dir=self._settings.embedding.output_dir,
                result=result,
                staging_artifact=prepared.artifact,
                operation_root=operation_root,
            )
            # 先备份会被原子替换的正式派生文件；错误时精确恢复这些已知文件。
            targets = (
                self._index_store.index_path,
                self._index_store.manifest_path,
                *(_artifact_files(self._settings.embedding.output_dir)),
            )
            backups = _backup_targets(targets=targets, backup_root=operation_root / "backup")

            imported = self._database.promote_chunking_result(
                result=result,
                vector_assignments=[
                    (item.chunk_id, item.vector_id, item.embedding_text_sha256)
                    for item in assignments
                ],
            )
            database_written = True
            self._index_store.publish_incremental_candidate(candidate, database=self._database)
            _publish_embedding_candidate(
                candidate=embedding_candidate,
                formal_output_dir=self._settings.embedding.output_dir,
            )
            # 最终在线验证只依赖正式 SQLite + FAISS + manifest，不依赖 staging。
            verified = self._index_store.verify_published_index(database=self._database)
            after = _formal_counts(self._database, self._index_store)
            if verified.vector_count != after.faiss_ntotal:
                raise PromotionError("Published FAISS count disagrees with the formal database.")
            _mark_workspace_promoted(prepared.root, paper_id=prepared.paper_id)
            return _success_result(
                prepared=prepared, imported=imported, formal_pdf=formal_pdf,
                before=before, after=after,
            )
        except DocumentAlreadyExistsError:
            # 并发情况下另一调用刚完成同一 paper 的提升，也保持安全 no-op。
            return _already_exists_result(prepared=prepared, counts=_formal_counts(self._database, self._index_store))
        except Exception:
            _restore_targets(backups)
            if database_written:
                self._database.delete_document_for_promotion_rollback(prepared.paper_id)
            if pdf_created:
                formal_pdf.unlink(missing_ok=True)
            if candidate is not None:
                candidate.candidate_index_path.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(operation_root, ignore_errors=True)


def promote_workspace(*, settings: AppSettings, workspace_id: str) -> PromotionResult:
    """供 CLI、未来 API 或前端复用的简洁入口。"""
    return PromotionService(settings=settings).promote(workspace_id)


@dataclass(frozen=True)
class _FormalCounts:
    documents: int
    sections: int
    chunks: int
    faiss_ntotal: int


def _formal_counts(database: MetadataDatabase, index_store: FaissIndexStore) -> _FormalCounts:
    """读取 promotion 前后的关键计数；无文档的空库没有 FAISS 文件也是合法状态。"""
    import sqlite3

    database.initialize()
    with sqlite3.connect(database.path) as connection:
        documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        sections = int(connection.execute("SELECT COUNT(*) FROM sections").fetchone()[0])
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    state = database.vector_index_state()
    if state.total_chunk_count == 0:
        return _FormalCounts(documents, sections, chunks, 0)
    verified = index_store.verify_published_index(database=database)
    return _FormalCounts(documents, sections, chunks, verified.vector_count)


def _load_workspace_for_promotion(*, staging_dir: Path, workspace_id: str) -> _PromotionInput:
    """在任何正式写入前读取并严格校验 staging 的原始、解析、分块和向量产物。"""
    normalized_id = _workspace_id(workspace_id)
    root = (staging_dir.resolve() / normalized_id).resolve()
    if root.parent != staging_dir.resolve() or not root.is_dir():
        raise PromotionError(f"Temporary workspace does not exist: {normalized_id}")
    manifest = _json_object(root / "workspace.json")
    if manifest.get("workspace_id") != normalized_id:
        raise PromotionError("workspace.json does not match its workspace directory.")
    paper_id = _required_string(manifest.get("paper_id"), "workspace.paper_id")
    original_pdf = root / "original" / "paper.pdf"
    if not original_pdf.is_file():
        raise PromotionError("Workspace original PDF is missing.")
    pdf_sha256 = _file_sha256(original_pdf)
    expected_paper_id = f"paper_{pdf_sha256[:16]}"
    if paper_id != expected_paper_id or manifest.get("source_sha256") != pdf_sha256:
        raise PromotionError("Workspace paper_id or PDF hash does not match the stable identity rule.")
    source_filename = _safe_pdf_filename(manifest.get("source_filename"), paper_id)
    parsed = _json_object(root / "parsed" / "parsed_paper.json")
    metadata = _json_object(root / "chunks" / "metadata.json")
    sections = _load_sections(parsed.get("sections"), paper_id=paper_id)
    chunks = _load_chunks(root / "chunks" / "chunks.jsonl", paper_id=paper_id)
    result = ChunkingResult(
        parsed_paper=ParsedPaper(
            source=original_pdf,
            parser_id=_required_string(parsed.get("parser_id"), "parsed.parser_id"),
            markdown="",
            page_furniture=(),
            front_matter=(),
            paper_title=_optional_string(parsed.get("paper_title")),
            title_source=_required_string(parsed.get("title_source"), "parsed.title_source"),
            title_candidates=tuple(
                item for item in parsed.get("title_candidates", []) if isinstance(item, str)
            ),
            diagnostics=_diagnostics(parsed.get("diagnostics"), "parsed.diagnostics"),
            native_document=object(),
            sections=sections,
        ),
        chunks=chunks,
        diagnostics=_diagnostics(metadata.get("chunking_diagnostics"), "chunking_diagnostics"),
    )
    # SQLite 的同一套严格结构校验会在写入前再次执行；这里提前检查关键跨文件关系。
    known_sections = {section.section_id for section in sections}
    if any(chunk.section_id is not None and chunk.section_id not in known_sections for chunk in chunks):
        raise PromotionError("A workspace chunk references a missing section_id.")
    try:
        artifact = load_embedding_artifact(root / "embeddings")
    except EmbeddingArtifactError as error:
        raise PromotionError(f"Workspace staging embedding artifact is invalid: {error}") from error
    _validate_staging_artifact(result=result, paper_id=paper_id, artifact=artifact)
    return _PromotionInput(
        workspace_id=normalized_id, root=root, paper_id=paper_id, original_pdf=original_pdf,
        source_filename=source_filename, result=result, artifact=artifact,
    )


def _validate_staging_artifact(*, result: ChunkingResult, paper_id: str, artifact: LoadedEmbeddingArtifact) -> None:
    """确认 vectors/records 仅覆盖该论文正文，并与固化 embedding_text 完全一致。"""
    if not artifact.manifest.get("normalized"):
        raise PromotionError("Promotion requires normalized staging embeddings for FAISS inner-product search.")
    if artifact.manifest.get("record_count") != len(artifact.records):
        raise PromotionError("Staging embedding manifest record count is inconsistent.")
    expected = {
        chunk.chunk_id: hashlib.sha256(chunk.embedding_text.encode("utf-8")).hexdigest()
        for chunk in result.chunks if chunk.section_type == "content"
    }
    actual = {record.chunk_id: record.embedding_text_sha256 for record in artifact.records}
    if set(actual) != set(expected) or any(actual[key] != expected[key] for key in expected):
        raise PromotionError("Staging embedding records do not exactly match current content chunks.")
    if any(record.paper_id != paper_id for record in artifact.records):
        raise PromotionError("Staging embedding records refer to a different paper_id.")


def _formal_vector_assignments(*, artifact: LoadedEmbeddingArtifact, start_after: int) -> tuple[VectorAssignment, ...]:
    """按正式 FAISS 当前最大/总数之后连续分配 ID，绝不复用 temporary index ID。"""
    return tuple(
        VectorAssignment(
            chunk_id=record.chunk_id,
            vector_id=start_after + offset,
            embedding_text_sha256=record.embedding_text_sha256,
        )
        for offset, record in enumerate(artifact.records, start=1)
    )


def _prepare_formal_embedding_candidate(
    *, database: MetadataDatabase, formal_output_dir: Path, result: ChunkingResult,
    staging_artifact: LoadedEmbeddingArtifact, operation_root: Path,
) -> _EmbeddingCandidate:
    """合成“旧正式 vectors + 新 staging vectors”的候选 artifact，不重新 embedding。"""
    state = database.vector_index_state()
    if state.total_chunk_count:
        formal_artifact = load_embedding_artifact(formal_output_dir)
        database.validate_embedding_artifact_records(
            tuple((record.chunk_id, record.embedding_text_sha256) for record in formal_artifact.records)
        )
        if (
            formal_artifact.manifest.get("model_id") != staging_artifact.manifest.get("model_id")
            or formal_artifact.vectors.shape[1] != staging_artifact.vectors.shape[1]
            or not formal_artifact.manifest.get("normalized")
        ):
            raise PromotionError("Staging embeddings are incompatible with the formal embedding artifact.")
        old_inputs_by_id = {
            str(row["chunk_id"]): EmbeddingInput(
                chunk_id=str(row["chunk_id"]), paper_id=str(row["paper_id"]),
                chunk_index=int(row["chunk_index"]), embedding_text=str(row["embedding_text"]),
            )
            for row in database.list_content_embedding_inputs_for_promotion()
        }
        old_inputs = tuple(old_inputs_by_id[record.chunk_id] for record in formal_artifact.records)
        vectors = np.ascontiguousarray(np.vstack((formal_artifact.vectors, staging_artifact.vectors)))
    else:
        old_inputs = ()
        vectors = staging_artifact.vectors
    new_inputs = tuple(
        EmbeddingInput(
            chunk_id=chunk.chunk_id, paper_id=chunk.paper_id, chunk_index=chunk.chunk_index,
            embedding_text=chunk.embedding_text,
        )
        for chunk in result.chunks if chunk.section_type == "content"
    )
    output_dir = operation_root / "embedding_candidate"
    paths = write_embedding_artifact(
        output_dir=output_dir,
        inputs=(*old_inputs, *new_inputs),
        vectors=vectors,
        backend_id=str(staging_artifact.manifest["backend_id"]),
        model_id=str(staging_artifact.manifest["model_id"]),
        normalized=True,
        database_path=database.path,
    )
    return _EmbeddingCandidate(paths=paths)


def _copy_formal_pdf(*, source: Path, papers_dir: Path, source_filename: str, paper_id: str) -> tuple[Path, bool]:
    """将原 PDF 放入正式 library；同名不同内容时使用稳定 paper_id 规避覆盖。"""
    papers_dir.mkdir(parents=True, exist_ok=True)
    target = papers_dir / source_filename
    source_hash = _file_sha256(source)
    if target.is_file() and _file_sha256(target) != source_hash:
        target = papers_dir / f"{paper_id}.pdf"
    if target.is_file():
        if _file_sha256(target) != source_hash:
            raise PromotionError("Formal PDF target already exists with different content.")
        return target, False
    shutil.copy2(source, target)
    return target, True


def _with_formal_source(result: ChunkingResult, source: Path) -> ChunkingResult:
    """只替换保存路径，文本、chunk、section、embedding_text 均保持 staging 原样。"""
    parsed = result.parsed_paper
    return ChunkingResult(
        parsed_paper=ParsedPaper(
            source=source, parser_id=parsed.parser_id, markdown=parsed.markdown,
            page_furniture=parsed.page_furniture, front_matter=parsed.front_matter,
            paper_title=parsed.paper_title, title_source=parsed.title_source,
            title_candidates=parsed.title_candidates, diagnostics=parsed.diagnostics,
            native_document=parsed.native_document, sections=parsed.sections,
        ),
        chunks=tuple(
            PaperChunk(
                **{**chunk.__dict__, "source": source, "vector_id": None}
            )
            for chunk in result.chunks
        ),
        diagnostics=result.diagnostics,
    )


def _artifact_files(output_dir: Path) -> tuple[Path, Path, Path]:
    paths = artifact_paths(output_dir)
    return paths.vectors_path, paths.records_path, paths.manifest_path


def _backup_targets(*, targets: tuple[Path, ...], backup_root: Path) -> list[tuple[Path, Path | None]]:
    """备份固定目标文件；None 表示目标原本不存在，回滚时应删除新文件。"""
    backup_root.mkdir(parents=True, exist_ok=True)
    backups: list[tuple[Path, Path | None]] = []
    for index, target in enumerate(targets):
        if target.is_file():
            backup = backup_root / f"{index}_{target.name}"
            shutil.copy2(target, backup)
            backups.append((target, backup))
        elif target.exists():
            raise PromotionError(f"Promotion target is not a regular file: {target}")
        else:
            backups.append((target, None))
    return backups


def _restore_targets(backups: list[tuple[Path, Path | None]]) -> None:
    """异常路径中恢复 promotion 触及的正式派生文件，不扫描或删除其他工作区。"""
    for target, backup in backups:
        if backup is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)


def _publish_embedding_candidate(*, candidate: _EmbeddingCandidate, formal_output_dir: Path) -> None:
    """候选 artifact 三文件已完整校验；逐一 replace 后可被 loader 作为完整新批次读取。"""
    source = candidate.paths
    destination = artifact_paths(formal_output_dir)
    destination.output_dir.mkdir(parents=True, exist_ok=True)
    for candidate_file, target in (
        (source.vectors_path, destination.vectors_path),
        (source.records_path, destination.records_path),
        (source.manifest_path, destination.manifest_path),
    ):
        os.replace(candidate_file, target)


def _mark_workspace_promoted(root: Path, *, paper_id: str) -> None:
    """成功后保留 workspace，只在其 manifest 增加可审计的 promotion 状态。"""
    path = root / "workspace.json"
    manifest = _json_object(path)
    manifest["added_to_kb"] = True
    manifest["promotion"] = {"paper_id": paper_id, "promoted_at": _utc_timestamp()}
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _success_result(*, prepared: _PromotionInput, imported: PromotionImportSummary, formal_pdf: Path, before: _FormalCounts, after: _FormalCounts) -> PromotionResult:
    return PromotionResult(
        status="promoted", workspace_id=prepared.workspace_id, paper_id=prepared.paper_id,
        formal_pdf_path=formal_pdf, document_count_before=before.documents,
        document_count_after=after.documents, section_count_before=before.sections,
        section_count_after=after.sections, chunk_count_before=before.chunks,
        chunk_count_after=after.chunks, faiss_ntotal_before=before.faiss_ntotal,
        faiss_ntotal_after=after.faiss_ntotal, content_chunk_count=imported.content_chunk_count,
        bibliography_chunk_count=imported.bibliography_chunk_count,
    )


def _already_exists_result(*, prepared: _PromotionInput, counts: _FormalCounts) -> PromotionResult:
    return PromotionResult(
        status="already_exists", workspace_id=prepared.workspace_id, paper_id=prepared.paper_id,
        formal_pdf_path=None, document_count_before=counts.documents, document_count_after=counts.documents,
        section_count_before=counts.sections, section_count_after=counts.sections,
        chunk_count_before=counts.chunks, chunk_count_after=counts.chunks,
        faiss_ntotal_before=counts.faiss_ntotal, faiss_ntotal_after=counts.faiss_ntotal,
    )


def _load_sections(value: object, *, paper_id: str) -> tuple[SectionRecord, ...]:
    if not isinstance(value, list):
        raise PromotionError("parsed_paper.json is missing its sections list.")
    sections: list[SectionRecord] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise PromotionError(f"sections[{index}] must be an object.")
        try:
            section = SectionRecord(**raw)
        except TypeError as error:
            raise PromotionError(f"Invalid SectionRecord at index {index}.") from error
        if section.paper_id != paper_id:
            raise PromotionError("Section paper_id does not match workspace paper_id.")
        sections.append(section)
    if len({item.section_id for item in sections}) != len(sections) or len({item.section_index for item in sections}) != len(sections):
        raise PromotionError("Workspace sections contain duplicate section_id or section_index.")
    return tuple(sorted(sections, key=lambda item: item.section_index))


def _load_chunks(path: Path, *, paper_id: str) -> tuple[PaperChunk, ...]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError("Cannot read workspace chunks.jsonl.") from error
    chunks: list[PaperChunk] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise PromotionError(f"chunks[{index}] must be an object.")
        raw = dict(raw)
        raw["source"] = Path(_required_string(raw.get("source"), f"chunks[{index}].source"))
        try:
            chunk = PaperChunk(**raw)
        except TypeError as error:
            raise PromotionError(f"Invalid PaperChunk at index {index}.") from error
        if chunk.paper_id != paper_id:
            raise PromotionError("Chunk paper_id does not match workspace paper_id.")
        chunks.append(chunk)
    if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise PromotionError("Workspace chunks must be non-empty with unique chunk_id values.")
    if sorted(chunk.chunk_index for chunk in chunks) != list(range(len(chunks))):
        raise PromotionError("Workspace chunk_index values must be contiguous from zero.")
    return tuple(sorted(chunks, key=lambda item: item.chunk_index))


def _workspace_id(value: str) -> str:
    workspace_id = _required_string(value, "workspace_id")
    if not workspace_id.startswith("staging_") or any(char in workspace_id for char in "/\\"):
        raise PromotionError("Invalid workspace_id.")
    return workspace_id


def _safe_pdf_filename(value: object, paper_id: str) -> str:
    if not isinstance(value, str):
        return f"{paper_id}.pdf"
    name = Path(value).name
    if name != value or not name or Path(name).suffix.casefold() != ".pdf":
        return f"{paper_id}.pdf"
    return name


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"JSON artifact must be an object: {path}")
    return value


def _diagnostics(value: object, field_name: str) -> dict[str, int | str]:
    if not isinstance(value, dict):
        raise PromotionError(f"{field_name} must be a diagnostics object.")
    if any(not isinstance(key, str) or not isinstance(item, (str, int)) for key, item in value.items()):
        raise PromotionError(f"{field_name} must contain only string/int values.")
    return dict(value)


def _required_string(value: object, field_name: str) -> str:
    normalized = _optional_string(value)
    if normalized is None:
        raise PromotionError(f"{field_name} must be a non-empty string.")
    return normalized


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
