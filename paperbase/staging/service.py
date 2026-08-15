"""创建、保存和删除 v0.2 的单篇论文临时工作区。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from paperbase.chunking.base import ChunkingResult, PaperChunk, PaperChunker
from paperbase.chunking.factory import create_chunker
from paperbase.config import AppSettings
from paperbase.embedding.artifacts import EmbeddingArtifactPaths, write_embedding_artifact
from paperbase.embedding.base import DocumentEmbedder, EmbeddingInput
from paperbase.embedding.factory import create_document_embedder
from paperbase.indexing.faiss_store import write_flat_ip_index
from paperbase.parsing.base import ParsedPaper, PaperParser
from paperbase.parsing.factory import create_parser


class TemporaryWorkspaceError(RuntimeError):
    """临时工作区输入、产物或隔离边界不符合约定时抛出。"""


@dataclass(frozen=True)
class TemporaryPaperWorkspace:
    """一个已完成的临时论文工作区及其关键产物路径。"""

    workspace_id: str
    paper_id: str
    root_dir: Path
    original_pdf_path: Path
    parsed_markdown_path: Path
    parsed_data_path: Path
    chunks_path: Path
    chunk_metadata_path: Path
    embedding_paths: EmbeddingArtifactPaths
    index_path: Path
    index_manifest_path: Path
    total_chunk_count: int
    searchable_chunk_count: int


def run_temporary_workspace_stage(
    *, settings: AppSettings, source_pdf: Path
) -> TemporaryPaperWorkspace:
    """按现有配置创建临时工作区，不读取或写入正式知识库。"""
    return create_temporary_workspace(
        staging_dir=settings.storage.staging_dir,
        source_pdf=source_pdf,
        parser=create_parser(settings),
        chunker=create_chunker(settings),
        embedder=create_document_embedder(settings.embedding),
        normalized_embeddings=settings.embedding.normalize_embeddings,
    )


def create_temporary_workspace(
    *,
    staging_dir: Path,
    source_pdf: Path,
    parser: PaperParser,
    chunker: PaperChunker,
    embedder: DocumentEmbedder,
    normalized_embeddings: bool,
) -> TemporaryPaperWorkspace:
    """执行 ``PDF -> Parse -> Chunk -> Embedding -> Temporary FAISS``。

    所有可写路径均在 ``staging_dir/<workspace_id>`` 内。解析和分块仍使用 v0.1
    的统一数据类，因此未来“确认入库”可以直接读取 chunks 与 embedding 工件。
    """
    source = source_pdf.resolve()
    _validate_source_pdf(source)
    staging_root = staging_dir.resolve()
    workspace_id = f"staging_{uuid4().hex}"
    final_root = staging_root / workspace_id
    temporary_root = staging_root / f".{workspace_id}.tmp"
    if final_root.exists() or temporary_root.exists():
        raise TemporaryWorkspaceError(f"Temporary workspace ID already exists: {workspace_id}")

    try:
        # 先写入不可见的同级临时目录；所有步骤成功后才一次性发布工作区目录。
        temporary_root.mkdir(parents=True)
        original_pdf_path = temporary_root / "original" / "paper.pdf"
        original_pdf_path.parent.mkdir()
        shutil.copy2(source, original_pdf_path)

        # Parser 和 Chunker 接收原始源文件，保留 v0.1 的 source 哈希式 paper_id 规则。
        parsed_paper = parser.parse(source)
        result = chunker.chunk(parsed_paper)
        paper_id = _validate_chunking_result(result, source)

        parsed_markdown_path, parsed_data_path = _write_parsed_artifacts(
            parsed_paper, temporary_root / "parsed"
        )
        chunks_path, chunk_metadata_path = _write_chunk_artifacts(
            result, temporary_root / "chunks"
        )

        # bibliography chunk 完整保存在 chunks.jsonl，但绝不嵌入或写入临时主索引。
        searchable_chunks = tuple(
            chunk for chunk in result.chunks if chunk.section_type == "content"
        )
        if not searchable_chunks:
            raise TemporaryWorkspaceError("A temporary paper needs at least one content chunk.")
        inputs = tuple(_embedding_input(chunk) for chunk in searchable_chunks)
        vectors = embedder.embed_documents([item.embedding_text for item in inputs])
        embedding_paths = write_embedding_artifact(
            output_dir=temporary_root / "embeddings",
            inputs=inputs,
            vectors=vectors,
            backend_id=embedder.backend_id,
            model_id=embedder.model_id,
            normalized=normalized_embeddings,
            # 临时工作区没有 SQLite；该字段明确记录实际的工作区语义来源。
            database_path=temporary_root / "workspace.json",
        )
        index_path, index_manifest_path = _write_temporary_index(
            workspace_root=temporary_root,
            inputs=inputs,
            vectors=np.asarray(vectors),
            normalized=normalized_embeddings,
        )
        _write_workspace_manifest(
            root=temporary_root,
            workspace_id=workspace_id,
            paper_id=paper_id,
            source=source,
            result=result,
            searchable_chunk_count=len(searchable_chunks),
            embedding_paths=embedding_paths,
            index_path=index_path,
        )
        os.replace(temporary_root, final_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    # 目录发布后，统一将每条路径改为正式路径，调用方不会持有已不存在的 tmp 路径。
    return TemporaryPaperWorkspace(
        workspace_id=workspace_id,
        paper_id=paper_id,
        root_dir=final_root,
        original_pdf_path=final_root / "original" / "paper.pdf",
        parsed_markdown_path=final_root / "parsed" / "paper.md",
        parsed_data_path=final_root / "parsed" / "parsed_paper.json",
        chunks_path=final_root / "chunks" / "chunks.jsonl",
        chunk_metadata_path=final_root / "chunks" / "metadata.json",
        embedding_paths=EmbeddingArtifactPaths(
            output_dir=final_root / "embeddings",
            vectors_path=final_root / "embeddings" / "vectors.npy",
            records_path=final_root / "embeddings" / "records.jsonl",
            manifest_path=final_root / "embeddings" / "manifest.json",
        ),
        index_path=final_root / "index" / "paper.faiss",
        index_manifest_path=final_root / "index" / "manifest.json",
        total_chunk_count=len(result.chunks),
        searchable_chunk_count=len(searchable_chunks),
    )


def delete_temporary_workspace(*, staging_dir: Path, workspace_id: str) -> None:
    """只删除一个已验证位于 staging 根目录下的临时工作区。"""
    if not workspace_id.startswith("staging_") or any(
        separator in workspace_id for separator in ("/", "\\")
    ):
        raise TemporaryWorkspaceError("Invalid temporary workspace ID.")
    root = staging_dir.resolve()
    target = (root / workspace_id).resolve()
    if target.parent != root:
        raise TemporaryWorkspaceError("Temporary workspace path escapes the staging directory.")
    if target.exists():
        shutil.rmtree(target)


def _validate_source_pdf(source: Path) -> None:
    """在启动昂贵的 Docling/GPU 流程前验证单个本地 PDF 输入。"""
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    if source.suffix.casefold() != ".pdf":
        raise TemporaryWorkspaceError(f"Expected a PDF file, got: {source.name}")


def _validate_chunking_result(result: ChunkingResult, source: Path) -> str:
    """确认 Chunker 的 v0.1 身份链条仍对应本次上传的 PDF。"""
    if not result.chunks:
        raise TemporaryWorkspaceError("Parser completed but Chunker returned no chunks.")
    if result.parsed_paper.source.resolve() != source:
        raise TemporaryWorkspaceError("Parsed paper source does not match the uploaded PDF.")
    paper_ids = {chunk.paper_id for chunk in result.chunks}
    if len(paper_ids) != 1:
        raise TemporaryWorkspaceError("Temporary workspace chunks must belong to one paper.")
    return next(iter(paper_ids))


def _write_parsed_artifacts(parsed_paper: ParsedPaper, output_dir: Path) -> tuple[Path, Path]:
    """保存统一结构数据与 Markdown，并在可用时附带 Docling 原生 JSON。"""
    output_dir.mkdir(parents=True)
    markdown_path = output_dir / "paper.md"
    data_path = output_dir / "parsed_paper.json"
    markdown_path.write_text(parsed_paper.markdown, encoding="utf-8")
    data_path.write_text(
        json.dumps(_parsed_paper_record(parsed_paper), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Docling 的 export_to_dict 与 v0.1 inspection 的结构相同，供后续阅读工作区复用。
    export_to_dict = getattr(parsed_paper.native_document, "export_to_dict", None)
    if callable(export_to_dict):
        (output_dir / "docling_document.json").write_text(
            json.dumps(export_to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return markdown_path, data_path


def _write_chunk_artifacts(result: ChunkingResult, output_dir: Path) -> tuple[Path, Path]:
    """以 v0.1 inspection 相同的 PaperChunk JSONL 保存全部 chunk。"""
    output_dir.mkdir(parents=True)
    chunks_path = output_dir / "chunks.jsonl"
    metadata_path = output_dir / "metadata.json"
    chunks_path.write_text(
        "".join(
            json.dumps(_chunk_record(chunk), ensure_ascii=False, sort_keys=True) + "\n"
            for chunk in result.chunks
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "paper_id": result.chunks[0].paper_id,
                "chunk_count": len(result.chunks),
                "searchable_chunk_count": sum(
                    chunk.section_type == "content" for chunk in result.chunks
                ),
                "chunking_diagnostics": result.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return chunks_path, metadata_path


def _write_temporary_index(
    *,
    workspace_root: Path,
    inputs: tuple[EmbeddingInput, ...],
    vectors: np.ndarray,
    normalized: bool,
) -> tuple[Path, Path]:
    """写入仅含 content chunk 的独立 FAISS 与可审计 chunk 映射清单。"""
    if not normalized:
        raise TemporaryWorkspaceError("Temporary faiss_flat_ip index requires normalized embeddings.")
    if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[0] != len(inputs):
        raise TemporaryWorkspaceError("Embedding vectors do not match temporary index inputs.")
    index_dir = workspace_root / "index"
    index_path = index_dir / "paper.faiss"
    manifest_path = index_dir / "manifest.json"
    vector_ids = np.arange(1, len(inputs) + 1, dtype=np.int64)
    write_flat_ip_index(path=index_path, vectors=vectors, vector_ids=vector_ids)
    manifest_path.write_text(
        json.dumps(
            {
                "index_schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backend_id": "faiss_flat_ip",
                "index_type": "IndexIDMap2(IndexFlatIP)",
                "metric": "inner_product",
                "dimension": int(vectors.shape[1]),
                "vector_count": len(inputs),
                "included_section_type": "content",
                "records": [
                    {
                        "vector_id": int(vector_id),
                        "chunk_id": item.chunk_id,
                        "paper_id": item.paper_id,
                        "chunk_index": item.chunk_index,
                        "embedding_text_sha256": _text_sha256(item.embedding_text),
                    }
                    for vector_id, item in zip(vector_ids, inputs, strict=True)
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return index_path, manifest_path


def _write_workspace_manifest(
    *,
    root: Path,
    workspace_id: str,
    paper_id: str,
    source: Path,
    result: ChunkingResult,
    searchable_chunk_count: int,
    embedding_paths: EmbeddingArtifactPaths,
    index_path: Path,
) -> None:
    """记录后续阅读和增量入库所需的定位信息，不复制全文或向量。"""
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "workspace_schema_version": 1,
                "workspace_id": workspace_id,
                "paper_id": paper_id,
                "source_filename": source.name,
                "source_sha256": _file_sha256(source),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parser_id": result.parsed_paper.parser_id,
                "chunker_id": result.diagnostics.get("chunking.chunker_id"),
                "total_chunk_count": len(result.chunks),
                "searchable_chunk_count": searchable_chunk_count,
                "embedding_dir": str(embedding_paths.output_dir.relative_to(root)),
                "index_path": str(index_path.relative_to(root)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _parsed_paper_record(parsed_paper: ParsedPaper) -> dict[str, Any]:
    """把解析器无关对象转换为 JSON 基础类型，避免持久化第三方运行时对象。"""
    return {
        "source": str(parsed_paper.source),
        "parser_id": parsed_paper.parser_id,
        "paper_title": parsed_paper.paper_title,
        "title_source": parsed_paper.title_source,
        "title_candidates": list(parsed_paper.title_candidates),
        "diagnostics": dict(parsed_paper.diagnostics),
        "page_furniture": [asdict(item) for item in parsed_paper.page_furniture],
        "front_matter": [asdict(item) for item in parsed_paper.front_matter],
    }


def _chunk_record(chunk: PaperChunk) -> dict[str, Any]:
    """复用 v0.1 的 PaperChunk 字段名；Path 单独转为 JSON 字符串。"""
    record = asdict(chunk)
    record["source"] = str(chunk.source)
    return record


def _embedding_input(chunk: PaperChunk) -> EmbeddingInput:
    """仅从 v0.1 固化的 embedding_text 生成向量输入，不重新拼接论文文本。"""
    return EmbeddingInput(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        chunk_index=chunk.chunk_index,
        embedding_text=chunk.embedding_text,
    )


def _file_sha256(path: Path) -> str:
    """流式计算 PDF 哈希，避免大文件一次性占用内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    """保持与 embedding 工件相同的 UTF-8 文本哈希语义。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
