"""Step 4：从 SQLite chunk 输入生成本地 embedding staging 工件。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from paperbase.config import AppSettings, default_config_path, load_settings
from paperbase.database import MetadataDatabase

from .artifacts import EmbeddingArtifactPaths, write_embedding_artifact
from .base import DocumentEmbedder, EmbeddingInput
from .factory import create_document_embedder


@dataclass(frozen=True)
class EmbeddingRunSummary:
    """一次 Step 4 运行的轻量结果，方便 CLI、测试和未来服务层展示。"""

    chunk_count: int
    dimension: int
    backend_id: str
    model_id: str
    artifact_paths: EmbeddingArtifactPaths


def generate_database_embeddings(
    *,
    database: MetadataDatabase,
    embedder: DocumentEmbedder,
    output_dir: Path,
    normalized: bool,
) -> EmbeddingRunSummary:
    """读取当前 SQLite chunks，生成且仅生成 Step 4 的向量工件。

    这条链路刻意从正式数据库读取，而非重新解析 PDF 或读取检查 JSONL：Step 3 已经固定了
    当前知识库应使用的 chunk 集合，故 Step 4 可以独立重跑，并且不会额外消耗 Docling/GPU
    解析资源。它不创建 FAISS，也不更新 ``chunks.vector_id``。
    """
    rows = database.list_embedding_inputs()
    inputs = tuple(_embedding_input_from_row(row) for row in rows)
    vectors = embedder.embed_documents([item.embedding_text for item in inputs])
    paths = write_embedding_artifact(
        output_dir=output_dir,
        inputs=inputs,
        vectors=vectors,
        backend_id=embedder.backend_id,
        model_id=embedder.model_id,
        normalized=normalized,
        database_path=database.path,
    )
    return EmbeddingRunSummary(
        chunk_count=len(inputs),
        dimension=int(vectors.shape[1]),
        backend_id=embedder.backend_id,
        model_id=embedder.model_id,
        artifact_paths=paths,
    )


def run_embedding_stage(settings: AppSettings) -> EmbeddingRunSummary:
    """按全局配置执行 Step 4，集中创建数据库读取器与模型适配器。"""
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    embedder = create_document_embedder(settings.embedding)
    return generate_database_embeddings(
        database=database,
        embedder=embedder,
        output_dir=settings.embedding.output_dir,
        normalized=settings.embedding.normalize_embeddings,
    )


def _embedding_input_from_row(row: object) -> EmbeddingInput:
    """将 SQLite Row 转为无 SQLite 依赖的不可变输入对象，便于单测与未来替换存储层。"""
    try:
        return EmbeddingInput(
            chunk_id=str(row["chunk_id"]),  # type: ignore[index]
            paper_id=str(row["paper_id"]),  # type: ignore[index]
            chunk_index=int(row["chunk_index"]),  # type: ignore[index]
            embedding_text=str(row["embedding_text"]),  # type: ignore[index]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("SQLite embedding input row is missing a required field.") from error


def main() -> None:
    """运行 ``SQLite chunks -> local embedding artifacts`` 的 Step 4 命令行入口。"""
    argument_parser = argparse.ArgumentParser(
        description="Generate PaperBase embedding artifacts from SQLite chunks."
    )
    argument_parser.add_argument(
        "--config", type=Path, default=default_config_path()
    )
    args = argument_parser.parse_args()
    settings = load_settings(args.config)
    summary = run_embedding_stage(settings)
    print(
        f"embedded_chunks={summary.chunk_count}, dimension={summary.dimension}, "
        f"backend={summary.backend_id}, model={summary.model_id}\n"
        f"vectors={summary.artifact_paths.vectors_path}\n"
        f"records={summary.artifact_paths.records_path}\n"
        f"manifest={summary.artifact_paths.manifest_path}"
    )


if __name__ == "__main__":
    main()
