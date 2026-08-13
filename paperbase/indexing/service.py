"""Step 5：从 Step 4 staging 工件建立或验证正式全局 FAISS 索引。"""

from __future__ import annotations

import argparse
from pathlib import Path

from paperbase.config import AppSettings, default_config_path, load_settings
from paperbase.database import MetadataDatabase

from .faiss_store import FaissIndexStore, IndexBuildSummary


def run_initial_index_build(settings: AppSettings) -> IndexBuildSummary:
    """按全局配置执行首次全库建索引，不重跑 Parse、Chunk 或 Embedding。"""
    if settings.indexing.backend != "faiss_flat_ip":
        raise ValueError(
            f"Unsupported indexing backend: {settings.indexing.backend!r}"
        )
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    store = FaissIndexStore(settings.indexing)
    return store.build_initial_index(
        database=database,
        embedding_output_dir=settings.embedding.output_dir,
    )


def run_vector_reset_for_rebuild(settings: AppSettings) -> int:
    """只清空 SQLite 的派生 vector_id，不触碰 PDF、chunks 或 FTS5。"""
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    return database.reset_vector_ids_for_rebuild()


def run_rebuild(settings: AppSettings) -> IndexBuildSummary:
    """在已显式清空 vector_id 并重新生成 Step 4 工件后，原子替换旧 FAISS/manifest。"""
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    store = FaissIndexStore(settings.indexing)
    return store.build_initial_index(
        database=database,
        embedding_output_dir=settings.embedding.output_dir,
        replace_existing=True,
    )


def run_index_verification(settings: AppSettings) -> IndexBuildSummary:
    """只读验证 FAISS、SQLite 与 Step 4 向量工件三者的一致性。"""
    if settings.indexing.backend != "faiss_flat_ip":
        raise ValueError(
            f"Unsupported indexing backend: {settings.indexing.backend!r}"
        )
    database = MetadataDatabase(
        settings.database.path,
        busy_timeout_ms=settings.database.busy_timeout_ms,
    )
    store = FaissIndexStore(settings.indexing)
    return store.verify_against_sqlite(
        database=database,
        embedding_output_dir=settings.embedding.output_dir,
    )


def main() -> None:
    """运行首次建库，或通过 ``--verify`` 验证已发布的正式索引。"""
    argument_parser = argparse.ArgumentParser(
        description="Build or verify the PaperBase global FAISS index."
    )
    argument_parser.add_argument(
        "--config", type=Path, default=default_config_path()
    )
    argument_parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify the existing FAISS index, SQLite vector IDs, and embedding artifact.",
    )
    argument_parser.add_argument(
        "--reset-for-rebuild",
        action="store_true",
        help="Only clear SQLite vector_id mappings; preserve PDFs, chunks, and FTS5 for a subsequent rebuild.",
    )
    argument_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Atomically replace the existing FAISS index after --reset-for-rebuild and Step 4 embedding generation.",
    )
    args = argument_parser.parse_args()
    settings = load_settings(args.config)
    if args.verify and (args.reset_for_rebuild or args.rebuild) or args.reset_for_rebuild and args.rebuild:
        argument_parser.error("--verify, --reset-for-rebuild, and --rebuild are mutually exclusive.")
    if args.reset_for_rebuild:
        cleared = run_vector_reset_for_rebuild(settings)
        print(f"cleared_vector_ids={cleared}; PDFs, chunks, and FTS5 were preserved.")
        return
    summary = (
        run_index_verification(settings)
        if args.verify
        else (run_rebuild(settings) if args.rebuild else run_initial_index_build(settings))
    )
    action = "verified" if args.verify else ("rebuilt" if args.rebuild else "built")
    recovery_suffix = " (recovered pending publish)" if summary.recovered_pending_publish else ""
    print(
        f"{action}_vectors={summary.vector_count}, dimension={summary.dimension}, "
        f"vector_ids={summary.vector_id_min}..{summary.vector_id_max}{recovery_suffix}\n"
        f"index={summary.index_path}\n"
        f"manifest={summary.manifest_path}"
    )


if __name__ == "__main__":
    main()
