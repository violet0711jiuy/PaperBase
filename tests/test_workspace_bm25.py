"""Temporary Workspace 内存 BM25 的隔离、字段与缓存回归测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from paperbase.staging.bm25 import (
    WorkspaceBM25Error,
    WorkspaceBM25Index,
    WorkspaceBM25IndexCache,
)
from paperbase.staging.sections import WorkspaceChunk, WorkspaceSectionSnapshot


class WorkspaceBM25Tests(unittest.TestCase):
    """不读取 SQLite、FAISS、Embedding 或正式 Knowledge Base。"""

    def test_search_only_indexes_current_workspace_content_chunks(self) -> None:
        snapshot = _snapshot()
        index = WorkspaceBM25Index.build(snapshot)

        matches = index.search("dynamic time warping", top_k=5)

        self.assertEqual([match.chunk.chunk_id for match in matches], ["current_result"])
        self.assertEqual(matches[0].rank, 1)
        self.assertGreater(matches[0].bm25_score, 0)
        self.assertNotIn("bibliography_only", [match.chunk.chunk_id for match in matches])
        # 另一个 workspace 的 chunk 从未传入，无法被当前索引意外检索到。
        self.assertNotIn("other_workspace", [match.chunk.chunk_id for match in matches])

    def test_section_field_and_chinese_terms_participate_in_matching(self) -> None:
        index = WorkspaceBM25Index.build(_snapshot())

        matches = index.search("实验结果", top_k=2)

        self.assertEqual([match.chunk.chunk_id for match in matches], ["current_result"])

    def test_cache_reuses_same_content_and_rebuilds_after_content_change(self) -> None:
        cache = WorkspaceBM25IndexCache(max_entries=1)
        snapshot = _snapshot()
        first = cache.get_or_build(snapshot)
        second = cache.get_or_build(snapshot)
        self.assertIs(first, second)

        changed_chunk = replace(snapshot.chunks[0], raw_text="changed extrema method")
        changed = replace(snapshot, chunks=(changed_chunk, *snapshot.chunks[1:]))
        rebuilt = cache.get_or_build(changed)
        self.assertIsNot(first, rebuilt)
        self.assertEqual([item.chunk.chunk_id for item in rebuilt.search("extrema", top_k=2)], ["current_intro"])

        cache.invalidate(snapshot.workspace_id)
        after_invalidate = cache.get_or_build(changed)
        self.assertIsNot(rebuilt, after_invalidate)

    def test_invalid_query_and_top_k_are_explicit(self) -> None:
        index = WorkspaceBM25Index.build(_snapshot())
        with self.assertRaises(WorkspaceBM25Error):
            index.search("   ", top_k=1)
        with self.assertRaises(WorkspaceBM25Error):
            index.search("method", top_k=0)


def _snapshot() -> WorkspaceSectionSnapshot:
    """建立一个只含当前论文的最小 staging 快照。"""
    return WorkspaceSectionSnapshot(
        workspace_id="staging_current",
        root_dir=Path("/temporary/staging_current"),
        paper_id="paper_current",
        sections=(),
        chunks=(
            WorkspaceChunk(
                chunk_id="current_intro",
                chunk_index=1,
                raw_text="The method extracts local extrema before matching shapes.",
                raw_token_count=10,
                section="1 Introduction",
                section_id="section_1",
                section_type="content",
            ),
            WorkspaceChunk(
                chunk_id="current_result",
                chunk_index=2,
                raw_text="Our experiments report improved dynamic time warping accuracy.",
                raw_token_count=10,
                section="4 实验结果",
                section_id="section_4",
                section_type="content",
            ),
            WorkspaceChunk(
                chunk_id="bibliography_only",
                chunk_index=3,
                raw_text="Dynamic time warping reference citation.",
                raw_token_count=6,
                section="References",
                section_id=None,
                section_type="bibliography",
            ),
        ),
    )
