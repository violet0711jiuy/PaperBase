"""Section Hierarchy Step 1 的纯数据模型回归测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import unittest

from paperbase.chunking.base import PaperChunk
from paperbase.parsing.base import ParsedPaper, SectionRecord


class SectionModelTests(unittest.TestCase):
    """本轮只验证契约；不加载 Docling，也不创建 SQLite schema。"""

    def test_section_records_represent_parent_without_direct_chunk(self) -> None:
        """父章节可独立存在，子章节和 chunk 只引用自己的直属 section。"""
        paper_id = "paper_section_fixture"
        parent = SectionRecord(
            section_id="paper_section_fixture_section_0003",
            paper_id=paper_id,
            section_title="3. Method",
            section_number="3",
            section_level=1,
            parent_section_id=None,
            section_index=3,
            page_start=4,
            page_end=7,
        )
        child = SectionRecord(
            section_id="paper_section_fixture_section_0004",
            paper_id=paper_id,
            section_title="3.1. Encoder",
            section_number="3.1",
            section_level=2,
            parent_section_id=parent.section_id,
            section_index=4,
            page_start=4,
            page_end=5,
        )
        chunk = _chunk(paper_id=paper_id, section_id=child.section_id)
        parsed = _parsed_paper(sections=(parent, child))

        self.assertIsNone(parent.parent_section_id)
        self.assertEqual(child.parent_section_id, parent.section_id)
        self.assertEqual(chunk.section_id, child.section_id)
        self.assertNotIn(parent.section_id, {chunk.section_id})
        self.assertEqual(asdict(parsed)["sections"][1]["section_number"], "3.1")
        self.assertEqual(asdict(chunk)["section_id"], child.section_id)

    def test_new_fields_default_to_empty_or_none_for_legacy_callers(self) -> None:
        """旧 Parser 和 Chunker 不填 hierarchy 时，新增字段必须保持兼容默认值。"""
        parsed = _parsed_paper()
        chunk = _chunk()

        self.assertEqual(parsed.sections, ())
        self.assertIsNone(chunk.section_id)
        self.assertIn("sections", asdict(parsed))
        self.assertIn("section_id", asdict(chunk))


def _parsed_paper(*, sections: tuple[SectionRecord, ...] = ()) -> ParsedPaper:
    """构造不依赖真实 Parser 的最小 ParsedPaper，验证默认参数与序列化契约。"""
    return ParsedPaper(
        source=Path("fixture.pdf"),
        parser_id="fixture_parser",
        markdown="# Fixture paper",
        page_furniture=(),
        front_matter=(),
        paper_title="Fixture paper",
        title_source="fixture",
        title_candidates=("Fixture paper",),
        diagnostics={},
        native_document=object(),
        sections=sections,
    )


def _chunk(
    *, paper_id: str = "paper_section_fixture", section_id: str | None = None
) -> PaperChunk:
    """构造一块正文，section 旧字段与新的直属 section_id 可独立共存。"""
    return PaperChunk(
        chunk_id=f"{paper_id}_chunk_0000",
        vector_id=None,
        paper_id=paper_id,
        paper_title="Fixture paper",
        source=Path("fixture.pdf"),
        chunk_index=0,
        raw_text="A fixture paragraph.",
        embedding_text="Paper title: Fixture paper\nContent:\nA fixture paragraph.",
        section="3. Method > 3.1. Encoder",
        page_start=4,
        page_end=4,
        raw_token_count=5,
        embedding_token_count=10,
        prev_chunk_id=None,
        next_chunk_id=None,
        section_id=section_id,
    )


if __name__ == "__main__":
    unittest.main()
