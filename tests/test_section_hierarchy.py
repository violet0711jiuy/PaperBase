"""Section Hierarchy Step 2 的纯逻辑回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from docling_core.types.doc import DocItemLabel

from paperbase.chunking.docling_hybrid_chunker import _section_id_from_headings
from paperbase.parsing.base import FrontMatterBlock, SectionRecord
from paperbase.parsing.docling_parser import _build_section_records


class SectionHierarchyTests(unittest.TestCase):
    """不加载 PDF/GPU，只验证原生 level 到统一章节树和 chunk 映射的规则。"""

    def test_docling_levels_build_parent_tree_and_exclude_non_body_headers(self) -> None:
        """父章节无直属正文也保留；front matter、算法标签与参考文献不进入正文树。"""
        document = _document(
            _heading("Paper title", level=1, page=1),
            _heading("Authors and affiliations", level=1, page=1),
            _heading("Abstract", level=1, page=1),
            _heading("3 Methodology", level=2, page=3),
            _heading("3.1 Enhanced MVE", level=3, page=3),
            _heading("3.1.1 Standard MVE", level=4, page=4),
            _heading("3.1.2 Variant MVE", level=4, page=5),
            _heading("Algorithm 1 MVE", level=1, page=5),
            _heading("Input:", level=1, page=5),
            _heading("4 Experiments", level=2, page=6),
            _heading("References", level=1, page=10),
        )
        sections = _build_section_records(
            document=document,
            paper_id="paper_fixture",
            paper_title="Paper title",
            front_matter=(
                _front_matter("authors_affiliations", "Authors and affiliations"),
                _front_matter("abstract", "Abstract"),
            ),
        )

        self.assertEqual(
            [section.section_title for section in sections],
            [
                "3 Methodology",
                "3.1 Enhanced MVE",
                "3.1.1 Standard MVE",
                "3.1.2 Variant MVE",
                "4 Experiments",
            ],
        )
        self.assertEqual([section.section_level for section in sections], [1, 2, 3, 3, 1])
        self.assertIsNone(sections[0].parent_section_id)
        self.assertEqual(sections[1].parent_section_id, sections[0].section_id)
        self.assertEqual(sections[2].parent_section_id, sections[1].section_id)
        self.assertEqual(sections[3].parent_section_id, sections[1].section_id)
        self.assertIsNone(sections[4].parent_section_id)
        self.assertEqual(sections[0].page_start, 3)
        self.assertEqual(sections[2].section_number, "3.1.1")

    def test_chunk_mapping_uses_leaf_heading_and_resolves_duplicate_by_page(self) -> None:
        """chunk 只绑定最近的直属末级 heading，重复标题按页码选择阅读顺序最近节点。"""
        first = _section("paper_fixture_section_0000", "3.1 Details", 3, 1)
        second = _section("paper_fixture_section_0004", "3.1 Details", 7, 5)
        mapped = _section_id_from_headings(
            headings=["3 Methodology", "3.1 Details"],
            sections=(first, second),
            page_start=8,
        )
        missing = _section_id_from_headings(
            headings=["References"],
            sections=(first, second),
            page_start=10,
        )
        algorithm = _section_id_from_headings(
            headings=["3 Methodology", "Input:"],
            sections=(_section("paper_fixture_section_0005", "3 Methodology", 2, 0),),
            page_start=4,
        )

        self.assertEqual(mapped, second.section_id)
        self.assertIsNone(missing)
        self.assertEqual(algorithm, "paper_fixture_section_0005")


def _heading(text: str, *, level: int, page: int) -> SimpleNamespace:
    """构造带原生 heading level 与 provenance 的最小 Docling 标题替身。"""
    return SimpleNamespace(
        label=DocItemLabel.SECTION_HEADER,
        text=text,
        level=level,
        prov=[SimpleNamespace(page_no=page)],
        children=[],
    )


def _document(*items: SimpleNamespace) -> SimpleNamespace:
    """以固定 reading order 返回标题，模拟本轮实际消费的 Docling 接口。"""
    return SimpleNamespace(iterate_items=lambda: iter((item, 1) for item in items))


def _front_matter(block_type: str, canonical_section: str) -> FrontMatterBlock:
    """构造无正文的前置元数据块，仅验证其章节排除语义。"""
    return FrontMatterBlock(
        block_type=block_type,
        canonical_section=canonical_section,
        text="fixture",
        page_start=1,
        page_end=1,
        source_item_count=1,
        detection_method="fixture",
        confidence="high",
    )


def _section(section_id: str, title: str, page: int, index: int) -> SectionRecord:
    """构造最小 SectionRecord，验证 Chunker 不依赖 raw_text 进行映射。"""
    return SectionRecord(
        section_id=section_id,
        paper_id="paper_fixture",
        section_title=title,
        section_number="3.1",
        section_level=2,
        parent_section_id=None,
        section_index=index,
        page_start=page,
        page_end=page,
    )


if __name__ == "__main__":
    unittest.main()
