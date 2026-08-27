"""Section Hierarchy Step 2 的纯逻辑回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from docling_core.types.doc import DocItemLabel

from paperbase.chunking.docling_hybrid_chunker import (
    _front_matter_heading_for_chunk,
    _section_id_from_headings,
)
from paperbase.parsing.base import (
    FrontMatterBlock,
    FrontMatterHeading,
    SectionRecord,
)
from paperbase.parsing.docling_parser import (
    _build_section_records,
    _resolve_front_matter_headings,
)


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

    def test_nested_bibliography_never_falls_back_to_conclusion_section(self) -> None:
        """References 即使带有错误的 Conclusion 祖先，也不能绑定正文 section_id。"""
        conclusion = _section(
            "paper_fixture_section_0000", "7. Conclusion", 19, 0
        )

        mapped = _section_id_from_headings(
            headings=["7. Conclusion", "References"],
            sections=(conclusion,),
            page_start=20,
        )

        self.assertIsNone(mapped)

    def test_back_matter_headers_are_independent_roots_despite_docling_level(self) -> None:
        """受控后置标题即使被 Docling 标为三级，也应成为独立根章节。"""
        document = _document(
            _heading("7. Conclusion", level=1, page=19),
            _heading("CRediT authorship contribution statement", level=3, page=19),
            _heading("Declaration of competing interest", level=3, page=19),
            _heading("Acknowledgment", level=3, page=20),
            _heading("Data availability", level=3, page=20),
            _heading("References", level=3, page=20),
        )

        sections = _build_section_records(
            document=document,
            paper_id="paper_fixture",
            paper_title="Fixture paper",
            front_matter=(),
        )

        self.assertEqual(
            [section.section_title for section in sections],
            [
                "7. Conclusion",
                "CRediT authorship contribution statement",
                "Declaration of competing interest",
                "Acknowledgment",
                "Data availability",
            ],
        )
        self.assertTrue(all(section.section_level == 1 for section in sections))
        self.assertTrue(all(section.parent_section_id is None for section in sections))

    def test_late_reading_order_front_matter_parent_and_children_stay_out_of_body_tree(self) -> None:
        """双栏侧栏可在 Introduction 后出现，但其自身 provenance 仍优先于阅读顺序。"""
        document = _document(
            _heading("1. Introduction", level=1, page=2, tree_level=1),
            _heading("ARTICLEHISTORY", level=1, page=2, tree_level=1),
            _heading("Publisher note", level=2, page=2, tree_level=2),
            _heading("Keywords", level=2, page=2, tree_level=2),
            _heading("2. Related work", level=1, page=4, tree_level=1),
        )
        front_matter = (
            _front_matter("article_info", "Front matter > Article information"),
            _front_matter("keywords", "Front matter > Keywords"),
        )
        front_matter_headings = _resolve_front_matter_headings(
            document=document,
            front_matter=front_matter,
            max_pages=2,
        )
        sections = _build_section_records(
            document=document,
            paper_id="paper_fixture",
            paper_title="Fixture paper",
            front_matter=front_matter,
            front_matter_headings=front_matter_headings,
            front_matter_max_pages=2,
        )

        self.assertEqual(
            [(heading.heading_text, heading.block_type) for heading in front_matter_headings],
            [
                ("ARTICLEHISTORY", "article_info"),
                ("Publisher note", "article_info"),
                ("Keywords", "keywords"),
            ],
        )
        self.assertEqual(
            [section.section_title for section in sections],
            ["1. Introduction", "2. Related work"],
        )

    def test_front_matter_before_body_is_filtered_without_changing_body_hierarchy(self) -> None:
        """常规首页元数据也走同一条页码、语义和 provenance 规则。"""
        document = _document(
            _heading("ARTICLE HISTORY", level=1, page=1, tree_level=1),
            _heading("Keywords", level=2, page=1, tree_level=2),
            _heading("1. Introduction", level=1, page=2, tree_level=1),
            _heading("1.1 Contributions", level=2, page=2, tree_level=2),
        )
        front_matter = (
            _front_matter("article_info", "Front matter > Article information"),
            _front_matter("keywords", "Front matter > Keywords"),
        )
        headings = _resolve_front_matter_headings(
            document=document, front_matter=front_matter, max_pages=2
        )
        sections = _build_section_records(
            document=document,
            paper_id="paper_fixture",
            paper_title="Fixture paper",
            front_matter=front_matter,
            front_matter_headings=headings,
            front_matter_max_pages=2,
        )
        self.assertEqual(
            [section.section_title for section in sections],
            ["1. Introduction", "1.1 Contributions"],
        )
        self.assertEqual(sections[1].parent_section_id, sections[0].section_id)

    def test_normal_body_heading_on_early_page_is_not_removed_by_front_matter_filter(self) -> None:
        """前置页码窗口本身不能删除普通正文标题或字符串相近的标题。"""
        document = _document(
            _heading("1. Introduction", level=1, page=1, tree_level=1),
            _heading("Keywords analysis for model selection", level=2, page=1, tree_level=2),
            _heading("2. Methodology", level=1, page=3, tree_level=1),
        )
        sections = _build_section_records(
            document=document,
            paper_id="paper_fixture",
            paper_title="Fixture paper",
            front_matter=(),
            front_matter_headings=(),
            front_matter_max_pages=2,
        )
        self.assertEqual(
            [section.section_title for section in sections],
            [
                "1. Introduction",
                "Keywords analysis for model selection",
                "2. Methodology",
            ],
        )

    def test_chunk_front_matter_match_uses_parser_heading_provenance(self) -> None:
        """Chunker 仅消费 Parser 决议，前置 chunk 不会错误绑定 Introduction。"""
        record = FrontMatterHeading(
            heading_text="Article information",
            block_type="article_info",
            canonical_section="Front matter > Article information",
            page_start=2,
            page_end=2,
            reading_order=8,
        )
        matched = _front_matter_heading_for_chunk(
            headings=["1. Introduction", "Article information"],
            page_start=2,
            page_end=2,
            front_matter_headings=(record,),
        )
        self.assertEqual(matched, record)


def _heading(
    text: str, *, level: int, page: int, tree_level: int = 1
) -> SimpleNamespace:
    """构造带原生 heading level 与 provenance 的最小 Docling 标题替身。"""
    return SimpleNamespace(
        label=DocItemLabel.SECTION_HEADER,
        text=text,
        level=level,
        tree_level=tree_level,
        prov=[SimpleNamespace(page_no=page)],
        children=[],
    )


def _document(*items: SimpleNamespace) -> SimpleNamespace:
    """以固定 reading order 返回标题，模拟本轮实际消费的 Docling 接口。"""
    return SimpleNamespace(
        iterate_items=lambda: iter((item, item.tree_level) for item in items)
    )


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
