"""Step 2 的无模型纯函数回归测试。"""

from __future__ import annotations

import unittest

from paperbase.chunking.docling_hybrid_chunker import (
    _build_embedding_text,
    _chunk_id,
    _front_matter_type_for_chunk,
    _is_layout_noise_chunk,
    _section_from_headings,
    _section_type_from_headings,
)
from paperbase.parsing.base import FrontMatterBlock


class ChunkingHelperTests(unittest.TestCase):
    """验证 ID、章节路径及 embedding 上下文的稳定语义。"""

    def test_chunk_id_is_stable_and_sortable(self) -> None:
        """同一 paper_id 应生成带零填充序号的稳定 chunk ID。"""
        self.assertEqual(
            _chunk_id("paper_abc", 7),
            "paper_abc_chunk_0007",
        )

    def test_section_path_preserves_heading_hierarchy(self) -> None:
        """标题路径应保留层级，并只规范化其展示空白。"""
        self.assertEqual(
            _section_from_headings(["3. Results", " 3.1.  Evaluation "]),
            "3. Results > 3.1. Evaluation",
        )
        self.assertIsNone(_section_from_headings(None))

    def test_only_controlled_reference_headings_are_bibliography(self) -> None:
        self.assertEqual(_section_type_from_headings(["2. Related Work"]), "content")
        self.assertEqual(_section_type_from_headings(["7. References"]), "bibliography")
        self.assertEqual(_section_type_from_headings(["Works Cited"]), "bibliography")
        self.assertEqual(_section_type_from_headings(["Literature Cited"]), "bibliography")

    def test_embedding_text_adds_context_without_changing_raw_text(self) -> None:
        """检索文本应显式带标题与章节，正文内容完整保留。"""
        raw_text = "The model uses a graph neural network."
        embedding_text = _build_embedding_text(
            paper_title="Traffic Forecasting", section="2. Method", raw_text=raw_text
        )
        self.assertIn("Paper title: Traffic Forecasting", embedding_text)
        self.assertIn("Section: 2. Method", embedding_text)
        self.assertTrue(embedding_text.endswith(raw_text))

    def test_only_exact_unsectioned_publication_label_is_filtered(self) -> None:
        """不应因文本短就丢弃公式、图注或有章节归属的有效内容。"""
        self.assertTrue(
            _is_layout_noise_chunk(raw_text="Article", section=None, raw_token_count=1)
        )
        self.assertFalse(
            _is_layout_noise_chunk(raw_text="Figure 1.", section=None, raw_token_count=3)
        )

    def test_front_matter_label_uses_normalized_section_and_page_range(self) -> None:
        """前置块标记应复用 Step 1 语义结果，而不依赖某篇论文的正文文案。"""
        abstract = FrontMatterBlock(
            block_type="abstract",
            canonical_section="Front matter > Abstract",
            text="A short abstract.",
            page_start=1,
            page_end=1,
            source_item_count=1,
            detection_method="explicit_heading",
            confidence="high",
        )
        self.assertEqual(
            _front_matter_type_for_chunk(
                section="A B S T R A C T",
                page_start=1,
                page_end=1,
                front_matter_blocks=(abstract,),
            ),
            "abstract",
        )
        self.assertIsNone(
            _front_matter_type_for_chunk(
                section="1. Introduction",
                page_start=1,
                page_end=1,
                front_matter_blocks=(abstract,),
            )
        )
        self.assertFalse(
            _is_layout_noise_chunk(
                raw_text="Article", section="1. Introduction", raw_token_count=1
            )
        )

if __name__ == "__main__":
    unittest.main()
