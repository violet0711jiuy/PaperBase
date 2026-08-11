"""页眉/页脚与合并标题的纯函数回归测试。"""

from __future__ import annotations

import unittest

from paperbase.parsing.docling_parser import (
    _DANGLING_PAGE_COUNTER_IN_TABLE,
    _PAGE_COUNTER,
    _split_merged_numbered_heading,
    _split_repeated_caption,
    _strip_page_counter_prefix,
)


class DoclingCleanupHelperTests(unittest.TestCase):
    """验证保守清理规则不会把常见正文形式误判为页码或合并标题。"""

    def test_page_counter_accepts_only_pagination_forms(self) -> None:
        """纯分页号应被忽略，含期刊信息的页眉不能被误判为页码。"""
        self.assertIsNotNone(_PAGE_COUNTER.fullmatch("3 of 11"))
        self.assertIsNotNone(_PAGE_COUNTER.fullmatch("Page 3"))
        self.assertIsNone(_PAGE_COUNTER.fullmatch("Toxics 2023, 11, 777"))

    def test_page_counter_prefix_only_removes_prefix(self) -> None:
        """清理分页碎片时必须完整保留其后的正文。"""
        self.assertEqual(
            _strip_page_counter_prefix("5 of 11 (Figure 2c) remains relevant."),
            "(Figure 2c) remains relevant.",
        )
        self.assertEqual(
            _strip_page_counter_prefix("5 methods are compared."),
            "5 methods are compared.",
        )

    def test_only_unambiguous_merged_numbered_heading_is_split(self) -> None:
        """两个连续编号标题可拆分；带数字的普通正文不能拆分。"""
        self.assertEqual(
            _split_merged_numbered_heading(
                "3.3. Forecast Accuracy 3.3.1. Weather Classification"
            ),
            ("3.3. Forecast Accuracy", "3.3.1. Weather Classification"),
        )
        self.assertIsNone(
            _split_merged_numbered_heading("The value is 3.3.1. in this experiment.")
        )

    def test_only_complete_repeated_caption_is_deduplicated(self) -> None:
        """同编号、长且高度相似的双图注可合并；正文式引用不能触发。"""
        caption = (
            "Figure 1. A detailed caption that contains enough words to describe the "
            "experimental setup and the corresponding observations in full. "
        )
        self.assertEqual(_split_repeated_caption(caption + caption), caption.strip())
        self.assertIsNone(
            _split_repeated_caption(
                "Figure 1. A short caption. The comparison is shown in Figure 1."
            )
        )

    def test_only_dangling_table_pagination_fragment_is_removed(self) -> None:
        """页码残片可删除；含完整数量语义的单元格内容必须保留。"""
        self.assertEqual(_DANGLING_PAGE_COUNTER_IN_TABLE.sub("", "- 8 of").rstrip(), "-")
        self.assertEqual(
            _DANGLING_PAGE_COUNTER_IN_TABLE.sub("", "8 of 11 samples"),
            "8 of 11 samples",
        )


if __name__ == "__main__":
    unittest.main()
