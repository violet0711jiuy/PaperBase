"""论文前置元数据标准化规则的纯逻辑回归测试。"""

from __future__ import annotations

import unittest

from paperbase.parsing.docling_parser import (
    _availability_content_block_type,
    _front_matter_heading_type,
    _inline_publication_block_type,
    _split_inline_content_label,
)


class FrontMatterHelperTests(unittest.TestCase):
    """验证跨出版社别名与行内标签，而不是依赖某篇论文的固定文本。"""

    def test_heading_aliases_are_normalized_to_stable_types(self) -> None:
        """大小写、空格和冒号差异不应影响标准元数据类型。"""
        self.assertEqual(_front_matter_heading_type("A B S T R A C T"), "abstract")
        self.assertEqual(_front_matter_heading_type("PVLDBReference Format:"), "publication_info")
        self.assertEqual(_front_matter_heading_type("Data Availability"), "availability")
        self.assertIsNone(_front_matter_heading_type("2. Materials and Methods"))

    def test_inline_abstract_and_keywords_need_explicit_labels(self) -> None:
        """只提升明确的行内标签，未标注首段不能凭长度猜成摘要。"""
        self.assertEqual(
            _split_inline_content_label("Abstract: This study evaluates a model."),
            ("abstract", "This study evaluates a model."),
        )
        self.assertEqual(
            _split_inline_content_label("Keywords: traffic forecasting; graph network"),
            ("keywords", "traffic forecasting; graph network"),
        )
        self.assertIsNone(
            _split_inline_content_label("This study evaluates a model without a label.")
        )

    def test_inline_publication_labels_do_not_match_regular_sentences(self) -> None:
        """Citation/Received 等必须处于行首标签位置，避免误提取正文引文。"""
        self.assertEqual(
            _inline_publication_block_type("Citation: Doe et al. (2026)."),
            "publication_info",
        )

    def test_availability_container_is_split_by_supported_content_signals(self) -> None:
        """代码、通讯作者、许可证和期刊脚注不应作为同一种检索语义。"""
        self.assertEqual(
            _availability_content_block_type("The source code is available at https://example.org."),
            "availability",
        )
        self.assertEqual(
            _availability_content_block_type("* Corresponding author."),
            "correspondence",
        )
        self.assertEqual(
            _availability_content_block_type("This work is licensed under Creative Commons."),
            "rights",
        )
        self.assertEqual(
            _availability_content_block_type("Proceedings of the VLDB Endowment, ISSN 2150-8097."),
            "publication_info",
        )
        self.assertEqual(
            _inline_publication_block_type("Copyright: © 2026 by the authors."),
            "rights",
        )
        self.assertIsNone(
            _inline_publication_block_type(
                "The citation was used to compare the two approaches."
            )
        )


if __name__ == "__main__":
    unittest.main()
