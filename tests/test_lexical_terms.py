"""统一确定性 lexical extractor 的科研实体保真回归测试。"""

from __future__ import annotations

import unittest

from paperbase.retrieval.lexical_terms import (
    extract_lexical_terms,
    merge_lexical_terms,
)


class LexicalTermExtractionTests(unittest.TestCase):
    """验证复杂科研实体、专业短语、停用词和合并上限。"""

    def test_method_007_preserves_all_algorithm_names_without_how(self) -> None:
        """英文问题必须保留四个比较对象，不能让 How 占名额。"""
        terms = extract_lexical_terms(
            "How does ESDTW differ from traditional DTW and variants like LEDTW and shapeDTW?"
        )
        self.assertTrue({"ESDTW", "DTW", "LEDTW", "shapeDTW"}.issubset(terms))
        self.assertNotIn("How", terms)

    def test_pm25_and_72_hour_remain_complete(self) -> None:
        """带点实体和中文时长约束必须成为完整关键词。"""
        terms = extract_lexical_terms("WRF与CMAQ如何生成成都PM2.5的72小时预报？")
        self.assertIn("PM2.5", terms)
        self.assertIn("72-hour", terms)
        self.assertNotIn("PM2", terms)
        self.assertNotIn("72", terms)

    def test_hyphenated_and_mixed_case_entities_remain_complete(self) -> None:
        """连字符模型名和内部大小写模型名不得被拆开。"""
        terms = extract_lexical_terms("Compare LSTM-EMVE with shapeDTW and LEDTW.")
        self.assertIn("LSTM-EMVE", terms)
        self.assertIn("shapeDTW", terms)
        self.assertIn("LEDTW", terms)

    def test_capitalized_multiword_entity_is_not_split(self) -> None:
        """Estimation Gate 等名称必须整体保留，组成词不应重复占名额。"""
        terms = extract_lexical_terms("Estimation Gate如何利用time slots？")
        self.assertIn("Estimation Gate", terms)
        self.assertNotIn("Estimation", terms)
        self.assertNotIn("Gate", terms)

    def test_superscript_and_dagger_variants_remain_distinct(self) -> None:
        """两个 D²STGNN 消融变体必须保留各自的 †/‡ 标记。"""
        terms = extract_lexical_terms("比较D²STGNN†与D²STGNN‡的性能。")
        self.assertIn("D²STGNN†", terms)
        self.assertIn("D²STGNN‡", terms)
        self.assertNotIn("D", terms)
        self.assertNotIn("STGNN", terms)

    def test_result_006_prioritizes_method_dataset_and_comparators(self) -> None:
        """五条上限下仍要保留主方法、数据集及主要比较算法。"""
        terms = extract_lexical_terms(
            "与ED、最优窗口DTW和DDTW相比，ESDTW分别在84个UCR数据集上表现如何？"
        )
        self.assertEqual(len(terms), 5)
        self.assertTrue({"ESDTW", "DDTW", "UCR", "DTW"}.issubset(terms))

    def test_parenthetical_and_multiword_phrases_are_supported(self) -> None:
        """括号术语和英文专业短语可以作为一个精确词法单元。"""
        quantile = extract_lexical_terms("量化违反（quantile violation）阈值是多少？")
        bibliography = extract_lexical_terms(
            "Which studies discuss support vector regression and atmospheric extinction?"
        )
        self.assertIn("quantile violation", quantile)
        self.assertTrue(
            "support vector regression" in bibliography
            or "atmospheric extinction" in bibliography
        )

    def test_stopwords_never_become_terms(self) -> None:
        """How、What、Which 等问句词必须稳定过滤。"""
        terms = extract_lexical_terms("How What Which Who paper study result method")
        self.assertEqual(terms, ())

    def test_merge_deduplicates_and_caps_five_terms(self) -> None:
        """原问题实体优先，LLM 关键词补充，最终去重且不超过五条。"""
        merged = merge_lexical_terms(
            ["How", "ESDTW", "shape descriptors", "extra term"],
            ["ESDTW", "DTW", "LEDTW", "shapeDTW"],
        )
        self.assertEqual(merged[:4], ("ESDTW", "DTW", "LEDTW", "shapeDTW"))
        self.assertEqual(len(merged), 5)
        self.assertNotIn("How", merged)


if __name__ == "__main__":
    unittest.main()
