"""Parser 结构修复规则的纯逻辑回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from docling_core.types.doc import DocItemLabel

from paperbase.parsing.docling_parser import (
    _LIST_STYLE_ITEM,
    _have_stable_provenance_positions,
    _has_next_list_item,
    _join_adjacent_hard_word_break,
    _normalize_pdf_word_breaks_in_text,
)


def _root_item(label: DocItemLabel, text: str) -> SimpleNamespace:
    """构造只含本测试所需字段的文档根节点替身。"""
    return SimpleNamespace(label=label, text=text, children=[])


class DoclingParserHelperTests(unittest.TestCase):
    """验证伪章节修复必须依赖连续正文列表，而非具体论文内容。"""

    def test_long_parenthesized_item_has_number_and_content(self) -> None:
        """编号括号格式需要精确匹配，普通 ``1. Introduction`` 不属于该规则。"""
        match = _LIST_STYLE_ITEM.fullmatch("1) A long contribution statement")
        self.assertIsNotNone(match)
        self.assertEqual(match.group("number"), "1")
        self.assertEqual(match.group("content"), "A long contribution statement")
        self.assertIsNone(_LIST_STYLE_ITEM.fullmatch("1. Introduction"))

    def test_next_numbered_item_is_required_as_structural_evidence(self) -> None:
        """只有紧随正文中出现连续的下一项，才可证明前一个标题属于列表。"""
        roots = [
            _root_item(DocItemLabel.SECTION_HEADER, "1) A very long list item"),
            _root_item(DocItemLabel.PAGE_HEADER, "Journal header"),
            _root_item(DocItemLabel.TEXT, "- 2) The next contribution item"),
        ]
        self.assertTrue(
            _has_next_list_item(
                document=SimpleNamespace(),
                root_items=roots,
                start_index=0,
                expected_number=2,
            )
        )

    def test_next_numbered_item_can_be_inside_docling_list_group(self) -> None:
        """Docling 的后续列表项在 ListGroup 子树时，也应作为连续列表证据。"""
        next_item = SimpleNamespace(
            label=DocItemLabel.LIST_ITEM,
            enumerated=True,
            marker="2)",
        )
        list_group = SimpleNamespace(
            label="list",
            text="",
            children=[SimpleNamespace(resolve=lambda _document: next_item)],
        )
        roots = [
            _root_item(DocItemLabel.SECTION_HEADER, "1) A very long list item"),
            list_group,
        ]
        self.assertTrue(
            _has_next_list_item(
                document=SimpleNamespace(),
                root_items=roots,
                start_index=0,
                expected_number=2,
            )
        )

    def test_next_real_section_stops_list_evidence_search(self) -> None:
        """下一个真实章节之后的数字不能反向证明此前标题是列表项。"""
        roots = [
            _root_item(DocItemLabel.SECTION_HEADER, "1) A very long list item"),
            _root_item(DocItemLabel.SECTION_HEADER, "2. Methods"),
            _root_item(DocItemLabel.TEXT, "- 2) A method substep"),
        ]
        self.assertFalse(
            _has_next_list_item(
                document=SimpleNamespace(),
                root_items=roots,
                start_index=0,
                expected_number=2,
            )
        )

    def test_running_furniture_requires_stable_coordinates(self) -> None:
        """重复文字还必须位于稳定位置，不能只因内容相同就误删正文。"""
        stable_items = [
            SimpleNamespace(
                prov=[
                    SimpleNamespace(
                        bbox=SimpleNamespace(l=30.0, t=756.0, r=90.0, b=750.0)
                    )
                ]
            ),
            SimpleNamespace(
                prov=[
                    SimpleNamespace(
                        bbox=SimpleNamespace(l=30.5, t=756.3, r=90.5, b=750.3)
                    )
                ]
            ),
        ]
        moved_items = [
            stable_items[0],
            SimpleNamespace(
                prov=[
                    SimpleNamespace(
                        bbox=SimpleNamespace(l=60.0, t=700.0, r=120.0, b=694.0)
                    )
                ]
            ),
        ]
        self.assertTrue(_have_stable_provenance_positions(stable_items))
        self.assertFalse(_have_stable_provenance_positions(moved_items))

    def test_pdf_line_end_word_breaks_are_repaired_before_chunking(self) -> None:
        """硬连字符断词应在 Step 1 修复，供 Markdown 与全部下游共用。"""
        self.assertEqual(
            _normalize_pdf_word_breaks_in_text(
                "relative humid-\nity and low-pres-\n\nsure conditions"
            ),
            "relative humidity and low-pressure conditions",
        )
        self.assertEqual(
            _normalize_pdf_word_breaks_in_text("The fore-\ncasting error declined."),
            "The forecasting error declined.",
        )

    def test_existing_compound_hyphen_is_preserved_while_line_break_is_removed(self) -> None:
        """模糊情形优先保留真实术语的连字符，不能为修复断词制造新错误。"""
        self.assertEqual(
            _normalize_pdf_word_breaks_in_text("a model-\nbased state-of-the-\nart system"),
            "a model-based state-of-the-art system",
        )

    def test_adjacent_text_items_restore_one_split_word_without_joining_sentences(self) -> None:
        """跨 Docling 文本项的断词需合并；无强断词证据时禁止拼接。"""
        self.assertEqual(
            _join_adjacent_hard_word_break(
                left_text="2 m relative humid-",
                right_text="ity, and 10 m wind speed",
            ),
            "2 m relative humidity, and 10 m wind speed",
        )
        self.assertEqual(
            _join_adjacent_hard_word_break(
                left_text="a model-",
                right_text="based approach",
            ),
            "a model-based approach",
        )
        self.assertIsNone(
            _join_adjacent_hard_word_break(
                left_text="The first sentence ends.",
                right_text="the next item begins here.",
            )
        )


if __name__ == "__main__":
    unittest.main()
