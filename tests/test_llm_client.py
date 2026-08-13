"""非流式 ModelScope/OpenAI 兼容客户端的无网络回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from paperbase.llm.client import LLMRequestError, _extract_non_stream_content


def _response(content: str | None) -> SimpleNamespace:
    """构造最小非流式响应替身；额外 reasoning_content 不应被业务代码读取。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content="内部推理")
            )
        ]
    )


class NonStreamExtractionTests(unittest.TestCase):
    def test_extracts_only_final_content_not_reasoning_content(self) -> None:
        self.assertEqual(
            _extract_non_stream_content(_response('{"ok": true}')),
            '{"ok": true}',
        )

    def test_rejects_response_without_any_choices(self) -> None:
        with self.assertRaisesRegex(LLMRequestError, "no choices"):
            _extract_non_stream_content(SimpleNamespace(choices=[]))

    def test_rejects_response_without_final_content(self) -> None:
        with self.assertRaisesRegex(LLMRequestError, "no final text"):
            _extract_non_stream_content(_response(None))


if __name__ == "__main__":
    unittest.main()
