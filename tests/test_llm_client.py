"""非流式 ModelScope/OpenAI 兼容客户端的无网络回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from paperbase.llm.client import (
    LLMRequestError,
    LLMRuntimeSettings,
    OpenAICompatibleChatClient,
    _extract_non_stream_completion,
    _extract_non_stream_content,
)


def _response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning_content: str = "内部推理",
) -> SimpleNamespace:
    """构造最小非流式响应替身；额外 reasoning_content 不应被业务代码读取。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=17, completion_tokens=23),
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
        with self.assertRaisesRegex(LLMRequestError, "no final text") as raised:
            _extract_non_stream_content(
                _response(None, finish_reason="length", reasoning_content="推理用完预算")
            )
        self.assertEqual(raised.exception.error_stage, "empty_content")
        self.assertEqual(raised.exception.finish_reason, "length")
        self.assertEqual(raised.exception.prompt_tokens, 17)
        self.assertEqual(raised.exception.completion_tokens, 23)
        self.assertEqual(raised.exception.reasoning_chars, 6)

    def test_completion_exposes_safe_metrics_without_reasoning_text(self) -> None:
        completion = _extract_non_stream_completion(
            _response('{"ok": true}', reasoning_content="不应持久化的推理")
        )
        self.assertEqual(completion.content_chars, 12)
        self.assertEqual(completion.reasoning_chars, 8)
        self.assertEqual(completion.finish_reason, "stop")
        self.assertFalse(hasattr(completion, "reasoning_content"))

    def test_json_object_mode_injects_schema_and_keeps_local_validation_contract(self) -> None:
        completions = _FakeCompletions()
        client = object.__new__(OpenAICompatibleChatClient)
        client._settings = LLMRuntimeSettings(
            api_key="test",
            base_url="https://example.test",
            model="deepseek-test",
            timeout_seconds=30,
            temperature=0,
            max_tokens=500,
            structured_output_mode="json_object",
        )
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        schema = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        }

        result = client.complete_json(
            system_prompt="系统提示",
            user_prompt="连接测试",
            json_schema=schema,
            schema_name="smoke",
        )

        self.assertEqual(result, '{"message":"ok"}')
        self.assertEqual(completions.kwargs["response_format"], {"type": "json_object"})
        system_prompt = completions.kwargs["messages"][0]["content"]
        self.assertIn('"additionalProperties": false', system_prompt)
        self.assertIn("只输出一个 JSON 对象", system_prompt)


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _response('{"message":"ok"}')


if __name__ == "__main__":
    unittest.main()
