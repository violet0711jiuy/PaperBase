"""非流式 ModelScope/OpenAI 兼容客户端的无网络回归测试。"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paperbase.llm.client import (
    LLMRequestError,
    LLMRuntimeSettings,
    OpenAICompatibleChatClient,
    ZhipuAIChatClient,
    _extract_non_stream_completion,
    _extract_non_stream_content,
    load_llm_runtime_settings,
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
        client._extra_body = {"enable_thinking": False}
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
        self.assertEqual(completions.kwargs["extra_body"], {"enable_thinking": False})
        system_prompt = completions.kwargs["messages"][0]["content"]
        self.assertIn('"additionalProperties": false', system_prompt)
        self.assertIn("只输出一个 JSON 对象", system_prompt)

    def test_thinking_disabled_is_the_default_and_is_preserved_by_views(self) -> None:
        completions = _FakeCompletions()
        client = object.__new__(OpenAICompatibleChatClient)
        client._settings = LLMRuntimeSettings(
            api_key="test", base_url="https://example.test", model="qwen-test",
            timeout_seconds=30, temperature=0, max_tokens=500,
        )
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client._extra_body = {"enable_thinking": False}

        client.with_max_tokens(300).complete_json(
            system_prompt="system", user_prompt="question", json_schema={"type": "object"},
            schema_name="smoke",
        )

        self.assertEqual(completions.kwargs["extra_body"], {"enable_thinking": False})
        self.assertEqual(client._extra_body, {"enable_thinking": False})

    def test_schema_critical_view_omits_caps_preserves_base_and_supports_strict(self) -> None:
        from paperbase.llm.client import structured_client_view
        for cls in (OpenAICompatibleChatClient, ZhipuAIChatClient):
            completions = _FakeCompletions()
            client = object.__new__(cls)
            client._settings = LLMRuntimeSettings(api_key="test",base_url="https://example.test",
                model="test",timeout_seconds=60,temperature=0.2,max_tokens=1400,
                structured_output_mode="json_object")
            client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
            if cls is OpenAICompatibleChatClient:client._extra_body={"enable_thinking":False}
            else:client._thinking={"type":"disabled"}
            view = structured_client_view(client, temperature=0,
                structured_output_mode="json_schema" if cls is OpenAICompatibleChatClient else "json_object")
            view.complete_json(system_prompt="s",user_prompt="u",json_schema={"type":"object"},schema_name="schema")
            self.assertNotIn("max_tokens", completions.kwargs)
            self.assertNotIn("max_output_tokens", completions.kwargs)
            self.assertNotIn("max_completion_tokens", completions.kwargs)
            self.assertEqual(completions.kwargs["temperature"],0)
            self.assertEqual(client._settings.temperature,0.2)
            self.assertEqual(client._settings.max_tokens,1400)
            if cls is OpenAICompatibleChatClient:
                self.assertTrue(completions.kwargs["response_format"]["json_schema"]["strict"])

    def test_strict_wire_schema_materializes_defaulted_fields_without_mutating_dto(self) -> None:
        from paperbase.llm.client import _strict_response_schema
        from paperbase.agentic.schemas import GroundedAnswerDraft
        schema=GroundedAnswerDraft.model_json_schema()
        original_required=list(schema['$defs']['GroundedRequirementAnswerDraft']['required'])
        strict=_strict_response_schema(schema)
        self.assertIn('remaining_gap', original_required)
        self.assertEqual(
            strict['$defs']['GroundedRequirementAnswerDraft']['required'],
            original_required,
        )
        self.assertEqual(
            schema['$defs']['GroundedRequirementAnswerDraft']['required'],
            original_required,
        )

    def test_zhipu_official_request_is_non_streaming_and_disables_thinking(self) -> None:
        completions = _FakeCompletions()
        client = object.__new__(ZhipuAIChatClient)
        client._settings = LLMRuntimeSettings(
            api_key="test-zhipu-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4.5-air",
            timeout_seconds=30,
            temperature=0,
            max_tokens=500,
            structured_output_mode="json_object",
            provider="zhipuai",
        )
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client._thinking = {"type": "disabled"}

        result = client.complete_json(
            system_prompt="system",
            user_prompt="question",
            json_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            schema_name="ignored_by_json_object",
        )

        self.assertEqual(result, '{"message":"ok"}')
        self.assertEqual(completions.kwargs["model"], "glm-4.5-air")
        self.assertFalse(completions.kwargs["stream"])
        self.assertEqual(completions.kwargs["thinking"], {"type": "disabled"})
        self.assertEqual(completions.kwargs["response_format"], {"type": "json_object"})
        self.assertIn("只输出一个 JSON 对象", completions.kwargs["messages"][0]["content"])


class RuntimeSettingsTests(unittest.TestCase):
    def test_zhipu_settings_require_own_key_and_json_object(self) -> None:
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        "LLM_PROVIDER=zhipuai",
                        "ZHIPUAI_API_KEY=test-key",
                        "LLM_MODEL=glm-4.5-air",
                        "LLM_STRUCTURED_OUTPUT_MODE=json_object",
                        "LLM_THINKING_ENABLED=false",
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_runtime_settings(env_path)

        self.assertEqual(settings.provider, "zhipuai")
        self.assertEqual(settings.model, "glm-4.5-air")
        self.assertEqual(settings.base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertFalse(settings.thinking_enabled)


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _response('{"message":"ok"}')


if __name__ == "__main__":
    unittest.main()
