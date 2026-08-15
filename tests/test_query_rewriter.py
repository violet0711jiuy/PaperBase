"""Step 6 Query Rewrite 的结构化输出、Pydantic 校验、清洗与降级测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from paperbase.config import QueryRewriteSettings
from paperbase.llm.client import LLMRequestError
from paperbase.retrieval.query_rewriter import (
    LLMQueryRewriter,
    NoopQueryRewriter,
    QueryRewriteResult,
    normalize_query_rewrite,
    resolve_bibliography_search_rule,
)


class _FakeClient:
    """顺序返回结构化 JSON 文本的 LLM 替身；不访问网络。"""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        schema_name: str,
    ) -> str:
        # 记录每一次模型调用的参数。这里主要用于确认结构化输出失败时不会悄悄再发起
        # 第二次“修复 JSON”调用；这样测试能防止未来修改又把不可控的修复路径加回来。
        self.calls.append(
            {
                "schema_name": schema_name,
                "json_schema": json_schema,
                # 保存 Prompt 仅用于确认当前问题与历史的边界传递，不会记录真实网络请求。
                "user_prompt": user_prompt,
            }
        )
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class QueryRewriterTests(unittest.TestCase):
    def test_schema_validates_then_normalizes_whitespace_duplicates_and_config_limit(self) -> None:
        client = _FakeClient(
            [
                """{
                "semantic_query": "  dynamic graph construction  ",
                "lexical_keywords_en": [
                  "dynamic graph", "", "dynamic graph", " adjacency matrix ",
                  "spatial dependency", "extra keyword"
                ],
                "search_bibliography": false
                }"""
            ]
        )
        rewriter = LLMQueryRewriter(
            settings=QueryRewriteSettings(max_lexical_keywords_en=3),
            client=client,
        )

        plan = rewriter.rewrite("动态图构建方法是什么？")

        self.assertEqual(plan.status, "success")
        self.assertEqual(plan.semantic_query, "dynamic graph construction")
        self.assertEqual(
            plan.lexical_keywords_en,
            ("dynamic graph", "adjacency matrix", "spatial dependency"),
        )
        # 原生 Schema 必须禁止额外字段，且两个字段都在 required 中。
        schema = client.calls[0]["json_schema"]
        self.assertEqual(schema["additionalProperties"], False)  # type: ignore[index]
        self.assertEqual(
            set(schema["required"]),  # type: ignore[index]
            {"semantic_query", "lexical_keywords_en", "search_bibliography"},
        )

    def test_invalid_structured_output_falls_back_without_a_second_llm_call(self) -> None:
        client = _FakeClient(
            [
                # extra 字段违反 QueryRewriteResult 的 extra="forbid" 契约。
                # 即使 API 层已请求 JSON Schema，本地仍必须把供应商返回视为外部输入并校验。
                '{"semantic_query": null, "lexical_keywords_en": [], "search_bibliography": false, "extra": true}',
            ]
        )
        rewriter = LLMQueryRewriter(
            settings=QueryRewriteSettings(max_lexical_keywords_en=3),
            client=client,
        )

        plan = rewriter.rewrite("作者是谁？")

        # 结构化输出不合法时不再启动 JSON Repair。Query Rewrite 是可选增强功能，
        # 所以系统应只使用一次 LLM 调用后退回原始问题，不让错误 JSON 中断检索。
        self.assertEqual(plan.status, "fallback")
        self.assertIsNone(plan.semantic_query)
        self.assertEqual(plan.lexical_keywords_en, ())
        self.assertEqual([call["schema_name"] for call in client.calls], ["query_rewrite_result"])

    def test_invalid_structured_output_always_falls_back_to_original_query(self) -> None:
        client = _FakeClient(
            [
                '{"semantic_query": 123, "lexical_keywords_en": [], "search_bibliography": false}',
            ]
        )
        rewriter = LLMQueryRewriter(
            # 即使历史配置值为 false，Query Rewrite 失败也不能中断 Retrieval。
            settings=QueryRewriteSettings(fallback_to_original=False),
            client=client,
        )

        plan = rewriter.rewrite("  作者是谁？  ")

        self.assertEqual(plan.status, "fallback")
        self.assertEqual(plan.original_query, "作者是谁？")
        self.assertEqual(len(client.calls), 1)

    def test_unavailable_llm_falls_back_to_original_query(self) -> None:
        client = _FakeClient([LLMRequestError("network unavailable")])
        rewriter = LLMQueryRewriter(
            settings=QueryRewriteSettings(fallback_to_original=True),
            client=client,
        )

        plan = rewriter.rewrite("作者是谁？")

        self.assertEqual(plan.status, "fallback")
        self.assertEqual(len(client.calls), 1)

    def test_pydantic_rejects_extra_fields_before_normalization(self) -> None:
        with self.assertRaises(ValidationError):
            QueryRewriteResult.model_validate(
                {
                    "semantic_query": None,
                    "lexical_keywords_en": [],
                    "unexpected": "field",
                }
            )

        normalized = normalize_query_rewrite(
            QueryRewriteResult(
                semantic_query="   ",
                lexical_keywords_en=[" A ", "a", "", "B"],
                search_bibliography=False,
            ),
            max_lexical_keywords_en=2,
        )
        self.assertIsNone(normalized.semantic_query)
        self.assertEqual(normalized.lexical_keywords_en, ["A", "B"])

    def test_citation_intent_can_enable_bibliography_but_comparison_cannot(self) -> None:
        citation_client = _FakeClient(
            ['{"semantic_query": "Does this paper cite Graph WaveNet?", "lexical_keywords_en": ["Graph WaveNet"], "search_bibliography": true}']
        )
        citation_plan = LLMQueryRewriter(
            settings=QueryRewriteSettings(), client=citation_client
        ).rewrite("这篇论文有没有引用 Graph WaveNet？")
        self.assertTrue(citation_plan.search_bibliography)

        comparison_client = _FakeClient(
            ['{"semantic_query": "How does Graph WaveNet differ from the proposed model?", "lexical_keywords_en": ["Graph WaveNet"], "search_bibliography": false}']
        )
        comparison_plan = LLMQueryRewriter(
            settings=QueryRewriteSettings(), client=comparison_client
        ).rewrite("Graph WaveNet 和本文模型有什么区别？")
        self.assertFalse(comparison_plan.search_bibliography)

    def test_bibliography_rules_classify_explicit_true_false_and_ambiguous_queries(self) -> None:
        # 明确询问参考文献、引用关系或编号条目时，不依赖 LLM 的不稳定判断。
        self.assertTrue(resolve_bibliography_search_rule("这篇论文有没有引用 Graph WaveNet？"))
        self.assertTrue(resolve_bibliography_search_rule("作者为什么引用 Graph WaveNet？"))
        self.assertTrue(resolve_bibliography_search_rule("[15] 是哪篇论文？"))
        # 方法比较、baseline 和实现机制问题应优先从正文找证据。
        self.assertFalse(resolve_bibliography_search_rule("Graph WaveNet 和本文方法有什么区别？"))
        self.assertFalse(resolve_bibliography_search_rule("这篇论文用了哪些 baseline？"))
        self.assertFalse(resolve_bibliography_search_rule("动态图是怎么构建的？"))
        # 不包含高精度信号的问题不由规则猜测，保留给 LLM 结合上下文判断。
        self.assertIsNone(resolve_bibliography_search_rule("请解释 Graph WaveNet 的作用。"))

    def test_explicit_rule_overrides_llm_bibliography_decision(self) -> None:
        # 即使 LLM 漏判引用意图，正向规则仍强制开启 bibliography FTS5。
        citation_client = _FakeClient(
            ['{"semantic_query": "Does this paper cite Graph WaveNet?", '
             '"lexical_keywords_en": ["Graph WaveNet"], "search_bibliography": false}']
        )
        citation_plan = LLMQueryRewriter(
            settings=QueryRewriteSettings(), client=citation_client
        ).rewrite("这篇论文有没有引用 Graph WaveNet？")
        self.assertTrue(citation_plan.search_bibliography)

        # 即使 LLM 因论文名误判为引用问题，明确的正文比较规则仍强制关闭该索引。
        comparison_client = _FakeClient(
            ['{"semantic_query": "How does Graph WaveNet differ from the proposed model?", '
             '"lexical_keywords_en": ["Graph WaveNet"], "search_bibliography": true}']
        )
        comparison_plan = LLMQueryRewriter(
            settings=QueryRewriteSettings(), client=comparison_client
        ).rewrite("Graph WaveNet 和本文方法有什么区别？")
        self.assertFalse(comparison_plan.search_bibliography)

    def test_explicit_citation_rule_survives_llm_failure_and_disabled_rewriter(self) -> None:
        # 外部 LLM 不可用时，引用问题仍走正文检索加 bibliography FTS5，不会退化为正文-only。
        fallback_plan = LLMQueryRewriter(
            settings=QueryRewriteSettings(),
            client=_FakeClient([LLMRequestError("network unavailable")]),
        ).rewrite("这篇论文有没有引用 Graph WaveNet？")
        self.assertEqual(fallback_plan.status, "fallback")
        self.assertTrue(fallback_plan.search_bibliography)

        disabled_plan = NoopQueryRewriter().rewrite("[15] 是哪篇论文？")
        self.assertEqual(disabled_plan.status, "disabled")
        self.assertTrue(disabled_plan.search_bibliography)

    def test_recent_context_is_bounded_and_passed_to_rewrite_prompt(self) -> None:
        client = _FakeClient(
            [
                '{"semantic_query": "How does EMVE differ from LSTM for wind speed forecasting?", '
                '"lexical_keywords_en": ["EMVE", "LSTM", "wind speed forecasting"], '
                '"search_bibliography": false}'
            ]
        )
        rewriter = LLMQueryRewriter(
            settings=QueryRewriteSettings(max_context_turns=2),
            client=client,
        )

        plan = rewriter.rewrite(
            "那这个和 LSTM 有什么区别？",
            conversation_context=[
                "很早的无关历史",
                "上一轮：介绍 EMVE 模型",
                "当前上下文：EMVE 用于风速预测",
            ],
        )

        self.assertEqual(plan.status, "success")
        self.assertEqual(
            plan.semantic_query,
            "How does EMVE differ from LSTM for wind speed forecasting?",
        )
        user_prompt = client.calls[0]["user_prompt"]
        self.assertNotIn("很早的无关历史", user_prompt)  # type: ignore[operator]
        self.assertIn("上一轮：介绍 EMVE 模型", user_prompt)  # type: ignore[operator]
        self.assertIn("当前上下文：EMVE 用于风速预测", user_prompt)  # type: ignore[operator]


if __name__ == "__main__":
    unittest.main()
