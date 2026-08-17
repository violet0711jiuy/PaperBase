"""Query Planning 的 Resolution、Retrieval Rewrite 与路由离线回归测试。"""

from __future__ import annotations

import unittest

from paperbase.config import QueryRewriteSettings
from paperbase.llm.client import LLMRequestError
from paperbase.retrieval.query_rewriter import (
    LLMQueryPlanner,
    QueryResolutionResult,
    RetrievalRewriteResult,
    TrustedPaperScope,
    resolve_bibliography_search_rule,
)


class _FakeClient:
    """顺序返回结构化 JSON，不访问网络，并记录两类阶段调用。"""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response

    def complete(self, **kwargs: object) -> str:
        raise AssertionError("Query Planning 必须使用结构化输出。")


def _rewrite_response(
    semantic_query_en: str = "What is the method framework?",
    keywords: str = '["D2STGNN", "framework"]',
) -> str:
    """构造 Retrieval Rewrite 成功响应，避免每个测试重复长 JSON。"""
    return (
        '{"semantic_query_en":"'
        f"{semantic_query_en}"
        '","lexical_keywords_en":'
        f"{keywords}"
        "}"
    )


class QueryPlanningTests(unittest.TestCase):
    def test_self_contained_abbreviation_is_resolved_without_history_resolution(self) -> None:
        client = _FakeClient(
            [_rewrite_response("What is the overall framework of D2STGNN?")]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("D2STGNN的整体框架是什么？")

        self.assertEqual(plan.resolution_status, "resolved")
        self.assertEqual(plan.resolved_query, "D2STGNN的整体框架是什么？")
        self.assertEqual(plan.rewrite_status, "success")
        self.assertEqual(plan.semantic_query_en, "What is the overall framework of D2STGNN?")
        self.assertEqual([call["schema_name"] for call in client.calls], ["retrieval_rewrite_result"])

    def test_inline_paper_title_resolves_inside_current_query(self) -> None:
        title_query = "ESDTW: Extrema-based shape dynamic time warping 这篇论文有什么局限性？"
        client = _FakeClient([_rewrite_response("What limitations does ESDTW have?")])
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan(title_query)

        self.assertEqual(plan.resolution_status, "resolved")
        self.assertEqual(plan.resolved_query, title_query)
        self.assertEqual([call["schema_name"] for call in client.calls], ["retrieval_rewrite_result"])

    def test_trusted_active_paper_scope_resolves_only_paper_reference(self) -> None:
        client = _FakeClient([_rewrite_response("What limitations does ESDTW have?")])
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)
        scope = TrustedPaperScope("paper-esdtw", "ESDTW: Extrema-based shape dynamic time warping")

        plan = planner.plan("这篇论文有什么局限性？", trusted_scope=scope)

        self.assertEqual(plan.resolution_status, "resolved")
        self.assertIn("论文《ESDTW: Extrema-based shape dynamic time warping》", plan.resolved_query or "")
        self.assertEqual([call["schema_name"] for call in client.calls], ["retrieval_rewrite_result"])

    def test_context_resolves_generic_reference_before_retrieval_rewrite(self) -> None:
        client = _FakeClient(
            [
                '{"resolution_status":"resolved",'
                '"resolved_query":"ESDTW 和 DTW 有什么区别？"}',
                _rewrite_response("How does ESDTW differ from DTW?", '["ESDTW", "DTW"]'),
            ]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan(
            "它和DTW有什么区别？",
            conversation_context=("上一轮正在讨论 ESDTW 方法。",),
        )

        self.assertEqual(plan.resolved_query, "ESDTW 和 DTW 有什么区别？")
        self.assertEqual(plan.semantic_query_en, "How does ESDTW differ from DTW?")
        self.assertEqual(
            [call["schema_name"] for call in client.calls],
            ["query_resolution_result", "retrieval_rewrite_result"],
        )
        self.assertIn("ESDTW", client.calls[0]["user_prompt"])  # type: ignore[operator]
        self.assertNotIn("conversation_context", client.calls[1]["user_prompt"])  # type: ignore[operator]

    def test_ambiguous_reference_stops_before_retrieval_rewrite(self) -> None:
        client = _FakeClient(['{"resolution_status":"unresolved","resolved_query":null}'])
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan(
            "它的结果怎么样？",
            conversation_context=("上一轮讨论 ESDTW。", "上一轮也讨论 D2STGNN。"),
        )

        self.assertEqual(plan.resolution_status, "unresolved")
        self.assertIsNone(plan.resolved_query)
        self.assertEqual(plan.rewrite_status, "not_run")
        self.assertEqual([call["schema_name"] for call in client.calls], ["query_resolution_result"])

    def test_broad_paper_discovery_question_is_not_misclassified_as_unresolved(self) -> None:
        client = _FakeClient(
            [_rewrite_response("Which papers in the knowledge base concern wind speed forecasting?")]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("知识库中有哪些论文是讲风速预测的？")

        self.assertEqual(plan.resolution_status, "resolved")
        self.assertEqual(plan.resolved_query, "知识库中有哪些论文是讲风速预测的？")

    def test_retrieval_rewrite_failure_degrades_without_invented_english_translation(self) -> None:
        client = _FakeClient([LLMRequestError("network unavailable")])
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("D2STGNN的整体框架是什么？")

        self.assertEqual(plan.resolution_status, "resolved")
        self.assertEqual(plan.rewrite_status, "degraded")
        self.assertIsNone(plan.semantic_query_en)
        self.assertEqual(plan.lexical_keywords_en, ("D2STGNN",))

    def test_bibliography_router_is_program_owned(self) -> None:
        self.assertTrue(resolve_bibliography_search_rule("引用了哪些论文？"))
        self.assertTrue(resolve_bibliography_search_rule("[15] 是什么？"))
        self.assertTrue(resolve_bibliography_search_rule("是否引用 Graph WaveNet？"))
        self.assertTrue(resolve_bibliography_search_rule("Graph WaveNet 对应哪篇参考文献？"))
        self.assertFalse(resolve_bibliography_search_rule("Graph WaveNet 和本文模型有什么区别？"))
        self.assertFalse(resolve_bibliography_search_rule("D2STGNN 怎么构建动态图？"))

    def test_two_llm_schemas_have_single_responsibility(self) -> None:
        resolution_schema = QueryResolutionResult.model_json_schema()
        rewrite_schema = RetrievalRewriteResult.model_json_schema()

        self.assertEqual(set(resolution_schema["required"]), {"resolution_status", "resolved_query"})
        self.assertEqual(
            set(rewrite_schema["required"]), {"semantic_query_en"}
        )
        self.assertNotIn("search_bibliography", rewrite_schema["properties"])


if __name__ == "__main__":
    unittest.main()
