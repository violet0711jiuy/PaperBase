"""Query Planning 的 Resolution、Retrieval Rewrite 与路由离线回归测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

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
        self.assertEqual(plan.rewrite_status, "partial")
        self.assertEqual(plan.semantic_status, "unavailable")
        self.assertEqual(plan.lexical_status, "valid_fallback")
        self.assertIsNone(plan.semantic_query_en)
        self.assertEqual(plan.lexical_keywords_en, ("D2STGNN",))

    def test_missing_lexical_field_uses_deterministic_fallback(self) -> None:
        """字段缺失必须触发 Schema 失败，但不能连带丢掉原问题中的确定性实体。"""
        client = _FakeClient(['{"semantic_query_en":"How does ESDTW differ from DTW?"}'])
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("How does ESDTW differ from DTW?")

        self.assertEqual(plan.semantic_status, "unavailable")
        self.assertEqual(plan.lexical_status, "valid_fallback")
        self.assertEqual(plan.rewrite_status, "partial")
        self.assertEqual(plan.lexical_keywords_en, ("ESDTW", "DTW"))
        self.assertTrue(any(item.startswith("rewrite_error:ValidationError") for item in plan.validation_diagnostics))

    def test_explicit_empty_lexical_field_is_legal_and_uses_fallback(self) -> None:
        """显式空数组是合法 LLM 输出，随后由确定性实体补足 BM25 通道。"""
        client = _FakeClient([_rewrite_response("How does ESDTW differ from DTW?", "[]")])
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("How does ESDTW differ from DTW?")

        self.assertEqual(plan.semantic_status, "valid")
        self.assertEqual(plan.lexical_status, "valid_fallback")
        self.assertEqual(plan.rewrite_status, "success")
        self.assertEqual(plan.lexical_keywords_en, ("ESDTW", "DTW"))

    def test_llm_and_deterministic_terms_merge_without_duplicates(self) -> None:
        """合并时保护原问题实体、过滤问句词、去重并严格限制为五条。"""
        client = _FakeClient(
            [_rewrite_response(
                "How does ESDTW differ from DTW variants?",
                '["How", "ESDTW", "shape descriptors", "DTW"]',
            )]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan(
            "How does ESDTW differ from DTW, LEDTW, shapeDTW, and DDTW?"
        )

        self.assertEqual(plan.lexical_status, "valid_merged")
        self.assertLessEqual(len(plan.lexical_keywords_en), 5)
        self.assertNotIn("How", plan.lexical_keywords_en)
        self.assertEqual(len(plan.lexical_keywords_en), len(set(plan.lexical_keywords_en)))
        self.assertTrue({"ESDTW", "DTW", "LEDTW", "shapeDTW"}.issubset(plan.lexical_keywords_en))

    def test_cjk_contamination_invalidates_only_semantic_channel(self) -> None:
        """明显中文污染不能进入 Dense，但 lexical fallback 与 Original Dense 仍可继续。"""
        client = _FakeClient(
            [_rewrite_response(
                "What are the complexities of ESDTW and DTW?的眼动实验数据集？",
                '["ESDTW", "DTW"]',
            )]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("ESDTW和DTW的复杂度分别是什么？")

        self.assertIsNone(plan.semantic_query_en)
        self.assertEqual(plan.semantic_status, "invalid")
        self.assertEqual(plan.lexical_status, "valid_fallback")
        self.assertEqual(plan.rewrite_status, "partial")
        self.assertIn("semantic_contains_cjk", plan.validation_diagnostics)

    def test_new_ablation_marker_invalidates_semantic_channel(self) -> None:
        """LLM 不得把主模型名擅自改成带 †/‡ 的特定消融变体。"""
        client = _FakeClient(
            [_rewrite_response(
                "How does the estimation gate in D²STGNN† use time slots?",
                '["D²STGNN†", "estimation gate"]',
            )]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("D2STGNN中的估计门如何利用时间槽？")

        self.assertIsNone(plan.semantic_query_en)
        self.assertEqual(plan.semantic_status, "invalid")
        self.assertEqual(plan.rewrite_status, "partial")
        self.assertIn("semantic_entity_variant_marker_added", plan.validation_diagnostics)
        self.assertNotIn("D²STGNN†", plan.lexical_keywords_en)
        self.assertTrue(
            any(
                item.startswith("llm_lexical_variant_marker_removed:")
                for item in plan.validation_diagnostics
            )
        )
        self.assertEqual(plan.lexical_status, "valid_fallback")

    def test_unanchored_llm_lexical_noise_is_removed(self) -> None:
        """原问题和合法 semantic 中都不存在的 PM2.5 不得进入最终 BM25 关键词。"""
        client = _FakeClient(
            [_rewrite_response(
                "How do D²STGNN† and D²STGNN‡ compare?",
                '["D²STGNN†", "D²STGNN‡", "PM2.5"]',
            )]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("D²STGNN†与D²STGNN‡相比表现如何？")

        self.assertNotIn("PM2.5", plan.lexical_keywords_en)
        self.assertIn("llm_lexical_unanchored_removed:PM2.5", plan.validation_diagnostics)

    def test_missing_paired_season_invalidates_semantic_channel(self) -> None:
        """“春夏季”不能被改写成仅 summer，否则问题范围被无声缩小。"""
        client = _FakeClient(
            [_rewrite_response(
                "What parameters change when applying CMAQ during summer?",
                '["CMAQ", "summer"]',
            )]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("CMAQ用于成都春夏季时需要修改哪些参数？")

        self.assertIsNone(plan.semantic_query_en)
        self.assertEqual(plan.semantic_status, "invalid")
        self.assertIn("semantic_season_constraint_missing:spring", plan.validation_diagnostics)

    def test_single_paper_scope_cannot_be_pluralized(self) -> None:
        """单篇目标论文不能被改写成 papers，从而无声扩大检索范围。"""
        client = _FakeClient(
            [_rewrite_response(
                "How do papers pair observations with model forecasts?",
                '["model forecasts"]',
            )]
        )
        planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

        plan = planner.plan("论文如何把观测值与模型预报配对？")

        self.assertIsNone(plan.semantic_query_en)
        self.assertEqual(plan.semantic_status, "invalid")
        self.assertEqual(plan.lexical_status, "empty")
        self.assertEqual(plan.rewrite_status, "degraded")
        self.assertIn("semantic_single_paper_scope_pluralized", plan.validation_diagnostics)

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
            set(rewrite_schema["required"]), {"semantic_query_en", "lexical_keywords_en"}
        )
        self.assertEqual(rewrite_schema["properties"]["lexical_keywords_en"]["maxItems"], 5)
        self.assertNotIn("minItems", rewrite_schema["properties"]["lexical_keywords_en"])
        self.assertNotIn("search_bibliography", rewrite_schema["properties"])

    def test_retrieval_rewrite_schema_accepts_empty_but_rejects_missing_or_six_terms(self) -> None:
        """Schema 区分显式空数组、字段缺失和超过上限三种情况。"""
        parsed = RetrievalRewriteResult.model_validate_json(
            '{"semantic_query_en":"What is ESDTW?","lexical_keywords_en":[]}'
        )
        self.assertEqual(parsed.lexical_keywords_en, [])
        with self.assertRaises(ValidationError):
            RetrievalRewriteResult.model_validate_json(
                '{"semantic_query_en":"What is ESDTW?"}'
            )
        with self.assertRaises(ValidationError):
            RetrievalRewriteResult.model_validate(
                {
                    "semantic_query_en": "What is ESDTW?",
                    "lexical_keywords_en": ["a", "b", "c", "d", "e", "f"],
                }
            )


if __name__ == "__main__":
    unittest.main()
