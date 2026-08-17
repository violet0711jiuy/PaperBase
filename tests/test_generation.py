"""Step 8 的离线回归测试：邻居扩展边界、答案结构化校验和故障降级。"""

from __future__ import annotations

import unittest

from paperbase.config import AnswerGenerationSettings, ContextExpansionSettings
from paperbase.generation.answer_generator import GroundedAnswerGenerator
from paperbase.generation.section_expander import EvidenceUnit, SectionAwareNeighborExpander
from paperbase.prompts.answer_generation import ANSWER_GENERATION_SYSTEM_PROMPT
from paperbase.retrieval.hybrid_retriever import RetrievedChunk, RetrievalResult
from paperbase.retrieval.query_rewriter import QueryRewritePlan


def _retrieved_chunk(
    *,
    chunk_id: str,
    chunk_index: int,
    section: str = "3.2 Method",
    section_type: str = "content",
    rank: int = 1,
    rerank_score: float | None = 0.9,
) -> RetrievedChunk:
    """构造带完整元数据的检索结果替身，避免测试加载真实模型和 SQLite 文件。"""
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        vector_id=chunk_index,
        paper_id="paper-a",
        paper_title="Paper A",
        section=section,
        content_kind="body",
        front_matter_type=None,
        section_type=section_type,
        page_start=2,
        page_end=2,
        raw_text=f"text for {chunk_id}",
        pre_rerank_rank=rank if section_type == "content" else None,
        rerank_score=rerank_score if section_type == "content" else None,
        fused_score=0.1,
        source_matches=(),
    )


class _FakeDatabase:
    """返回同节连续块的内存替身；References 即使存在也不应在正文扩展中读取。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.rows = {
            ("paper-a", "3.2 Method"): tuple(
                {
                    "chunk_id": f"c{index}",
                    "chunk_index": index,
                    "raw_text": f"method chunk {index}",
                    "raw_token_count": 10,
                    "page_start": 2,
                    "page_end": 2,
                    "paper_title": "Paper A",
                }
                for index in range(1, 5)
            ),
            ("paper-a", "4. Results"): (
                {
                    "chunk_id": "r1",
                    "chunk_index": 5,
                    "raw_text": "results chunk",
                    "raw_token_count": 10,
                    "page_start": 3,
                    "page_end": 3,
                    "paper_title": "Paper A",
                },
            ),
        }

    def list_content_chunks_in_section(
        self, *, paper_id: str, section: str
    ) -> tuple[dict[str, object], ...]:
        self.calls.append((paper_id, section))
        return self.rows.get((paper_id, section), ())


class _FakeClient:
    """可控的 JSON Schema 客户端替身，记录结构化调用但不访问网络。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def complete(self, **kwargs: object) -> str:
        raise AssertionError("Step 8 应使用 complete_json 而非普通文本调用")


class SectionAwareNeighborExpansionTests(unittest.TestCase):
    def _result(self, chunks: tuple[RetrievedChunk, ...]) -> RetrievalResult:
        return RetrievalResult(
            query="How is the method implemented?",
            rewrite_plan=QueryRewritePlan(original_query="How is the method implemented?"),
            reranking_status="success",
            chunks=chunks,
        )

    def test_expands_seed_to_same_section_contiguous_neighbors(self) -> None:
        database = _FakeDatabase()
        expander = SectionAwareNeighborExpander(
            database=database,
            settings=ContextExpansionSettings(
                neighbor_window=1,
                max_total_tokens=256,
            ),
        )
        result = expander.expand(self._result((_retrieved_chunk(chunk_id="c2", chunk_index=2),)))

        self.assertEqual(database.calls, [("paper-a", "3.2 Method")])
        self.assertEqual(len(result.content_evidence), 1)
        self.assertEqual(result.content_evidence[0].chunk_ids, ("c1", "c2", "c3"))
        self.assertEqual(result.content_evidence[0].seed_chunk_ids, ("c2",))
        self.assertEqual(result.content_evidence[0].token_count, 30)

    def test_does_not_cross_section_and_keeps_bibliography_as_separate_evidence(self) -> None:
        database = _FakeDatabase()
        expander = SectionAwareNeighborExpander(
            database=database,
            settings=ContextExpansionSettings(
                neighbor_window=1,
                max_total_tokens=256,
            ),
        )
        result = expander.expand(
            self._result(
                (
                    _retrieved_chunk(chunk_id="c2", chunk_index=2),
                    _retrieved_chunk(
                        chunk_id="reference-1",
                        chunk_index=99,
                        section="References",
                        section_type="bibliography",
                        rank=2,
                        rerank_score=None,
                    ),
                )
            )
        )

        self.assertEqual(database.calls, [("paper-a", "3.2 Method")])
        self.assertEqual(result.content_evidence[0].chunk_ids, ("c1", "c2", "c3"))
        self.assertEqual(len(result.bibliography_evidence), 1)
        self.assertEqual(result.bibliography_evidence[0].evidence_id, "R1")
        self.assertEqual(result.bibliography_evidence[0].chunk_ids, ("reference-1",))

    def test_overlapping_windows_keep_all_reranker_seed_chunks(self) -> None:
        expander = SectionAwareNeighborExpander(
            database=_FakeDatabase(),
            settings=ContextExpansionSettings(neighbor_window=1, max_total_tokens=256),
        )
        result = expander.expand(
            self._result(
                (
                    _retrieved_chunk(chunk_id="c2", chunk_index=2, rank=1, rerank_score=0.9),
                    _retrieved_chunk(chunk_id="c3", chunk_index=3, rank=2, rerank_score=0.8),
                )
            )
        )

        self.assertEqual(len(result.content_evidence), 1)
        self.assertEqual(result.content_evidence[0].chunk_ids, ("c1", "c2", "c3", "c4"))
        self.assertEqual(result.content_evidence[0].seed_chunk_ids, ("c2", "c3"))

    def test_token_budget_keeps_complete_high_priority_group_only(self) -> None:
        expander = SectionAwareNeighborExpander(
            database=_FakeDatabase(),
            settings=ContextExpansionSettings(
                neighbor_window=1,
                max_total_tokens=256,
            ),
        )
        # 让第二个章节的完整组超过预算，验证程序不截断文本而是跳过该组。
        expander._database.rows[("paper-a", "4. Results")][0]["raw_token_count"] = 300
        result = expander.expand(
            self._result(
                (
                    _retrieved_chunk(chunk_id="c2", chunk_index=2, rank=1, rerank_score=0.9),
                    _retrieved_chunk(
                        chunk_id="r1",
                        chunk_index=5,
                        section="4. Results",
                        rank=2,
                        rerank_score=0.8,
                    ),
                )
            )
        )
        self.assertEqual(len(result.content_evidence), 1)
        self.assertEqual(result.content_evidence[0].section, "3.2 Method")


class GroundedAnswerGeneratorTests(unittest.TestCase):
    @staticmethod
    def _evidence() -> tuple[EvidenceUnit, ...]:
        return (
            EvidenceUnit(
                evidence_id="E1",
                kind="content",
                paper_id="paper-a",
                paper_title="Paper A",
                section="3.2 Method",
                page_start=2,
                page_end=2,
                seed_chunk_ids=("c2",),
                chunk_ids=("c1", "c2", "c3"),
                text="The method uses dynamic graphs.",
                token_count=30,
            ),
        )

    def test_uses_json_schema_and_accepts_only_known_citations(self) -> None:
        client = _FakeClient(
            ['{"direct_answer":"该方法使用动态图。[E1]","evidence_explanation":"证据说明其使用动态图。[E1]","reading_interpretation":"这表示动态图是该方法的核心机制。[E1]","citations":[" E1 ","E1"],"insufficient_evidence":false}']
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )
        outcome = generator.generate(query="方法是什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.citations, ("E1",))
        self.assertIn("### 论文中的依据与推导", outcome.answer or "")
        self.assertEqual(client.calls[0]["schema_name"], "paperbase_grounded_answer")
        self.assertEqual(client.calls[0]["json_schema"]["additionalProperties"], False)

    def test_prompt_requests_reading_level_explanation(self) -> None:
        """回答提示词应要求解释推导与含义，而不是只复制一行检索结论。"""
        self.assertIn("阅读式解释深度", ANSWER_GENERATION_SYSTEM_PROMPT)
        self.assertIn("公式、复杂度或定量结果问题", ANSWER_GENERATION_SYSTEM_PROMPT)
        self.assertIn("### 论文中的依据与推导", ANSWER_GENERATION_SYSTEM_PROMPT)
        self.assertIn("无关证据同样属于证据不足", ANSWER_GENERATION_SYSTEM_PROMPT)

    def test_rejects_fabricated_citation_and_falls_back_to_evidence(self) -> None:
        client = _FakeClient(
            ['{"direct_answer":"模型编造了来源。[E9]","evidence_explanation":"该结论来自不存在的来源。[E9]","reading_interpretation":null,"citations":["E9"],"insufficient_evidence":false}']
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )
        outcome = generator.generate(query="方法是什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "fallback")
        self.assertIsNotNone(outcome.answer)
        self.assertIn("回答生成未完成", outcome.answer or "")
        self.assertEqual(outcome.citations, ())
        self.assertTrue(outcome.insufficient_evidence)

    def test_invalid_answer_json_falls_back_without_json_repair_call(self) -> None:
        """Pydantic 拒绝额外字段后，系统只保留证据，不再发起第二次 LLM 修复请求。"""
        client = _FakeClient(
            [
                '{"direct_answer":"有额外字段的回答 [E1]",'
                '"evidence_explanation":"依据说明。[E1]","reading_interpretation":null,'
                '"citations":["E1"],"insufficient_evidence":false,"unexpected":true}',
            ]
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )

        outcome = generator.generate(query="方法是什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "fallback")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["schema_name"], "paperbase_grounded_answer")

    def test_deictic_single_paper_question_with_multiple_sources_is_not_guessed(self) -> None:
        client = _FakeClient([])
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )
        other_paper_evidence = EvidenceUnit(
            **{**self._evidence()[0].__dict__, "evidence_id": "E2", "paper_id": "paper-b"}
        )
        outcome = generator.generate(
            query="这篇论文使用了哪些数据集？",
            evidence=(*self._evidence(), other_paper_evidence),
        )

        self.assertEqual(outcome.status, "ambiguous_paper")
        self.assertTrue(outcome.insufficient_evidence)
        self.assertEqual(client.calls, [])

    def test_conversation_context_is_passed_to_prompt_for_follow_up_resolution(self) -> None:
        client = _FakeClient(
            ['{"direct_answer":"该论文使用动态图。[E1]","evidence_explanation":"证据描述了动态图的使用。[E1]","reading_interpretation":null,"citations":["E1"],"insufficient_evidence":false}']
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )
        outcome = generator.generate(
            query="那这篇论文呢？",
            evidence=self._evidence(),
            conversation_context=("上一轮正在讨论 Paper A。",),
        )

        self.assertEqual(outcome.status, "success")
        self.assertIn("<conversation_context>", client.calls[0]["user_prompt"])
        self.assertIn("Paper A", client.calls[0]["user_prompt"])

    def test_derives_citations_and_partial_status_from_grounded_answer_text(self) -> None:
        """模型漏填 citations、误标证据不足时，程序保留有依据的部分回答。"""
        client = _FakeClient(
            [
                '{"direct_answer":"该方法使用动态图。[E1]",'
                '"evidence_explanation":"证据说明其使用动态图。[E1]",'
                '"reading_interpretation":null,"citations":[],'
                '"insufficient_evidence":true}',
            ]
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )

        outcome = generator.generate(query="方法是什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "success")
        self.assertFalse(outcome.insufficient_evidence)
        self.assertTrue(outcome.partial_answer)
        self.assertEqual(outcome.citations, ("E1",))
        self.assertIsNotNone(outcome.coverage_note)
        self.assertIn("### 证据覆盖说明", outcome.answer or "")

    def test_true_insufficient_answer_keeps_no_citations(self) -> None:
        """完全无关的证据不能被误升级为部分回答，也要给前端稳定正文。"""
        client = _FakeClient(
            [
                '{"direct_answer":"提供的论文证据与该问题无关，无法据此回答。",'
                '"evidence_explanation":"当前证据只描述动态图方法，未涉及所问对象。",'
                '"reading_interpretation":null,"citations":["E1"],'
                '"insufficient_evidence":true}',
            ]
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )

        outcome = generator.generate(query="猫喜欢吃什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "success")
        self.assertTrue(outcome.insufficient_evidence)
        self.assertFalse(outcome.partial_answer)
        self.assertEqual(outcome.citations, ())
        self.assertTrue(outcome.answer)


if __name__ == "__main__":
    unittest.main()
