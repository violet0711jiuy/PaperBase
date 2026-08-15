"""Step 8 的离线回归测试：邻居扩展边界、答案结构化校验和故障降级。"""

from __future__ import annotations

import unittest

from paperbase.config import AnswerGenerationSettings, ContextExpansionSettings
from paperbase.generation.answer_generator import GroundedAnswerGenerator
from paperbase.generation.section_expander import EvidenceUnit, SectionAwareNeighborExpander
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
            ['{"answer":"该方法使用动态图。[E1]","citations":[" E1 ","E1"],"insufficient_evidence":false}']
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )
        outcome = generator.generate(query="方法是什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.citations, ("E1",))
        self.assertEqual(client.calls[0]["schema_name"], "paperbase_grounded_answer")
        self.assertEqual(client.calls[0]["json_schema"]["additionalProperties"], False)

    def test_rejects_fabricated_citation_and_falls_back_to_evidence(self) -> None:
        client = _FakeClient(
            ['{"answer":"模型编造了来源。[E9]","citations":["E9"],"insufficient_evidence":false}']
        )
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(),
            client=client,
        )
        outcome = generator.generate(query="方法是什么？", evidence=self._evidence())

        self.assertEqual(outcome.status, "fallback")
        self.assertIsNone(outcome.answer)
        self.assertEqual(outcome.citations, ())

    def test_invalid_answer_json_falls_back_without_json_repair_call(self) -> None:
        """Pydantic 拒绝额外字段后，系统只保留证据，不再发起第二次 LLM 修复请求。"""
        client = _FakeClient(
            [
                '{"answer":"有额外字段的回答 [E1]","citations":["E1"],'
                '"insufficient_evidence":false,"unexpected":true}',
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
            ['{"answer":"该论文使用动态图。[E1]","citations":["E1"],"insufficient_evidence":false}']
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


if __name__ == "__main__":
    unittest.main()
