"""Conversation Store 的持久化、范围隔离与 Query Resolution 接入测试。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paperbase.config import QueryRewriteSettings
from paperbase.conversations import (
    ConversationScopeError,
    ConversationStore,
    format_turns_as_context,
)
from paperbase.retrieval.query_rewriter import LLMQueryPlanner
from paperbase.retrieval.hybrid_retriever import RetrievalResult
from paperbase.retrieval.query_rewriter import QueryRewritePlan
from paperbase.generation.answer_generator import AnswerGenerationOutcome
from paperbase.generation.section_expander import ExpansionResult
from paperbase.generation.service import AnswerService


class _FakeClient:
    """按顺序返回 Resolution 与 Retrieval Rewrite JSON，不访问外部 LLM。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return next(self._responses)


class _ContextRecordingRetriever:
    """只记录服务传给 Query Planning 的历史，不执行真实索引检索。"""

    def __init__(self) -> None:
        self.contexts: list[tuple[str, ...]] = []

    def retrieve(self, query: str, *, conversation_context: object, result_limit: object) -> RetrievalResult:
        _ = result_limit
        self.contexts.append(tuple(conversation_context or ()))
        return RetrievalResult(
            query=query,
            rewrite_plan=QueryRewritePlan(original_query=query, resolved_query=query),
            reranking_status="disabled",
            chunks=(),
        )


class _EmptyExpander:
    def expand(self, result: object) -> ExpansionResult:
        _ = result
        return ExpansionResult(content_evidence=(), bibliography_evidence=())


class _RecordingGenerator:
    def __init__(self) -> None:
        self.contexts: list[object] = []

    def generate(self, *, query: str, evidence: object, conversation_context: object) -> AnswerGenerationOutcome:
        _ = query, evidence
        self.contexts.append(conversation_context)
        return AnswerGenerationOutcome(
            status="success",
            answer="最终展示答案。",
            citations=(),
            insufficient_evidence=True,
        )


class ConversationStoreTests(unittest.TestCase):
    def test_turns_survive_store_reinstantiation_and_keep_recent_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "conversations.sqlite3"
            store = ConversationStore(path)
            conversation = store.create_conversation("knowledge_base", "default")
            for index in range(6):
                store.append_turn(
                    conversation.conversation_id,
                    user_query=f"问题 {index}",
                    assistant_answer=f"回答 {index}",
                )

            # 模拟程序重启：重新实例化后仍从同一独立 SQLite 读取。
            restarted_store = ConversationStore(path)
            recent = restarted_store.get_recent_turns(conversation.conversation_id, 4)

            self.assertEqual([turn.turn_index for turn in recent], [2, 3, 4, 5])
            self.assertEqual([turn.user_query for turn in recent], ["问题 2", "问题 3", "问题 4", "问题 5"])
            self.assertEqual(restarted_store.get_conversation(conversation.conversation_id).scope_id, "default")

    def test_scope_isolation_rejects_kb_and_other_workspace(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = ConversationStore(Path(temporary_directory) / "conversations.sqlite3")
            kb = store.create_conversation("knowledge_base", "default")
            workspace_a = store.create_conversation("workspace", "staging_a")
            workspace_b = store.create_conversation("workspace", "staging_b")

            store.append_turn(
                workspace_a.conversation_id,
                user_query="workspace A 问题",
                assistant_answer="workspace A 回答",
            )
            self.assertEqual(store.get_recent_turns(kb.conversation_id, 4), ())
            self.assertEqual(store.get_recent_turns(workspace_b.conversation_id, 4), ())
            with self.assertRaises(ConversationScopeError):
                store.get_conversation(
                    workspace_a.conversation_id,
                    expected_scope_type="workspace",
                    expected_scope_id="staging_b",
                )
            with self.assertRaises(ConversationScopeError):
                store.get_conversation(
                    kb.conversation_id,
                    expected_scope_type="workspace",
                    expected_scope_id="staging_a",
                )

    def test_persisted_user_and_assistant_turn_resolves_second_step_then_rewrites(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = ConversationStore(Path(temporary_directory) / "conversations.sqlite3")
            conversation = store.create_conversation("knowledge_base", "default")
            store.append_turn(
                conversation.conversation_id,
                user_query="ESDTW 的整体流程是什么？",
                assistant_answer="ESDTW 的第二步会计算 shape descriptors。",
                resolved_query="ESDTW 的整体流程是什么？",
                resolution_status="resolved",
            )
            context = format_turns_as_context(
                store.get_recent_turns(conversation.conversation_id, 4)
            )
            client = _FakeClient(
                [
                    '{"resolution_status":"resolved",'
                    '"resolved_query":"ESDTW 的第二步具体如何计算 shape descriptors？"}',
                    '{"semantic_query_en":"How does ESDTW compute shape descriptors in its second step?",'
                    '"lexical_keywords_en":["ESDTW", "shape descriptors"]}',
                ]
            )
            planner = LLMQueryPlanner(settings=QueryRewriteSettings(), client=client)

            plan = planner.plan("第二步具体怎么做？", conversation_context=context)

            self.assertEqual(plan.resolution_status, "resolved")
            self.assertEqual(
                plan.resolved_query, "ESDTW 的第二步具体如何计算 shape descriptors？"
            )
            self.assertEqual(plan.rewrite_status, "success")
            self.assertEqual(
                [call["schema_name"] for call in client.calls],
                ["query_resolution_result", "retrieval_rewrite_result"],
            )
            self.assertIn("User: ESDTW 的整体流程是什么？", client.calls[0]["user_prompt"])  # type: ignore[operator]
            self.assertIn("Assistant: ESDTW 的第二步会计算 shape descriptors。", client.calls[0]["user_prompt"])  # type: ignore[operator]
            # 第二次 LLM 调用只接收 resolved_query，历史文本不进入 Retrieval Rewrite。
            self.assertNotIn("Assistant:", client.calls[1]["user_prompt"])  # type: ignore[operator]

    def test_kb_service_loads_persisted_context_but_never_passes_it_to_answer_llm(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = ConversationStore(Path(temporary_directory) / "conversations.sqlite3")
            retriever = _ContextRecordingRetriever()
            generator = _RecordingGenerator()
            service = AnswerService(
                retriever=retriever,
                expander=_EmptyExpander(),  # type: ignore[arg-type]
                generator=generator,  # type: ignore[arg-type]
                conversation_store=store,
                conversation_max_context_turns=4,
            )

            first = service.answer_query("ESDTW 的整体流程是什么？")
            second = service.answer_query("第二步具体怎么做？", conversation_id=first.conversation_id)

            self.assertEqual(first.conversation_id, second.conversation_id)
            self.assertIn("User: ESDTW 的整体流程是什么？", retriever.contexts[1][0])
            self.assertIn("Assistant: 最终展示答案。", retriever.contexts[1][0])
            self.assertEqual(generator.contexts, [None, None])


if __name__ == "__main__":
    unittest.main()
