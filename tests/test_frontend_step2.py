"""Frontend Step 2 的会话、历史恢复和 Evidence Service 测试。"""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

from app.services.paperbase_service import (
    FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
    PaperBaseService,
)
from paperbase.config import load_settings
from paperbase.conversations import ConversationStore
from paperbase.generation.service import create_answer_service


def test_streamlit_and_cli_factory_share_formal_kb_scope() -> None:
    """产品入口默认使用同一个正式 scope，避免 UI 创建的会话无法被 CLI 读取。"""
    default_scope = signature(create_answer_service).parameters[
        "conversation_scope_id"
    ].default
    assert default_scope == FORMAL_KNOWLEDGE_BASE_SCOPE_ID


def test_kb_conversation_list_titles_and_history_are_persistent(tmp_path) -> None:
    service = PaperBaseService(_temporary_settings(tmp_path))
    first = service.create_kb_conversation()
    second = service.create_kb_conversation()

    service._conversations.append_turn(  # type: ignore[attr-defined]
        first.conversation_id,
        user_query="ESDTW 和 DTW 有什么区别？",
        assistant_answer="回答 [E1]",
    )

    conversations = service.list_kb_conversations()
    assert [item.conversation_id for item in conversations] == [
        first.conversation_id,
        second.conversation_id,
    ]
    assert conversations[0].title == "ESDTW 和 DTW 有什么区别？"
    assert conversations[1].title == "新会话"
    turns = service.get_conversation_turns(first.conversation_id)
    assert [turn.user_query for turn in turns] == ["ESDTW 和 DTW 有什么区别？"]


def test_kb_answer_service_receives_query_and_persists_real_evidence_snapshot(tmp_path) -> None:
    settings = _temporary_settings(tmp_path)
    service = PaperBaseService(settings)
    conversation = service.create_kb_conversation()
    fake = _FakeAnswerService(service._conversations)  # type: ignore[attr-defined]
    service._answer_service = fake  # type: ignore[attr-defined]

    result = service.ask_knowledge_base("  第一轮问题  ", conversation.conversation_id)

    assert fake.calls == [("第一轮问题", conversation.conversation_id)]
    turns = service.get_conversation_turns(conversation.conversation_id)
    assert turns[0].assistant_answer == "回答 [E1]"
    evidence = service.get_conversation_turn_evidence(turns[0])
    assert evidence[0].evidence_id == "E1"
    assert evidence[0].paper_title == "ESDTW"
    assert evidence[0].section == "方法"
    assert evidence[0].page_start == 3
    assert evidence[0].page_end is None
    assert evidence[0].text == "真实证据原文"


def test_kb_scope_does_not_accept_default_or_workspace_conversation(tmp_path) -> None:
    service = PaperBaseService(_temporary_settings(tmp_path))
    default_store = ConversationStore(service._settings.conversation.path)  # type: ignore[attr-defined]
    wrong = default_store.create_conversation("knowledge_base", "default")
    with_workspace = default_store.create_conversation("workspace", "staging_a")

    for conversation_id in (wrong.conversation_id, with_workspace.conversation_id):
        try:
            service.get_conversation_turns(conversation_id)
        except Exception as error:
            assert "scope" in str(error).lower()
        else:
            raise AssertionError("cross-scope conversation must be rejected")


class _FakeAnswerService:
    """模拟正式 QA 链路，只验证 Service 传递 query + conversation_id。"""

    def __init__(self, store: ConversationStore) -> None:
        self.store = store
        self.calls: list[tuple[str, str]] = []

    def answer_query(self, query: str, *, conversation_id: str) -> object:
        self.calls.append((query, conversation_id))
        self.store.append_turn(
            conversation_id,
            user_query=query,
            assistant_answer="回答 [E1]",
            resolved_query=query,
            resolution_status="resolved",
        )
        evidence = SimpleNamespace(
            evidence_id="E1",
            kind="content",
            paper_title="ESDTW",
            section="方法",
            page_start=3,
            page_end=None,
            text="真实证据原文",
        )
        return SimpleNamespace(
            conversation_id=conversation_id,
            answer=SimpleNamespace(
                status="success",
                answer="回答 [E1]",
                citations=("E1",),
                insufficient_evidence=False,
                partial_answer=False,
                coverage_note=None,
            ),
            expansion=SimpleNamespace(evidence=(evidence,)),
        )


def _temporary_settings(tmp_path):
    settings = load_settings()
    storage = settings.storage.model_copy(
        update={
            "papers_dir": tmp_path / "papers",
            "parsed_dir": tmp_path / "parsed",
            "staging_dir": tmp_path / "staging",
        }
    )
    database = settings.database.model_copy(update={"path": tmp_path / "paperbase.sqlite3"})
    conversation = settings.conversation.model_copy(
        update={"path": tmp_path / "conversations.sqlite3"}
    )
    return settings.model_copy(
        update={"storage": storage, "database": database, "conversation": conversation}
    )
