"""Frontend Step 2 的会话、历史恢复和 Evidence Service 测试。"""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace
import json

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
    metadata = service.get_conversation_turn_metadata(turns[0])
    assert metadata is not None
    assert metadata.mode == "quick"
    assert not service._answer_artifact_path(conversation.conversation_id, 0).exists()


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


def test_legacy_kb_sidecar_remains_readable(tmp_path) -> None:
    service = PaperBaseService(_temporary_settings(tmp_path))
    conversation = service.create_kb_conversation()
    turn = service._conversations.append_turn(  # type: ignore[attr-defined]
        conversation.conversation_id,
        user_query="旧问题",
        assistant_answer="旧回答 [E1]",
    )
    path = service._answer_artifact_path(conversation.conversation_id, turn.turn_index)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "conversation_id": conversation.conversation_id,
        "turn_index": turn.turn_index,
        "answer_status": "success",
        "partial_answer": True,
        "coverage_note": "只覆盖一篇论文。",
        "evidence": [{
            "evidence_id": "E1",
            "type": "content",
            "paper_title": "旧论文",
            "raw_text": "旧证据",
        }],
    }, ensure_ascii=False), encoding="utf-8")

    restored = service.get_conversation_turns(conversation.conversation_id)[0]
    metadata = service.get_conversation_turn_metadata(restored)
    evidence = service.get_conversation_turn_evidence(restored)

    assert metadata is not None
    assert metadata.partial_answer
    assert metadata.coverage_note == "只覆盖一篇论文。"
    assert evidence[0].text == "旧证据"


def test_agentic_stream_persists_answer_thread_and_evidence(tmp_path) -> None:
    fake = _FakeAgenticService()
    service = PaperBaseService(
        _temporary_settings(tmp_path),
        agentic_service=fake,
    )
    conversation = service.create_kb_conversation()

    events = list(
        service.stream_knowledge_base_agentic(
            "  比较 A 和 B  ",
            conversation.conversation_id,
            selected_paper_ids=("A", "B"),
            knowledge_base_all=False,
            thread_id="research-test",
        )
    )

    assert [event.kind for event in events] == ["started", "completed"]
    assert fake.calls == [("比较 A 和 B", ("A", "B"), False)]
    turn = service.get_conversation_turns(conversation.conversation_id)[0]
    assert turn.assistant_answer == "A 与 B 的方法不同 [E1]。"
    evidence = service.get_conversation_turn_evidence(turn)
    assert evidence[0].paper_title == "Paper A"
    metadata = service.get_conversation_turn_metadata(turn)
    assert metadata is not None
    assert metadata.mode == "agentic"
    assert metadata.thread_id == "research-test"
    assert turn.metadata_json is not None
    assert not service._answer_artifact_path(conversation.conversation_id, 0).exists()
    audit = service.get_conversation_turn_audit(turn)
    assert audit is not None
    assert audit.thread_id == "research-test"
    assert audit.schema_version == "grounded-answer-v1"
    assert service.export_conversation_turn_audit(turn) == audit.to_json(indent=2)
    assert service.get_pending_agentic_task(conversation.conversation_id) is None


def test_agentic_interruption_can_resume_without_starting_a_new_thread(tmp_path) -> None:
    fake = _RecoverableFakeAgenticService()
    service = PaperBaseService(
        _temporary_settings(tmp_path),
        agentic_service=fake,
    )
    conversation = service.create_kb_conversation()

    first_events = list(
        service.stream_knowledge_base_agentic(
            "比较 A 和 B",
            conversation.conversation_id,
            selected_paper_ids=("A", "B"),
            knowledge_base_all=False,
            thread_id="resume-thread",
        )
    )
    pending = service.get_pending_agentic_task(conversation.conversation_id)

    assert first_events[-1].kind == "interrupted"
    assert pending is not None
    assert pending.thread_id == "resume-thread"
    assert service.get_conversation_turns(conversation.conversation_id) == ()

    resumed_events = list(service.resume_knowledge_base_agentic(conversation.conversation_id))

    assert resumed_events[-1].kind == "completed"
    assert fake.started_threads == ["resume-thread"]
    assert fake.resumed_threads == ["resume-thread"]
    assert service.get_pending_agentic_task(conversation.conversation_id) is None
    assert len(service.get_conversation_turns(conversation.conversation_id)) == 1


class _FakeAnswerService:
    """模拟正式 QA 链路，只验证 Service 传递 query + conversation_id。"""

    def __init__(self, store: ConversationStore) -> None:
        self.store = store
        self.calls: list[tuple[str, str]] = []

    def answer_query(self, query: str, *, conversation_id: str) -> object:
        self.calls.append((query, conversation_id))
        evidence = SimpleNamespace(
            evidence_id="E1",
            kind="content",
            paper_title="ESDTW",
            section="方法",
            page_start=3,
            page_end=None,
            text="真实证据原文",
        )
        self.store.append_turn(
            conversation_id,
            user_query=query,
            assistant_answer="回答 [E1]",
            resolved_query=query,
            resolution_status="resolved",
            evidence_json=json.dumps([{
                "evidence_id": evidence.evidence_id,
                "type": evidence.kind,
                "paper_title": evidence.paper_title,
                "section": evidence.section,
                "page_start": evidence.page_start,
                "page_end": evidence.page_end,
                "raw_text": evidence.text,
            }], ensure_ascii=False),
            metadata_json=json.dumps({
                "answer_status": "success",
                "insufficient_evidence": False,
                "partial_answer": False,
                "coverage_note": None,
                "mode": "quick",
            }),
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


class _FakeAgenticService:
    def __init__(self) -> None:
        self.calls = []

    def stream_research(
        self,
        query,
        *,
        selected_paper_ids,
        knowledge_base_all,
        thread_id=None,
    ):
        self.calls.append((query, selected_paper_ids, knowledge_base_all))
        yield SimpleNamespace(
            kind="started",
            message="研究任务已开始",
            thread_id="research-test",
            result=None,
        )
        source = SimpleNamespace(
            source_type=SimpleNamespace(value="content"),
            paper_id="A",
            paper_title="Paper A",
            section="方法",
            page_start=2,
            page_end=3,
            content="Paper A 的方法证据。",
        )
        result = SimpleNamespace(
            answer=SimpleNamespace(
                status="success",
                answer="A 与 B 的方法不同 [E1]。",
                citations=("E1",),
                insufficient_evidence=False,
                partial_answer=False,
                coverage_note=None,
            ),
            selected_evidence=(
                SimpleNamespace(evidence_id="E1", evidence=source),
            ),
            query_context=SimpleNamespace(resolved_query=query),
            thread_id="research-test",
            error_codes=(),
        )
        yield SimpleNamespace(
            kind="completed",
            message="研究任务已完成",
            thread_id="research-test",
            result=result,
        )


class _RecoverableFakeAgenticService:
    """模拟同一 checkpoint 的中断与恢复，不重复执行新任务入口。"""

    def __init__(self) -> None:
        self.started_threads: list[str] = []
        self.resumed_threads: list[str] = []

    def stream_research(
        self,
        query,
        *,
        selected_paper_ids,
        knowledge_base_all,
        thread_id=None,
    ):
        _ = query, selected_paper_ids, knowledge_base_all
        self.started_threads.append(thread_id)
        yield SimpleNamespace(
            kind="started",
            message="研究任务已开始",
            thread_id=thread_id,
            result=None,
        )
        yield SimpleNamespace(
            kind="interrupted",
            message="研究任务已暂停",
            thread_id=thread_id,
            result=None,
        )

    def resume_stream(self, thread_id):
        self.resumed_threads.append(thread_id)
        yield SimpleNamespace(
            kind="started",
            message="正在恢复研究任务",
            thread_id=thread_id,
            result=None,
        )
        source = SimpleNamespace(
            source_type=SimpleNamespace(value="content"),
            paper_id="A",
            paper_title="Paper A",
            section="方法",
            page_start=2,
            page_end=3,
            content="Paper A 的方法证据。",
        )
        evidence = SimpleNamespace(evidence_id="E1", evidence=source)
        result = SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            stop_reason=SimpleNamespace(value="agent_finished"),
            answer=SimpleNamespace(
                status="success",
                answer="恢复后的回答 [E1]。",
                citations=("E1",),
                insufficient_evidence=False,
                partial_answer=False,
                coverage_note=None,
                llm_diagnostics=(),
            ),
            selected_evidence=(evidence,),
            evidence=(evidence,),
            observations=(),
            query_context=SimpleNamespace(
                resolved_query="比较 A 和 B",
                scope=SimpleNamespace(value="selected_papers"),
            ),
            plan=None,
            coverage=None,
            candidate_selection=None,
            budget_usage=SimpleNamespace(),
            thread_id=thread_id,
            error_codes=(),
            clarification=None,
            llm_diagnostics=(),
        )
        yield SimpleNamespace(
            kind="completed",
            message="研究任务已完成",
            thread_id=thread_id,
            result=result,
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
