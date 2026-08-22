"""Ask This Paper 前端适配层的会话隔离与 Evidence 恢复测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.paperbase_service import PaperBaseService
from paperbase.config import load_settings
from paperbase.conversations import ConversationScopeError, ConversationStore


def test_paper_conversations_are_scoped_and_evidence_is_restored(tmp_path) -> None:
    settings = _temporary_settings(tmp_path)
    service = PaperBaseService(settings)
    workspace_a = _write_workspace(settings.storage.staging_dir, "staging_a", "Paper A")
    workspace_b = _write_workspace(settings.storage.staging_dir, "staging_b", "Paper B")

    first = service.create_paper_conversation(workspace_a)
    second = service.create_paper_conversation(workspace_a)
    service._paper_ask_services[workspace_a] = _FakePaperAskService(  # type: ignore[attr-defined]
        service._conversations
    )

    result = service.ask_this_paper(workspace_a, first.conversation_id, "第一轮问题")
    assert result.conversation_id == first.conversation_id
    summaries = service.list_paper_conversations(workspace_a)
    assert {item.conversation_id for item in summaries} == {
        first.conversation_id,
        second.conversation_id,
    }
    assert next(
        item for item in summaries if item.conversation_id == first.conversation_id
    ).title == "第一轮问题"

    turns = service.get_paper_conversation_turns(
        first.conversation_id, workspace_id=workspace_a
    )
    assert json.loads(turns[0].evidence_json or "[]")[0]["evidence_id"] == "E1"
    evidence = service.get_paper_conversation_turn_evidence(turns[0])
    assert evidence[0].paper_title == "Paper A"
    assert evidence[0].section == "方法"
    assert evidence[0].page_start == 2

    with pytest.raises(ConversationScopeError):
        service.get_paper_conversation_turns(
            first.conversation_id, workspace_id=workspace_b
        )


def test_legacy_paper_turn_can_restore_evidence_from_audit_artifact(tmp_path) -> None:
    settings = _temporary_settings(tmp_path)
    service = PaperBaseService(settings)
    workspace_id = _write_workspace(settings.storage.staging_dir, "staging_a", "Paper A")
    conversation = service.create_paper_conversation(workspace_id)
    artifact = settings.storage.staging_dir / workspace_id / "ask_paper" / "turn.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "answer_status": "success",
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "kind": "content",
                        "paper_title": "Paper A",
                        "section": "实验",
                        "page_start": 4,
                        "page_end": 5,
                        "text": "历史证据原文。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service._conversations.append_turn(  # type: ignore[attr-defined]
        conversation.conversation_id,
        user_query="历史问题",
        assistant_answer="历史回答 [E1]",
        audit_path=str(artifact),
    )

    turn = service.get_paper_conversation_turns(
        conversation.conversation_id, workspace_id=workspace_id
    )[0]
    assert service.get_paper_conversation_turn_evidence(turn)[0].text == "历史证据原文。"


class _FakePaperAskService:
    def __init__(self, store: ConversationStore) -> None:
        self.store = store

    def ask(self, query: str, *, conversation_id: str) -> object:
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
            paper_title="Paper A",
            section="方法",
            page_start=2,
            page_end=None,
            text="当前论文的真实证据。",
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


def _write_workspace(staging_root, workspace_id: str, title: str) -> str:
    root = staging_root / workspace_id
    (root / "parsed").mkdir(parents=True)
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "paper_id": f"paper_{workspace_id}",
                "source_filename": f"{workspace_id}.pdf",
                "total_chunk_count": 2,
                "added_to_kb": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "parsed" / "parsed_paper.json").write_text(
        json.dumps({"paper_title": title, "sections": [{"title": "方法"}]}),
        encoding="utf-8",
    )
    return workspace_id


def _temporary_settings(tmp_path):
    settings = load_settings()
    storage = settings.storage.model_copy(update={"staging_dir": tmp_path / "staging"})
    database = settings.database.model_copy(update={"path": tmp_path / "paperbase.sqlite3"})
    conversation = settings.conversation.model_copy(
        update={"path": tmp_path / "conversations.sqlite3"}
    )
    return settings.model_copy(
        update={"storage": storage, "database": database, "conversation": conversation}
    )
