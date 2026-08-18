"""Focused tests for the Step 1 Streamlit service and UI-only state helpers."""

from __future__ import annotations

import json

from app.services.paperbase_service import (
    FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
    PaperBaseService,
)
from app.state import activate_workspace, initialize_session_state, set_workspace_conversation
from paperbase.config import load_settings


def test_service_lists_workspace_and_keeps_conversation_scopes_separate(tmp_path) -> None:
    service = PaperBaseService(_temporary_settings(tmp_path))
    workspace_id = "staging_example"
    workspace_root = tmp_path / "staging" / workspace_id
    (workspace_root / "parsed").mkdir(parents=True)
    (workspace_root / "workspace.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "paper_id": "paper_example",
                "source_filename": "example.pdf",
                "total_chunk_count": 12,
                "added_to_kb": False,
            }
        ),
        encoding="utf-8",
    )
    (workspace_root / "parsed" / "parsed_paper.json").write_text(
        json.dumps({"paper_title": "Example Paper", "sections": [{}, {}]}),
        encoding="utf-8",
    )

    assert service.list_knowledge_base_papers() == ()
    workspace = service.get_workspace(workspace_id)
    assert workspace is not None
    assert workspace.display_title == "Example Paper"
    assert workspace.section_count == 2
    assert service.get_workspace("../outside") is None

    kb_conversation = service.create_kb_conversation()
    workspace_conversation = service.create_workspace_conversation(workspace_id)
    assert kb_conversation.scope_type == "knowledge_base"
    assert kb_conversation.scope_id == FORMAL_KNOWLEDGE_BASE_SCOPE_ID
    assert workspace_conversation.scope_type == "workspace"
    assert workspace_conversation.scope_id == workspace_id


def test_workspace_state_restores_only_the_matching_workspace_conversation() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    activate_workspace(state, "staging_one")
    set_workspace_conversation(
        state, workspace_id="staging_one", conversation_id="conversation_one"
    )
    activate_workspace(state, "staging_two")
    assert state["paper_conversation_id"] is None
    set_workspace_conversation(
        state, workspace_id="staging_two", conversation_id="conversation_two"
    )
    activate_workspace(state, "staging_one")
    assert state["paper_conversation_id"] == "conversation_one"


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
