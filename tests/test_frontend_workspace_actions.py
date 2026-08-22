"""Paper Workspace 的加入知识库与删除 staging 工作区回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.services import paperbase_service as service_module
from app.services.paperbase_service import PaperBaseService
from paperbase.config import load_settings


def test_add_workspace_delegates_to_existing_promotion_service(
    tmp_path: Path, monkeypatch
) -> None:
    """前端 Service 只转交 settings/workspace_id，不复制 promotion 流程。"""
    settings = _temporary_settings(tmp_path)
    workspace_id = "staging_promote"
    _write_workspace(settings, workspace_id)
    captured: dict[str, object] = {}

    def fake_promote_workspace(*, settings, workspace_id: str):
        captured["settings"] = settings
        captured["workspace_id"] = workspace_id
        return SimpleNamespace(status="promoted", workspace_id=workspace_id)

    monkeypatch.setattr(service_module, "promote_workspace", fake_promote_workspace)

    result = PaperBaseService(settings).add_workspace_to_knowledge_base(workspace_id)

    assert result.status == "promoted"
    assert captured == {"settings": settings, "workspace_id": workspace_id}


def test_delete_workspace_only_removes_the_selected_staging_directory(tmp_path: Path) -> None:
    """删除临时论文不触碰正式 papers 目录，也不删除同级其他 workspace。"""
    settings = _temporary_settings(tmp_path)
    selected_id = "staging_delete_me"
    other_id = "staging_keep_me"
    selected_root = _write_workspace(settings, selected_id)
    other_root = _write_workspace(settings, other_id)
    formal_pdf = settings.storage.papers_dir / "formal-paper.pdf"
    formal_pdf.parent.mkdir(parents=True, exist_ok=True)
    formal_pdf.write_bytes(b"formal")

    PaperBaseService(settings).delete_workspace(selected_id)

    assert not selected_root.exists()
    assert other_root.is_dir()
    assert formal_pdf.read_bytes() == b"formal"


def _write_workspace(settings, workspace_id: str) -> Path:
    """创建通过 Service manifest 校验的最小 staging workspace。"""
    root = settings.storage.staging_dir / workspace_id
    (root / "parsed").mkdir(parents=True)
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "paper_id": f"paper_{workspace_id}",
                "source_filename": "example.pdf",
                "total_chunk_count": 3,
                "added_to_kb": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "parsed" / "parsed_paper.json").write_text(
        json.dumps({"paper_title": "Example Paper", "sections": [{}]}),
        encoding="utf-8",
    )
    return root


def _temporary_settings(tmp_path: Path):
    """构造隔离的 storage/SQLite 配置，避免测试接触真实开发数据。"""
    settings = load_settings()
    storage = settings.storage.model_copy(
        update={
            "staging_dir": tmp_path / "staging",
            "papers_dir": tmp_path / "papers",
            "parsed_dir": tmp_path / "parsed",
        }
    )
    database = settings.database.model_copy(update={"path": tmp_path / "paperbase.sqlite3"})
    conversation = settings.conversation.model_copy(
        update={"path": tmp_path / "conversations.sqlite3"}
    )
    return settings.model_copy(
        update={"storage": storage, "database": database, "conversation": conversation}
    )
