"""Streamlit 上传入口到既有 staging pipeline 的 Service 层回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import paperbase_service as service_module
from app.services.paperbase_service import PaperBaseService, PaperUploadError
from paperbase.config import load_settings


def test_pdf_upload_delegates_to_staging_pipeline_and_returns_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """上传内容只经临时副本进入既有后端 stage，成功后恢复展示摘要。"""
    settings = _temporary_settings(tmp_path)
    captured: dict[str, object] = {}

    def fake_stage(*, settings, source_pdf: Path):
        # 后端 stage 收到的是原始安全文件名与完整 bytes，而不是 staging 内部路径。
        captured["source"] = source_pdf
        captured["filename"] = source_pdf.name
        captured["content"] = source_pdf.read_bytes()
        workspace_id = "staging_uploaded"
        root = settings.storage.staging_dir / workspace_id
        (root / "parsed").mkdir(parents=True)
        (root / "workspace.json").write_text(
            json.dumps(
                {
                    "workspace_id": workspace_id,
                    "paper_id": "paper_uploaded",
                    "source_filename": source_pdf.name,
                    "total_chunk_count": 3,
                    "added_to_kb": False,
                }
            ),
            encoding="utf-8",
        )
        (root / "parsed" / "parsed_paper.json").write_text(
            json.dumps({"paper_title": "Uploaded Paper", "sections": [{}, {}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(workspace_id=workspace_id)

    monkeypatch.setattr(service_module, "run_temporary_workspace_stage", fake_stage)
    service = PaperBaseService(settings)
    content = b"%PDF-1.7\nminimal test PDF"

    result = service.create_workspace_from_pdf_upload(
        filename=r"C:\\fake-path\\uploaded-paper.pdf", content=content
    )

    assert captured["filename"] == "uploaded-paper.pdf"
    assert captured["content"] == content
    assert not Path(captured["source"]).exists()
    assert result.workspace_id == "staging_uploaded"
    assert result.display_title == "Uploaded Paper"
    assert result.section_count == 2


@pytest.mark.parametrize(
    ("filename", "content"),
    (
        ("not-a-pdf.txt", b"%PDF-1.7"),
        ("empty.pdf", b""),
        ("invalid.pdf", b"not really a pdf"),
    ),
)
def test_pdf_upload_rejects_invalid_input_before_staging_stage(
    tmp_path: Path, filename: str, content: bytes
) -> None:
    """扩展名、空文件和无 PDF 签名输入都不应启动昂贵的后端流程。"""
    with pytest.raises(PaperUploadError):
        PaperBaseService(_temporary_settings(tmp_path)).create_workspace_from_pdf_upload(
            filename=filename, content=content
        )


def _temporary_settings(tmp_path: Path):
    """构造隔离的 Storage / SQLite 配置，不接触开发目录真实数据。"""
    settings = load_settings()
    storage = settings.storage.model_copy(
        update={
            "staging_dir": tmp_path / "staging",
            "papers_dir": tmp_path / "papers",
            "parsed_dir": tmp_path / "parsed",
        }
    )
    database = settings.database.model_copy(update={"path": tmp_path / "paperbase.sqlite3"})
    conversation = settings.conversation.model_copy(update={"path": tmp_path / "conversations.sqlite3"})
    return settings.model_copy(
        update={"storage": storage, "database": database, "conversation": conversation}
    )
