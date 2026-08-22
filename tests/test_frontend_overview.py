"""Paper Workspace Overview Service 适配和工件恢复测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import paperbase_service as service_module
from app.services.paperbase_service import PaperBaseService, PaperOverviewArtifactError
from paperbase.config import load_settings
from paperbase.overview.service import PaperOverview, OverviewField


def test_overview_artifact_and_sources_restore_after_new_service(tmp_path: Path) -> None:
    settings = _temporary_settings(tmp_path)
    workspace_id = "staging_overview"
    root = _write_workspace(settings, workspace_id)
    overview = _overview()
    (root / "overview").mkdir()
    (root / "overview" / "overview.json").write_text(
        json.dumps(overview.model_dump(), ensure_ascii=False), encoding="utf-8"
    )
    (root / "chunks" / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk_1",
                "section": "3. Method",
                "page_start": 6,
                "page_end": 7,
                "raw_text": "原文证据。第二句。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    service = PaperBaseService(settings)
    restored = service.get_paper_overview(workspace_id)
    assert restored is not None
    assert restored.paper_title == "Example Paper"
    sources = service.get_paper_overview_sources(workspace_id, restored.main_method.source_chunk_ids)
    assert [(source.section, source.page_start, source.page_end, source.text) for source in sources] == [
        ("3. Method", 6, 7, "原文证据。第二句。")
    ]

    # 新建 Service 仍从 workspace artifact 读取，不需要重新生成。
    restarted = PaperBaseService(settings)
    assert restarted.get_paper_overview(workspace_id) == restored


def test_corrupt_overview_artifact_is_not_silently_accepted(tmp_path: Path) -> None:
    settings = _temporary_settings(tmp_path)
    root = _write_workspace(settings, "staging_broken")
    (root / "overview").mkdir()
    (root / "overview" / "overview.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(PaperOverviewArtifactError):
        PaperBaseService(settings).get_paper_overview("staging_broken")


def test_generate_overview_delegates_to_existing_backend_stage(tmp_path: Path, monkeypatch) -> None:
    settings = _temporary_settings(tmp_path)
    _write_workspace(settings, "staging_generate")
    expected = _overview()
    calls: list[tuple[object, str]] = []

    def fake_stage(*, settings, workspace_id):
        calls.append((settings, workspace_id))
        return SimpleNamespace(overview=expected)

    monkeypatch.setattr(service_module, "run_paper_overview_stage", fake_stage)
    result = PaperBaseService(settings).generate_paper_overview("staging_generate")

    assert result == expected
    assert calls == [(settings, "staging_generate")]


def _write_workspace(settings, workspace_id: str) -> Path:
    root = settings.storage.staging_dir / workspace_id
    (root / "parsed").mkdir(parents=True)
    (root / "chunks").mkdir()
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "paper_id": "paper_example",
                "source_filename": "example.pdf",
                "total_chunk_count": 1,
                "added_to_kb": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "parsed" / "parsed_paper.json").write_text(
        json.dumps({"paper_title": "Example Paper", "sections": [{}, {}]}),
        encoding="utf-8",
    )
    return root


def _overview() -> PaperOverview:
    field = lambda text, ids=(): OverviewField(content=text, source_chunk_ids=list(ids))
    return PaperOverview(
        paper_title="Example Paper",
        research_problem=field("问题"),
        main_method=field("方法", ("chunk_1",)),
        contributions=field("贡献"),
        datasets=field("数据集"),
        experimental_setup=field("实验"),
        main_results=field("结果"),
        limitations=field("论文未明确说明"),
    )


def _temporary_settings(tmp_path: Path):
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

