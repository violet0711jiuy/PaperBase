"""Explain Section 前端 Service 适配测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from app.services import paperbase_service as service_module
from app.services.paperbase_service import PaperBaseService
from app.components.explain_section import _extract_template_points, _split_explanation_points
from app.components.scientific_text import normalize_scientific_text
from paperbase.config import load_settings
from paperbase.explain_section.service import ExplainSection


def test_section_tree_uses_persisted_hierarchy_and_source_chunks(tmp_path: Path) -> None:
    settings = _temporary_settings(tmp_path)
    workspace_id = "staging_sections"
    root = _write_section_workspace(settings, workspace_id)
    service = PaperBaseService(settings)

    sections = service.get_section_tree(workspace_id)
    parent = next(item for item in sections if item.title == "3. Method")
    leaf = next(item for item in sections if item.title == "3.2 Descriptors")
    assert parent.has_children is True
    assert leaf.has_children is False
    assert parent.has_content is True
    assert leaf.has_content is True
    assert leaf.level == 2
    assert leaf.parent_section_id == parent.section_id

    sources = service.get_section_explanation_sources(workspace_id, ["chunk_leaf"])
    assert [(item.section, item.page_start, item.page_end, item.text) for item in sources] == [
        ("3.2 Descriptors", 4, 5, "Descriptor evidence.")
    ]

    explanation = ExplainSection(
        section_id=parent.section_id,
        section_title=parent.title,
        mode="section_overview",
        explanation="本章说明整体方法。",
        key_points=["先提取局部极值", "再计算描述符"],
        source_chunk_ids=["chunk_parent"],
        insufficient_evidence=False,
    )
    artifact = root / "explain_sections" / f"{hashlib.sha256(parent.section_id.encode()).hexdigest()[:16]}.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps(explanation.model_dump(), ensure_ascii=False), encoding="utf-8")
    restored = service.get_section_explanation(workspace_id, parent.section_id)
    assert restored is not None
    assert restored.mode == "section_overview"
    assert restored.key_points == ("先提取局部极值", "再计算描述符")
    restarted = PaperBaseService(settings)
    assert restarted.get_section_explanation(workspace_id, parent.section_id) == restored


def test_generate_section_explanation_delegates_existing_backend(tmp_path: Path, monkeypatch) -> None:
    settings = _temporary_settings(tmp_path)
    _write_section_workspace(settings, "staging_generate_section")
    calls: list[tuple[str, str]] = []
    expected = ExplainSection(
        section_id="section_parent",
        section_title="3. Method",
        mode="section_overview",
        explanation="章节解释。",
        key_points=[],
        source_chunk_ids=["chunk_parent"],
        insufficient_evidence=False,
    )

    def fake_stage(*, settings, workspace_id, section_id):
        calls.append((workspace_id, section_id))
        return SimpleNamespace(explanation=expected)

    monkeypatch.setattr(service_module, "run_explain_section_stage", fake_stage)
    result = PaperBaseService(settings).generate_section_explanation(
        "staging_generate_section", "section_parent"
    )
    assert result.mode == "section_overview"
    assert calls == [("staging_generate_section", "section_parent")]


def test_scientific_renderer_keeps_math_readable() -> None:
    rendered = normalize_scientific_text(r"\(O(N^2)\) and O(l_p^2)")
    assert rendered == "$O(N^2)$ and $O(l_p^2)$"


def test_explanation_points_prefer_existing_lines() -> None:
    points = _split_explanation_points(
        "本节任务：提取局部极值。\n处理流程：构建描述符后进行 DTW 对齐。"
    )
    assert points == (
        "本节任务：提取局部极值。",
        "处理流程：构建描述符后进行 DTW 对齐。",
    )


def test_explanation_points_split_legacy_compacted_text_by_semantic_labels() -> None:
    points = _split_explanation_points(
        "本节任务：提取局部极值。核心方法：以极值构建形状描述符。"
        "设计作用与边界：降低复杂度，但依赖极值质量。"
    )
    assert points == (
        "本节任务：提取局部极值。",
        "核心方法：以极值构建形状描述符。",
        "设计作用与边界：降低复杂度，但依赖极值质量。",
    )


def test_overview_template_fields_keep_the_defined_display_order() -> None:
    points = _extract_template_points(
        "整体流程：先构建描述符。\n\n本章目标与范围：解释方法流程。"
        "\n\n章节关系与作用：连接前后章节。\n\n子章节职责：分别完成不同步骤。",
        (
            "本章目标与范围",
            "整体流程",
            "子章节职责",
            "章节关系与作用",
        ),
    )
    assert points == (
        ("本章目标与范围", "解释方法流程。"),
        ("整体流程", "先构建描述符。"),
        ("子章节职责", "分别完成不同步骤。"),
        ("章节关系与作用", "连接前后章节。"),
    )


def test_detail_template_fields_extract_each_explanation_point() -> None:
    points = _extract_template_points(
        "本节任务：识别局部极值。\n\n关键对象与输入输出：输入为时间序列。"
        "\n\n处理过程与判断条件：按符号变化判断。\n\n公式、符号或结果如何阅读："
        "$O(N^2)$ 表示复杂度。\n\n设计作用与边界：依赖极值质量。",
        (
            "本节任务",
            "关键对象与输入输出",
            "处理过程与判断条件",
            "公式、符号或结果如何阅读",
            "设计作用与边界",
        ),
    )
    assert [title for title, _ in points] == [
        "本节任务",
        "关键对象与输入输出",
        "处理过程与判断条件",
        "公式、符号或结果如何阅读",
        "设计作用与边界",
    ]


def _write_section_workspace(settings, workspace_id: str) -> Path:
    root = settings.storage.staging_dir / workspace_id
    (root / "parsed").mkdir(parents=True)
    (root / "chunks").mkdir()
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "workspace_id": workspace_id,
                "paper_id": "paper_sections",
                "source_filename": "sections.pdf",
                "total_chunk_count": 2,
                "added_to_kb": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "parsed" / "parsed_paper.json").write_text(
        json.dumps(
            {
                "paper_title": "Section Paper",
                "sections": [
                    {
                        "section_id": "section_parent",
                        "paper_id": "paper_sections",
                        "section_title": "3. Method",
                        "section_number": "3",
                        "section_level": 1,
                        "parent_section_id": None,
                        "section_index": 0,
                        "page_start": 3,
                        "page_end": 3,
                    },
                    {
                        "section_id": "section_leaf",
                        "paper_id": "paper_sections",
                        "section_title": "3.2 Descriptors",
                        "section_number": "3.2",
                        "section_level": 2,
                        "parent_section_id": "section_parent",
                        "section_index": 1,
                        "page_start": 4,
                        "page_end": 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    chunks = [
        {
            "chunk_id": "chunk_parent",
            "paper_id": "paper_sections",
            "chunk_index": 0,
            "raw_text": "Parent evidence.",
            "raw_token_count": 2,
            "section": "3. Method",
            "section_id": "section_parent",
            "section_type": "content",
            "page_start": 3,
            "page_end": 3,
        },
        {
            "chunk_id": "chunk_leaf",
            "paper_id": "paper_sections",
            "chunk_index": 1,
            "raw_text": "Descriptor evidence.",
            "raw_token_count": 2,
            "section": "3.2 Descriptors",
            "section_id": "section_leaf",
            "section_type": "content",
            "page_start": 4,
            "page_end": 5,
        },
    ]
    (root / "chunks" / "chunks.jsonl").write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    return root


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
