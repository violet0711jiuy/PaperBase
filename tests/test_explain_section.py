"""v0.2 Explain Section 的工作区隔离、层级访问、Context 与来源校验测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paperbase.config import ExplainSectionSettings
from paperbase.explain_section.service import (
    ExplainSectionError,
    build_explain_section_context,
    create_explain_section,
)
from paperbase.llm.client import LLMRequestError
from paperbase.staging.sections import WorkspaceSectionError, WorkspaceSectionRepository


class _FakeExplainClient:
    """记录唯一一次结构化请求并返回可控 JSON。"""

    def __init__(self, response: dict[str, object] | Exception) -> None:
        self.response = response
        self.call_count = 0
        self.user_prompt = ""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        schema_name: str,
    ) -> str:
        self.call_count += 1
        self.user_prompt = user_prompt
        if isinstance(self.response, Exception):
            raise self.response
        if schema_name != "paperbase_explain_section" or not json_schema:
            raise AssertionError("Explain service did not request its structured schema.")
        return json.dumps(self.response, ensure_ascii=False)


class ExplainSectionTests(unittest.TestCase):
    """所有测试都只读写临时目录，不加载 Parser、FAISS、Embedding 或正式 SQLite。"""

    def test_repository_exposes_exact_tree_children_descendants_and_chunks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository, workspace_id = _repository(Path(temporary_directory))
            tree = repository.get_section_tree(workspace_id)
            parent_id = tree[0].section_id
            child_id = tree[1].section_id

            self.assertEqual([item.section_title for item in repository.get_children(workspace_id, parent_id)], ["3.1 Extract local extrema", "3.2 Match extrema"])
            self.assertEqual(
                [item.section_title for item in repository.get_descendants(workspace_id, parent_id)],
                ["3.1 Extract local extrema", "3.2 Match extrema", "3.2.1 Local alignment"],
            )
            self.assertEqual(
                [item.chunk_id for item in repository.get_direct_chunks(workspace_id, child_id)],
                ["chunk_010"],
            )
            self.assertEqual(
                [item.chunk_id for item in repository.get_subtree_chunks(workspace_id, parent_id)],
                ["chunk_010", "chunk_020", "chunk_021"],
            )
            # section_id 为 null 的正文不应被程序强行归入任何节点。
            self.assertNotIn("chunk_099", [item.chunk_id for item in repository.get_subtree_chunks(workspace_id, parent_id)])

    def test_parent_context_covers_each_direct_child_without_parent_direct_chunks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository, workspace_id = _repository(Path(temporary_directory))
            snapshot = repository.load(workspace_id)
            parent_id = snapshot.sections[0].section_id
            context = build_explain_section_context(
                snapshot=snapshot, section_id=parent_id, settings=_settings()
            )

            self.assertEqual(context.mode, "section_overview")
            self.assertEqual(context.selection_debug["selected_section_direct_chunk_ids"], [])
            selected_ids = [chunk.chunk_id for chunk in context.chunks]
            self.assertIn("chunk_010", selected_ids)
            self.assertIn("chunk_020", selected_ids)
            self.assertIn("chunk_021", selected_ids)
            self.assertLessEqual(context.final_token_count, _settings().max_context_tokens)
            self.assertNotIn("chunk_099", selected_ids)

    def test_leaf_context_uses_only_direct_chunks_and_validates_program_owned_fields(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository, workspace_id = _repository(Path(temporary_directory))
            snapshot = repository.load(workspace_id)
            leaf = snapshot.sections[1]
            context = build_explain_section_context(
                snapshot=snapshot, section_id=leaf.section_id, settings=_settings()
            )
            client = _FakeExplainClient(
                {
                    "explanation": "本节从时间序列中提取局部极值，作为后续匹配的结构特征。",
                    "key_points": ["输出是局部极值序列。"],
                    "source_chunk_ids": ["chunk_010"],
                    "insufficient_evidence": False,
                }
            )
            outcome = create_explain_section(
                snapshot=snapshot,
                section_id=leaf.section_id,
                settings=_settings(),
                client=client,  # type: ignore[arg-type]
            )

            self.assertEqual(context.mode, "section_explanation")
            self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["chunk_010"])
            self.assertNotIn("chunk_020", client.user_prompt)
            self.assertEqual(client.call_count, 1)
            self.assertEqual(outcome.explanation.section_id, leaf.section_id)
            self.assertEqual(outcome.explanation.section_title, leaf.section_title)
            self.assertTrue(outcome.explanation_path.is_file())
            self.assertTrue(outcome.context_path.is_file())

    def test_context_hard_budget_and_fabricated_source_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository, workspace_id = _repository(Path(temporary_directory), oversized=True)
            snapshot = repository.load(workspace_id)
            parent_id = snapshot.sections[0].section_id
            settings = ExplainSectionSettings(
                max_tokens_per_chunk=300,
                target_context_tokens=700,
                max_context_tokens=1_000,
                max_representative_chunks_per_branch=3,
                max_key_points=5,
                max_output_tokens=800,
            )
            context = build_explain_section_context(
                snapshot=snapshot, section_id=parent_id, settings=settings
            )
            self.assertLessEqual(context.final_token_count, settings.max_context_tokens)
            client = _FakeExplainClient(
                {
                    "explanation": "伪造来源。",
                    "key_points": [],
                    "source_chunk_ids": ["not_in_context"],
                    "insufficient_evidence": False,
                }
            )
            with self.assertRaises(ExplainSectionError):
                create_explain_section(
                    snapshot=snapshot,
                    section_id=parent_id,
                    settings=settings,
                    client=client,  # type: ignore[arg-type]
                )

    def test_invalid_section_and_llm_failure_are_explicit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository, workspace_id = _repository(Path(temporary_directory))
            snapshot = repository.load(workspace_id)
            with self.assertRaises(WorkspaceSectionError):
                build_explain_section_context(
                    snapshot=snapshot, section_id="missing", settings=_settings()
                )
            with self.assertRaises(ExplainSectionError):
                create_explain_section(
                    snapshot=snapshot,
                    section_id=snapshot.sections[1].section_id,
                    settings=_settings(),
                    client=_FakeExplainClient(LLMRequestError("network unavailable")),  # type: ignore[arg-type]
                )


def _settings() -> ExplainSectionSettings:
    return ExplainSectionSettings(
        max_tokens_per_chunk=300,
        target_context_tokens=1_200,
        max_context_tokens=2_000,
        max_representative_chunks_per_branch=3,
        max_key_points=5,
        max_output_tokens=800,
    )


def _repository(root: Path, *, oversized: bool = False) -> tuple[WorkspaceSectionRepository, str]:
    """写入一棵 3 → 3.1 / 3.2 → 3.2.1 的真实持久化形态工作区。"""
    workspace_id = "staging_example"
    workspace = root / "staging" / workspace_id
    (workspace / "parsed").mkdir(parents=True)
    (workspace / "chunks").mkdir()
    paper_id = "paper_example"
    sections = (
        _section("section_030", paper_id, "3. Extrema-based shape dynamic time warping", None, 1, 0),
        _section("section_031", paper_id, "3.1 Extract local extrema", "section_030", 2, 1),
        _section("section_032", paper_id, "3.2 Match extrema", "section_030", 2, 2),
        _section("section_0321", paper_id, "3.2.1 Local alignment", "section_032", 3, 3),
    )
    (workspace / "workspace.json").write_text(
        json.dumps({"workspace_id": workspace_id, "paper_id": paper_id}), encoding="utf-8"
    )
    (workspace / "parsed" / "parsed_paper.json").write_text(
        json.dumps({"sections": sections}), encoding="utf-8"
    )
    token_count = 900 if oversized else 40
    chunks = (
        _chunk("chunk_010", paper_id, 10, "section_031", "3.1 Extract local extrema", token_count),
        _chunk("chunk_020", paper_id, 20, "section_032", "3.2 Match extrema", token_count),
        _chunk("chunk_021", paper_id, 21, "section_0321", "3.2 Match extrema > 3.2.1 Local alignment", token_count),
        _chunk("chunk_099", paper_id, 99, None, None, token_count),
    )
    (workspace / "chunks" / "chunks.jsonl").write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8"
    )
    return WorkspaceSectionRepository(root / "staging"), workspace_id


def _section(
    section_id: str, paper_id: str, title: str, parent_id: str | None, level: int, index: int
) -> dict[str, object]:
    return {
        "section_id": section_id,
        "paper_id": paper_id,
        "section_title": title,
        "section_number": title.split(maxsplit=1)[0].rstrip("."),
        "section_level": level,
        "parent_section_id": parent_id,
        "section_index": index,
        "page_start": 3,
        "page_end": 3,
    }


def _chunk(
    chunk_id: str,
    paper_id: str,
    chunk_index: int,
    section_id: str | None,
    section: str | None,
    token_count: int,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "chunk_index": chunk_index,
        "raw_text": f"{section or 'unsectioned'} contains equation x = y and Result 91.2%.",
        "raw_token_count": token_count,
        "section_id": section_id,
        "section": section,
        "section_type": "content",
    }


if __name__ == "__main__":
    unittest.main()
