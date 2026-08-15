"""v0.2 Paper Overview 的章节选择、来源校验与隔离回归测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paperbase.config import PaperOverviewSettings
from paperbase.llm.client import LLMRuntimeSettings
from paperbase.overview.service import (
    PaperOverviewError,
    build_overview_context,
    create_paper_overview,
    _with_overview_output_budget,
)


class _FakeOverviewClient:
    """返回固定 JSON，同时保留 Prompt 以检查 References 没有进入 Overview Context。"""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        schema_name: str,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        if schema_name != "paperbase_paper_overview" or not json_schema:
            raise AssertionError("Overview service did not request the expected structured output.")
        return json.dumps(self.response, ensure_ascii=False)


class PaperOverviewTests(unittest.TestCase):
    """不加载 Parser、Embedder 或 FAISS，直接验证已保存工作区上的纯读取流程。"""

    def test_overview_uses_core_sections_and_preserves_sources(self) -> None:
        """核心章节进入 Prompt、References 排除，且所有非缺失字段可回溯。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = _write_workspace(root / "staging_example")
            formal_files = _write_formal_kb_sentinels(root)
            before = {path: _sha256(path) for path in formal_files}
            response = _valid_response()
            client = _FakeOverviewClient(response)

            context = build_overview_context(
                workspace_root=workspace,
                settings=PaperOverviewSettings(
                    max_chunks_per_section=2,
                    max_chars_per_chunk=1_000,
                    max_total_context_chars=8_000,
                    max_fallback_chunks=1,
                ),
            )
            outcome = create_paper_overview(
                workspace_root=workspace,
                settings=PaperOverviewSettings(
                    max_chunks_per_section=2,
                    max_chars_per_chunk=1_000,
                    max_total_context_chars=8_000,
                    max_fallback_chunks=1,
                ),
                client=client,  # type: ignore[arg-type]
            )

            self.assertEqual(
                [chunk.category for chunk in context.chunks],
                ["abstract", "introduction", "method", "experiments", "experiments", "conclusion"],
            )
            self.assertNotIn("reference-only-text", client.user_prompt)
            self.assertNotIn("[chunk_id: chunk_006]", client.user_prompt)
            self.assertEqual(outcome.overview.paper_title, "Example paper")
            self.assertEqual(outcome.overview.datasets.content, "在 BenchSet 数据集上评估。")
            self.assertEqual(outcome.overview.datasets.source_sections, ["4 Experiments"])
            self.assertEqual(outcome.overview.main_results.content, "Accuracy 达到 91.2%。")
            self.assertEqual(outcome.overview.limitations.content, "论文未明确说明")
            self.assertEqual(outcome.overview.limitations.source_chunk_ids, [])
            self.assertTrue(outcome.overview_path.is_file())
            self.assertTrue(outcome.context_path.is_file())
            self.assertEqual(before, {path: _sha256(path) for path in formal_files})

    def test_overview_rejects_fabricated_source_chunk(self) -> None:
        """模型不能给字段引用未进入 Prompt 的 chunk，也不能借此伪造来源。"""
        with TemporaryDirectory() as temporary_directory:
            workspace = _write_workspace(Path(temporary_directory) / "staging_example")
            response = _valid_response()
            response["main_method"] = {
                "content": "伪造来源。",
                "source_chunk_ids": ["not_in_context"],
            }
            with self.assertRaises(PaperOverviewError):
                create_paper_overview(
                    workspace_root=workspace,
                    settings=PaperOverviewSettings(),
                    client=_FakeOverviewClient(response),  # type: ignore[arg-type]
                )

    def test_overview_output_budget_does_not_change_shared_runtime_settings(self) -> None:
        """Overview 的长 JSON 预算只作用于其专用客户端，不能改变其他 v0.1 调用。"""
        shared = LLMRuntimeSettings(
            api_key="test", base_url="https://example.test", model="test-model",
            timeout_seconds=30.0, temperature=0.0, max_tokens=500,
        )
        overview = _with_overview_output_budget(shared, 1_600)
        self.assertEqual(shared.max_tokens, 500)
        self.assertEqual(overview.max_tokens, 1_600)
        self.assertEqual(overview.model, shared.model)


def _write_workspace(root: Path) -> Path:
    """写出与 Temporary Workspace 相同的最小 parsed/chunks 文件，不创建向量或索引。"""
    (root / "parsed").mkdir(parents=True)
    (root / "chunks").mkdir()
    (root / "parsed" / "parsed_paper.json").write_text(
        json.dumps({"paper_title": "Example paper"}, ensure_ascii=False), encoding="utf-8"
    )
    records = (
        _chunk("chunk_000", "Front matter > Abstract", "This work studies forecasting errors.", front_matter_type="abstract"),
        _chunk("chunk_001", "1 Introduction", "Existing systems lack reliable uncertainty estimates."),
        # 不含 Method/Approach 关键词，模拟论文以算法名或自定义名称命名的第 3 章。
        _chunk("chunk_002", "3 Core Design", "Our MethodX combines two encoders."),
        _chunk("chunk_003", "4 Experiments", "We evaluate MethodX on BenchSet using Accuracy."),
        _chunk("chunk_004", "4.2 Results", "MethodX reaches Accuracy 91.2%, better than BaselineY."),
        _chunk("chunk_005", "5 Conclusion", "The method improves forecasting reliability."),
        _chunk("chunk_006", "References", "reference-only-text", section_type="bibliography"),
        # 无章节名的正文可能由 PDF 版面解析产生；它不能阻断编号方法章节的结构回退。
        _chunk("chunk_007", "", "An unsectioned layout fragment."),
    )
    (root / "chunks" / "chunks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return root


def _chunk(
    chunk_id: str,
    section: str,
    raw_text: str,
    *,
    section_type: str = "content",
    front_matter_type: str | None = None,
) -> dict[str, object]:
    """构造 v0.1 PaperChunk JSONL 中 Overview 需要读取的稳定字段。"""
    return {
        "chunk_id": chunk_id,
        "section": section,
        "raw_text": raw_text,
        "section_type": section_type,
        "front_matter_type": front_matter_type,
    }


def _valid_response() -> dict[str, object]:
    """模拟受控模型输出：所有字段均引用当前论文可见的相关 chunk。"""
    return {
        "paper_title": "Example paper",
        "research_problem": {
            "content": "解决预测系统缺乏可靠不确定性估计的问题。",
            "source_chunk_ids": ["chunk_000", "chunk_001"],
        },
        "main_method": {
            "content": "提出 MethodX，并结合两个 encoders。",
            "source_chunk_ids": ["chunk_002"],
        },
        "contributions": {
            "content": "提出面向可靠预测的方法。",
            "source_chunk_ids": ["chunk_002", "chunk_005"],
        },
        "datasets": {
            "content": "在 BenchSet 数据集上评估。",
            "source_chunk_ids": ["chunk_003"],
        },
        "experimental_setup": {
            "content": "使用 Accuracy 评估，并与 BaselineY 比较。",
            "source_chunk_ids": ["chunk_003", "chunk_004"],
        },
        "main_results": {
            "content": "Accuracy 达到 91.2%。",
            "source_chunk_ids": ["chunk_004"],
        },
        "limitations": {
            "content": "论文未明确说明",
            "source_chunk_ids": [],
        },
    }


def _write_formal_kb_sentinels(root: Path) -> tuple[Path, ...]:
    """建立正式 KB 文件哨兵，证明 Overview 服务不会触碰它们。"""
    paths = (
        root / "storage" / "paperbase.sqlite3",
        root / "storage" / "paperbase.faiss",
        root / "storage" / "paperbase.faiss.manifest.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"formal:{path.name}".encode("utf-8"))
    return paths


def _sha256(path: Path) -> str:
    """为正式 KB 文件创建不可变字节快照。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
