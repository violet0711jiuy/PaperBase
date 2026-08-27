"""Golden Dataset Validator 的确定性回归测试。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from eval.scripts.validate_golden_dataset import run_validation


def test_safe_location_sync_keeps_semantics_and_returns_pass(tmp_path: Path) -> None:
    """现存 chunk 与 paper 一致时，只同步 section/page，不改变语义和 chunk_ids。"""
    database = tmp_path / "paperbase.sqlite3"
    _write_database(
        database,
        [
            (
                "chunk_1",
                "paper_1",
                "References",
                "bibliography",
                9,
                10,
            )
        ],
    )
    record = _record(
        primary_type="bibliography",
        tags=[
            "zh_query_en_doc",
            "exact_term",
            "single_hop",
            "bibliography_intent",
            "easy",
        ],
        intent=True,
        evidence={
            "paper_id": "paper_1",
            "section": "7. Conclusion",
            "page_start": 8,
            "page_end": 8,
            "chunk_ids": ["chunk_1"],
        },
    )
    source, validated, report = _write_inputs_and_run(tmp_path, database, [record])

    assert report["status"] == "PASS"
    assert report["outputs"]["validated_dataset_written"] is True
    written = json.loads(validated.read_text(encoding="utf-8").strip())
    assert written["question"] == record["question"]
    assert written["required_facts"] == record["required_facts"]
    assert written["relevant_evidence"][0] == {
        "paper_id": "paper_1",
        "section": "References",
        "page_start": 9,
        "page_end": 10,
        "chunk_ids": ["chunk_1"],
    }
    # 原始文件必须保持完全不变，保证同步始终是非覆盖式操作。
    assert json.loads(source.read_text(encoding="utf-8").strip()) == record


def test_missing_chunk_requires_manual_remap_and_writes_no_validated_copy(
    tmp_path: Path,
) -> None:
    """chunk_id 不存在时禁止猜测替代 chunk，并明确要求人工 remap。"""
    database = tmp_path / "paperbase.sqlite3"
    _write_database(database, [])
    record = _record(
        evidence={
            "paper_id": "paper_1",
            "section": "Method",
            "page_start": 3,
            "page_end": 3,
            "chunk_ids": ["missing_chunk"],
        }
    )
    _, validated, report = _write_inputs_and_run(tmp_path, database, [record])

    assert report["status"] == "NEEDS_MANUAL_REVIEW"
    assert report["evidence_integrity"]["missing_chunk_ids"][0]["chunk_id"] == "missing_chunk"
    assert report["evidence_integrity"]["needs_manual_remap"][0]["status"] == "NEEDS_MANUAL_REMAP"
    assert not validated.exists()


def test_bibliography_chunk_must_be_classified_as_bibliography(tmp_path: Path) -> None:
    """bibliography Case 指向正文 chunk 时必须失败，且不能靠改 section 掩盖分类错误。"""
    database = tmp_path / "paperbase.sqlite3"
    _write_database(
        database,
        [("chunk_1", "paper_1", "References", "content", 9, 9)],
    )
    record = _record(
        primary_type="bibliography",
        tags=[
            "zh_query_en_doc",
            "exact_term",
            "single_hop",
            "bibliography_intent",
            "easy",
        ],
        intent=True,
        evidence={
            "paper_id": "paper_1",
            "section": "References",
            "page_start": 9,
            "page_end": 9,
            "chunk_ids": ["chunk_1"],
        },
    )
    _, _, report = _write_inputs_and_run(tmp_path, database, [record])

    assert report["status"] == "NEEDS_MANUAL_REVIEW"
    assert len(report["evidence_integrity"]["bibliography_classification_errors"]) == 1


def test_single_hop_with_multiple_chunks_is_only_a_review_checklist(tmp_path: Path) -> None:
    """single_hop 多 chunk 只进入清单，不应自动删 Evidence 或导致校验失败。"""
    database = tmp_path / "paperbase.sqlite3"
    _write_database(
        database,
        [
            ("chunk_1", "paper_1", "Method", "content", 3, 3),
            ("chunk_2", "paper_1", "Method", "content", 3, 4),
        ],
    )
    record = _record(
        evidence={
            "paper_id": "paper_1",
            "section": "Method",
            "page_start": 3,
            "page_end": 4,
            "chunk_ids": ["chunk_1", "chunk_2"],
        }
    )
    _, _, report = _write_inputs_and_run(tmp_path, database, [record])

    assert report["status"] == "PASS"
    assert report["single_hop_multi_chunk_cases"][0]["chunk_ids"] == ["chunk_1", "chunk_2"]


def _record(
    *,
    primary_type: str = "fact",
    tags: list[str] | None = None,
    intent: bool = False,
    evidence: dict[str, object],
) -> dict[str, object]:
    """构造满足设计文档基础 Schema 的最小可回答测试 Case。"""
    return {
        "id": "case_1",
        "question": "这项方法的关键事实是什么？",
        "primary_type": primary_type,
        "tags": tags
        or ["zh_query_en_doc", "semantic_paraphrase", "single_hop", "easy"],
        "paper_id": "paper_1",
        "answerable": True,
        "reference_answer": "关键事实。",
        "required_facts": ["关键事实。"],
        "relevant_evidence": [evidence],
        "expected_bibliography_intent": intent,
    }


def _write_database(
    path: Path,
    chunks: list[tuple[str, str, str, str, int | None, int | None]],
) -> None:
    """创建只包含 Validator 所需正式字段的临时 SQLite。"""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (paper_id TEXT PRIMARY KEY);
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                section TEXT,
                section_type TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER
            );
            INSERT INTO documents (paper_id) VALUES ('paper_1');
            """
        )
        connection.executemany(
            "INSERT INTO chunks "
            "(chunk_id, paper_id, section, section_type, page_start, page_end) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            chunks,
        )


def _write_inputs_and_run(
    root: Path,
    database: Path,
    records: list[dict[str, object]],
) -> tuple[Path, Path, dict[str, object]]:
    """写入临时 JSONL 并以显式路径执行一次完整 Validator。"""
    source = root / "golden.jsonl"
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    validated = root / "golden_validated.jsonl"
    args = argparse.Namespace(
        dataset=source,
        database=database,
        json_report=root / "report.json",
        markdown_report=root / "report.md",
        validated_output=validated,
    )
    return source, validated, run_validation(args)
