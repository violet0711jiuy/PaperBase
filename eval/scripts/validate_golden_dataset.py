"""校验 Golden Dataset 与当前正式 SQLite 的一致性，并安全同步定位元数据。

本脚本只以只读方式访问正式知识库，不调用也不修改 PaperBase 的 Retrieval、
Generation 或正式 RAG 链路。原始 Golden Dataset 永远不会被覆盖；只有能够由
现存 chunk_id 和一致的 paper_id 唯一确定时，才会生成带新定位元数据的副本。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "eval" / "datasets" / "golden_dataset_v1_2.jsonl"
DEFAULT_DATABASE = ROOT / "storage" / "paperbase.sqlite3"
DEFAULT_JSON_REPORT = (
    ROOT / "eval" / "results" / "validation" / "golden_validation_report.json"
)
DEFAULT_MARKDOWN_REPORT = (
    ROOT / "eval" / "results" / "validation" / "golden_validation_report.md"
)
DEFAULT_VALIDATED_DATASET = (
    ROOT / "eval" / "datasets" / "golden_dataset_v1_validated.jsonl"
)

REQUIRED_FIELDS = (
    "id",
    "question",
    "primary_type",
    "tags",
    "paper_id",
    "answerable",
    "reference_answer",
    "required_facts",
    "relevant_evidence",
    "expected_bibliography_intent",
)
EVIDENCE_FIELDS = ("paper_id", "section", "page_start", "page_end", "chunk_ids")
PRIMARY_TYPES = (
    "fact",
    "method",
    "experiment",
    "result",
    "synthesis",
    "bibliography",
    "unanswerable",
)
LEGAL_TAGS = (
    "zh_query_en_doc",
    "english_query",
    "exact_term",
    "semantic_paraphrase",
    "single_hop",
    "multi_hop",
    "bibliography_intent",
    "easy",
    "medium",
    "hard",
)
SEMANTIC_FIELDS = (
    "id",
    "question",
    "primary_type",
    "tags",
    "paper_id",
    "answerable",
    "reference_answer",
    "required_facts",
    "expected_bibliography_intent",
)


class ValidationInputError(RuntimeError):
    """表示输入路径、SQLite schema 等前置条件不满足，无法可靠执行校验。"""


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐行读取 JSONL，并保留所有解析错误以便生成完整报告。"""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    {
                        "line": line_number,
                        "message": str(error),
                        "content_preview": line.rstrip()[:160],
                    }
                )
                continue
            if not isinstance(value, dict):
                errors.append(
                    {
                        "line": line_number,
                        "message": "JSONL row must be an object",
                        "content_preview": line.rstrip()[:160],
                    }
                )
                continue
            value["__line_number__"] = line_number
            records.append(value)
    return records, errors


def open_database_read_only(path: Path) -> sqlite3.Connection:
    """以 SQLite read-only URI 打开正式数据库，避免校验器产生任何数据库写入。"""
    if not path.is_file():
        raise ValidationInputError(f"SQLite database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def validate_sqlite_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    """确认正式 chunks/documents 表包含 Evidence 校验所需的真实字段。"""
    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    required_tables = {"chunks", "documents"}
    missing_tables = sorted(required_tables - table_names)
    chunk_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
    } if "chunks" in table_names else set()
    required_chunk_columns = {
        "chunk_id",
        "paper_id",
        "section",
        "section_type",
        "page_start",
        "page_end",
    }
    missing_columns = sorted(required_chunk_columns - chunk_columns)
    return {
        "valid": not missing_tables and not missing_columns,
        "missing_tables": missing_tables,
        "missing_chunk_columns": missing_columns,
        "available_chunk_columns": sorted(chunk_columns),
    }


def _case_label(record: Mapping[str, Any], index: int) -> str:
    """为缺少合法 id 的行生成稳定、可定位的报告标签。"""
    value = record.get("id")
    return str(value) if value not in (None, "") else f"line_{record.get('__line_number__', index)}"


def _is_nonempty_string(value: Any) -> bool:
    """判断一个值是否为去除空白后仍有内容的字符串。"""
    return isinstance(value, str) and bool(value.strip())


def validate_dataset(
    records: Sequence[Mapping[str, Any]], parse_errors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """校验 Case schema、合法枚举、不可回答约束与 bibliography intent 逻辑。"""
    missing_fields: dict[str, list[str]] = {}
    invalid_primary_types: dict[str, Any] = {}
    illegal_tags: dict[str, list[Any]] = {}
    schema_errors: list[dict[str, Any]] = []
    logic_errors: list[dict[str, Any]] = []
    ids: list[Any] = []

    for index, record in enumerate(records, 1):
        case_id = _case_label(record, index)
        ids.append(record.get("id"))
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            missing_fields[case_id] = missing

        if "id" in record and not _is_nonempty_string(record.get("id")):
            schema_errors.append({"case_id": case_id, "message": "id must be a non-empty string"})
        if "question" in record and not _is_nonempty_string(record.get("question")):
            schema_errors.append({"case_id": case_id, "message": "question must be a non-empty string"})
        if "paper_id" in record and not _is_nonempty_string(record.get("paper_id")):
            schema_errors.append({"case_id": case_id, "message": "paper_id must be a non-empty string"})

        primary_type = record.get("primary_type")
        if primary_type not in PRIMARY_TYPES:
            invalid_primary_types[case_id] = primary_type

        tags = record.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            schema_errors.append({"case_id": case_id, "message": "tags must be a list of strings"})
            tag_values: list[Any] = tags if isinstance(tags, list) else []
        else:
            tag_values = tags
        invalid_tags = sorted(
            (tag for tag in tag_values if tag not in LEGAL_TAGS), key=str
        )
        if invalid_tags:
            illegal_tags[case_id] = invalid_tags
        if len(tag_values) != len(set(map(str, tag_values))):
            schema_errors.append({"case_id": case_id, "message": "tags contain duplicates"})

        answerable = record.get("answerable")
        intent = record.get("expected_bibliography_intent")
        if "answerable" in record and not isinstance(answerable, bool):
            schema_errors.append({"case_id": case_id, "message": "answerable must be boolean"})
        if "expected_bibliography_intent" in record and not isinstance(intent, bool):
            schema_errors.append(
                {"case_id": case_id, "message": "expected_bibliography_intent must be boolean"}
            )
        if "required_facts" in record and not isinstance(record.get("required_facts"), list):
            schema_errors.append({"case_id": case_id, "message": "required_facts must be a list"})
        if "relevant_evidence" in record and not isinstance(record.get("relevant_evidence"), list):
            schema_errors.append({"case_id": case_id, "message": "relevant_evidence must be a list"})

        if answerable is False and (
            record.get("reference_answer") is not None
            or record.get("required_facts") != []
            or record.get("relevant_evidence") != []
        ):
            logic_errors.append(
                {
                    "case_id": case_id,
                    "rule": "unanswerable_contract",
                    "message": "answerable=false requires null answer and empty facts/evidence",
                }
            )
        if answerable is True and (
            not _is_nonempty_string(record.get("reference_answer"))
            or not isinstance(record.get("required_facts"), list)
            or not record.get("required_facts")
            or not isinstance(record.get("relevant_evidence"), list)
            or not record.get("relevant_evidence")
        ):
            logic_errors.append(
                {
                    "case_id": case_id,
                    "rule": "answerable_contract",
                    "message": "answerable=true requires answer, required facts, and evidence",
                }
            )

        is_bibliography = primary_type == "bibliography"
        if is_bibliography and intent is not True:
            logic_errors.append(
                {
                    "case_id": case_id,
                    "rule": "bibliography_intent",
                    "message": "bibliography case requires expected_bibliography_intent=true",
                }
            )
        if not is_bibliography and intent is True:
            logic_errors.append(
                {
                    "case_id": case_id,
                    "rule": "bibliography_intent",
                    "message": "non-bibliography case must not enable bibliography intent",
                }
            )
        if ("bibliography_intent" in tag_values) != is_bibliography:
            logic_errors.append(
                {
                    "case_id": case_id,
                    "rule": "bibliography_intent_tag",
                    "message": "bibliography_intent tag must agree with primary_type",
                }
            )

        evidence_list = record.get("relevant_evidence", [])
        if isinstance(evidence_list, list):
            for evidence_index, evidence in enumerate(evidence_list):
                if not isinstance(evidence, dict):
                    schema_errors.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "message": "evidence must be an object",
                        }
                    )
                    continue
                missing_evidence = [field for field in EVIDENCE_FIELDS if field not in evidence]
                if missing_evidence:
                    schema_errors.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "message": f"evidence missing fields: {missing_evidence}",
                        }
                    )
                chunk_ids = evidence.get("chunk_ids")
                if not isinstance(chunk_ids, list) or not chunk_ids or any(
                    not _is_nonempty_string(chunk_id) for chunk_id in chunk_ids
                ):
                    schema_errors.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "message": "chunk_ids must be a non-empty list of strings",
                        }
                    )
                elif len(chunk_ids) != len(set(chunk_ids)):
                    schema_errors.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "message": "chunk_ids contain duplicates",
                        }
                    )

    duplicate_ids = sorted(
        str(case_id)
        for case_id, count in Counter(ids).items()
        if case_id is not None and count > 1
    )
    type_counts = Counter(str(record.get("primary_type")) for record in records)
    paper_counts = Counter(str(record.get("paper_id")) for record in records)
    tag_counts = Counter(
        str(tag)
        for record in records
        if isinstance(record.get("tags"), list)
        for tag in record["tags"]
    )
    answerable_counts = Counter(str(record.get("answerable")).lower() for record in records)
    intent_counts = Counter(
        str(record.get("expected_bibliography_intent")).lower() for record in records
    )
    valid = not (
        parse_errors
        or duplicate_ids
        or missing_fields
        or invalid_primary_types
        or illegal_tags
        or schema_errors
        or logic_errors
    )
    return {
        "valid": valid,
        "jsonl_parseable": not parse_errors,
        "parse_errors": list(parse_errors),
        "total_cases": len(records),
        "duplicate_ids": duplicate_ids,
        "missing_fields": missing_fields,
        "invalid_primary_types": invalid_primary_types,
        "illegal_tags": illegal_tags,
        "schema_errors": schema_errors,
        "logic_errors": logic_errors,
        "counts": {
            "primary_type": {kind: type_counts.get(kind, 0) for kind in PRIMARY_TYPES},
            "paper_id": dict(sorted(paper_counts.items())),
            "tags": {tag: tag_counts.get(tag, 0) for tag in LEGAL_TAGS},
            "answerable": {
                "true": answerable_counts.get("true", 0),
                "false": answerable_counts.get("false", 0),
            },
            "expected_bibliography_intent": {
                "true": intent_counts.get("true", 0),
                "false": intent_counts.get("false", 0),
            },
        },
    }


def _all_evidence_chunk_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """按 Golden 中的出现顺序收集所有结构合法的 Evidence chunk 引用。"""
    result: list[str] = []
    for record in records:
        evidence_list = record.get("relevant_evidence", [])
        if not isinstance(evidence_list, list):
            continue
        for evidence in evidence_list:
            if not isinstance(evidence, dict) or not isinstance(evidence.get("chunk_ids"), list):
                continue
            result.extend(
                str(chunk_id)
                for chunk_id in evidence["chunk_ids"]
                if _is_nonempty_string(chunk_id)
            )
    return result


def load_chunks(
    connection: sqlite3.Connection, chunk_ids: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """一次性从 SQLite 读取所有被引用 chunk 的规范定位元数据。"""
    unique_ids = sorted(set(chunk_ids))
    if not unique_ids:
        return {}
    rows: list[sqlite3.Row] = []
    # SQLite 默认参数上限常见为 999，分批查询可以兼容更大的 Golden Dataset。
    for offset in range(0, len(unique_ids), 900):
        batch = unique_ids[offset : offset + 900]
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            connection.execute(
                "SELECT chunk_id, paper_id, section, section_type, page_start, page_end "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                batch,
            ).fetchall()
        )
    return {
        str(row["chunk_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "paper_id": str(row["paper_id"]),
            "section": row["section"],
            "section_type": str(row["section_type"]),
            "page_start": row["page_start"],
            "page_end": row["page_end"],
        }
        for row in rows
    }


def load_paper_ids(connection: sqlite3.Connection) -> set[str]:
    """读取正式知识库中的论文标识，用于发现指向已不存在论文的 Case。"""
    return {
        str(row[0]) for row in connection.execute("SELECT paper_id FROM documents").fetchall()
    }


def _canonical_location(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """将一个 Evidence 组归并成唯一 section 和页范围；跨 section 时返回 None。"""
    sections = {row.get("section") for row in rows}
    if len(sections) != 1:
        return None
    starts = [int(row["page_start"]) for row in rows if row.get("page_start") is not None]
    ends = [int(row["page_end"]) for row in rows if row.get("page_end") is not None]
    return {
        "section": next(iter(sections)),
        "page_start": min(starts) if starts else None,
        "page_end": max(ends) if ends else None,
    }


def validate_evidence(
    records: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]],
    database_paper_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """逐组核对 Evidence，并只对无歧义的定位字段构造安全同步副本。"""
    validated_records = copy.deepcopy([dict(record) for record in records])
    for record in validated_records:
        record.pop("__line_number__", None)

    missing_chunks: list[dict[str, Any]] = []
    paper_mismatches: list[dict[str, Any]] = []
    section_mismatches: list[dict[str, Any]] = []
    page_mismatches: list[dict[str, Any]] = []
    section_type_mismatches: list[dict[str, Any]] = []
    bibliography_errors: list[dict[str, Any]] = []
    manual_remaps: list[dict[str, Any]] = []
    safe_updates: list[dict[str, Any]] = []
    missing_paper_ids = sorted(
        {
            str(record.get("paper_id"))
            for record in records
            if _is_nonempty_string(record.get("paper_id"))
            and str(record.get("paper_id")) not in database_paper_ids
        }
    )

    referenced_ids = _all_evidence_chunk_ids(records)
    evidence_bearing_cases = 0
    for record_index, record in enumerate(records):
        case_id = _case_label(record, record_index + 1)
        evidence_list = record.get("relevant_evidence", [])
        if not isinstance(evidence_list, list):
            continue
        if evidence_list:
            evidence_bearing_cases += 1
        for evidence_index, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                continue
            raw_chunk_ids = evidence.get("chunk_ids", [])
            if not isinstance(raw_chunk_ids, list):
                continue
            chunk_ids = [str(value) for value in raw_chunk_ids if _is_nonempty_string(value)]
            missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in chunks]
            for chunk_id in missing:
                missing_chunks.append(
                    {"case_id": case_id, "evidence_index": evidence_index, "chunk_id": chunk_id}
                )

            expected_paper_id = record.get("paper_id")
            evidence_paper_id = evidence.get("paper_id")
            group_paper_errors: list[dict[str, Any]] = []
            if evidence_paper_id != expected_paper_id:
                group_paper_errors.append(
                    {
                        "case_id": case_id,
                        "evidence_index": evidence_index,
                        "chunk_id": None,
                        "case_paper_id": expected_paper_id,
                        "evidence_paper_id": evidence_paper_id,
                        "sqlite_paper_id": None,
                    }
                )
            rows = [chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks]
            for row in rows:
                if row["paper_id"] != expected_paper_id or row["paper_id"] != evidence_paper_id:
                    group_paper_errors.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "chunk_id": row["chunk_id"],
                            "case_paper_id": expected_paper_id,
                            "evidence_paper_id": evidence_paper_id,
                            "sqlite_paper_id": row["paper_id"],
                        }
                    )
            paper_mismatches.extend(group_paper_errors)

            canonical = _canonical_location(rows) if rows and len(rows) == len(chunk_ids) else None
            if canonical is not None:
                if evidence.get("section") != canonical["section"]:
                    section_mismatches.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "chunk_ids": chunk_ids,
                            "golden": evidence.get("section"),
                            "sqlite": canonical["section"],
                        }
                    )
                if (
                    evidence.get("page_start") != canonical["page_start"]
                    or evidence.get("page_end") != canonical["page_end"]
                ):
                    page_mismatches.append(
                        {
                            "case_id": case_id,
                            "evidence_index": evidence_index,
                            "chunk_ids": chunk_ids,
                            "golden": {
                                "page_start": evidence.get("page_start"),
                                "page_end": evidence.get("page_end"),
                            },
                            "sqlite": {
                                "page_start": canonical["page_start"],
                                "page_end": canonical["page_end"],
                            },
                        }
                    )

            if missing or group_paper_errors or canonical is None:
                if missing:
                    reason = "chunk_id does not exist in current SQLite"
                elif group_paper_errors:
                    reason = "paper_id conflict prevents unambiguous mapping"
                else:
                    reason = "current chunks in one Evidence group span multiple sections"
                manual_remaps.append(
                    {
                        "status": "NEEDS_MANUAL_REMAP",
                        "case_id": case_id,
                        "evidence_index": evidence_index,
                        "reason": reason,
                        "chunk_ids": chunk_ids,
                        "golden_location": {
                            "paper_id": evidence.get("paper_id"),
                            "section": evidence.get("section"),
                            "page_start": evidence.get("page_start"),
                            "page_end": evidence.get("page_end"),
                        },
                        "sqlite_chunks": rows,
                    }
                )
            else:
                # 只改 Evidence 里的纯定位字段；语义字段和 Ground Truth chunk 集合保持不动。
                target_evidence = validated_records[record_index]["relevant_evidence"][evidence_index]
                for field in ("section", "page_start", "page_end"):
                    old_value = evidence.get(field)
                    new_value = canonical[field]
                    if old_value != new_value:
                        target_evidence[field] = new_value
                        safe_updates.append(
                            {
                                "case_id": case_id,
                                "evidence_index": evidence_index,
                                "field": field,
                                "old": old_value,
                                "new": new_value,
                                "chunk_ids": chunk_ids,
                            }
                        )

            expected_section_type = (
                "bibliography" if record.get("primary_type") == "bibliography" else "content"
            )
            for row in rows:
                if row["section_type"] != expected_section_type:
                    mismatch = {
                        "case_id": case_id,
                        "evidence_index": evidence_index,
                        "chunk_id": row["chunk_id"],
                        "primary_type": record.get("primary_type"),
                        "actual_section_type": row["section_type"],
                        "expected_section_type": expected_section_type,
                    }
                    section_type_mismatches.append(mismatch)
                    if record.get("primary_type") == "bibliography":
                        bibliography_errors.append(mismatch)

    # 防御性断言：validated 副本绝不能改变任何 Case 语义字段或 chunk_id 集合。
    for original, validated in zip(records, validated_records):
        for field in SEMANTIC_FIELDS:
            if original.get(field) != validated.get(field):
                raise AssertionError(f"semantic field changed unexpectedly: {field}")
        original_chunks = [
            evidence.get("chunk_ids")
            for evidence in original.get("relevant_evidence", [])
            if isinstance(evidence, dict)
        ]
        validated_chunks = [
            evidence.get("chunk_ids")
            for evidence in validated.get("relevant_evidence", [])
            if isinstance(evidence, dict)
        ]
        if original_chunks != validated_chunks:
            raise AssertionError("Ground Truth chunk_ids changed unexpectedly")

    result = {
        "valid": not (
            missing_chunks
            or paper_mismatches
            or section_type_mismatches
            or bibliography_errors
            or manual_remaps
            or missing_paper_ids
        ),
        "evidence_bearing_cases": evidence_bearing_cases,
        "referenced_chunk_total": len(referenced_ids),
        "referenced_unique_chunk_total": len(set(referenced_ids)),
        "missing_chunk_ids": missing_chunks,
        "paper_id_mismatches": paper_mismatches,
        "section_mismatches": section_mismatches,
        "page_mismatches": page_mismatches,
        "section_type_mismatches": section_type_mismatches,
        "bibliography_classification_errors": bibliography_errors,
        "missing_paper_ids": missing_paper_ids,
        "needs_manual_remap": manual_remaps,
        "safe_metadata_updates": safe_updates,
    }
    return result, validated_records


def find_single_hop_multi_chunk_cases(
    records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """列出 single_hop 但含多个 Ground Truth chunks 的人工审查清单。"""
    checklist: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        tags = record.get("tags", [])
        if not isinstance(tags, list) or "single_hop" not in tags:
            continue
        evidence_list = record.get("relevant_evidence", [])
        if not isinstance(evidence_list, list):
            continue
        chunk_ids = [
            str(chunk_id)
            for evidence in evidence_list
            if isinstance(evidence, dict) and isinstance(evidence.get("chunk_ids"), list)
            for chunk_id in evidence["chunk_ids"]
        ]
        if len(chunk_ids) <= 1:
            continue
        checklist.append(
            {
                "case_id": _case_label(record, index),
                "question": record.get("question"),
                "required_facts": record.get("required_facts"),
                "chunk_ids": chunk_ids,
                "locations": [
                    {
                        "section": evidence.get("section"),
                        "page_start": evidence.get("page_start"),
                        "page_end": evidence.get("page_end"),
                        "chunk_ids": evidence.get("chunk_ids"),
                    }
                    for evidence in evidence_list
                    if isinstance(evidence, dict)
                ],
            }
        )
    return checklist


def write_json_atomic(path: Path, value: Any) -> None:
    """以同目录临时文件原子写入 JSON 报告，避免留下半写入文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """以原子替换方式写入 Markdown，确保报告始终是完整文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """写出 validated JSONL 副本，并确保每行都是可独立解析的 JSON 对象。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _markdown_table(rows: Sequence[tuple[Any, Any]], headers: tuple[str, str]) -> list[str]:
    """把两列统计转为简单 Markdown 表格。"""
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """将机器可读校验结果渲染为便于人工复核的 Markdown 报告。"""
    dataset = report["dataset_validation"]
    evidence = report["evidence_integrity"]
    counts = dataset["counts"]
    lines = [
        "# Golden Dataset Validation Report",
        "",
        f"**Final result: {report['status']}**",
        "",
        f"- Dataset: `{report['inputs']['dataset']}`",
        f"- SQLite: `{report['inputs']['database']}`",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Dataset validation",
        "",
        f"- JSONL parseable: `{dataset['jsonl_parseable']}`",
        f"- Total cases: `{dataset['total_cases']}`",
        f"- Duplicate IDs: `{len(dataset['duplicate_ids'])}`",
        f"- Missing-field cases: `{len(dataset['missing_fields'])}`",
        f"- Invalid primary types: `{len(dataset['invalid_primary_types'])}`",
        f"- Illegal-tag cases: `{len(dataset['illegal_tags'])}`",
        f"- Schema errors: `{len(dataset['schema_errors'])}`",
        f"- Logic errors: `{len(dataset['logic_errors'])}`",
        "",
        "### Primary types",
        "",
        *_markdown_table(list(counts["primary_type"].items()), ("primary_type", "count")),
        "",
        "### Papers",
        "",
        *_markdown_table(list(counts["paper_id"].items()), ("paper_id", "count")),
        "",
        "### Tags",
        "",
        *_markdown_table(list(counts["tags"].items()), ("tag", "count")),
        "",
        "### Boolean labels",
        "",
        f"- answerable: true `{counts['answerable']['true']}`, false `{counts['answerable']['false']}`",
        "- expected_bibliography_intent: "
        f"true `{counts['expected_bibliography_intent']['true']}`, "
        f"false `{counts['expected_bibliography_intent']['false']}`",
        "",
        "## Evidence integrity",
        "",
        f"- Evidence-bearing cases: `{evidence['evidence_bearing_cases']}`",
        f"- Referenced chunks: `{evidence['referenced_chunk_total']}` "
        f"(unique `{evidence['referenced_unique_chunk_total']}`)",
        f"- Missing chunk IDs: `{len(evidence['missing_chunk_ids'])}`",
        f"- paper_id mismatches: `{len(evidence['paper_id_mismatches'])}`",
        f"- section mismatches before safe sync: `{len(evidence['section_mismatches'])}`",
        f"- page mismatches before safe sync: `{len(evidence['page_mismatches'])}`",
        f"- section_type mismatches: `{len(evidence['section_type_mismatches'])}`",
        "- bibliography classification errors: "
        f"`{len(evidence['bibliography_classification_errors'])}`",
        f"- NEEDS_MANUAL_REMAP: `{len(evidence['needs_manual_remap'])}`",
        "",
        "## Safe metadata synchronization",
        "",
        f"- Updated fields: `{len(evidence['safe_metadata_updates'])}`",
        f"- Validated dataset written: `{report['outputs']['validated_dataset_written']}`",
    ]
    if evidence["safe_metadata_updates"]:
        lines.extend(["", "| Case | Evidence | Field | Old | New |", "|---|---:|---|---|---|"])
        for item in evidence["safe_metadata_updates"]:
            lines.append(
                f"| {item['case_id']} | {item['evidence_index']} | {item['field']} | "
                f"`{item['old']}` | `{item['new']}` |"
            )

    lines.extend(["", "## Manual remap", ""])
    if evidence["needs_manual_remap"]:
        for item in evidence["needs_manual_remap"]:
            lines.append(
                f"- `{item['case_id']}` evidence {item['evidence_index']}: "
                f"{item['reason']} — chunks `{item['chunk_ids']}`"
            )
    else:
        lines.append("No Evidence requires manual remapping.")

    lines.extend(["", "## single_hop cases with multiple Ground Truth chunks", ""])
    checklist = report["single_hop_multi_chunk_cases"]
    if checklist:
        for item in checklist:
            lines.extend(
                [
                    f"### {item['case_id']}",
                    "",
                    f"- Question: {item['question']}",
                    f"- Required facts: `{json.dumps(item['required_facts'], ensure_ascii=False)}`",
                    f"- Chunk IDs: `{json.dumps(item['chunk_ids'], ensure_ascii=False)}`",
                    f"- Section/page: `{json.dumps(item['locations'], ensure_ascii=False)}`",
                    "",
                ]
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Detailed errors", ""])
    detail_groups = (
        ("Parse errors", dataset["parse_errors"]),
        ("Schema errors", dataset["schema_errors"]),
        ("Logic errors", dataset["logic_errors"]),
        ("Missing chunks", evidence["missing_chunk_ids"]),
        ("paper_id mismatches", evidence["paper_id_mismatches"]),
        ("section_type mismatches", evidence["section_type_mismatches"]),
        ("Bibliography classification errors", evidence["bibliography_classification_errors"]),
    )
    any_detail = False
    for title, values in detail_groups:
        if values:
            any_detail = True
            lines.extend([f"### {title}", "", "```json", json.dumps(values, ensure_ascii=False, indent=2), "```", ""])
    if not any_detail:
        lines.append("No blocking errors.")
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    *,
    dataset_path: Path,
    database_path: Path,
    validated_path: Path,
    dataset_validation: Mapping[str, Any],
    sqlite_schema: Mapping[str, Any],
    evidence_integrity: Mapping[str, Any],
    checklist: Sequence[Mapping[str, Any]],
    validated_written: bool,
) -> dict[str, Any]:
    """汇总最终 PASS/NEEDS_MANUAL_REVIEW 结论及全部机器可读结果。"""
    status = (
        "PASS"
        if dataset_validation["valid"]
        and sqlite_schema["valid"]
        and evidence_integrity["valid"]
        else "NEEDS_MANUAL_REVIEW"
    )
    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset": str(dataset_path.resolve()),
            "database": str(database_path.resolve()),
        },
        "sqlite_schema": dict(sqlite_schema),
        "dataset_validation": dict(dataset_validation),
        "evidence_integrity": dict(evidence_integrity),
        "single_hop_multi_chunk_cases": list(checklist),
        "outputs": {
            "validated_dataset": str(validated_path.resolve()),
            "validated_dataset_written": validated_written,
        },
        "sync_policy": {
            "allowed_fields": ["section", "page_start", "page_end"],
            "semantic_fields_changed": False,
            "ground_truth_chunk_ids_changed": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行路径参数，所有参数都有项目内安全默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--validated-output", type=Path, default=DEFAULT_VALIDATED_DATASET)
    return parser.parse_args(argv)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整校验、可选安全同步，并生成 JSON 与 Markdown 两份报告。"""
    dataset_path = args.dataset.resolve()
    validated_path = args.validated_output.resolve()
    if dataset_path == validated_path:
        raise ValidationInputError("validated output must not overwrite the source dataset")
    if not dataset_path.is_file():
        raise ValidationInputError(f"Golden Dataset does not exist: {dataset_path}")

    records, parse_errors = load_jsonl(dataset_path)
    dataset_validation = validate_dataset(records, parse_errors)
    with open_database_read_only(args.database) as connection:
        sqlite_schema = validate_sqlite_schema(connection)
        if not sqlite_schema["valid"]:
            raise ValidationInputError(
                "SQLite schema is missing required tables/columns: "
                f"{sqlite_schema['missing_tables']} {sqlite_schema['missing_chunk_columns']}"
            )
        chunks = load_chunks(connection, _all_evidence_chunk_ids(records))
        evidence_integrity, validated_records = validate_evidence(
            records, chunks, load_paper_ids(connection)
        )

    checklist = find_single_hop_multi_chunk_cases(validated_records)
    # 解析错误会导致输出丢行，因此此时即使局部字段可同步，也绝不生成不完整副本。
    validated_written = bool(evidence_integrity["safe_metadata_updates"]) and not parse_errors
    if validated_written:
        write_jsonl_atomic(validated_path, validated_records)

    report = build_report(
        dataset_path=dataset_path,
        database_path=args.database,
        validated_path=validated_path,
        dataset_validation=dataset_validation,
        sqlite_schema=sqlite_schema,
        evidence_integrity=evidence_integrity,
        checklist=checklist,
        validated_written=validated_written,
    )
    write_json_atomic(args.json_report, report)
    write_text_atomic(args.markdown_report, render_markdown(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：打印简洁统计，并用退出码 0/1 表示 PASS/需人工复核。"""
    args = parse_args(argv)
    try:
        report = run_validation(args)
    except (OSError, sqlite3.Error, ValidationInputError) as error:
        print(f"ERROR: {error}")
        return 2

    dataset = report["dataset_validation"]
    evidence = report["evidence_integrity"]
    print(report["status"])
    print(f"Total cases: {dataset['total_cases']}")
    print(f"Evidence-bearing cases: {evidence['evidence_bearing_cases']}")
    print(f"Referenced chunks: {evidence['referenced_chunk_total']}")
    print(f"Safe metadata updates: {len(evidence['safe_metadata_updates'])}")
    print(f"Missing chunks: {len(evidence['missing_chunk_ids'])}")
    print(f"NEEDS_MANUAL_REMAP: {len(evidence['needs_manual_remap'])}")
    print(f"JSON report: {args.json_report.resolve()}")
    print(f"Markdown report: {args.markdown_report.resolve()}")
    if report["outputs"]["validated_dataset_written"]:
        print(f"Validated dataset: {args.validated_output.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
