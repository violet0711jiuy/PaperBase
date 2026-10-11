"""为 synthesis_001 增加人工审核的 Evidence 替代路径并冻结 Golden v1.3。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "eval" / "datasets" / "golden_dataset_v1_2.jsonl"
DEFAULT_OUTPUT = ROOT / "eval" / "datasets" / "golden_dataset_v1_3.jsonl"

EXPECTED_TYPE_COUNTS = {
    "fact": 8,
    "method": 8,
    "experiment": 6,
    "result": 6,
    "synthesis": 4,
    "bibliography": 4,
    "unanswerable": 4,
}

# 每个内层列表是一条 all-of 替代路径；同一 canonical chunk 的多条路径为 OR。
# 路径来自对 SQLite 原文和 2026-10-11 synthesis_001 Answer 引用的人工复核。
SYNTHESIS_001_ALTERNATIVES = {
    "paper_5b6a1007fa7514bf_chunk_0026": [[
        "paper_5b6a1007fa7514bf_chunk_0014",
        "paper_5b6a1007fa7514bf_chunk_0015",
        "paper_5b6a1007fa7514bf_chunk_0032",
    ]],
    "paper_5b6a1007fa7514bf_chunk_0038": [[
        "paper_5b6a1007fa7514bf_chunk_0030",
        "paper_5b6a1007fa7514bf_chunk_0036",
    ]],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON解析失败：{error}") from error
    return records


def build_v1_3(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = copy.deepcopy(source_records)
    matches = [record for record in records if record.get("id") == "synthesis_001"]
    if len(matches) != 1:
        raise ValueError(f"预期唯一 synthesis_001，实际找到 {len(matches)} 条")
    matches[0]["evidence_alternatives"] = copy.deepcopy(SYNTHESIS_001_ALTERNATIVES)
    return records


def validate_frozen_records(
    source_records: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    if len(records) != 40:
        raise ValueError(f"预期40条Golden，实际为{len(records)}条")

    ids = [str(record.get("id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("v1.3存在重复ID")

    counts = Counter(str(record.get("primary_type")) for record in records)
    if dict(counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"primary_type分布异常：{dict(counts)}")

    source_by_id = {str(record["id"]): record for record in source_records}
    for record in records:
        case_id = str(record["id"])
        source = source_by_id.get(case_id)
        if source is None:
            raise ValueError(f"v1.3出现v1.2不存在的Case：{case_id}")
        expected = copy.deepcopy(source)
        if case_id == "synthesis_001":
            expected["evidence_alternatives"] = copy.deepcopy(
                SYNTHESIS_001_ALTERNATIVES
            )
        if record != expected:
            raise ValueError(f"{case_id}出现冻结规则之外的字段变化")

        if record["answerable"] is False and any((
            record["reference_answer"] is not None,
            bool(record["required_facts"]),
            bool(record["relevant_evidence"]),
            bool(record.get("evidence_alternatives")),
        )):
            raise ValueError(f"{case_id}违反unanswerable逻辑约束")


def serialize_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="只验证现有v1.3，不写文件。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_records = load_jsonl(args.source)
    expected_records = build_v1_3(source_records)
    validate_frozen_records(source_records, expected_records)
    expected_text = serialize_jsonl(expected_records)

    if args.check:
        actual_records = load_jsonl(args.output)
        validate_frozen_records(source_records, actual_records)
        if serialize_jsonl(actual_records) != expected_text:
            raise ValueError("现有v1.3与冻结规则不一致")
        print(
            f"PASS：{args.output}，共{len(actual_records)}条，"
            f"SHA256={sha256_text(expected_text)}"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected_text, encoding="utf-8", newline="\n")
    print(
        f"已冻结{args.output}，共{len(expected_records)}条，"
        f"修改Case=['synthesis_001']，SHA256={sha256_text(expected_text)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
