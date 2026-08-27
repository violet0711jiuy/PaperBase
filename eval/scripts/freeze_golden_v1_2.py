"""根据失败案例人工复核结果，将 Context-free Golden v1.1 冻结为 v1.2。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "eval" / "datasets" / "golden_dataset_v1_1.jsonl"
DEFAULT_OUTPUT = ROOT / "eval" / "datasets" / "golden_dataset_v1_2.jsonl"

EXPECTED_TYPE_COUNTS = {
    "fact": 8,
    "method": 8,
    "experiment": 6,
    "result": 6,
    "synthesis": 4,
    "bibliography": 4,
    "unanswerable": 4,
}

# 这些变更均来自逐块人工复核；未列出的 Case 必须与 v1.1 完全一致。
CASE_PATCHES: dict[str, dict[str, Any]] = {
    "method_002": {
        "question": "在LSTM-EMVE的EMVE损失函数中，MPIWcapt和校准项C_i分别如何控制预测区间宽度与覆盖率？",
        "tags": [
            "zh_query_en_doc",
            "exact_term",
            "multi_hop",
            "medium",
        ],
        "reference_answer": (
            "MPIWcapt只计算并压缩已经覆盖真实值的预测区间，避免继续收窄本来就漏覆盖的区间；"
            "校准项C_i在真实值落到区间外时，按照越界距离施加惩罚，促使模型维持覆盖率。"
            "两者与MLE-QV项共同组成最终EMVE损失，从而平衡区间宽度与覆盖率。"
        ),
        "required_facts": [
            "MPIWcapt只压缩成功覆盖真实值的预测区间。",
            "C_i按照真实值落在区间外的距离施加惩罚，以维持覆盖率。",
            "MPIWcapt和C_i与MLE-QV项共同进入最终EMVE损失。",
        ],
        "relevant_evidence": [
            {
                "paper_id": "paper_5b6a1007fa7514bf",
                "section": "3. Methodology > 3.1. Enhanced mean-variance approach > 3.1.5. PI optimization",
                "page_start": 7,
                "page_end": 8,
                "chunk_ids": [
                    "paper_5b6a1007fa7514bf_chunk_0031",
                    "paper_5b6a1007fa7514bf_chunk_0032",
                    "paper_5b6a1007fa7514bf_chunk_0033",
                ],
            }
        ],
    },
    "method_007": {
        "question": (
            "How does ESDTW combine ideas from LEDTW and shapeDTW, "
            "and how does its alignment procedure differ from each?"
        ),
        "reference_answer": (
            "LEDTW separates local maxima and minima into two sequences and applies DTW to them independently, "
            "which can destroy the temporal relationship between maxima and minima and produce cross-alignments. "
            "ShapeDTW computes a shape descriptor for every point in the full sequence and aligns the resulting "
            "descriptor sequences. ESDTW instead selects local extrema, samples subsequences centered on them, "
            "computes descriptors for those extrema, and jointly aligns the extrema-descriptor sequences with DTW."
        ),
        "required_facts": [
            "LEDTW separates maxima and minima and aligns the two extrema sequences independently.",
            "ShapeDTW computes descriptors for every point in the full sequence.",
            "ESDTW computes descriptors only for subsequences centered on local extrema and aligns those extrema descriptors jointly.",
        ],
        "relevant_evidence": [
            {
                "paper_id": "paper_b7a064b63171eaee",
                "section": "1. Introduction",
                "page_start": 2,
                "page_end": 2,
                "chunk_ids": ["paper_b7a064b63171eaee_chunk_0009"],
            },
            {
                "paper_id": "paper_b7a064b63171eaee",
                "section": "2. Background > 2.3. Shape dynamic time warping",
                "page_start": 3,
                "page_end": 3,
                "chunk_ids": ["paper_b7a064b63171eaee_chunk_0015"],
            },
            {
                "paper_id": "paper_b7a064b63171eaee",
                "section": "3. Extrema-based shape dynamic time warping > 3.3. Algorithm overview",
                "page_start": 4,
                "page_end": 4,
                "chunk_ids": ["paper_b7a064b63171eaee_chunk_0023"],
            },
        ],
    },
    "synthesis_001": {
        "question": (
            "LSTM-EMVE如何将MLE-QV、MPIWcapt和C_i与branched multi-activation architecture结合起来，"
            "同时输出点预测、标准差和预测区间？"
        )
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行读取 Golden JSONL，并保留明确的错误行号。"""
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


def build_v1_2(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在 v1.1 深拷贝上严格应用三条人工审核补丁。"""
    records = copy.deepcopy(source_records)
    ids = {str(record.get("id")) for record in records}
    missing = sorted(set(CASE_PATCHES) - ids)
    if missing:
        raise ValueError(f"v1.1缺少待修改Case：{missing}")

    for record in records:
        patch = CASE_PATCHES.get(str(record["id"]))
        if patch:
            record.update(copy.deepcopy(patch))
    return records


def validate_frozen_records(
    source_records: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    """确认数量分布不变，且所有字段变化都来自审核补丁。"""
    if len(records) != 40:
        raise ValueError(f"预期40条Golden，实际为{len(records)}条")

    ids = [str(record.get("id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("v1.2存在重复ID")

    counts = Counter(str(record.get("primary_type")) for record in records)
    if dict(counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"primary_type分布异常：{dict(counts)}")

    source_by_id = {str(record["id"]): record for record in source_records}
    for record in records:
        case_id = str(record["id"])
        source = source_by_id.get(case_id)
        if source is None:
            raise ValueError(f"v1.2出现v1.1不存在的Case：{case_id}")
        expected = copy.deepcopy(source)
        expected.update(copy.deepcopy(CASE_PATCHES.get(case_id, {})))
        if record != expected:
            raise ValueError(f"{case_id}出现审核补丁之外的字段变化")

        if record["answerable"] is False and any(
            (
                record["reference_answer"] is not None,
                bool(record["required_facts"]),
                bool(record["relevant_evidence"]),
            )
        ):
            raise ValueError(f"{case_id}违反unanswerable逻辑约束")


def serialize_jsonl(records: list[dict[str, Any]]) -> str:
    """用稳定的UTF-8单行JSON格式生成可复算内容。"""
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def sha256_text(text: str) -> str:
    """计算冻结内容的SHA256，供实验记录绑定精确数据版本。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析v1.1源文件、v1.2输出文件与只检查模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="只验证现有v1.2，不写文件。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """冻结或复核Golden Dataset v1.2。"""
    args = parse_args(argv)
    source_records = load_jsonl(args.source)
    expected_records = build_v1_2(source_records)
    validate_frozen_records(source_records, expected_records)
    expected_text = serialize_jsonl(expected_records)

    if args.check:
        actual_records = load_jsonl(args.output)
        validate_frozen_records(source_records, actual_records)
        if serialize_jsonl(actual_records) != expected_text:
            raise ValueError("现有v1.2与冻结规则不一致")
        print(
            f"PASS：{args.output}，共{len(actual_records)}条，"
            f"SHA256={sha256_text(expected_text)}"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected_text, encoding="utf-8", newline="\n")
    print(
        f"已冻结{args.output}，共{len(expected_records)}条，"
        f"修改Case={sorted(CASE_PATCHES)}，SHA256={sha256_text(expected_text)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
