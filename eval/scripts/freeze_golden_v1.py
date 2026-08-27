"""Freeze the manually reviewed Candidate Goldens as Golden Dataset v1.

The review workbook is the source of truth for selection, rewrites, final
types, and tags.  This script intentionally stays outside PaperBase's formal
retrieval and generation path: it reads Candidate JSONL plus canonical chunk
metadata from SQLite opened in read-only mode, applies the reviewed decisions,
and writes a validated JSONL snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / "eval" / "candidates" / "candidate_goldens.jsonl"
DEFAULT_DATABASE = ROOT / "storage" / "paperbase.sqlite3"
DEFAULT_OUTPUT = ROOT / "eval" / "datasets" / "golden_dataset_v1.jsonl"

FIELDS = (
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
TYPE_ORDER = (
    "fact",
    "method",
    "experiment",
    "result",
    "synthesis",
    "bibliography",
    "unanswerable",
)
TAG_ORDER = (
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
LEGAL_TAGS = set(TAG_ORDER)
TARGET_TYPE_COUNTS = {
    "fact": 8,
    "method": 8,
    "experiment": 6,
    "result": 6,
    "synthesis": 4,
    "bibliography": 4,
    "unanswerable": 4,
}

SELECTED_IDS = (
    "fact_001",
    "fact_002",
    "fact_004",
    "fact_005",
    "fact_007",
    "fact_008",
    "fact_010",
    "fact_011",
    "method_001",
    "method_002",
    "method_004",
    "method_005",
    "method_006",
    "method_007",
    "method_010",
    "method_011",
    "experiment_001",
    "experiment_003",
    "experiment_004",
    "experiment_005",
    "experiment_007",
    "experiment_008",
    "result_001",
    "result_002",
    "result_003",
    "result_006",
    "result_007",
    "result_008",
    "synthesis_001",
    "synthesis_002",
    "synthesis_004",
    "synthesis_006",
    "bibliography_001",
    "bibliography_003",
    "bibliography_005",
    "bibliography_006",
    "unanswerable_001",
    "unanswerable_002",
    "unanswerable_003",
    "unanswerable_006",
)

# Used only if bibliography_001 fails the review's VERIFY condition.
BIBLIOGRAPHY_FALLBACK_ID = "bibliography_004"

AUDIT_MODIFY_IDS = {
    "fact_001",
    "fact_002",
    "fact_005",
    "fact_007",
    "fact_008",
    "fact_011",
    "method_001",
    "method_002",
    "method_007",
    "method_011",
    "experiment_001",
    "experiment_005",
    "experiment_007",
    "result_002",
    "result_007",
    "synthesis_001",
}

# KEEP rows with a concrete reviewer suggestion to tighten wording/evidence.
ADDITIONAL_TIGHTEN_IDS = {"result_001", "synthesis_002", "synthesis_004"}

FINAL_TAGS: dict[str, list[str]] = {
    "fact_001": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "fact_002": ["zh_query_en_doc", "semantic_paraphrase", "single_hop", "easy"],
    "fact_004": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "fact_005": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "fact_007": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "fact_008": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "fact_010": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "fact_011": ["zh_query_en_doc", "semantic_paraphrase", "single_hop", "medium"],
    "method_001": ["english_query", "semantic_paraphrase", "multi_hop", "medium"],
    "method_002": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "method_004": ["english_query", "semantic_paraphrase", "single_hop", "medium"],
    "method_005": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "medium"],
    "method_006": ["zh_query_en_doc", "exact_term", "multi_hop", "medium"],
    "method_007": ["english_query", "exact_term", "multi_hop", "medium"],
    "method_010": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "method_011": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "experiment_001": ["english_query", "exact_term", "single_hop", "medium"],
    "experiment_003": ["english_query", "exact_term", "multi_hop", "medium"],
    "experiment_004": ["zh_query_en_doc", "semantic_paraphrase", "single_hop", "easy"],
    "experiment_005": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "medium"],
    "experiment_007": ["zh_query_en_doc", "exact_term", "multi_hop", "medium"],
    "experiment_008": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "result_001": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "result_002": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "hard"],
    "result_003": ["zh_query_en_doc", "exact_term", "single_hop", "easy"],
    "result_006": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "result_007": ["zh_query_en_doc", "exact_term", "single_hop", "medium"],
    "result_008": ["zh_query_en_doc", "semantic_paraphrase", "single_hop", "medium"],
    "synthesis_001": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "hard"],
    "synthesis_002": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "medium"],
    "synthesis_004": ["zh_query_en_doc", "exact_term", "multi_hop", "hard"],
    "synthesis_006": ["zh_query_en_doc", "exact_term", "multi_hop", "hard"],
    "bibliography_001": [
        "zh_query_en_doc",
        "exact_term",
        "single_hop",
        "bibliography_intent",
        "easy",
    ],
    "bibliography_003": [
        "zh_query_en_doc",
        "exact_term",
        "single_hop",
        "bibliography_intent",
        "easy",
    ],
    "bibliography_004": [
        "zh_query_en_doc",
        "exact_term",
        "multi_hop",
        "bibliography_intent",
        "medium",
    ],
    "bibliography_005": [
        "zh_query_en_doc",
        "exact_term",
        "multi_hop",
        "bibliography_intent",
        "medium",
    ],
    "bibliography_006": [
        "zh_query_en_doc",
        "exact_term",
        "multi_hop",
        "bibliography_intent",
        "medium",
    ],
    "unanswerable_001": ["zh_query_en_doc", "semantic_paraphrase", "single_hop", "hard"],
    "unanswerable_002": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "hard"],
    "unanswerable_003": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "hard"],
    "unanswerable_006": ["zh_query_en_doc", "semantic_paraphrase", "multi_hop", "medium"],
}

FIELD_OVERRIDES: dict[str, dict[str, Any]] = {
    "fact_001": {
        "question": "生成90%预测区间时，量化违反（quantile violation）阈值设置为多少？",
        "reference_answer": "0.8σ。",
        "required_facts": ["量化违反阈值设置为0.8σ。"],
    },
    "fact_002": {
        "required_facts": ["标准差的三个并行分支分别使用softplus、ReLU和指数激活函数。"],
    },
    "fact_005": {
        "required_facts": ["高压底部条件下，24小时细颗粒物浓度预报与监测值的相关系数R为0.67。"],
    },
    "fact_007": {
        "question": "ESDTW在分类实验中与什么分类器结合，并在多少个UCR数据集上进行主要比较？",
        "reference_answer": "与1-NN结合，在初始84个UCR数据集上进行主要比较。",
        "required_facts": [
            "ESDTW在分类实验中与1-NN分类器结合。",
            "主要比较使用初始84个UCR数据集。",
        ],
    },
    "fact_008": {
        "question": "ESDTW和传统DTW的理论时间复杂度分别是什么？",
        "reference_answer": "ESDTW为O(l_p²)，DTW为O(N²)，且l_p≪N。",
        "required_facts": ["ESDTW为O(l_p²)，DTW为O(N²)，且l_p≪N。"],
    },
    "fact_011": {
        "question": "Inherent Model使用哪两类模块分别捕捉局部和全局时间依赖？",
        "reference_answer": "GRU捕捉局部依赖，多头自注意力捕捉全局依赖。",
        "required_facts": [
            "GRU捕捉局部（短期）时间依赖。",
            "多头自注意力捕捉全局（长期）时间依赖。",
        ],
    },
    "method_002": {
        "reference_answer": (
            "EMVE用MPIWcapt压缩成功覆盖真实值的预测区间宽度；用C_i对真实值落在区间外的情况按偏离程度惩罚，"
            "以维持覆盖率；同时MLE与L_QV约束分布参数和均值误差，最终在式(21)中联合优化。"
        ),
        "required_facts": [
            "MPIWcapt只压缩成功覆盖真实值的预测区间宽度。",
            "C_i按真实值落在区间外的偏离程度施加惩罚，以维持覆盖率。",
            "MLE、L_QV、MPIWcapt和C_i在式(21)中被联合优化。",
        ],
    },
    "experiment_005": {
        "question": "ESDTW如何构造具有ground truth的模拟对齐序列，并用什么指标评价对齐质量？",
        "reference_answer": (
            "从原序列中随机选择θ%的点，并将每个选中点物理拉伸（等价于重复t次），得到S_stretch及其与原序列的真实匹配对/"
            "warping path；对齐质量用mean absolute deviation评价，即比较测试warping path与ground truth的平均绝对偏差。"
        ),
        "required_facts": [
            "随机选择原序列中θ%的点，并将选中点拉伸或重复t次以生成S_stretch。",
            "拉伸索引与原序列尺度信息给出S_stretch和原序列之间的ground-truth匹配对或warping path。",
            "用mean absolute deviation衡量测试warping path与ground truth之间的平均偏差。",
        ],
    },
    "experiment_007": {
        "question": "D²STGNN在四个数据集上的数据划分、输入/预测窗口、评估时效和指标如何设置？",
        "reference_answer": (
            "METR-LA和PEMS-BAY约按70%训练、20%测试、10%验证划分，PEMS04和PEMS08约按60%训练、20%测试、20%验证划分；"
            "滑动窗口宽度为24步，前12步作为输入、后12步作为ground truth；评估horizon 3、6、12，对应15、30、60分钟预测，"
            "指标为MAE、RMSE和MAPE。"
        ),
        "required_facts": [
            "METR-LA和PEMS-BAY约按70%训练、20%测试、10%验证划分。",
            "PEMS04和PEMS08约按60%训练、20%测试、20%验证划分。",
            "滑动窗口为24步，前12步输入、后12步作为ground truth。",
            "评估horizon 3、6、12，即15、30、60分钟预测。",
            "评价指标为MAE、RMSE和MAPE。",
        ],
    },
    "result_002": {
        "question": "LSTM-EMVE在多步风速预测中，随着预测时长增加，概率区间质量和确定性误差表现如何？",
        "reference_answer": (
            "LSTM-EMVE通常保持更窄的PINRW和有竞争力的coverage，因此在多数预测时域取得最低CWC。随着预测距离增加，各方法的"
            "确定性误差总体会恶化，但LSTM-EMVE的MSE分布更紧凑、horizon robustness更好；在Lake Huron数据上的方差范围约为"
            "0.90–2.84。"
        ),
        "required_facts": [
            "LSTM-EMVE通常以更窄PINRW和有竞争力的coverage在多数预测时域取得最低CWC。",
            "确定性误差随预测距离增加总体恶化。",
            "LSTM-EMVE的MSE分布更紧凑、horizon robustness更好，Lake Huron方差范围约为0.90–2.84。",
        ],
    },
    "result_007": {
        "required_facts": ["D²STGNN在所有四个数据集和所有预测时域上均取得最佳性能。"],
    },
    "synthesis_001": {
        "question": "EMVE如何通过损失函数设计与branched multi-activation architecture协同实现点预测和预测区间的联合优化？",
        "reference_answer": (
            "损失函数先用MLE与L_QV共同约束预测分布参数并修正均值误差，再用MPIWcapt只收紧成功覆盖真实值的区间，并以C_i按"
            "越界距离惩罚漏覆盖，从而兼顾sharpness与coverage。网络结构则用线性mean branch预测μ，用softplus、ReLU和exponential"
            "三个variance branches估计σ并平均。最终由μ和σ计算上下界，输出mean、standard deviation及prediction interval。"
        ),
        "required_facts": [
            "MLE与L_QV共同约束预测分布参数并修正均值误差。",
            "MPIWcapt和C_i分别优化成功覆盖区间的宽度与漏覆盖惩罚，以平衡sharpness和coverage。",
            "线性mean branch预测μ，softplus、ReLU和exponential三个variance branches的输出经平均得到σ。",
            "模型用μ和σ计算预测区间上下界，并联合输出mean、standard deviation、upper bound和lower bound。",
        ],
    },
    "synthesis_002": {
        "reference_answer": (
            "低气压和同压条件下压力梯度较小、风速较弱，污染物不易扩散，细颗粒物超标发生率分别为87.5%和52.8%；高压底部"
            "条件下风速和压力梯度较强，超标发生率最低，为25.2%。该天气型下2米相对湿度和10米风速预报也更准确，论文据此认为"
            "这些气象因子的预报准确性可能与细颗粒物预报准确性相关。"
        ),
        "required_facts": [
            "低气压和同压条件下细颗粒物超标发生率分别为87.5%和52.8%，并与较弱的压力梯度和风速相关。",
            "高压底部条件下超标发生率最低（25.2%），污染物扩散条件较好。",
            "高压底部条件下2米相对湿度和10米风速预报更准确，论文认为这可能与细颗粒物预报准确性相关。",
        ],
    },
}

EVIDENCE_CHUNK_IDS: dict[str, list[str]] = {
    "fact_001": ["paper_5b6a1007fa7514bf_chunk_0038"],
    "fact_002": ["paper_5b6a1007fa7514bf_chunk_0036"],
    "fact_005": ["paper_b12197625a863197_chunk_0019"],
    "fact_007": [
        "paper_b7a064b63171eaee_chunk_0048",
        "paper_b7a064b63171eaee_chunk_0049",
    ],
    "fact_008": ["paper_b7a064b63171eaee_chunk_0024"],
    "fact_011": ["paper_c162376bc253ae7d_chunk_0028"],
    "method_001": [
        "paper_5b6a1007fa7514bf_chunk_0036",
        "paper_5b6a1007fa7514bf_chunk_0037",
    ],
    "method_002": [
        "paper_5b6a1007fa7514bf_chunk_0026",
        "paper_5b6a1007fa7514bf_chunk_0032",
        "paper_5b6a1007fa7514bf_chunk_0033",
    ],
    "method_007": [
        "paper_b7a064b63171eaee_chunk_0014",
        "paper_b7a064b63171eaee_chunk_0015",
        "paper_b7a064b63171eaee_chunk_0016",
    ],
    "experiment_001": [
        "paper_5b6a1007fa7514bf_chunk_0067",
        "paper_5b6a1007fa7514bf_chunk_0073",
    ],
    "experiment_005": [
        "paper_b7a064b63171eaee_chunk_0026",
        "paper_b7a064b63171eaee_chunk_0040",
    ],
    "experiment_007": ["paper_c162376bc253ae7d_chunk_0045"],
    "result_001": ["paper_5b6a1007fa7514bf_chunk_0054"],
    "result_002": [
        "paper_5b6a1007fa7514bf_chunk_0076",
        "paper_5b6a1007fa7514bf_chunk_0077",
    ],
    "result_007": ["paper_c162376bc253ae7d_chunk_0046"],
    "synthesis_001": [
        "paper_5b6a1007fa7514bf_chunk_0026",
        "paper_5b6a1007fa7514bf_chunk_0032",
        "paper_5b6a1007fa7514bf_chunk_0033",
        "paper_5b6a1007fa7514bf_chunk_0036",
        "paper_5b6a1007fa7514bf_chunk_0038",
    ],
    "synthesis_002": [
        "paper_b12197625a863197_chunk_0018",
        "paper_b12197625a863197_chunk_0024",
    ],
    "synthesis_004": [
        "paper_b7a064b63171eaee_chunk_0024",
        "paper_b7a064b63171eaee_chunk_0049",
    ],
    "bibliography_001": ["paper_5b6a1007fa7514bf_chunk_0124"],
}

SECTION_OVERRIDES = {
    "paper_5b6a1007fa7514bf_chunk_0124": "References",
}


class FreezeError(RuntimeError):
    """Raised when the reviewed snapshot cannot be frozen safely."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FreezeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise FreezeError(f"JSONL row {line_number} is not an object")
            records.append(record)
    return records


def open_database_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_chunks(
    connection: sqlite3.Connection, chunk_ids: Iterable[str]
) -> dict[str, sqlite3.Row]:
    unique_ids = sorted(set(chunk_ids))
    if not unique_ids:
        return {}
    placeholders = ",".join("?" for _ in unique_ids)
    rows = connection.execute(
        "SELECT chunk_id, paper_id, chunk_index, section, section_type, "
        "page_start, page_end FROM chunks WHERE chunk_id IN (" + placeholders + ")",
        unique_ids,
    ).fetchall()
    found = {str(row["chunk_id"]): row for row in rows}
    missing = sorted(set(unique_ids) - found.keys())
    if missing:
        raise FreezeError(f"Missing SQLite chunks: {missing}")
    return found


def bibliography_001_status(connection: sqlite3.Connection) -> dict[str, Any]:
    """核对 bibliography_001 的数据库分类与展示章节是否已经一致。"""
    row = connection.execute(
        "SELECT chunk_id, paper_id, section, section_type, page_start, page_end "
        "FROM chunks WHERE chunk_id = ?",
        ("paper_5b6a1007fa7514bf_chunk_0124",),
    ).fetchone()
    if row is None:
        return {"kept": False, "reason": "chunk missing from SQLite"}
    kept = str(row["section_type"]).lower() == "bibliography"
    database_section = str(row["section"])
    section_already_correct = database_section.casefold().strip() == "references"
    return {
        "kept": kept,
        "chunk_id": str(row["chunk_id"]),
        "database_section": database_section,
        "section_type": str(row["section_type"]),
        "golden_section": "References" if kept else None,
        "reason": (
            (
                "SQLite bibliography classification and References section are both correct"
                if section_already_correct
                else "SQLite classifies the chunk as bibliography; Golden section is corrected to References"
            )
            if kept
            else "SQLite does not classify the chunk as bibliography"
        ),
    }


def make_evidence(
    paper_id: str,
    chunk_ids: Sequence[str],
    chunks: Mapping[str, sqlite3.Row],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    ordered_rows = sorted((chunks[chunk_id] for chunk_id in chunk_ids), key=lambda row: row["chunk_index"])
    for row in ordered_rows:
        if str(row["paper_id"]) != paper_id:
            raise FreezeError(
                f"Evidence {row['chunk_id']} belongs to {row['paper_id']}, expected {paper_id}"
            )
        section = SECTION_OVERRIDES.get(str(row["chunk_id"]), str(row["section"] or "Unknown section"))
        grouped[section].append(row)

    evidence: list[dict[str, Any]] = []
    for section, rows in grouped.items():
        starts = [int(row["page_start"]) for row in rows if row["page_start"] is not None]
        ends = [int(row["page_end"]) for row in rows if row["page_end"] is not None]
        evidence.append(
            {
                "paper_id": paper_id,
                "section": section,
                "page_start": min(starts) if starts else None,
                "page_end": max(ends) if ends else None,
                "chunk_ids": [str(row["chunk_id"]) for row in rows],
            }
        )
    return evidence


def freeze(
    candidates: Sequence[Mapping[str, Any]], connection: sqlite3.Connection
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(candidate.get("id")): candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise FreezeError("Candidate JSONL contains duplicate IDs")

    bibliography_status = bibliography_001_status(connection)
    selected_ids = list(SELECTED_IDS)
    if not bibliography_status["kept"]:
        selected_ids[selected_ids.index("bibliography_001")] = BIBLIOGRAPHY_FALLBACK_ID

    missing = sorted(set(selected_ids) - by_id.keys())
    if missing:
        raise FreezeError(f"Selected Candidate IDs are missing: {missing}")

    all_evidence_ids: list[str] = []
    for case_id in selected_ids:
        candidate = by_id[case_id]
        if case_id in EVIDENCE_CHUNK_IDS:
            all_evidence_ids.extend(EVIDENCE_CHUNK_IDS[case_id])
        else:
            for evidence in candidate.get("relevant_evidence", []):
                all_evidence_ids.extend(evidence.get("chunk_ids", []))
    chunks = load_chunks(connection, all_evidence_ids)

    frozen: list[dict[str, Any]] = []
    for case_id in selected_ids:
        case = copy.deepcopy(dict(by_id[case_id]))
        case.update(copy.deepcopy(FIELD_OVERRIDES.get(case_id, {})))
        case["tags"] = copy.deepcopy(FINAL_TAGS[case_id])
        case["expected_bibliography_intent"] = case["primary_type"] == "bibliography"

        if case["primary_type"] == "unanswerable":
            case["answerable"] = False
            case["reference_answer"] = None
            case["required_facts"] = []
            case["relevant_evidence"] = []
        else:
            case["answerable"] = True
            if case_id in EVIDENCE_CHUNK_IDS:
                case["relevant_evidence"] = make_evidence(
                    str(case["paper_id"]), EVIDENCE_CHUNK_IDS[case_id], chunks
                )

        frozen.append({field: case[field] for field in FIELDS})

    return frozen, bibliography_status


def validate(
    records: Sequence[Mapping[str, Any]],
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    errors: list[str] = []
    ids = [record.get("id") for record in records]
    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    missing_fields: dict[str, list[str]] = {}
    illegal_tags: dict[str, list[str]] = {}
    conflicts: list[str] = []

    for index, record in enumerate(records, 1):
        case_id = str(record.get("id", f"row_{index}"))
        missing = [field for field in FIELDS if field not in record]
        if missing:
            missing_fields[case_id] = missing
            continue
        invalid = sorted(set(record["tags"]) - LEGAL_TAGS)
        if invalid:
            illegal_tags[case_id] = invalid
        if record["answerable"] is False and (
            record["reference_answer"] is not None
            or record["required_facts"]
            or record["relevant_evidence"]
        ):
            conflicts.append(case_id)
        if record["answerable"] is True and (
            not record["reference_answer"]
            or not record["required_facts"]
            or not record["relevant_evidence"]
        ):
            errors.append(f"{case_id}: answerable case lacks answer/facts/evidence")
        is_bibliography = record["primary_type"] == "bibliography"
        if bool(record["expected_bibliography_intent"]) != is_bibliography:
            errors.append(f"{case_id}: bibliography intent flag conflicts with primary_type")
        if ("bibliography_intent" in record["tags"]) != is_bibliography:
            errors.append(f"{case_id}: bibliography_intent tag conflicts with primary_type")

    if duplicate_ids:
        errors.append(f"duplicate IDs: {duplicate_ids}")
    if missing_fields:
        errors.append(f"missing fields: {missing_fields}")
    if illegal_tags:
        errors.append(f"illegal tags: {illegal_tags}")
    if conflicts:
        errors.append(f"unanswerable conflicts: {conflicts}")

    type_counts = Counter(str(record.get("primary_type")) for record in records)
    if dict(type_counts) != TARGET_TYPE_COUNTS:
        errors.append(f"primary_type counts {dict(type_counts)} != {TARGET_TYPE_COUNTS}")

    evidence_ids = [
        chunk_id
        for record in records
        for evidence in record.get("relevant_evidence", [])
        for chunk_id in evidence.get("chunk_ids", [])
    ]
    chunk_rows = load_chunks(connection, evidence_ids)
    for record in records:
        expected_section_type = (
            "bibliography" if record.get("primary_type") == "bibliography" else "content"
        )
        for evidence in record.get("relevant_evidence", []):
            if evidence.get("paper_id") != record.get("paper_id"):
                errors.append(f"{record.get('id')}: evidence paper_id mismatch")
            for chunk_id in evidence.get("chunk_ids", []):
                row = chunk_rows[chunk_id]
                if str(row["paper_id"]) != record.get("paper_id"):
                    errors.append(f"{record.get('id')}: chunk {chunk_id} belongs to another paper")
                if str(row["section_type"]) != expected_section_type:
                    errors.append(
                        f"{record.get('id')}: chunk {chunk_id} is {row['section_type']}, "
                        f"expected {expected_section_type}"
                    )

    paper_counts = Counter(str(record.get("paper_id")) for record in records)
    tag_counts = Counter(tag for record in records for tag in record.get("tags", []))
    difficulty_counts = Counter(
        tag
        for record in records
        for tag in record.get("tags", [])
        if tag in {"easy", "medium", "hard"}
    )
    answerable_counts = Counter("answerable" if record.get("answerable") else "unanswerable" for record in records)
    intent_counts = Counter(
        "true" if record.get("expected_bibliography_intent") else "false" for record in records
    )
    return {
        "total": len(records),
        "primary_type": {kind: type_counts.get(kind, 0) for kind in TYPE_ORDER},
        "paper_id": dict(sorted(paper_counts.items())),
        "difficulty": {kind: difficulty_counts.get(kind, 0) for kind in ("easy", "medium", "hard")},
        "tags": {tag: tag_counts.get(tag, 0) for tag in TAG_ORDER},
        "answerable": {
            "true": answerable_counts.get("answerable", 0),
            "false": answerable_counts.get("unanswerable", 0),
        },
        "expected_bibliography_intent": {
            "true": intent_counts.get("true", 0),
            "false": intent_counts.get("false", 0),
        },
        "jsonl_parseable": True,
        "duplicate_ids": duplicate_ids,
        "illegal_tags": illegal_tags,
        "missing_fields": missing_fields,
        "unanswerable_conflicts": conflicts,
        "errors": errors,
    }


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_path.replace(path)


def actual_drop_ids(candidates: Sequence[Mapping[str, Any]], frozen: Sequence[Mapping[str, Any]]) -> list[str]:
    frozen_ids = {str(record["id"]) for record in frozen}
    return sorted(str(candidate["id"]) for candidate in candidates if str(candidate["id"]) not in frozen_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate the existing output without rewriting it")
    args = parser.parse_args()

    candidates = load_jsonl(args.candidates)
    with open_database_read_only(args.database) as connection:
        if args.check:
            frozen = load_jsonl(args.output)
            bibliography_status = bibliography_001_status(connection)
        else:
            frozen, bibliography_status = freeze(candidates, connection)
            write_jsonl_atomic(args.output, frozen)
            # Re-parse exactly what was written before reporting success.
            frozen = load_jsonl(args.output)
        report = validate(frozen, connection)

    report["actual_drop_ids"] = actual_drop_ids(candidates, frozen)
    report["audit_modify_ids"] = sorted(AUDIT_MODIFY_IDS)
    report["additional_tighten_ids"] = sorted(ADDITIONAL_TIGHTEN_IDS)
    report["actual_modified_ids"] = sorted(AUDIT_MODIFY_IDS | ADDITIONAL_TIGHTEN_IDS)
    report["bibliography_001"] = bibliography_status
    report["output"] = str(args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
