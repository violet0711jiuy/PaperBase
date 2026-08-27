"""从 Golden Dataset v1 生成问题自包含的 v1.1 数据集。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "eval" / "datasets" / "golden_dataset_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "datasets" / "golden_dataset_v1_1.jsonl"

# 本次只为问题补充目标论文、模型或方法锚点，不改变问题要考查的事实。
QUESTION_REWRITES = {
    "fact_001": "LSTM-EMVE生成90%预测区间时，量化违反（quantile violation）阈值设置为多少？",
    "fact_002": "LSTM-EMVE用于估计标准差的三个并行分支分别采用了哪些激活函数？",
    "fact_005": "在成都秋冬PM2.5研究的高气压底条件下，CMAQ模型对细颗粒物浓度的24小时预报相关系数是多少？",
    "fact_010": "D²STGNN中的估计门（Estimation Gate）主要作用是什么？",
    "fact_011": "D²STGNN的Inherent Model使用哪两类模块分别捕捉局部和全局时间依赖？",
    "method_001": "In the LSTM-EMVE wind-speed forecasting framework, what is the purpose of the branched ensemble architecture with multi-activation functions, and how does it enhance uncertainty estimation?",
    "method_004": "In the Chengdu autumn-winter PM2.5 study, how did the COST733 workflow derive objective synoptic weather patterns from ERA5 sea-level pressure and 10 m wind fields?",
    "method_005": "成都秋冬PM2.5研究中的WRF与CMAQ如何衔接，以生成72小时预报？",
    "method_006": "成都秋冬PM2.5研究如何把观测值与CMAQ及气象预报配对，并用什么指标判断预报准确性？",
    "method_010": "在D²STGNN的DSTF框架中，残差分解机制如何通过两个残差链接分离扩散信号和固有信号？",
    "method_011": "D²STGNN的估计门（Estimation Gate）如何利用时间槽和节点嵌入自动估算扩散信号的比例？",
    "experiment_003": "What time periods and station-aggregated data did the Chengdu autumn-winter PM2.5 study use for COST733 weather-pattern classification and forecast evaluation?",
    "experiment_004": "在成都秋冬PM2.5研究中，为什么低气压天气没有进入CMAQ预报准确性的进一步分组分析？其余天气模式评估了哪些预报时效？",
    "result_003": "在成都秋冬PM2.5研究的高气压底条件下，CMAQ的24、48和72小时预报相关系数分别是多少？整体上哪个时效更准确？",
    "synthesis_001": "LSTM-EMVE如何通过损失函数设计与branched multi-activation architecture协同实现点预测和预测区间的联合优化？",
    "bibliography_001": "在LSTM-EMVE风速预测论文的参考文献中，2009年的风速与风电预测综述和2015年的支持向量回归风速预测研究分别是哪两篇？",
    "bibliography_003": "在成都秋冬PM2.5预报论文的参考文献中，哪篇2019年的研究直接分析了2001—2016年成都和重庆冬季大气消光与环流、气象参数的关系？",
    "bibliography_005": "在ESDTW论文的参考文献中，Itakura 1975年的语音识别论文与shapeDTW原始论文各自的题名、期刊和年份是什么？",
    "bibliography_006": "在D²STGNN论文的参考文献中，DCRNN与GMAN两篇交通预测工作的完整题名、作者和发表年份是什么？",
    "unanswerable_006": "D²STGNN论文是否评估了模型在不同城市之间的迁移能力，并给出跨域微调的具体策略和结果？",
}

# rebuild 后 SQLite 已确认的无歧义页码修正；chunk_id、paper_id 和 section 均未变化。
EVIDENCE_LOCATION_UPDATES = {
    "fact_007": {0: {"page_end": 8}},
    "fact_010": {0: {"page_end": 4}},
    "method_010": {0: {"page_start": 4}},
}

# 每篇论文至少出现一个稳定锚点，避免依赖隐藏的 paper_id 或会话上下文。
PAPER_ANCHORS = {
    "paper_5b6a1007fa7514bf": ("LSTM-EMVE", "EMVE"),
    "paper_b12197625a863197": ("成都", "Chengdu", "COST733", "WRF", "CMAQ"),
    "paper_b7a064b63171eaee": ("ESDTW",),
    "paper_c162376bc253ae7d": ("D²STGNN", "D2STGNN"),
}

REQUIRED_FIELDS = {
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
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在行号级别报告解析错误。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON 解析失败：{error}") from error
    return records


def build_records(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """复制 v1，改写问题并同步已由 Validator 确认的纯定位元数据。"""
    records = copy.deepcopy(source_records)
    known_ids = {str(record.get("id")) for record in records}
    missing_rewrites = sorted(set(QUESTION_REWRITES) - known_ids)
    if missing_rewrites:
        raise ValueError(f"v1 中缺少待改写 ID：{missing_rewrites}")

    for record in records:
        case_id = str(record["id"])
        if case_id in QUESTION_REWRITES:
            record["question"] = QUESTION_REWRITES[case_id]
        _apply_evidence_location_updates(record)
    return records


def _apply_evidence_location_updates(record: dict[str, Any]) -> None:
    """按 Case 和 Evidence 索引应用确定性的 section/page 定位修正。"""
    case_id = str(record.get("id"))
    for evidence_index, fields in EVIDENCE_LOCATION_UPDATES.get(case_id, {}).items():
        evidence = record["relevant_evidence"][evidence_index]
        evidence.update(fields)


def validate_records(
    source_records: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    """验证数量、Schema、自包含锚点及改动严格符合既定清单。"""
    if len(records) != 40:
        raise ValueError(f"预期 40 条 Golden，实际为 {len(records)} 条")

    ids = [str(record.get("id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Golden Dataset 中存在重复 ID")

    source_by_id = {str(record["id"]): record for record in source_records}
    for record in records:
        case_id = str(record.get("id"))
        missing_fields = sorted(REQUIRED_FIELDS - set(record))
        if missing_fields:
            raise ValueError(f"{case_id} 缺少字段：{missing_fields}")

        paper_id = str(record["paper_id"])
        anchors = PAPER_ANCHORS.get(paper_id)
        if not anchors or not any(anchor in str(record["question"]) for anchor in anchors):
            raise ValueError(f"{case_id} 的问题缺少目标论文/模型锚点")

        # 每条输出必须等于“原记录 + 审核过的问题改写 + 安全定位修正”。
        source = source_by_id.get(case_id)
        if source is None:
            raise ValueError(f"v1.1 出现 v1 中不存在的 ID：{case_id}")
        expected = copy.deepcopy(source)
        if case_id in QUESTION_REWRITES:
            expected["question"] = QUESTION_REWRITES[case_id]
        _apply_evidence_location_updates(expected)
        if record != expected:
            raise ValueError(f"{case_id} 出现改写清单外的字段变化")


def serialize_jsonl(records: list[dict[str, Any]]) -> str:
    """以稳定的 UTF-8 单行 JSON 格式序列化数据集。"""
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析源文件、输出文件和只检查模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只验证现有 v1.1 是否与生成规则一致，不写文件。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """生成 v1.1，或在 --check 模式下验证现有文件。"""
    args = parse_args(argv)
    source_records = load_jsonl(args.source)
    expected_records = build_records(source_records)
    validate_records(source_records, expected_records)
    expected_text = serialize_jsonl(expected_records)

    if args.check:
        if not args.output.exists():
            raise FileNotFoundError(f"待检查文件不存在：{args.output}")
        actual_records = load_jsonl(args.output)
        validate_records(source_records, actual_records)
        if serialize_jsonl(actual_records) != expected_text:
            raise ValueError("现有 v1.1 与预期改写结果不一致")
        print(f"PASS：{args.output} 共 {len(actual_records)} 条，内容与生成规则一致。")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected_text, encoding="utf-8", newline="\n")
    print(
        f"已生成 {args.output}：共 {len(expected_records)} 条，"
        f"改写 question {len(QUESTION_REWRITES)} 条，"
        f"安全同步 Evidence 定位 metadata {len(EVIDENCE_LOCATION_UPDATES)} 条。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
