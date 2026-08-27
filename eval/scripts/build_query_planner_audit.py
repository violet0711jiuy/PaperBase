"""基于现有 Full Retrieval trace 生成 Query Planner Audit。

本脚本只读取 ``case_results.jsonl``，不调用 LLM、Embedding、FAISS、BM25 或
Reranker，也不创建新的 Golden Dataset。自动指标从 40 条 trace 直接计算；人工判断
绑定到已审阅 trace 的 SHA256，防止新一轮 Rewrite 输出变化后继续套用旧结论。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperbase.retrieval.lexical_terms import LEXICAL_STOPWORDS


DEFAULT_CASE_RESULTS = ROOT / "eval" / "results" / "retrieval_full" / "case_results.jsonl"
DEFAULT_JSON_REPORT = (
    ROOT / "eval" / "results" / "retrieval_full" / "query_planner_audit.json"
)
DEFAULT_MARKDOWN_REPORT = (
    ROOT / "eval" / "results" / "retrieval_full" / "query_planner_audit.md"
)

# 这份人工复核只对应修复后 Planner-only Audit 的 40 条 trace。只要原文件
# 发生变化，自动统计仍可重算，但人工字段必须重新审阅，不能静默沿用。
MANUAL_REVIEW_SOURCE_SHA256 = (
    "eb1c5bb330d38f17fbf361f2eb9a01a3290adac0e4cbe6933986d3b8e2988ba8"
)

# 保留修复前人工发现，作为 before/after 依据；修复后结论使用下方独立字典，绝不复用。
BEFORE_FIX_MANUAL_FINDINGS: dict[str, dict[str, Any]] = {
    "fact_001": {
        "entities_lost": ["90%", "quantile violation"],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "Rewrite 降级后 semantic query 与 lexical keywords 均为空，关键阈值和术语只剩 Original Dense 可用。",
    },
    "method_005": {
        "entities_lost": ["PM2.5", "72-hour"],
        "meaning_changed": None,
        "noise_introduced": ["PM2"],
        "malformed_or_contaminated": True,
        "note": "WRF、CMAQ 被保留，但 PM2.5 被拆成 PM2，72 小时约束未进入关键词。",
    },
    "method_007": {
        "entities_lost": ["LEDTW", "shapeDTW"],
        "meaning_changed": None,
        "noise_introduced": ["How"],
        "malformed_or_contaminated": False,
        "note": "How 占用最多 3 个关键词中的一个位置，导致两个关键比较对象被截断。",
    },
    "experiment_001": {
        "entities_lost": [],
        "meaning_changed": None,
        "noise_introduced": ["What"],
        "malformed_or_contaminated": False,
        "note": "LSTM-EMVE 被保留，但 What 是无区分度的问句噪声。",
    },
    "experiment_004": {
        "entities_lost": [],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "仅 CMAQ 进入关键词；低气压原因与其余预报时效仍只能依赖 Original Dense。",
    },
    "result_001": {
        "entities_lost": [],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "LSTM-EMVE 与 CWC 均被正确保留，是 degraded fallback 中质量较好的一条。",
    },
    "result_006": {
        "entities_lost": ["ESDTW", "84", "UCR", "optimal-window DTW qualifier"],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "前三个名额被 ED、DTW、DDTW 占满，主方法、数据集规模和窗口限定丢失。",
    },
    "result_008": {
        "entities_lost": ["D²STGNN†", "D²STGNN‡"],
        "meaning_changed": None,
        "noise_introduced": ["D", "STGNN"],
        "malformed_or_contaminated": True,
        "note": "模型名被拆坏，并丢失区分两个消融变体所必需的 †/‡ 标记。",
    },
    "bibliography_003": {
        "entities_lost": ["2019", "2001", "2016", "Chengdu", "Chongqing"],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "Planner 没有生成语义查询或关键词；本次命中来自后置 Bibliography 年份 fallback，而非 Planner 本身。",
    },
    "unanswerable_002": {
        "entities_lost": [],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "COST733、WRF、CMAQ 均保留；成都和春夏季约束仍只存在于 resolved query。",
    },
    "unanswerable_006": {
        "entities_lost": [],
        "meaning_changed": None,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "没有可提取英文实体，semantic query 与关键词均缺失，只剩 Original Dense。",
    },
    "fact_008": {
        "entities_lost": [],
        "meaning_changed": True,
        "noise_introduced": ["的眼动实验数据集？"],
        "malformed_or_contaminated": True,
        "note": "英文复杂度问题末尾混入无关中文眼动数据集问题，是明确的跨语种污染。",
    },
    "experiment_007": {
        "entities_lost": [],
        "meaning_changed": True,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "“评估时效/预测时域”被改写成 evaluation timeliness，检索含义发生漂移。",
    },
    "bibliography_001": {
        "entities_lost": [],
        "meaning_changed": True,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "改写把两篇不同工作部分合并为共同 review/using SVR 描述，弱化了分别查找两篇文献的结构。",
    },
    "bibliography_006": {
        "entities_lost": [],
        "meaning_changed": True,
        "noise_introduced": [],
        "malformed_or_contaminated": False,
        "note": "把 DCRNN、GMAN 两个工作简称表述为论文题名，属于轻微但真实的语义假设。",
    },
}

BEFORE_FIX_DEGRADED_CASE_IDS = {
    "fact_001",
    "method_005",
    "method_007",
    "experiment_001",
    "experiment_004",
    "result_001",
    "result_006",
    "result_008",
    "bibliography_003",
    "unanswerable_002",
    "unanswerable_006",
}

# 修复后的人工审阅只记录仍未被程序防护拦截的问题；空字典表示 40 条均未发现残留问题。
MANUAL_FINDINGS: dict[str, dict[str, Any]] = {}

BEFORE_FIX_BASELINE = {
    "success_rate": 0.725,
    "degraded_rate": 0.275,
    "semantic_query_coverage_rate": 0.725,
    "lexical_keyword_coverage_rate": 0.2,
    "success_cases_with_lexical_keywords": 0,
    "entity_preservation_errors": 6,
    "meaning_changed": 4,
    "noise_introduced": 5,
    "malformed_or_contaminated": 3,
    "avg_rewrite_latency_ms": 1993.944,
}

REGRESSION_ENTITY_REQUIREMENTS = {
    "method_005": {"PM2.5", "72-hour"},
    "method_007": {"ESDTW", "DTW", "LEDTW", "shapeDTW"},
    "result_006": {"ESDTW", "DTW", "DDTW", "UCR"},
    "result_008": {"D²STGNN†", "D²STGNN‡"},
}


class PlannerAuditInputError(RuntimeError):
    """表示 trace 缺失、损坏或与人工复核快照不一致。"""


def load_case_results(path: Path) -> tuple[list[dict[str, Any]], str]:
    """读取 JSONL 并返回记录及文件指纹，坏行、空集和重复 ID 立即失败。"""
    if not path.is_file():
        raise PlannerAuditInputError(f"Case results do not exist: {path}")
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PlannerAuditInputError(
                f"Invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise PlannerAuditInputError(
                f"Case result row {line_number} must be an object"
            )
        records.append(value)
    if not records:
        raise PlannerAuditInputError("Case results are empty")
    ids = [str(record.get("case_id", "")) for record in records]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if "" in ids or duplicates:
        raise PlannerAuditInputError(f"Missing or duplicate case IDs: {duplicates}")
    return records, hashlib.sha256(raw).hexdigest()


def _nonempty_text(value: Any) -> bool:
    """统一判断可选字符串是否真正包含内容。"""
    return isinstance(value, str) and bool(value.strip())


def _keywords(record: Mapping[str, Any]) -> list[str]:
    """清理 trace 中的关键词字段，避免字符串被误当作字符序列。"""
    value = record.get("lexical_keywords_en")
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _diagnostics(record: Mapping[str, Any]) -> list[str]:
    """读取 Planner 程序诊断，坏类型按空列表处理。"""
    value = record.get("validation_diagnostics")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def build_planner_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """计算 Planner 状态、输出覆盖率和 Rewrite 延迟等确定性指标。"""
    total = len(records)
    statuses = Counter(str(record.get("rewrite_status", "unknown")) for record in records)
    success = statuses.get("success", 0)
    partial = statuses.get("partial", 0)
    degraded = statuses.get("degraded", 0)
    semantic_cases = [record for record in records if _nonempty_text(record.get("semantic_query_en"))]
    lexical_cases = [record for record in records if _keywords(record)]
    success_with_keywords = [
        record
        for record in records
        if record.get("rewrite_status") == "success" and _keywords(record)
    ]
    latencies = []
    for record in records:
        latency = record.get("latency")
        if isinstance(latency, Mapping):
            value = latency.get("rewrite_latency_ms")
            if isinstance(value, (int, float)):
                latencies.append(float(value))
    return {
        "total_cases": total,
        "success_cases": success,
        "success_rate": round(success / total, 6) if total else 0.0,
        "partial_cases": partial,
        "partial_rate": round(partial / total, 6) if total else 0.0,
        "degraded_cases": degraded,
        "degraded_rate": round(degraded / total, 6) if total else 0.0,
        "not_run_cases": statuses.get("not_run", 0),
        "semantic_query_coverage": {
            "cases": len(semantic_cases),
            "total": total,
            "rate": round(len(semantic_cases) / total, 6) if total else 0.0,
        },
        "lexical_keyword_coverage": {
            "cases": len(lexical_cases),
            "total": total,
            "rate": round(len(lexical_cases) / total, 6) if total else 0.0,
        },
        "success_cases_with_lexical_keywords": {
            "cases": len(success_with_keywords),
            "success_total": success,
            "rate": round(len(success_with_keywords) / success, 6) if success else 0.0,
        },
        "avg_rewrite_latency_ms": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "latency_observation_count": len(latencies),
        "semantic_status_distribution": dict(
            sorted(Counter(str(record.get("semantic_status", "unknown")) for record in records).items())
        ),
        "lexical_status_distribution": dict(
            sorted(Counter(str(record.get("lexical_status", "unknown")) for record in records).items())
        ),
    }


def build_correctness_gates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """计算不依赖人工猜测的 correctness gate，并给出失败 Case。"""
    valid_contaminated = [
        str(record.get("case_id"))
        for record in records
        if record.get("semantic_status") == "valid"
        and (
            re.search(r"[\u3400-\u9fff]", str(record.get("semantic_query_en") or ""))
            or any(
                diagnostic.startswith("semantic_")
                and diagnostic not in {"semantic_entity_missing"}
                for diagnostic in _diagnostics(record)
            )
        )
    ]
    stopword_cases = [
        str(record.get("case_id"))
        for record in records
        if any(keyword.casefold() in LEXICAL_STOPWORDS for keyword in _keywords(record))
    ]
    regression_failures: dict[str, list[str]] = {}
    records_by_id = {str(record.get("case_id")): record for record in records}
    for case_id, required in REGRESSION_ENTITY_REQUIREMENTS.items():
        actual = set(_keywords(records_by_id.get(case_id, {})))
        missing = sorted(required - actual)
        if missing:
            regression_failures[case_id] = missing
    metrics = build_planner_metrics(records)
    checks = {
        "malformed_or_contaminated_reaching_valid_semantic": {
            "passed": not valid_contaminated,
            "case_ids": valid_contaminated,
        },
        "lexical_question_stopwords": {
            "passed": not stopword_cases,
            "case_ids": stopword_cases,
        },
        "regression_entity_preservation": {
            "passed": not regression_failures,
            "failures": regression_failures,
        },
        "success_has_lexical": {
            "passed": metrics["success_cases"] > 0
            and metrics["success_cases_with_lexical_keywords"]["cases"]
            == metrics["success_cases"],
        },
        "lexical_coverage_improved": {
            "passed": metrics["lexical_keyword_coverage"]["rate"]
            > BEFORE_FIX_BASELINE["lexical_keyword_coverage_rate"],
        },
        "degraded_rate_reduced": {
            "passed": metrics["degraded_rate"] < BEFORE_FIX_BASELINE["degraded_rate"],
        },
        "rewrite_latency_not_significantly_higher": {
            "passed": metrics["avg_rewrite_latency_ms"] is not None
            and metrics["avg_rewrite_latency_ms"]
            <= BEFORE_FIX_BASELINE["avg_rewrite_latency_ms"] * 1.25,
        },
    }
    return {
        "passed": all(bool(item["passed"]) for item in checks.values()),
        "checks": checks,
    }


def build_manual_review_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """把人工判断与 Planner 原始字段合并为一条可审计记录。"""
    case_id = str(record.get("case_id"))
    finding = MANUAL_FINDINGS.get(case_id, {})
    return {
        "case_id": case_id,
        "rewrite_status": str(record.get("rewrite_status", "unknown")),
        "semantic_status": str(record.get("semantic_status", "unknown")),
        "lexical_status": str(record.get("lexical_status", "unknown")),
        "original_query": str(record.get("original_query") or ""),
        "resolved_query": str(record.get("resolved_query") or ""),
        "semantic_query": (
            str(record.get("semantic_query_en"))
            if _nonempty_text(record.get("semantic_query_en"))
            else None
        ),
        "lexical_keywords": _keywords(record),
        "validation_diagnostics": _diagnostics(record),
        "entities_lost": list(finding.get("entities_lost", [])),
        # degraded 没有 semantic query，无法判断“改写后含义是否变化”，因此使用 null。
        "meaning_changed": finding.get("meaning_changed", False),
        "noise_introduced": list(finding.get("noise_introduced", [])),
        "malformed_or_contaminated": bool(
            finding.get("malformed_or_contaminated", False)
        ),
        "manual_note": str(finding.get("note", "人工检查未发现明显问题。")),
    }


def _manual_issue_summary(review_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按人工字段统计错误 Case，避免把同一 Case 的多个实体重复计数。"""
    entity_errors = [row for row in review_rows if row.get("entities_lost")]
    meaning_changes = [row for row in review_rows if row.get("meaning_changed") is True]
    noise_cases = [row for row in review_rows if row.get("noise_introduced")]
    malformed = [row for row in review_rows if row.get("malformed_or_contaminated") is True]
    return {
        "entity_preservation_errors": {
            "cases": len(entity_errors),
            "case_ids": [str(row["case_id"]) for row in entity_errors],
        },
        "meaning_changed": {
            "cases": len(meaning_changes),
            "case_ids": [str(row["case_id"]) for row in meaning_changes],
        },
        "noise_introduced": {
            "cases": len(noise_cases),
            "case_ids": [str(row["case_id"]) for row in noise_cases],
        },
        "malformed_or_contaminated_rewrite": {
            "cases": len(malformed),
            "case_ids": [str(row["case_id"]) for row in malformed],
        },
    }


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    """组合自动指标、人工快照状态、11 条 degraded 详审和全量审阅记录。"""
    review_rows = [build_manual_review_row(record) for record in records]
    non_success_rows = [
        row for row in review_rows if row["rewrite_status"] in {"partial", "degraded"}
    ]
    snapshot_matches = (
        source_sha256 == MANUAL_REVIEW_SOURCE_SHA256
        and len(records) == 40
    )
    manual_summary = _manual_issue_summary(review_rows) if snapshot_matches else None
    correctness_gates = build_correctness_gates(records)
    manual_passed = bool(
        manual_summary
        and all(item["cases"] == 0 for item in manual_summary.values())
    )
    audit_passed = snapshot_matches and correctness_gates["passed"] and manual_passed
    return {
        "status": "PASS" if audit_passed else (
            "NEEDS_FIX" if snapshot_matches else "MANUAL_REVIEW_STALE"
        ),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_case_results": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "manual_review": {
            "snapshot_matches": snapshot_matches,
            "reviewed_source_sha256": MANUAL_REVIEW_SOURCE_SHA256,
            "reviewed_cases": len(review_rows) if snapshot_matches else 0,
            "non_success_cases_reviewed": len(non_success_rows) if snapshot_matches else 0,
            "definitions": {
                "entity_preservation_error": (
                    "原问题中的模型名、缩写、年份、数值约束或高区分度技术词，"
                    "未被 semantic_query_en 或 lexical_keywords_en 正确保留。"
                ),
                "meaning_changed": "semantic query 增加、删减或误译了会改变检索目标的含义。",
                "noise_introduced": "Planner 输出包含原问题没有且不利于检索的词或片段。",
                "malformed_or_contaminated": "输出被错误拆词，或混入无关语言/问题片段。",
            },
            "issue_summary": manual_summary,
        },
        "before_fix_baseline": BEFORE_FIX_BASELINE,
        "planner_metrics": build_planner_metrics(records),
        "correctness_gates": correctness_gates,
        "non_success_case_review": non_success_rows if snapshot_matches else [],
        "all_case_review": review_rows if snapshot_matches else [],
    }


def _display_boolean(value: Any) -> str:
    """将三态人工字段渲染为 Yes、No 或 N/A。"""
    if value is None:
        return "N/A"
    return "Yes" if value is True else "No"


def _display_list(values: Sequence[Any]) -> str:
    """将人工列表转成适合 Markdown 的紧凑文本。"""
    return ", ".join(str(value) for value in values) if values else "None"


def render_markdown(report: Mapping[str, Any]) -> str:
    """将 Planner Audit 渲染为便于人工复核的 Markdown。"""
    metrics = report["planner_metrics"]
    manual = report["manual_review"]
    lines = [
        "# Query Planner Audit",
        "",
        f"**Status: {report['status']}**",
        "",
        f"- Source: `{report['source_case_results']}`",
        f"- Source SHA256: `{report['source_sha256']}`",
        f"- Manual snapshot matches: `{manual['snapshot_matches']}`",
        "",
        "## Planner Metrics",
        "",
        "| Planner Metric | Result |",
        "|---|---:|",
        f"| Success Rate | {metrics['success_rate']:.1%} ({metrics['success_cases']}/{metrics['total_cases']}) |",
        f"| Partial Rate | {metrics['partial_rate']:.1%} ({metrics['partial_cases']}/{metrics['total_cases']}) |",
        f"| Degraded Rate | {metrics['degraded_rate']:.1%} ({metrics['degraded_cases']}/{metrics['total_cases']}) |",
        "| Semantic Query Coverage | "
        f"{metrics['semantic_query_coverage']['cases']}/{metrics['semantic_query_coverage']['total']} |",
        "| Lexical Keyword Coverage | "
        f"{metrics['lexical_keyword_coverage']['cases']}/{metrics['lexical_keyword_coverage']['total']} |",
        "| Success cases with lexical keywords | "
        f"{metrics['success_cases_with_lexical_keywords']['cases']}/"
        f"{metrics['success_cases_with_lexical_keywords']['success_total']} |",
        f"| Avg Rewrite Latency | {metrics['avg_rewrite_latency_ms']:.3f} ms |",
    ]
    issue_summary = manual.get("issue_summary")
    if issue_summary:
        lines.extend(
            [
                "| Entity Preservation Errors | "
                f"{issue_summary['entity_preservation_errors']['cases']} |",
                "| Meaning Changed | " f"{issue_summary['meaning_changed']['cases']} |",
                "| Noise Introduced | " f"{issue_summary['noise_introduced']['cases']} |",
                "| Malformed / Contaminated Rewrite | "
                f"{issue_summary['malformed_or_contaminated_rewrite']['cases']} |",
            ]
        )
    if not manual["snapshot_matches"]:
        lines.extend(
            [
                "",
                "## Manual Review Stale",
                "",
                "当前 trace 与已审阅快照不一致。自动指标有效，但实体、语义和噪声字段必须重新人工检查。",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "",
            "## Correctness Gates",
            "",
            "| Gate | Result |",
            "|---|---:|",
        ]
    )
    for name, item in report["correctness_gates"]["checks"].items():
        lines.append(f"| {name} | {'PASS' if item['passed'] else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## Audit Definitions",
            "",
            "Entity Preservation Errors 只评估 Planner 新增的 semantic/lexical 检索补充；"
            "resolved_query 仍会进入 Original Dense，因此不表示原问题已从整个检索链路消失。",
            "",
            "## Partial / Degraded Cases",
            "",
        ]
    )
    for row in report["non_success_case_review"]:
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                f"- Original Query: `{row['original_query']}`",
                f"- Semantic Query: `{row['semantic_query'] or 'None'}`",
                f"- Lexical Keywords: `{_display_list(row['lexical_keywords'])}`",
                f"- Semantic Status: `{row['semantic_status']}`",
                f"- Lexical Status: `{row['lexical_status']}`",
                f"- Validation Diagnostics: `{_display_list(row['validation_diagnostics'])}`",
                f"- Entities lost?: `{_display_list(row['entities_lost'])}`",
                f"- Meaning changed?: `{_display_boolean(row['meaning_changed'])}`",
                f"- Noise introduced?: `{_display_list(row['noise_introduced'])}`",
                f"- Malformed / contaminated?: `{_display_boolean(row['malformed_or_contaminated'])}`",
                f"- Review: {row['manual_note']}",
                "",
            ]
        )

    exception_rows = [
        row
        for row in report["all_case_review"]
        if row["rewrite_status"] == "success"
        and (
            row["entities_lost"]
            or row["meaning_changed"] is True
            or row["noise_introduced"]
            or row["malformed_or_contaminated"]
        )
    ]
    lines.extend(["## Success Cases Requiring Attention", ""])
    for row in exception_rows:
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                f"- Original Query: `{row['original_query']}`",
                f"- Semantic Query: `{row['semantic_query']}`",
                f"- Lexical Keywords: `{_display_list(row['lexical_keywords'])}`",
                f"- Entities lost?: `{_display_list(row['entities_lost'])}`",
                f"- Meaning changed?: `{_display_boolean(row['meaning_changed'])}`",
                f"- Noise introduced?: `{_display_list(row['noise_introduced'])}`",
                f"- Malformed / contaminated?: `{_display_boolean(row['malformed_or_contaminated'])}`",
                f"- Review: {row['manual_note']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Full 40-case Review Index",
            "",
            "| Case | Overall | Semantic Status | Lexical Status | Semantic | Keywords | Entity Error | Meaning Changed | Noise | Malformed |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["all_case_review"]:
        lines.append(
            f"| {row['case_id']} | {row['rewrite_status']} | "
            f"{row['semantic_status']} | {row['lexical_status']} | "
            f"{'Yes' if row['semantic_query'] else 'No'} | {len(row['lexical_keywords'])} | "
            f"{'Yes' if row['entities_lost'] else 'No'} | "
            f"{_display_boolean(row['meaning_changed'])} | "
            f"{'Yes' if row['noise_introduced'] else 'No'} | "
            f"{_display_boolean(row['malformed_or_contaminated'])} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_json_atomic(path: Path, value: Any) -> None:
    """通过同目录临时文件原子写入 JSON，避免中断后留下半份报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    """通过同目录临时文件原子写入 Markdown 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 trace 与两份审计报告路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-results", type=Path, default=DEFAULT_CASE_RESULTS)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：生成自动统计，并仅在指纹一致时附加人工审计结论。"""
    args = parse_args(argv)
    try:
        records, source_sha256 = load_case_results(args.case_results)
        report = build_report(
            records,
            source_path=args.case_results,
            source_sha256=source_sha256,
        )
        write_json_atomic(args.json_report, report)
        write_text_atomic(args.markdown_report, render_markdown(report))
    except (OSError, UnicodeError, ValueError, PlannerAuditInputError) as error:
        print(f"ERROR: {error}")
        return 2
    print(report["status"])
    print(f"Cases: {report['planner_metrics']['total_cases']}")
    print(f"JSON report: {args.json_report.resolve()}")
    print(f"Markdown report: {args.markdown_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
