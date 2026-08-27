"""基于 Full Retrieval Case trace 生成确定性的 Retrieval Diagnostic Report。

本脚本只读取 ``case_results.jsonl``，不调用 Query Rewrite、Embedding、FAISS、BM25
或 Reranker，也不会修改正式 Retrieval 链路。所有指标都从已保存的阶段排名与 Golden
Ground Truth chunk IDs 直接计算。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scripts.run_retrieval_eval import (
    aggregate_ranking_metrics,
    build_expansion_evaluation,
)


DEFAULT_CASE_RESULTS = (
    ROOT / "eval" / "results" / "retrieval_full" / "case_results.jsonl"
)
DEFAULT_JSON_REPORT = (
    ROOT / "eval" / "results" / "retrieval_full" / "diagnostic_report.json"
)
DEFAULT_MARKDOWN_REPORT = (
    ROOT / "eval" / "results" / "retrieval_full" / "diagnostic_report.md"
)

TYPE_SLICES = ("fact", "method", "experiment", "result", "synthesis")
DIFFICULTY_SLICES = ("easy", "medium", "hard")
HOP_SLICES = ("single_hop", "multi_hop")
ROUTE_DEFINITIONS = (
    ("original_dense", "Original Dense", True),
    ("original_bm25", "Original BM25", False),
    ("rewritten_dense", "Rewritten Dense", True),
    ("rewritten_bm25", "Rewritten BM25", True),
)


class DiagnosticInputError(RuntimeError):
    """表示 Case trace 缺失、损坏或不能安全用于诊断。"""


def load_case_results(path: Path) -> list[dict[str, Any]]:
    """严格读取 Case JSONL，坏行或重复 ID 立即失败，避免改变诊断分母。"""
    if not path.is_file():
        raise DiagnosticInputError(f"Case results do not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DiagnosticInputError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise DiagnosticInputError(
                    f"Case result row {line_number} must be an object"
                )
            records.append(value)
    ids = [record.get("case_id") for record in records]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise DiagnosticInputError(f"Duplicate case IDs: {duplicates}")
    return records


def valid_content_cases(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """选择可参与正文 Retrieval 指标的 answerable、非 bibliography、有效 Case。"""
    return [
        record
        for record in records
        if record.get("answerable") is True
        and record.get("expected_bibliography_intent") is False
        and record.get("invalid_ground_truth") is False
        and record.get("retrieval_error") is None
    ]


def _slice_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    values: Sequence[str],
    selector: str,
) -> dict[str, dict[str, float | int]]:
    """按类型、难度或 Hop 输出最终 Top-5 的固定 Slice 指标。"""
    result: dict[str, dict[str, float | int]] = {}
    for value in values:
        if selector == "primary_type":
            selected = [record for record in records if record.get("primary_type") == value]
        elif selector == "tag":
            selected = [
                record
                for record in records
                if isinstance(record.get("tags"), list) and value in record["tags"]
            ]
        else:
            raise ValueError(f"Unsupported selector: {selector}")
        result[value] = aggregate_ranking_metrics(
            selected,
            ranking_field="final_results",
            cutoffs=(5,),
            mrr_k=5,
        )
    return result


def build_slice_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """汇总 Type、Difficulty 与 Hop 三组 Slice Evaluation。"""
    return {
        "type": _slice_metrics(records, values=TYPE_SLICES, selector="primary_type"),
        "difficulty": _slice_metrics(
            records, values=DIFFICULTY_SLICES, selector="tag"
        ),
        "hop": _slice_metrics(records, values=HOP_SLICES, selector="tag"),
    }


def build_route_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """计算每条正式召回路径的可用率及 Overall/Available-only Hit、Recall。"""
    result: dict[str, dict[str, Any]] = {}
    for field, label, implemented in ROUTE_DEFINITIONS:
        available = [
            record
            for record in records
            if isinstance(record.get("stage_availability"), dict)
            and record["stage_availability"].get(field) is True
        ]
        route: dict[str, Any] = {
            "label": label,
            "implemented": implemented,
            "eligible_cases": len(records),
            "available_cases": len(available),
            "availability_rate": (
                round(len(available) / len(records), 6) if records else 0.0
            ),
        }
        if not implemented:
            route["overall"] = None
            route["when_available"] = None
        else:
            # Overall 将路径缺失视为未命中，反映 Rewrite 降级对真实产品的影响。
            route["overall"] = aggregate_ranking_metrics(
                records,
                ranking_field=field,
                cutoffs=(5, 10, 20),
                mrr_k=20,
            )
            route["when_available"] = (
                aggregate_ranking_metrics(
                    available,
                    ranking_field=field,
                    cutoffs=(5, 10, 20),
                    mrr_k=20,
                )
                if available
                else None
            )
        result[field] = route
    return result


def build_rrf_curve(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """输出 RRF 在 5、10、20、40 深度的 Hit 与 Recall 曲线。"""
    metrics = aggregate_ranking_metrics(
        records,
        ranking_field="rrf_results",
        cutoffs=(5, 10, 20, 40),
        mrr_k=40,
    )
    return {
        f"at_{cutoff}": {
            "hit": float(metrics[f"hit_at_{cutoff}"]),
            "recall": float(metrics[f"recall_at_{cutoff}"]),
        }
        for cutoff in (5, 10, 20, 40)
    }


def build_reranker_comparison(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """在同一 K 下比较 RRF 与 Reranker，并输出绝对指标差值。"""
    result: dict[str, dict[str, Any]] = {}
    for cutoff in (5, 10, 20):
        rrf = aggregate_ranking_metrics(
            records,
            ranking_field="rrf_results",
            cutoffs=(cutoff,),
            mrr_k=cutoff,
        )
        reranker = aggregate_ranking_metrics(
            records,
            ranking_field="reranked_results",
            cutoffs=(cutoff,),
            mrr_k=cutoff,
        )
        keys = (f"hit_at_{cutoff}", f"recall_at_{cutoff}", f"mrr_at_{cutoff}")
        result[f"at_{cutoff}"] = {
            "rrf": {key: rrf[key] for key in keys},
            "reranker": {key: reranker[key] for key in keys},
            "delta": {
                key: round(float(reranker[key]) - float(rrf[key]), 6)
                for key in keys
            },
        }
    return result


def build_rewrite_diagnostics(
    all_records: Sequence[Mapping[str, Any]],
    content_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """统计 Rewrite 状态分布，并比较各状态下最终 Top-5 表现。"""
    def distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
        counts = Counter(str(record.get("rewrite_status", "unknown")) for record in records)
        return {
            status: {
                "cases": counts.get(status, 0),
                "rate": round(counts.get(status, 0) / len(records), 6) if records else 0.0,
            }
            for status in ("success", "partial", "degraded", "not_run", "unknown")
        }

    by_status: dict[str, Any] = {}
    for status in ("success", "partial", "degraded", "not_run", "unknown"):
        selected = [
            record
            for record in content_records
            if str(record.get("rewrite_status", "unknown")) == status
        ]
        by_status[status] = {
            **aggregate_ranking_metrics(
                selected,
                ranking_field="final_results",
                cutoffs=(5,),
                mrr_k=5,
            ),
            "case_ids": [str(record.get("case_id")) for record in selected],
        }
    return {
        "all_cases": distribution(all_records),
        "content_cases": distribution(content_records),
        "content_metrics_by_status": by_status,
    }


def _cross_paper_stage(
    records: Sequence[Mapping[str, Any]], *, field: str, cutoff: int
) -> dict[str, Any]:
    """计算一个阶段前 K 条候选中指向其他论文的数量、比例与受影响 Case。"""
    total = 0
    cross_paper = 0
    affected: list[str] = []
    for record in records:
        target_paper = record.get("paper_id")
        ranking = record.get(field, [])
        if not isinstance(ranking, list):
            ranking = []
        selected = ranking[:cutoff]
        wrong = sum(item.get("paper_id") != target_paper for item in selected)
        total += len(selected)
        cross_paper += wrong
        if wrong:
            affected.append(str(record.get("case_id")))
    return {
        "cutoff": cutoff,
        "candidate_count": total,
        "cross_paper_count": cross_paper,
        "cross_paper_rate": round(cross_paper / total, 6) if total else 0.0,
        "affected_case_count": len(affected),
        "affected_case_ids": affected,
    }


def build_cross_paper_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """比较各阶段跨论文候选噪声，确认 Reranker 是否在最终 Top-5 清除噪声。"""
    return {
        "original_dense_at_20": _cross_paper_stage(
            records, field="original_dense", cutoff=20
        ),
        "rewritten_dense_at_20": _cross_paper_stage(
            records, field="rewritten_dense", cutoff=20
        ),
        "rewritten_bm25_at_20": _cross_paper_stage(
            records, field="rewritten_bm25", cutoff=20
        ),
        "rrf_at_40": _cross_paper_stage(records, field="rrf_results", cutoff=40),
        "reranker_at_40": _cross_paper_stage(
            records, field="reranked_results", cutoff=40
        ),
        "final_at_5": _cross_paper_stage(records, field="final_results", cutoff=5),
    }


def _rank_map(ranking: Any) -> dict[str, int]:
    """把阶段 ranking 转为 chunk_id → 首次出现 rank 的映射。"""
    if not isinstance(ranking, list):
        return {}
    result: dict[str, int] = {}
    for index, item in enumerate(ranking, 1):
        if not isinstance(item, dict) or not isinstance(item.get("chunk_id"), str):
            continue
        declared_rank = item.get("rank")
        rank = int(declared_rank) if isinstance(declared_rank, int) else index
        result.setdefault(item["chunk_id"], rank)
    return result


def build_ground_truth_rank_changes(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """逐 Ground Truth chunk 比较 RRF、Reranker 与最终 Top-5 的名次变化。"""
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    movement_counts: Counter[str] = Counter()
    deltas: list[int] = []
    for record in records:
        rrf_ranks = _rank_map(record.get("rrf_results"))
        reranker_ranks = _rank_map(record.get("reranked_results"))
        final_ranks = _rank_map(record.get("final_results"))
        ground_truth = [str(value) for value in record.get("ground_truth_chunk_ids", [])]
        case_rows: list[dict[str, Any]] = []
        for chunk_id in ground_truth:
            pre_rank = rrf_ranks.get(chunk_id)
            post_rank = reranker_ranks.get(chunk_id)
            final_rank = final_ranks.get(chunk_id)
            delta = post_rank - pre_rank if pre_rank is not None and post_rank is not None else None
            if pre_rank is None and post_rank is None:
                movement = "missing_both"
            elif pre_rank is None:
                movement = "new_after_rerank"
            elif post_rank is None:
                movement = "missing_after_rerank"
            elif post_rank < pre_rank:
                movement = "promoted"
            elif post_rank > pre_rank:
                movement = "demoted"
            else:
                movement = "unchanged"
            movement_counts[movement] += 1
            if delta is not None:
                deltas.append(delta)
            item = {
                "case_id": record.get("case_id"),
                "primary_type": record.get("primary_type"),
                "chunk_id": chunk_id,
                "rrf_rank": pre_rank,
                "reranker_rank": post_rank,
                "rank_delta_post_minus_pre": delta,
                "final_top_5_rank": final_rank if final_rank is not None and final_rank <= 5 else None,
                "movement": movement,
            }
            rows.append(item)
            case_rows.append(item)
        case_summaries.append(
            {
                "case_id": record.get("case_id"),
                "primary_type": record.get("primary_type"),
                "ground_truth_count": len(ground_truth),
                "rrf_found_count": sum(item["rrf_rank"] is not None for item in case_rows),
                "reranker_found_count": sum(
                    item["reranker_rank"] is not None for item in case_rows
                ),
                "final_top_5_found_count": sum(
                    item["final_top_5_rank"] is not None for item in case_rows
                ),
                "best_rrf_rank": min(
                    (item["rrf_rank"] for item in case_rows if item["rrf_rank"] is not None),
                    default=None,
                ),
                "best_reranker_rank": min(
                    (
                        item["reranker_rank"]
                        for item in case_rows
                        if item["reranker_rank"] is not None
                    ),
                    default=None,
                ),
            }
        )
    return {
        "summary": {
            "ground_truth_chunk_count": len(rows),
            "movement_counts": dict(sorted(movement_counts.items())),
            "average_rank_delta_post_minus_pre": (
                round(sum(deltas) / len(deltas), 6) if deltas else None
            ),
        },
        "case_summaries": case_summaries,
        "chunk_rank_changes": rows,
    }


def validate_trace_structure(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """检查诊断依赖字段是否存在，并列出缺失 trace 的 Case。"""
    required = (
        "case_id",
        "paper_id",
        "ground_truth_chunk_ids",
        "stage_availability",
        "original_dense",
        "rewritten_dense",
        "rewritten_bm25",
        "rrf_results",
        "reranked_results",
        "final_results",
        "rewrite_status",
        "expansion",
    )
    missing: dict[str, list[str]] = {}
    empty_required_rankings: dict[str, list[str]] = {}
    for index, record in enumerate(records, 1):
        case_id = str(record.get("case_id", f"row_{index}"))
        absent = [field for field in required if field not in record]
        if absent:
            missing[case_id] = absent
        if record.get("answerable") is True and not record.get(
            "expected_bibliography_intent"
        ):
            empty = [
                field
                for field in ("original_dense", "rrf_results", "reranked_results", "final_results")
                if not record.get(field)
            ]
            if empty:
                empty_required_rankings[case_id] = empty
    return {
        "valid": not missing and not empty_required_rankings,
        "missing_fields": missing,
        "empty_required_rankings": empty_required_rankings,
    }


def build_report(records: Sequence[Mapping[str, Any]], *, source_path: Path) -> dict[str, Any]:
    """构造完整机器可读 Diagnostic Report。"""
    content = valid_content_cases(records)
    trace_validation = validate_trace_structure(records)
    return {
        "status": "PASS" if trace_validation["valid"] else "INCOMPLETE_TRACE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_case_results": str(source_path.resolve()),
        "counts": {
            "total_cases": len(records),
            "content_answerable_cases": len(content),
            "bibliography_cases": sum(
                record.get("expected_bibliography_intent") is True for record in records
            ),
            "unanswerable_cases": sum(record.get("answerable") is False for record in records),
        },
        "trace_validation": trace_validation,
        "slice_evaluation": build_slice_diagnostics(content),
        "evidence_expansion_evaluation": build_expansion_evaluation(content),
        "retrieval_routes": build_route_diagnostics(content),
        "rrf_depth_curve": build_rrf_curve(content),
        "reranker_before_after": build_reranker_comparison(content),
        "query_rewrite": build_rewrite_diagnostics(records, content),
        "cross_paper_noise": build_cross_paper_diagnostics(content),
        "ground_truth_rank_changes": build_ground_truth_rank_changes(content),
        "interpretation_notes": [
            "Route overall 指标把路径不可用视为未命中；when_available 只评实际执行该路径的 Case。",
            "Reranker before/after 使用同一份已保存候选集合，只表示排序变化，不替代正式 Ablation。",
            "rank_delta_post_minus_pre < 0 表示提升，> 0 表示下降。",
            "跨论文噪声以每条 Golden 的 paper_id 为目标论文判断。",
        ],
    }


def _metric(value: Any) -> str:
    """把指标转为紧凑 Markdown 文本。"""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _slice_table(
    title: str, label: str, values: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """渲染一组最终 Top-5 Slice 表格。"""
    lines = [
        f"### {title}",
        "",
        f"| {label} | Cases | Hit@5 | Recall@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in values.items():
        lines.append(
            f"| {name} | {item['cases']} | {_metric(item['hit_at_5'])} | "
            f"{_metric(item['recall_at_5'])} | {_metric(item['mrr_at_5'])} |"
        )
    return lines


def _expansion_tables(expansion: Mapping[str, Any]) -> list[str]:
    """渲染独立Expansion质量、Context规模及Before→After Slice表。"""
    before = expansion["before_expansion"]
    after = expansion["after_expansion"]
    delta = expansion["delta"]
    size = expansion["context_size"]
    lines = [
        "## Evidence Expansion Evaluation",
        "",
        "| Metric | Before Expansion | After Expansion | Delta |",
        "|---|---:|---:|---:|",
        f"| Hit | {_metric(before['hit_at_5'])} | {_metric(after['expanded_hit'])} | "
        f"{_metric(delta['hit'])} |",
        f"| Recall | {_metric(before['recall_at_5'])} | "
        f"{_metric(after['expanded_recall'])} | {_metric(delta['recall'])} |",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Expansion Recovery Cases | {expansion['recovery_cases']['count']} |",
        f"| Avg Expansion Latency | {_metric(expansion['avg_expansion_latency_ms'])} ms |",
        f"| Avg Expanded Chunks | {_metric(size['avg_expanded_chunks'])} |",
        f"| Max Expanded Chunks | {size['max_expanded_chunks']} |",
        f"| Avg Expanded Context Chars | {_metric(size['avg_expanded_context_chars'])} |",
        f"| Avg Expanded Context Tokens | {_metric(size['avg_expanded_context_tokens'])} |",
        f"| Expansion Errors | {expansion['expansion_errors']['count']} |",
        "",
        "### Expansion Recall by Type",
        "",
        "| Slice | Cases | Raw Recall@5 | Expanded Recall | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in expansion["slice_before_after"]["type"].items():
        lines.append(
            f"| {name} | {item['cases']} | {_metric(item['raw_recall_at_5'])} | "
            f"{_metric(item['expanded_recall'])} | {_metric(item['delta'])} |"
        )
    lines.extend(
        [
            "",
            "### Expansion Recall by Hop",
            "",
            "| Slice | Cases | Raw Recall@5 | Expanded Recall | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in expansion["slice_before_after"]["hop"].items():
        lines.append(
            f"| {name} | {item['cases']} | {_metric(item['raw_recall_at_5'])} | "
            f"{_metric(item['expanded_recall'])} | {_metric(item['delta'])} |"
        )
    lines.extend(
        [
            "",
            "Expansion Recovery Case IDs: "
            f"`{json.dumps(expansion['recovery_cases']['case_ids'], ensure_ascii=False)}`",
        ]
    )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """把 Diagnostic JSON 渲染为可直接审阅的 Markdown 报告。"""
    lines = [
        "# Retrieval Diagnostic Report",
        "",
        f"**Status: {report['status']}**",
        "",
        f"- Total cases: `{report['counts']['total_cases']}`",
        f"- Content answerable cases: `{report['counts']['content_answerable_cases']}`",
        f"- Source: `{report['source_case_results']}`",
        "",
        "## Slice Evaluation",
        "",
        *_slice_table("Type", "Type", report["slice_evaluation"]["type"]),
        "",
        *_slice_table(
            "Difficulty", "Difficulty", report["slice_evaluation"]["difficulty"]
        ),
        "",
        *_slice_table("Hop", "Hop", report["slice_evaluation"]["hop"]),
        "",
        *_expansion_tables(report["evidence_expansion_evaluation"]),
        "",
        "## Retrieval Route Diagnostics",
        "",
        "Overall 指标把未执行路径视为未命中；Available-only 只统计实际执行该路径的 Case。",
        "",
        "| Route | Available | Overall Hit@5 | Overall Recall@5 | Available Hit@5 | Available Recall@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in report["retrieval_routes"].values():
        overall = route["overall"] or {}
        available = route["when_available"] or {}
        lines.append(
            f"| {route['label']} | {route['available_cases']}/{route['eligible_cases']} | "
            f"{_metric(overall.get('hit_at_5'))} | {_metric(overall.get('recall_at_5'))} | "
            f"{_metric(available.get('hit_at_5'))} | "
            f"{_metric(available.get('recall_at_5'))} |"
        )

    lines.extend(
        [
            "",
            "## RRF Depth Curve",
            "",
            "| K | Hit@K | Recall@K |",
            "|---:|---:|---:|",
        ]
    )
    for cutoff in (5, 10, 20, 40):
        item = report["rrf_depth_curve"][f"at_{cutoff}"]
        lines.append(f"| {cutoff} | {_metric(item['hit'])} | {_metric(item['recall'])} |")

    lines.extend(
        [
            "",
            "## Reranker Before / After",
            "",
            "| K | Metric | RRF | Reranker | Delta |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for cutoff in (5, 10, 20):
        item = report["reranker_before_after"][f"at_{cutoff}"]
        for metric_name, label in (
            (f"hit_at_{cutoff}", f"Hit@{cutoff}"),
            (f"recall_at_{cutoff}", f"Recall@{cutoff}"),
            (f"mrr_at_{cutoff}", f"MRR@{cutoff}"),
        ):
            lines.append(
                f"| {cutoff} | {label} | {_metric(item['rrf'][metric_name])} | "
                f"{_metric(item['reranker'][metric_name])} | "
                f"{_metric(item['delta'][metric_name])} |"
            )

    lines.extend(
        [
            "",
            "## Query Rewrite Status",
            "",
            "| Scope | Status | Cases | Rate | Hit@5 | Recall@5 | MRR@5 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    rewrite = report["query_rewrite"]
    for status, distribution in rewrite["all_cases"].items():
        lines.append(
            f"| All | {status} | {distribution['cases']} | "
            f"{_metric(distribution['rate'])} | — | — | — |"
        )
    for status, metrics in rewrite["content_metrics_by_status"].items():
        lines.append(
            f"| Content | {status} | {metrics['cases']} | "
            f"{_metric(rewrite['content_cases'][status]['rate'])} | "
            f"{_metric(metrics['hit_at_5'])} | {_metric(metrics['recall_at_5'])} | "
            f"{_metric(metrics['mrr_at_5'])} |"
        )

    lines.extend(
        [
            "",
            "## Cross-paper Candidate Noise",
            "",
            "| Stage | Candidates | Cross-paper | Rate | Affected Cases |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["cross_paper_noise"].items():
        lines.append(
            f"| {name} | {item['candidate_count']} | {item['cross_paper_count']} | "
            f"{_metric(item['cross_paper_rate'])} | {item['affected_case_count']} |"
        )

    rank_changes = report["ground_truth_rank_changes"]
    lines.extend(
        [
            "",
            "## Ground Truth Pre/Post Rank Changes",
            "",
            f"- Ground Truth chunks: `{rank_changes['summary']['ground_truth_chunk_count']}`",
            "- Movement counts: "
            f"`{json.dumps(rank_changes['summary']['movement_counts'], ensure_ascii=False)}`",
            "- Average post-pre delta: "
            f"`{_metric(rank_changes['summary']['average_rank_delta_post_minus_pre'])}`",
            "",
            "负 delta 表示 Reranker 提升名次，正 delta 表示名次下降。",
            "",
            "| Case | Ground Truth Chunk | RRF Rank | Reranker Rank | Delta | Final Top-5 | Movement |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in rank_changes["chunk_rank_changes"]:
        lines.append(
            f"| {item['case_id']} | {item['chunk_id']} | {_metric(item['rrf_rank'])} | "
            f"{_metric(item['reranker_rank'])} | "
            f"{_metric(item['rank_delta_post_minus_pre'])} | "
            f"{_metric(item['final_top_5_rank'])} | {item['movement']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_json_atomic(path: Path, value: Any) -> None:
    """使用同目录临时文件原子写入 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """使用同目录临时文件原子写入 Markdown 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析输入 trace 和两份报告路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-results", type=Path, default=DEFAULT_CASE_RESULTS)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：构建报告并用退出码表示 trace 是否完整。"""
    args = parse_args(argv)
    try:
        records = load_case_results(args.case_results)
        report = build_report(records, source_path=args.case_results)
        write_json_atomic(args.json_report, report)
        write_text_atomic(args.markdown_report, render_markdown(report))
    except (OSError, ValueError, DiagnosticInputError) as error:
        print(f"ERROR: {error}")
        return 2
    print(report["status"])
    print(f"Content cases: {report['counts']['content_answerable_cases']}")
    print(f"JSON report: {args.json_report.resolve()}")
    print(f"Markdown report: {args.markdown_report.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
