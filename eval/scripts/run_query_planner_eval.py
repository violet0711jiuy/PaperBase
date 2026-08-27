"""只运行正式 Query Planner，为 40 条 Golden 生成独立 Audit trace。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from time import perf_counter
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperbase.config import default_config_path, load_settings
from paperbase.retrieval.query_rewriter import TrustedPaperScope, create_query_planner


DEFAULT_DATASET = ROOT / "eval" / "datasets" / "golden_dataset_v1_2.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "eval" / "results" / "query_planner_fix" / "planner_case_results.jsonl"
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取 Golden JSONL，并拒绝坏行、空数据或重复 Case ID。"""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Golden row {line_number} must be an object")
        records.append(value)
    case_ids = [str(record.get("id", "")) for record in records]
    if not records or "" in case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Golden Dataset is empty or contains missing/duplicate IDs")
    return records


def load_paper_titles(database_path: Path) -> dict[str, str | None]:
    """从正式 SQLite 读取可信 paper_id 与标题，供“本文”指代消解使用。"""
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT paper_id, paper_title FROM documents").fetchall()
    return {
        str(paper_id): str(title) if title is not None else None
        for paper_id, title in rows
    }


def write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """原子写出 Planner trace，避免中途失败留下半份审计输入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    """逐条调用正式 Planner；不加载 Embedding、FAISS、BM25 或 Reranker。"""
    settings = load_settings(args.config)
    cases = load_cases(args.dataset.resolve())
    titles = load_paper_titles(settings.database.path)
    planner = create_query_planner(
        settings=settings.retrieval.query_rewrite,
        env_path=settings.config_path.parent / ".env",
    )
    output: list[dict[str, Any]] = []
    for index, record in enumerate(cases, 1):
        case_id = str(record["id"])
        print(f"[{index}/{len(cases)}] {case_id}")
        started = perf_counter()
        plan = planner.plan(
            str(record["question"]),
            trusted_scope=TrustedPaperScope(
                paper_id=str(record["paper_id"]),
                paper_title=titles.get(str(record["paper_id"])),
            ),
        )
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        output.append(
            {
                "case_id": case_id,
                "question": record["question"],
                "primary_type": record["primary_type"],
                "tags": record["tags"],
                "paper_id": record["paper_id"],
                "original_query": plan.original_query,
                "resolved_query": plan.resolved_query,
                "semantic_query_en": plan.semantic_query_en,
                "lexical_keywords_en": list(plan.lexical_keywords_en),
                "resolution_status": plan.resolution_status,
                "semantic_status": plan.semantic_status,
                "lexical_status": plan.lexical_status,
                "rewrite_status": plan.rewrite_status,
                "validation_diagnostics": list(plan.validation_diagnostics),
                "predicted_bibliography_intent": plan.search_bibliography,
                "latency": {"rewrite_latency_ms": latency_ms},
            }
        )
    write_jsonl_atomic(args.output.resolve(), output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析数据集、正式配置与 Planner trace 输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：执行 40 条 Planner 并输出可复核 trace。"""
    args = parse_args(argv)
    try:
        records = run(args)
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    print(f"Completed: {len(records)} cases")
    print(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    print(f"Trace: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
