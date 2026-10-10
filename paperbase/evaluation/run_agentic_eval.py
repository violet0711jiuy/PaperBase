"""Run deterministic Retrieval + Agentic evaluation on the frozen Golden dataset."""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.scripts.run_retrieval_eval import ProductionRetrieverObserver, load_golden
from langgraph.checkpoint.memory import InMemorySaver
from paperbase.agentic.audit import build_agentic_audit_snapshot
from paperbase.agentic.factory import create_agentic_research_service
from paperbase.config import default_config_path, load_settings
from paperbase.llm.answer_output_capture import capture_answer_outputs
from paperbase.retrieval.service import create_hybrid_retriever
from .agentic import (
    aggregate_action_control_metrics,
    aggregate_case_metrics,
    evaluate_case_metrics,
    metrics_by_type,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "eval/datasets/golden_dataset_v1_2.jsonl"

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--case-id", action="append", default=[])
    p.add_argument("--type", dest="primary_type", action="append", default=[])
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

def _load_completed(path: Path, *, evaluation_mode: str) -> dict[str, dict[str, Any]]:
    if not path.is_file(): return {}
    return {
        str(row["id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if row.get("evaluation_mode", "combined") == evaluation_mode
    }

def _result_payload(result: object) -> dict[str, Any]:
    return {"run_status": result.status.value, "stop_reason": result.stop_reason.value,
            "answer": ({"status": result.answer.status, "insufficient_evidence": result.answer.insufficient_evidence,
                        "partial_answer": result.answer.partial_answer, "citations": list(result.answer.citations),
                        "overall_status": result.answer.overall_status,
                        "requirement_coverage": [
                            {"requirement_id": item.requirement_id, "status": item.status,
                             "evidence_ids": list(item.evidence_ids)}
                            for item in result.answer.requirement_coverage
                        ]} if result.answer else {}),
            "selected_evidence_ids": [item.evidence_id for item in result.selected_evidence],
            "answer_input_evidence_ids": list(result.answer_input_evidence_ids),
            "evidence_source_chunk_ids": {
                item.evidence_id: list(item.evidence.source_chunk_ids)
                for item in result.evidence
            },
            "agent_retrieved_source_chunk_ids": sorted({chunk for item in result.evidence for chunk in item.evidence.source_chunk_ids}),
            "selected_source_chunk_ids": sorted({chunk for item in result.selected_evidence for chunk in item.evidence.source_chunk_ids}),
            "error_codes": list(result.error_codes), "thread_id": result.thread_id}

def _public_audit_for(row: dict[str, Any]) -> dict[str, Any]:
    """Read a persisted public audit projection, never a prompt or model response."""
    raw_path = row.get("audit_path")
    if not isinstance(raw_path, str):
        return {}
    try:
        value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _report(
    run_id: str,
    metrics: dict[str, Any],
    action_control: dict[str, Any],
    rows: list[dict[str, Any]],
    dataset: Path,
    *,
    agentic_only: bool = False,
) -> str:
    failures = [row for row in rows if row["failure_labels"] != ["none"]]
    labels: dict[str, list[str]] = {}
    for row in failures:
        for label in row["failure_labels"]: labels.setdefault(label, []).append(str(row["id"]))
    table = "\n".join(f"| {name} | {len(ids)} | {', '.join(ids)} |" for name, ids in sorted(labels.items())) or "| none | 0 | — |"
    representative = "".join(
        f"- `{row['id']}`: stop={row['stop_reason']}, labels={', '.join(row['failure_labels'])}; audit={row['audit_path']}\n"
        for row in failures[:8]
    ) or "No deterministic failures."
    stage_diagnosis = (
        "This run measures only the Agentic service. Agent retrieval coverage uses all kept Evidence; "
        "final Evidence coverage uses the selected set. One-shot Retriever ranking metrics are N/A."
        if agentic_only else
        "`retrieval_miss` points to the standalone Retriever ranking. A retrieval hit followed by an "
        "uncited gold chunk points to Registry, Answer-input, grounding, or answer completeness."
    )
    recommendation = (
        "Inspect low Agent retrieval coverage, then answer-input, citation grounding, and open Requirement gaps."
        if agentic_only else
        "Prioritize only labels that recur in this run: retrieval misses first when Hit@5 is low; "
        "then false no-support/no-selected-evidence cases; then cost/stagnation labels."
    )
    return f"""# PaperBase Agentic RAG Evaluation Report

## 1. Executive Summary

Run `{run_id}` evaluated {len(rows)} frozen Golden cases from `{dataset}`. This report is deterministic: it uses canonical chunk IDs, public Agent audit fields, and no LLM judge.

## 2. Overall Metrics

```json
{json.dumps(metrics, ensure_ascii=False, indent=2)}
```

## 3. Metrics by Question Type

See `metrics_by_type.json` for the complete machine-readable breakdown. Metrics without a valid denominator are `null` (N/A), rather than zero.

## 4. Action Control Plane

```json
{json.dumps(action_control, ensure_ascii=False, indent=2)}
```

## 5. Failure Analysis

| Label | Count | Case IDs |
|---|---:|---|
{table}

## 6. Representative Failure Cases

{representative}

## 7. Stage Diagnosis

{stage_diagnosis} Budget and invalid-action labels come from structured runtime audit only.

## 8. Recommendations

{recommendation} Do not infer semantic answer quality from this MVP.
"""

def main(argv=None, *, agentic_only: bool = False) -> int:
    args = parse_args(argv); dataset = args.dataset.resolve(); rows = load_golden(dataset)
    if args.case_id: rows = [r for r in rows if r["id"] in set(args.case_id)]
    if args.primary_type: rows = [r for r in rows if r["primary_type"] in set(args.primary_type)]
    if args.limit is not None: rows = rows[:args.limit]
    evaluation_mode = "agentic_only" if agentic_only else "combined"
    run_id = datetime.now(timezone.utc).strftime(
        ("agentic_only_eval_" if agentic_only else "agentic_eval_") + "%Y%m%dT%H%M%SZ"
    )
    output = (args.output or ROOT / "storage/evaluation" / run_id).resolve(); output.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(output / "case_results.jsonl", evaluation_mode=evaluation_mode) if args.resume else {}
    settings = load_settings(args.config)
    observer = None if agentic_only else ProductionRetrieverObserver(
        create_hybrid_retriever(config_path=settings.config_path)
    )
    # Evaluation owns only in-memory graph checkpoints.  Case artifacts below
    # provide resumability, so no production checkpoint / conversation state is
    # touched by this external caller.
    service = create_agentic_research_service(
        config_path=settings.config_path, checkpointer=InMemorySaver()
    )
    results: list[dict[str, Any]] = []
    try:
        for index, golden in enumerate(rows, 1):
            case_id = str(golden["id"])
            if case_id in completed:
                results.append(completed[case_id]); continue
            print(f"[{index}/{len(rows)}] {case_id}", flush=True)
            ranked, retrieval_error = None, None
            if observer is not None:
                try:
                    retrieved, _trace = observer.retrieve(str(golden["question"]))
                    ranked = [chunk.chunk_id for chunk in retrieved.chunks]
                except Exception as exc:
                    ranked = []
                    retrieval_error = f"{type(exc).__name__}: {exc}"
            result_path = output / "cases" / f"{case_id}.result.json"; audit_path = output / "cases" / f"{case_id}.audit.json"
            run_error = None; audit: dict[str, Any] = {}; result_payload: dict[str, Any] = {}
            try:
                with capture_answer_outputs(output / "cases" / f"{case_id}.answer_llm_outputs.jsonl", case_id=case_id):
                    result = service.research(str(golden["question"]), selected_paper_ids=(str(golden["paper_id"]),), thread_id=f"eval-{run_id}-{case_id}")
                result_payload = _result_payload(result); _write_json(result_path, result_payload)
                audit = build_agentic_audit_snapshot(result).to_dict(); _write_json(audit_path, audit)
            except Exception as exc: run_error = f"{type(exc).__name__}: {exc}"
            row = evaluate_case_metrics(golden, ranked_chunk_ids=ranked, run_status=result_payload.get("run_status"), stop_reason=result_payload.get("stop_reason"), audit=audit, agent_retrieved_source_chunk_ids=result_payload.get("agent_retrieved_source_chunk_ids", ()), evidence_source_chunk_ids=result_payload.get("evidence_source_chunk_ids", {}), selected_source_chunk_ids=result_payload.get("selected_source_chunk_ids", ()), answer=result_payload.get("answer"), result_path=str(result_path), audit_path=str(audit_path), retrieval_error=retrieval_error, run_error=run_error)
            row["evaluation_mode"] = evaluation_mode
            results.append(row); _write_jsonl(output / "case_results.jsonl", results)
    finally: service.close()
    metrics = aggregate_case_metrics(results); by_type = metrics_by_type(results)
    action_control = aggregate_action_control_metrics(
        (row, _public_audit_for(row)) for row in results
    )
    _write_json(output / "metrics.json", metrics); _write_json(output / "metrics_by_type.json", by_type)
    _write_json(output / "action_control_metrics.json", action_control)
    _write_jsonl(output / "failures.jsonl", [r for r in results if r["failure_labels"] != ["none"]])
    (output / "evaluation_report.md").write_text(_report(run_id, metrics, action_control, results, dataset, agentic_only=agentic_only), encoding="utf-8")
    print(output); return 0

if __name__ == "__main__": raise SystemExit(main())
