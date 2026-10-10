"""Reclassify saved Agentic-only case artifacts without running an Agent or LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .agentic import aggregate_case_metrics, evaluate_case_metrics, metrics_by_type
from paperbase.retrieval.lexical_terms import extract_lexical_terms, normalize_lexical_terms


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def reclassify(
    *, dataset: Path, source: Path, output: Path,
    database: Path | None = None,
) -> list[dict[str, object]]:
    golden = {
        str(item["id"]): item
        for item in (json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
                     if line.strip())
    }
    old_rows = [
        json.loads(line) for line in (source / "case_results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    if any(row.get("evaluation_mode") != "agentic_only" for row in old_rows):
        raise ValueError("Offline reclassification currently requires Agentic-only results.")
    rows: list[dict[str, object]] = []
    for old in old_rows:
        case_id = str(old["id"])
        result_path = source / "cases" / f"{case_id}.result.json"
        audit_path = source / "cases" / f"{case_id}.audit.json"
        result = _read_json(result_path)
        audit = _read_json(audit_path)
        provenance = result.get("evidence_source_chunk_ids", {})
        provenance_source = "saved_result" if provenance else "unavailable"
        if not provenance and database is not None:
            references = _recover_reference_provenance(
                database=database, result=result, audit=audit,
            )
            sections = _recover_section_context_provenance(
                database=database, result=result, audit=audit,
            )
            provenance = {**references, **sections}
            if references:
                provenance_source = "verified_bibliography_fts_replay"
            elif sections:
                provenance_source = "verified_section_context_replay_partial"
        row = evaluate_case_metrics(
            golden[case_id], ranked_chunk_ids=None,
            run_status=result.get("run_status"), stop_reason=result.get("stop_reason"),
            audit=audit,
            agent_retrieved_source_chunk_ids=result.get("agent_retrieved_source_chunk_ids", ()),
            evidence_source_chunk_ids=provenance,
            selected_source_chunk_ids=result.get("selected_source_chunk_ids", ()),
            answer=result.get("answer"), result_path=str(result_path.resolve()),
            audit_path=str(audit_path.resolve()),
            retrieval_error=old.get("retrieval_error"), run_error=old.get("run_error"),
        )
        row["evaluation_mode"] = "agentic_only"
        row["evidence_provenance_source"] = provenance_source
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    (output / "case_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(aggregate_case_metrics(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "metrics_by_type.json").write_text(
        json.dumps(metrics_by_type(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def _recover_reference_provenance(
    *, database: Path, result: dict[str, object], audit: dict[str, object],
) -> dict[str, list[str]]:
    """Replay one saved bibliography FTS action; accept only an exact ID-set match."""
    steps = audit.get("steps", ())
    evidence = audit.get("evidence", ())
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(evidence, list):
        return {}
    step = steps[0]
    if not isinstance(step, dict) or step.get("action_name") != "search_references":
        return {}
    arguments = step.get("action_arguments", {})
    if not isinstance(arguments, dict):
        return {}
    terms = normalize_lexical_terms(arguments.get("lexical_terms", ()))
    if not terms:
        terms = extract_lexical_terms(str(arguments.get("query", "")))
    if not terms or not evidence or not database.is_file():
        return {}
    query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """SELECT chunks.chunk_id, chunks.page_start,
                      bm25(bibliography_fts, 1.2, 0.8, 1.0) AS bm25_score
               FROM bibliography_fts
               JOIN chunks ON chunks.chunk_id = bibliography_fts.chunk_id
               WHERE bibliography_fts MATCH ?
                 AND chunks.section_type = 'bibliography'
                 AND chunks.paper_id = ?
               ORDER BY bm25_score ASC, chunks.chunk_id ASC LIMIT ?""",
            (query, str(arguments.get("paper_id", "")), len(evidence)),
        ).fetchall()
    saved_chunks = result.get("agent_retrieved_source_chunk_ids", ())
    if len(rows) != len(evidence) or set(row[0] for row in rows) != set(saved_chunks):
        return {}
    if any(
        not isinstance(item, dict)
        or item.get("evidence_id") != f"REF{index}"
        or item.get("page_start") != row[1]
        for index, (item, row) in enumerate(zip(evidence, rows), 1)
    ):
        return {}
    return {
        str(item["evidence_id"]): [str(row[0])]
        for item, row in zip(evidence, rows)
    }


def _recover_section_context_provenance(
    *, database: Path, result: dict[str, object], audit: dict[str, object],
) -> dict[str, list[str]]:
    """Recover only single-result section actions with verified saved provenance."""
    steps = audit.get("steps", ())
    evidence = audit.get("evidence", ())
    if not isinstance(steps, list) or not isinstance(evidence, list) or not database.is_file():
        return {}
    if sum(int(step.get("added_count", 0)) for step in steps if isinstance(step, dict)) != len(evidence):
        return {}
    retrieved = set(result.get("agent_retrieved_source_chunk_ids", ()))
    recovered: dict[str, list[str]] = {}
    position = 0
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for step in steps:
            if not isinstance(step, dict):
                return {}
            count = int(step.get("added_count", 0))
            added = evidence[position:position + count]
            position += count
            if step.get("action_name") != "get_section_context" or count != 1:
                continue
            arguments = step.get("action_arguments", {})
            if not isinstance(arguments, dict):
                continue
            rows = connection.execute(
                """SELECT chunk_id, page_start, page_end FROM chunks
                   WHERE paper_id = ? AND section_id = ?
                   ORDER BY chunk_index""",
                (str(arguments.get("paper_id", "")),
                 str(arguments.get("section_id", ""))),
            ).fetchall()
            if not rows or not {row[0] for row in rows}.issubset(retrieved):
                continue
            item = added[0]
            if not isinstance(item, dict) or item.get("paper_id") != arguments.get("paper_id"):
                continue
            pages_start = [row[1] for row in rows if row[1] is not None]
            pages_end = [row[2] for row in rows if row[2] is not None]
            if (item.get("page_start") != (min(pages_start) if pages_start else None)
                    or item.get("page_end") != (max(pages_end) if pages_end else None)):
                continue
            recovered[str(item["evidence_id"])] = [str(row[0]) for row in rows]
    return recovered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path,
                        help="Optional read-only bibliography FTS provenance replay for old artifacts")
    args = parser.parse_args(argv)
    rows = reclassify(
        dataset=args.dataset.resolve(), source=args.source.resolve(),
        output=args.output.resolve(),
        database=args.database.resolve() if args.database is not None else None,
    )
    print(f"Reclassified {len(rows)} saved cases without Agent calls: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
