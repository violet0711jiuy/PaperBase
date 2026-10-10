"""Deterministic metrics and failure classification for Agentic evaluation.

This module deliberately does not import or alter the production graph.  The
runner supplies the Golden row, a retrieval ranking, and the public result /
audit projection produced by the normal Agentic service.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any


TAXONOMY = ("fact", "method", "experiment", "result", "synthesis", "bibliography", "unanswerable")
FAILURE_STAGES = frozenset({
    "retrieval", "registry", "answer_input", "grounding", "control",
    "answer_generation", "answer_validation",
})
RESEARCH_FAILURE_STAGES = frozenset({
    "retrieval", "registry", "answer_input", "grounding", "control",
})


def gold_chunk_ids(golden: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(chunk_id)
        for item in golden.get("relevant_evidence", ())
        if isinstance(item, Mapping)
        for chunk_id in item.get("chunk_ids", ())
    ))


def ranking_metrics(ranked_chunk_ids: Sequence[str], gold_ids: Sequence[str]) -> dict[str, float | None]:
    """Canonical chunk-ID-only Top-5 metrics; empty Gold is not applicable."""
    gold = set(gold_ids)
    if not gold:
        return {"hit_at_5": None, "recall_at_5": None, "reciprocal_rank": None}
    top5 = list(ranked_chunk_ids[:5])
    hits = gold.intersection(top5)
    rank = next((index for index, item in enumerate(ranked_chunk_ids, 1) if item in gold), None)
    return {
        "hit_at_5": float(bool(hits)),
        "recall_at_5": len(hits) / len(gold),
        "reciprocal_rank": (1.0 / rank) if rank is not None else 0.0,
    }


def _useful_action(step: Mapping[str, Any]) -> bool:
    return (
        int(step.get("added_count", 0) or 0) > 0
        or bool(step.get("newly_exposed_section_ids"))
        or (str(step.get("action_name", "")).startswith(("get_section_context", "get_paper_overview")) and str(step.get("tool_status", "")) == "ok")
    )


def selected_gold_coverage(selected_source_chunk_ids: Iterable[str], gold_ids: Sequence[str]) -> float | None:
    gold = set(gold_ids)
    if not gold:
        return None
    return len(gold.intersection(selected_source_chunk_ids)) / len(gold)


def evaluate_case_metrics(
    golden: Mapping[str, Any],
    *,
    ranked_chunk_ids: Sequence[str] | None,
    run_status: str | None,
    stop_reason: str | None,
    audit: Mapping[str, Any] | None,
    agent_retrieved_source_chunk_ids: Iterable[str] = (),
    evidence_source_chunk_ids: Mapping[str, Sequence[str]] | None = None,
    selected_source_chunk_ids: Iterable[str] = (),
    answer: Mapping[str, Any] | None = None,
    result_path: str | None = None,
    audit_path: str | None = None,
    retrieval_error: str | None = None,
    run_error: str | None = None,
) -> dict[str, Any]:
    """Build one machine-readable row without text/semantic judging."""
    answerable = golden.get("answerable") is True
    gold = gold_chunk_ids(golden)
    retrieval = ranking_metrics(ranked_chunk_ids, gold) if answerable and ranked_chunk_ids is not None else {
        "hit_at_5": None, "recall_at_5": None, "reciprocal_rank": None,
    }
    audit = audit or {}
    steps = tuple(audit.get("steps", ()) or ())
    useful_actions = sum(_useful_action(step) for step in steps if isinstance(step, Mapping))
    total_actions = len(steps)
    selected = tuple(selected_source_chunk_ids)
    agent_retrieved = tuple(agent_retrieved_source_chunk_ids)
    insufficient = bool((answer or {}).get("insufficient_evidence", False))
    partial = bool((answer or {}).get("partial_answer", False))
    overall_status = (answer or {}).get("overall_status")
    no_selected = not selected
    normal_result = run_status == "completed" and stop_reason not in {
        "fatal_error", "budget_exhausted", "invalid_action_limit",
    }
    completed_answerable = bool(
        answerable and normal_result and not insufficient and not no_selected
        and overall_status in (None, "complete")
    )
    legacy_false_no_support = bool(answerable and (insufficient or no_selected))
    failure_diagnostics = _failure_diagnostics(
        answerable=answerable, gold=gold, retrieved=agent_retrieved,
        evidence_source_chunk_ids=evidence_source_chunk_ids or {}, audit=audit,
        stop_reason=stop_reason, answer=answer or {},
    )
    false_no_support = bool(
        answerable and insufficient
        and any(item["stage"] in RESEARCH_FAILURE_STAGES for item in failure_diagnostics)
        and not _required_support_selected(audit)
    )
    abstention_correct = (
        bool(insufficient or (no_selected and run_status in {"no_evidence", "partial", "completed"}))
        if not answerable else None
    )
    evaluation_outcome = (
        "success" if completed_answerable else
        "partial" if answerable and normal_result and overall_status == "partial" else
        "correct_abstention" if not answerable and abstention_correct else
        "failure"
    )
    labels = failure_labels(
        answerable=answerable, retrieval=retrieval, retrieval_observed=ranked_chunk_ids is not None, run_status=run_status,
        stop_reason=stop_reason, audit=audit, insufficient=insufficient,
        partial=partial, no_selected=no_selected, run_error=run_error,
        retrieval_error=retrieval_error, abstention_correct=abstention_correct,
        false_no_support=false_no_support, failure_diagnostics=failure_diagnostics,
    )
    budget = audit.get("budget", {}) if isinstance(audit.get("budget"), Mapping) else {}
    return {
        "id": golden.get("id"), "question": golden.get("question"),
        "primary_type": golden.get("primary_type"), "answerable": answerable,
        "gold_chunk_ids": list(gold), "retrieved_chunk_ids": list(ranked_chunk_ids[:5]) if ranked_chunk_ids is not None else None,
        "run_status": run_status, "stop_reason": stop_reason,
        **retrieval,
        "tool_calls": budget.get("tool_calls"),
        "retrieval_queries": budget.get("retrieval_queries"),
        "llm_calls": budget.get("llm_calls"),
        "useful_actions": useful_actions, "total_actions": total_actions,
        "useful_action_rate": useful_actions / total_actions if total_actions else None,
        "completed_answerable": completed_answerable if answerable else None,
        "legacy_false_no_support": legacy_false_no_support if answerable else None,
        "false_no_support": false_no_support if answerable else None,
        "evaluation_outcome": evaluation_outcome,
        "failure_diagnostics": failure_diagnostics,
        "abstention_correct": abstention_correct,
        "gold_evidence_coverage": selected_gold_coverage(selected, gold) if answerable else None,
        "agent_retrieved_gold_coverage": selected_gold_coverage(agent_retrieved, gold) if answerable else None,
        "agent_retrieved_source_chunk_ids": list(agent_retrieved),
        "selected_source_chunk_ids": list(selected), "partial_answer": partial,
        "overall_answer_status": overall_status,
        "answer_requirement_coverage": (answer or {}).get("requirement_coverage", []),
        "insufficient_evidence": insufficient, "failure_labels": labels,
        "retrieval_error": retrieval_error, "run_error": run_error,
        "audit_path": audit_path, "result_path": result_path,
    }


def _failure_diagnostics(
    *, answerable: bool, gold: Sequence[str], retrieved: Sequence[str],
    evidence_source_chunk_ids: Mapping[str, Sequence[str]],
    audit: Mapping[str, Any], stop_reason: str | None,
    answer: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Classify only observable failures; Gold knowledge stays in Evaluation."""
    diagnostics: list[dict[str, Any]] = []

    def add(
        stage: str, code: str, *, requirement_ids: Iterable[str] = (),
        evidence_ids: Iterable[str] = (), detail: str,
    ) -> None:
        if stage not in FAILURE_STAGES:
            raise ValueError(f"Unknown evaluation failure stage: {stage}")
        diagnostics.append({
            "stage": stage, "code": code,
            "requirement_ids": sorted(set(requirement_ids)),
            "evidence_ids": sorted(set(evidence_ids)), "detail": detail,
        })

    gold_set = set(gold) if answerable else set()
    missing = gold_set.difference(retrieved)
    if missing:
        add("retrieval", "RETRIEVAL_MISS",
            detail=f"missing_gold_chunks={len(missing)}/{len(gold_set)}")

    evidence = tuple(
        item for item in audit.get("evidence", ()) if isinstance(item, Mapping)
    )
    gold_evidence = tuple(
        item for item in evidence
        if gold_set.intersection(evidence_source_chunk_ids.get(str(item.get("evidence_id")), ()))
    )
    answer_input_ids = set(str(value) for value in audit.get("answer_input_evidence_ids", ()))
    missing_from_answer = tuple(
        item for item in gold_evidence if str(item.get("evidence_id")) not in answer_input_ids
    )
    if missing_from_answer:
        add("answer_input", "GOLD_EVIDENCE_NOT_SHOWN_TO_ANSWERER",
            evidence_ids=(str(item["evidence_id"]) for item in missing_from_answer),
            detail=f"gold_evidence_missing_from_answer_input={len(missing_from_answer)}")
    cited_ids = set(str(value) for value in audit.get("cited_evidence_ids", ()))
    uncited = tuple(
        item for item in gold_evidence
        if str(item.get("evidence_id")) in answer_input_ids
        and str(item.get("evidence_id")) not in cited_ids
    )
    if uncited:
        add("grounding", "GOLD_ANSWER_INPUT_NOT_CITED",
            evidence_ids=(str(item["evidence_id"]) for item in uncited),
            detail=f"gold_answer_input_not_cited={len(uncited)}")

    required_ids = {
        str(item.get("requirement_id")) for item in audit.get("requirements", ())
        if isinstance(item, Mapping) and item.get("required") is True
    }
    open_gap_ids = {
        str(item.get("requirement_id"))
        for item in audit.get("requirements", ())
        if isinstance(item, Mapping) and item.get("required") is True
        and item.get("remaining_gap")
    }
    if answerable and open_gap_ids:
        add("grounding", "REQUIRED_REQUIREMENT_GAP",
            requirement_ids=open_gap_ids,
            detail=f"required_obligations_with_open_gap={len(open_gap_ids)}")

    if stop_reason == "budget_exhausted":
        add("control", "BUDGET_EXHAUSTED", detail="global_agent_iteration_limit_reached")
    if stop_reason == "invalid_action_limit":
        add("control", "INVALID_ACTION_STOP", detail="invalid_action_limit_reached")

    if answer.get("status") == "fallback":
        add("answer_generation", "ANSWER_GENERATION_FAILED", detail="answer_fallback_returned")
    for stage, code in (
        ("schema_validation", "ANSWER_SCHEMA_VALIDATION_FAILED"),
        ("invalid_json", "ANSWER_SCHEMA_VALIDATION_FAILED"),
        ("domain_validation", "ANSWER_SCHEMA_VALIDATION_FAILED"),
    ):
        calls = tuple(
            item for item in audit.get("llm_calls", ())
            if isinstance(item, Mapping) and item.get("node") == "generate_answer"
            and item.get("error_stage") == stage
        )
        if calls:
            rules = sorted({
                str(item.get("error_code", item.get("code")))
                for item in audit.get("validation_errors", ())
                if isinstance(item, Mapping) and item.get("node") == "generate_answer"
                and item.get("error_stage", item.get("stage")) == stage
                and item.get("error_code", item.get("code"))
            })
            if stage == "domain_validation" and any(
                rule in {"ANSWER_UNKNOWN_CITATION", "ANSWER_CITATION_INVALID"}
                for rule in rules
            ):
                code = "ANSWER_CITATION_INVALID"
            add("answer_validation", code,
                detail=(f"validation_rules={','.join(rules)}" if rules
                        else f"validation_stage={stage};rule_unavailable_in_audit"))
    return diagnostics


def _required_support_selected(audit: Mapping[str, Any]) -> bool:
    required = tuple(
        item for item in audit.get("requirements", ())
        if isinstance(item, Mapping) and item.get("required") is True
    )
    if not required or any(item.get("status") != "satisfied" for item in required):
        return False
    return all(item.get("answer") and item.get("citations") for item in required)


def failure_labels(*, answerable: bool, retrieval: Mapping[str, Any], retrieval_observed: bool = True, run_status: str | None,
                   stop_reason: str | None, audit: Mapping[str, Any], insufficient: bool,
                   partial: bool, no_selected: bool, run_error: str | None,
                   retrieval_error: str | None, abstention_correct: bool | None,
                   false_no_support: bool = False,
                   failure_diagnostics: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    labels: list[str] = []
    if retrieval_observed and retrieval_error:
        labels.append("retrieval_miss")
    elif retrieval_observed and answerable and retrieval.get("hit_at_5") == 0.0:
        labels.append("retrieval_miss")
    if stop_reason == "budget_exhausted": labels.append("global_budget_exhausted")
    if stop_reason == "invalid_action_limit": labels.append("invalid_action_stop")
    if any(item.get("rejection_stage") == "budget" for item in audit.get("rejected_actions", ()) if isinstance(item, Mapping)):
        labels.append("local_budget_exhausted")
    steps = tuple(audit.get("steps", ()) or ())
    if len(steps) >= 2 and all(int(item.get("added_count", 0) or 0) == 0 for item in steps[-2:] if isinstance(item, Mapping)):
        labels.append("agent_search_stagnation")
    if false_no_support: labels.append("false_no_support")
    elif answerable and no_selected: labels.append("no_selected_evidence")
    if any(item.get("code") == "ANSWER_GENERATION_FAILED" for item in failure_diagnostics):
        labels.append("answer_generation_failed")
    if partial: labels.append("partial_answer")
    if not answerable and abstention_correct is False: labels.append("unanswerable_not_abstained")
    if run_error: labels.append("other")
    return labels or ["none"]


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 6) if values else None


def aggregate_case_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable") is True]
    unanswerable = [row for row in rows if row.get("answerable") is False]
    return {
        "counts": {"cases": len(rows), "answerable": len(answerable), "unanswerable": len(unanswerable)},
        "evaluation_outcomes": dict(sorted(Counter(
            str(row.get("evaluation_outcome", "failure")) for row in rows
        ).items())),
        "retrieval": {key: _mean(answerable, key) for key in ("hit_at_5", "recall_at_5", "reciprocal_rank")},
        "agent_efficiency": {key: _mean(rows, key) for key in ("useful_action_rate", "tool_calls", "retrieval_queries", "llm_calls")},
        "research_completion": {
            "answerable_completion_rate": _mean(answerable, "completed_answerable"),
            "false_no_support_rate": _mean(answerable, "false_no_support"),
            "legacy_false_no_support_rate": _mean(answerable, "legacy_false_no_support"),
        },
        "agent_retrieval": {"gold_evidence_coverage": _mean(answerable, "agent_retrieved_gold_coverage")},
        "final_evidence": {"gold_evidence_coverage": _mean(answerable, "gold_evidence_coverage")},
        "grounding_safety": {"abstention_accuracy": _mean(unanswerable, "abstention_correct")},
    }


def metrics_by_type(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: grouped[str(row.get("primary_type"))].append(row)
    return {kind: {"n": len(grouped[kind]), **aggregate_case_metrics(grouped[kind])} for kind in TAXONOMY if grouped[kind]}


def aggregate_action_control_metrics(
    cases: Iterable[tuple[Mapping[str, Any], Mapping[str, Any] | None]],
) -> dict[str, Any]:
    """Aggregate public P0 control-plane audit fields without model text.

    The result intentionally contains only stable codes, field paths, counts,
    and case IDs.  Prompts, responses, evidence text, and hidden reasoning are
    never read into this report.
    """

    validation = Counter[tuple[str, str, str]]()
    canonical_operations = Counter[str]()
    guard_rejections = Counter[str]()
    duplicate_guard_rejections = 0
    duplicate_guard_recoveries = 0
    duplicate_validation_rejections = 0
    zero_tool_invalid_cases: list[str] = []
    audited_cases = 0

    for row, audit in cases:
        audit = audit or {}
        if audit:
            audited_cases += 1
        if (
            row.get("stop_reason") == "invalid_action_limit"
            and int(row.get("tool_calls") or 0) == 0
        ):
            zero_tool_invalid_cases.append(str(row.get("id")))
        for item in audit.get("validation_errors", ()):
            if not isinstance(item, Mapping):
                continue
            stage = str(item.get("stage", "unknown"))
            code = str(item.get("code", "unknown"))
            path = str(item.get("field_path", "unknown"))
            validation[(stage, code, path)] += 1
            if code in {"DuplicateAction", "DuplicatePaperQuery"}:
                duplicate_validation_rejections += 1
        for item in audit.get("rejected_actions", ()):
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("rejection_code", "unknown"))
            guard_rejections[code] += 1
            if code not in {"DuplicateAction", "DuplicatePaperQuery"}:
                continue
            duplicate_guard_rejections += 1
            rejected_key = _audit_action_key(
                str(item.get("action_name", "")), item.get("action_arguments")
            )
            rejected_iteration = int(item.get("iteration") or -1)
            later_steps = (
                step
                for step in audit.get("steps", ())
                if isinstance(step, Mapping)
                and int(step.get("iteration") or -1) > rejected_iteration
            )
            if any(
                _audit_action_key(
                    str(step.get("action_name", "")), step.get("action_arguments")
                )
                != rejected_key
                for step in later_steps
            ):
                duplicate_guard_recoveries += 1
        for container in (audit.get("steps", ()), audit.get("rejected_actions", ())):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                canonicalization = item.get("action_canonicalization")
                if not isinstance(canonicalization, Mapping) or not canonicalization.get("changed"):
                    continue
                for operation in canonicalization.get("operations", ()):
                    if isinstance(operation, Mapping):
                        canonical_operations[str(operation.get("operation", "unknown"))] += int(
                            operation.get("count") or 0
                        )

    return {
        "audited_case_count": audited_cases,
        "zero_tool_invalid_action": {
            "count": len(zero_tool_invalid_cases),
            "case_ids": zero_tool_invalid_cases,
        },
        "validation_error_breakdown": [
            {"stage": stage, "code": code, "field_path": path, "count": count}
            for (stage, code, path), count in sorted(
                validation.items(), key=lambda value: (-value[1], value[0])
            )
        ],
        "canonicalization_summary": {
            "operation_counts": dict(sorted(canonical_operations.items()))
        },
        "guard_rejection_breakdown": dict(sorted(guard_rejections.items())),
        "duplicate_recovery": {
            "guard_rejections": duplicate_guard_rejections,
            "guard_recoveries_with_materially_different_executed_action": (
                duplicate_guard_recoveries
            ),
            "preflight_validation_rejections": duplicate_validation_rejections,
        },
    }


def _audit_action_key(action_name: str, action_arguments: object) -> str:
    """Stable, non-textual comparison key for audit action recovery metrics."""

    arguments = action_arguments if isinstance(action_arguments, Mapping) else {}
    if action_name == "search_papers":
        searches = arguments.get("searches", ())
        normalized = []
        if isinstance(searches, Sequence) and not isinstance(searches, (str, bytes)):
            for item in searches:
                if not isinstance(item, Mapping):
                    continue
                terms = item.get("lexical_terms", ())
                normalized_terms = (
                    sorted({" ".join(str(term).casefold().split()) for term in terms})
                    if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes))
                    else []
                )
                normalized.append(
                    (
                        str(item.get("paper_id", "")),
                        " ".join(str(item.get("query", "")).casefold().split()),
                        normalized_terms,
                    )
                )
        value: object = sorted(normalized)
    elif action_name in {"search_knowledge_base", "search_references"}:
        value = {
            "paper_id": str(arguments.get("paper_id", "")),
            "query": " ".join(str(arguments.get("query", "")).casefold().split()),
        }
    elif action_name == "get_section_context":
        value = {
            "paper_id": str(arguments.get("paper_id", "")),
            "section_id": str(arguments.get("section_id", "")),
        }
    elif action_name == "get_paper_overview":
        value = {"paper_id": str(arguments.get("paper_id", ""))}
    else:
        value = {}
    return json.dumps([action_name, value], ensure_ascii=False, sort_keys=True)
