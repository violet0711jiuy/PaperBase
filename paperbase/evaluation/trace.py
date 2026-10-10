"""Read-only adapters for current Grounded Answer results and saved artifacts."""
from __future__ import annotations

from dataclasses import replace
from pydantic import TypeAdapter

from .contracts import (
    AnswerTrace, CaseTrace, EvidenceTrace, RequirementTrace,
    ResearchStep, RetrievalCandidate, json_safe,
)

SEARCH_TOOLS = frozenset({"search_papers", "search_references", "search_knowledge_base"})
CONTEXT_TOOLS = frozenset({"get_section_context", "get_paper_overview"})
RETRIEVAL_TOOLS = SEARCH_TOOLS | CONTEXT_TOOLS
TRACE_SCHEMA_VERSION = "case-trace-grounded-answer-v2"


def _ids(value):
    return tuple(dict.fromkeys(str(v) for v in value)) if value is not None else None


def trace_from_artifacts(result: dict, audit: dict, *, legacy_ranked_chunk_ids=None) -> CaseTrace:
    if result.get("case_trace_schema_version") == TRACE_SCHEMA_VERSION:
        trace = TypeAdapter(CaseTrace).validate_python(result["case_trace"])
        return replace(trace, legacy_ranked_chunk_ids=tuple(legacy_ranked_chunk_ids)) if legacy_ranked_chunk_ids is not None else trace

    gaps = ["pre_admission_candidates", "per_action_chunk_provenance", "executed_retrieval_receipts"]
    provenance = result.get("evidence_source_chunk_ids")
    answer_input = set(audit.get("answer_input_evidence_ids", ())) if "answer_input_evidence_ids" in audit else None
    cited_by_requirement: dict[str, list[str]] = {}
    for requirement in audit.get("requirements", ()):
        for citation in requirement.get("citations", ()):
            cited_by_requirement.setdefault(str(citation.get("evidence_id")), []).append(
                str(requirement.get("requirement_id"))
            )
    cited_ids = set(audit.get("cited_evidence_ids", ())) if "cited_evidence_ids" in audit else None
    selected_ids = set(audit.get("selected_evidence_ids", result.get("selected_evidence_ids", ())))

    evidence = None
    if "evidence" in audit:
        rows = []
        for row in audit["evidence"]:
            evidence_id = str(row["evidence_id"])
            chunks = _ids(row.get("source_chunk_ids"))
            if chunks is None and isinstance(provenance, dict): chunks = _ids(provenance.get(evidence_id))
            if chunks is None: gaps.append("registry_chunk_provenance")
            rows.append(EvidenceTrace(
                evidence_id, row.get("paper_id"), chunks,
                presented_to_answerer=(evidence_id in answer_input if answer_input is not None else None),
                cited_requirement_ids=(tuple(dict.fromkeys(cited_by_requirement.get(evidence_id, ())))
                                       if "requirements" in audit else None),
                selected=evidence_id in selected_ids,
                cited=(evidence_id in cited_ids if cited_ids is not None else row.get("cited")),
            ))
        evidence = tuple(rows)
    if answer_input is None: gaps.append("answer_input_evidence_ids")

    steps = None
    if "steps" in audit:
        steps = tuple(ResearchStep(
            int(row.get("iteration", index)), str(row.get("action_name", "")), True,
            None, None, None, int(row.get("added_count", 0)),
        ) for index, row in enumerate(audit["steps"], 1))

    requirements = None
    if "requirements" in audit:
        requirements = tuple(RequirementTrace(
            str(row["requirement_id"]), str(row.get("description", "")),
            row.get("required") is True, row.get("status"),
            bool(row.get("answer")) if "answer" in row else None,
            _ids(c.get("evidence_id") for c in row.get("citations", ())) if "citations" in row else None,
            row.get("remaining_gap"),
        ) for row in audit["requirements"])

    answer = None
    if result.get("answer"):
        row = result["answer"]
        answer = AnswerTrace(
            bool(str(row.get("answer", "")).strip()) if "answer" in row else row.get("status") not in (None, ""),
            row.get("insufficient_evidence"), row.get("partial_answer"),
            _ids(row.get("citations")), row.get("status"),
        )
    elif "answer" in result:
        answer = AnswerTrace(False, None, None, (), None)

    counters = tuple((key, value) for key, value in audit.get("budget", {}).items() if type(value) is int)
    decision_rows = [row for row in audit.get("llm_calls", ()) if row.get("node") == "decide_action"]
    decision_calls = (len({row["decision_iteration"] for row in decision_rows})
                      if decision_rows and all(type(row.get("decision_iteration")) is int for row in decision_rows)
                      else None)
    return CaseTrace(
        result.get("run_status", audit.get("run_status")),
        result.get("stop_reason", audit.get("stop_reason")), evidence, steps, requirements,
        tuple(selected_ids), _ids(result.get("selected_source_chunk_ids")), answer, counters,
        tuple(str(row.get("rejection_code", "unknown")) for row in audit.get("rejected_actions", ()))
        if "rejected_actions" in audit else None,
        len(audit.get("validation_errors", ())) if "validation_errors" in audit else None,
        tuple((str(row["node"]), bool(row["success"])) for row in audit.get("llm_calls", ()))
        if "llm_calls" in audit else None,
        tuple(legacy_ranked_chunk_ids) if legacy_ranked_chunk_ids is not None else None,
        audit.get("planning_mode"),
        setup_failure_stage=("scope" if result.get("stop_reason", audit.get("stop_reason")) == "scope_unresolved" else None),
        observability_gaps=tuple(dict.fromkeys(gaps)), decision_calls=decision_calls,
        steps_complete=steps is not None and dict(counters).get("tool_calls") == len(steps),
    )


def trace_from_result(result) -> CaseTrace:
    selected_ids = tuple(record.evidence_id for record in result.selected_evidence)
    answer_input_ids = set(result.answer_input_evidence_ids)
    grounded = tuple(result.answer.grounded_requirements) if result.answer else ()
    cited_by_evidence: dict[str, list[str]] = {}
    for requirement in grounded:
        for citation in requirement.citations:
            cited_by_evidence.setdefault(citation.evidence_id, []).append(requirement.requirement_id)
    evidence = tuple(EvidenceTrace(
        record.evidence_id, record.evidence.paper_id,
        tuple(record.evidence.source_chunk_ids), tuple(record.evidence.seed_chunk_ids),
        record.evidence_id in answer_input_ids,
        tuple(dict.fromkeys(cited_by_evidence.get(record.evidence_id, ()))),
        record.evidence_id in selected_ids,
        record.evidence_id in cited_by_evidence,
    ) for record in result.evidence)

    gaps = []
    steps = []
    for row in result.observations:
        candidates = None
        report = row.evidence_filter_report
        if report is not None and len(report.decisions) == report.total_candidates:
            candidates = tuple(RetrievalCandidate(
                decision.paper_id, tuple(decision.source_chunk_ids),
                retrieval_rank=decision.retrieval_rank,
            ) for decision in report.decisions)
        else:
            gaps.append("pre_admission_candidates")
        executed_queries = None
        retrieval_executed = None
        if row.action_name in SEARCH_TOOLS and row.request_executions:
            executed_queries = sum(request.get("executed") is True for request in row.request_executions)
            retrieval_executed = bool(executed_queries)
        elif row.action_name in CONTEXT_TOOLS and row.tool_status.value in {"ok", "empty", "partial"}:
            executed_queries, retrieval_executed = 0, True
        elif row.action_name in SEARCH_TOOLS:
            gaps.append("executed_retrieval_receipts")
        steps.append(ResearchStep(
            row.iteration, row.action_name, True, retrieval_executed,
            executed_queries, candidates, len(row.added_evidence_ids),
        ))

    states = {item.requirement_id: item.status.value for item in result.requirement_states}
    answer_by_id = {item.requirement_id: item for item in grounded}
    requirements = tuple(RequirementTrace(
        requirement.requirement_id, requirement.description, requirement.required,
        states.get(requirement.requirement_id),
        (answer_by_id[requirement.requirement_id].answer is not None
         if requirement.requirement_id in answer_by_id else False),
        (tuple(dict.fromkeys(c.evidence_id for c in answer_by_id[requirement.requirement_id].citations))
         if requirement.requirement_id in answer_by_id else ()),
        (answer_by_id[requirement.requirement_id].remaining_gap
         if requirement.requirement_id in answer_by_id else None),
    ) for requirement in (result.plan.requirements if result.plan else ()))
    cited = tuple(result.answer.citations) if result.answer else ()
    answer = (AnswerTrace(bool(result.answer.answer.strip()), result.answer.insufficient_evidence,
                          result.answer.partial_answer, cited, result.answer.status)
              if result.answer else AnswerTrace(False, None, None, (), None))
    setup = "scope" if result.query_context is None else "planning" if result.plan is None else None
    decision_rows = [row for row in result.llm_diagnostics if row.node == "decide_action"]
    decision_calls = (len({row.decision_iteration for row in decision_rows})
                      if decision_rows and all(row.decision_iteration is not None for row in decision_rows) else None)
    steps_complete = len(steps) == result.budget_usage.tool_calls
    if not steps_complete: gaps.append("incomplete_tool_observations")
    return CaseTrace(
        result.status.value, result.stop_reason.value, evidence, tuple(steps), requirements,
        selected_ids,
        tuple(dict.fromkeys(chunk for record in result.selected_evidence
                            for chunk in (*record.evidence.source_chunk_ids, *record.evidence.seed_chunk_ids))),
        answer,
        tuple((key, value) for key, value in json_safe(result.budget_usage).items() if type(value) is int),
        tuple(row.rejection_code for row in result.action_rejections),
        sum(len(row.validation_diagnostics) for row in result.llm_diagnostics),
        tuple((row.node, row.success) for row in result.llm_diagnostics),
        planning_mode=result.plan.planning_mode.value if result.plan else None,
        setup_failure_stage=setup, observability_gaps=tuple(dict.fromkeys(gaps)),
        decision_calls=decision_calls, steps_complete=steps_complete,
    )


def result_artifact(result) -> dict:
    from .run_agentic_eval import _result_payload
    payload = _result_payload(result)
    if result.answer: payload["answer"]["answer"] = result.answer.answer
    payload["case_trace_schema_version"] = TRACE_SCHEMA_VERSION
    payload["case_trace"] = json_safe(trace_from_result(result))
    return payload
