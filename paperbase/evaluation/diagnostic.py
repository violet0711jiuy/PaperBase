"""Deterministic E1 metrics for retrieval, raw Registry and grounded citations."""
from __future__ import annotations

from .agentic import (
    covered_gold_evidence_ids,
    evidence_path_coverage,
    gold_chunk_ids,
    gold_evidence_alternatives,
    ranking_metrics,
)
from .contracts import (
    AgenticCaseEvaluation, AnswerEvaluation, CaseTrace, DatasetIssue,
    EfficiencyEvaluation, EvidenceFunnelEvaluation, RequirementEvaluation,
    ResearchEvaluation, RetrievalEvaluation,
)
from .trace import RETRIEVAL_TOOLS, SEARCH_TOOLS


def _chunks(item):
    return set(item.source_chunk_ids or ()) | set(item.seed_chunk_ids)


def evaluate_agentic_case(
    golden: dict, trace: CaseTrace, *, dataset_issues: tuple[DatasetIssue, ...] = (),
) -> AgenticCaseEvaluation:
    gold = gold_chunk_ids(golden)
    gold_set = set(gold)
    alternative_chunk_ids = {
        chunk_id
        for paths in gold_evidence_alternatives(golden).values()
        for path in paths
        for chunk_id in path
    }
    audited_chunk_ids = gold_set | alternative_chunk_ids
    gaps = list(trace.observability_gaps)
    records = trace.evidence

    def gold_ids(known, complete):
        overlap = gold_set.intersection(known)
        return tuple(chunk for chunk in gold if chunk in overlap) if complete or overlap == gold_set else None

    def path_covered_ids(known, complete):
        covered = covered_gold_evidence_ids(golden, known)
        return covered if complete or len(covered) == len(gold) else None

    def stage_ids(flag):
        known = set()
        exact_complete = path_complete = records is not None
        for record in records or ():
            value = flag(record)
            chunks = _chunks(record)
            if value is None:
                if gold_set.intersection(chunks):
                    exact_complete = False
                if audited_chunk_ids.intersection(chunks):
                    path_complete = False
            if value is True:
                if record.source_chunk_ids is None:
                    exact_complete = path_complete = False
                known.update(chunks)
        return (
            gold_ids(known, exact_complete),
            path_covered_ids(known, path_complete),
        )

    registered, registered_path = stage_ids(lambda _: True)
    answer_input, answer_input_path = stage_ids(
        lambda record: record.presented_to_answerer
    )
    cited, cited_path = stage_ids(lambda record: record.cited)
    for name, value in (("registered", registered), ("answer_input", answer_input), ("cited", cited)):
        if value is None: gaps.append(f"gold_{name}_provenance_or_authority")

    retrieved_known = {chunk for record in records or () for chunk in _chunks(record)}
    retrieval_complete = trace.steps is not None and trace.steps_complete
    first = None
    first_unknown = not trace.steps_complete
    query_count = 0
    queries_known = True
    until_gold = None
    until_possible = trace.steps_complete
    for step in trace.steps or ():
        if step.tool not in RETRIEVAL_TOOLS or step.executed is False: continue
        if step.retrieval_executed is None:
            retrieval_complete = False
            if step.tool in SEARCH_TOOLS and first is None: first_unknown = True
            until_possible = False
            continue
        if not step.retrieval_executed: continue
        if step.tool in SEARCH_TOOLS:
            if first is None and not first_unknown: first = step
            if step.executed_queries is None: queries_known = False
            else: query_count += step.executed_queries
        if step.candidates is None:
            retrieval_complete = False; until_possible = False; continue
        step_chunks = {chunk for candidate in step.candidates for chunk in _chunks(candidate)}
        retrieved_known.update(step_chunks)
        if gold_set.intersection(step_chunks) and until_gold is None and until_possible and queries_known:
            if step.tool in SEARCH_TOOLS and step.executed_queries == 1: until_gold = query_count
            else: gaps.append("queries_until_first_gold_attribution")
            until_possible = False
    retrieved = gold_ids(retrieved_known, retrieval_complete)
    retrieved_path = path_covered_ids(retrieved_known, retrieval_complete)
    if retrieved is None: gaps.append("cumulative_retrieval_provenance")
    initial_chunks = None
    if first is not None and first.candidates is not None:
        initial_chunks = {chunk for candidate in first.candidates for chunk in _chunks(candidate)}
    elif first_unknown or first is not None:
        gaps.append("initial_search_provenance")

    def recall(ids): return len(ids) / len(gold) if ids is not None and gold else None
    budget = dict(trace.counters)
    retrieval_queries = budget.get("retrieval_queries", query_count if queries_known and retrieval_complete else None)
    legacy = ranking_metrics(trace.legacy_ranked_chunk_ids, gold) if trace.legacy_ranked_chunk_ids is not None else {}
    retrieval = RetrievalEvaluation(
        legacy.get("hit_at_5"), legacy.get("recall_at_5"),
        bool(gold_set.intersection(initial_chunks)) if initial_chunks is not None and gold else None,
        len(gold_set.intersection(initial_chunks)) / len(gold) if initial_chunks is not None and gold else None,
        recall(retrieved), until_gold, retrieval_queries,
        evidence_path_coverage(golden, initial_chunks) if initial_chunks is not None else None,
        recall(retrieved_path),
    )
    funnel = EvidenceFunnelEvaluation(
        gold, retrieved, registered, answer_input, cited,
        tuple(chunk for chunk in gold if chunk in retrieved_known),
        retrieved_path, registered_path, answer_input_path, cited_path,
    )

    requirement_rows = tuple(RequirementEvaluation(
        row.requirement_id, row.description, row.status, row.required,
        row.answer_present, row.citation_evidence_ids, row.remaining_gap,
    ) for row in trace.requirements or ())
    requirement_count = len(requirement_rows) if trace.requirements is not None else None
    required_count = sum(row.required for row in requirement_rows) if requirement_count is not None else None
    answered_count = (sum(row.answer_present is True for row in requirement_rows)
                      if requirement_count is not None and all(row.answer_present is not None for row in requirement_rows)
                      else None)
    gap_count = (sum(bool(row.remaining_gap) for row in requirement_rows)
                 if requirement_count is not None else None)
    research = ResearchEvaluation(
        requirement_count, required_count, answered_count, gap_count,
        answered_count / required_count if required_count and answered_count is not None else None,
        requirement_rows,
    )
    if requirement_count is None or answered_count is None: gaps.append("requirement_answer_metrics")

    answer = trace.answer
    citations = answer.citations if answer else None
    registry_ids = {record.evidence_id for record in records} if records is not None else None
    input_ids = ({record.evidence_id for record in records if record.presented_to_answerer is True}
                 if records is not None and all(record.presented_to_answerer is not None for record in records)
                 else None)
    legal_ids = registry_ids.intersection(input_ids) if registry_ids is not None and input_ids is not None else None
    invalid = tuple(citation for citation in citations if citation not in legal_ids) if citations is not None and legal_ids is not None else None
    validity = ((len(citations) - len(invalid)) / len(citations)
                if citations and invalid is not None else None)
    input_coverage = (len(set(citations).intersection(input_ids)) / len(input_ids)
                      if input_ids and citations is not None else None)
    insufficient = answer.insufficient_evidence if answer else None
    has_answer = answer.has_answer if answer else None
    answerable = golden.get("answerable") is True
    correct_abstention = (insufficient is True and not citations
                          if not answerable and insufficient is not None and citations is not None else None)
    unsafe = (insufficient is False and has_answer is True
              if not answerable and insufficient is not None and has_answer is not None else None)
    false_abstention = insufficient if answerable else None
    answer_metrics = AnswerEvaluation(
        has_answer, insufficient, answer.partial_answer if answer else None,
        len(citations) if citations is not None else None, validity, invalid,
        input_coverage, recall(cited), recall(cited_path), correct_abstention,
        unsafe, false_abstention,
    )
    if answer is None or has_answer is None: gaps.append("answer_presence")
    if invalid is None: gaps.append("citation_validity")

    executed = [step for step in trace.steps or () if step.executed is True]
    execution_complete = trace.steps is not None and trace.steps_complete and all(step.executed is not None for step in trace.steps)
    deltas_known = execution_complete and all(step.new_evidence_count is not None for step in executed)
    peak = streak = 0
    for step in executed:
        streak = streak + 1 if step.new_evidence_count == 0 else 0
        peak = max(peak, streak)
    first_new = next((index for index, step in enumerate(executed, 1)
                      if (step.new_evidence_count or 0) > 0), None) if deltas_known else None
    rejected = trace.guard_rejection_codes
    duplicates = sum("Duplicate" in code for code in rejected) if rejected is not None else None
    decision_llm_calls = sum(node == "decide_action" for node, _ in trace.llm_nodes) if trace.llm_nodes is not None else None
    efficiency = EfficiencyEvaluation(
        budget.get("tool_calls", len(executed) if execution_complete else None), retrieval_queries,
        trace.decision_calls, len(rejected) if rejected is not None else None,
        duplicates, trace.validation_error_count, budget.get("agent_iterations"),
        budget.get("llm_calls"), len(executed) if execution_complete else None,
        sum(step.new_evidence_count == 0 for step in executed) if deltas_known else None,
        first_new, peak if deltas_known else None, decision_llm_calls,
    )
    if not deltas_known: gaps.append("per_action_evidence_delta")
    if any(getattr(efficiency, name) is None for name in
           ("decision_calls", "guard_rejections", "validation_errors", "iterations", "llm_calls")):
        gaps.append("control_counters")

    labels, primary = [], None
    def failure(stage, label):
        nonlocal primary
        if primary is None: primary = stage
        labels.append(label)

    if dataset_issues:
        primary, labels = "dataset", ["dataset_error"]
    else:
        if trace.setup_failure_stage: failure(trace.setup_failure_stage, f"{trace.setup_failure_stage}_setup_failure")
        if trace.requirements == (): failure("planning", "no_usable_plan")
        if answerable and gold:
            if retrieved_path == (): failure("retrieval", "retrieval_miss")
            elif retrieved_path is not None and len(retrieved_path) < len(gold):
                failure("retrieval", "retrieval_partial")
            if retrieved_path and registered_path == ():
                failure("registry", "gold_retrieved_not_registered")
            if registered_path and answer_input_path == ():
                failure("answer_input", "gold_registered_not_shown_to_answerer")
            if answer_input_path and cited_path == ():
                failure("grounding", "gold_answer_input_not_cited")
        if answerable and insufficient is True: failure("answer", "false_abstention")
        if unsafe is True: failure("answer", "unsafe_non_abstention")
        if answer_metrics.partial_answer is True: failure("answer", "partial_answer")
        if invalid: failure("grounding", "invalid_citation")
        required_gaps = [row for row in requirement_rows if row.required and row.remaining_gap]
        if required_gaps: failure("answer", "required_requirement_gap")
        if trace.stop_reason == "fatal_error": failure("control", "fatal_run_stop")
        if trace.stop_reason in {"budget_exhausted", "invalid_action_limit"}:
            labels.append(trace.stop_reason)
            if primary is None: primary = "control"
        if trace.stop_reason in {"budget_exhausted", "invalid_action_limit"} and deltas_known and peak >= 2:
            labels.append("agent_search_stagnation")
    return AgenticCaseEvaluation(
        str(golden.get("id", "unknown")), str(golden.get("primary_type", "unknown")),
        answerable, trace.run_status, trace.stop_reason, retrieval, research, funnel,
        answer_metrics, efficiency, primary, tuple(dict.fromkeys(labels)),
        tuple(dict.fromkeys(gaps)), dataset_issues,
    )
