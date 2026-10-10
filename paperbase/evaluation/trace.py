"""Read-only adapters: current public ResearchRunResult and saved v1/v2 artifacts."""
from __future__ import annotations

from pydantic import TypeAdapter

from .contracts import (CaseTrace, EvidenceTrace, RetrievalCandidate, ResearchStep,
    RequirementTrace, AnswerTrace, json_safe)

SEARCH_TOOLS = frozenset({'search_papers', 'search_references', 'search_knowledge_base'})
CONTEXT_TOOLS = frozenset({'get_section_context', 'get_paper_overview'})
RETRIEVAL_TOOLS = SEARCH_TOOLS | CONTEXT_TOOLS
TRACE_SCHEMA_VERSION = 'case-trace-e1-v1'


def _ids(value):
    return tuple(dict.fromkeys(str(v) for v in value)) if value is not None else None


def trace_from_artifacts(result: dict, audit: dict, *, legacy_ranked_chunk_ids=None) -> CaseTrace:
    if result.get('case_trace_schema_version') == TRACE_SCHEMA_VERSION:
        trace = TypeAdapter(CaseTrace).validate_python(result['case_trace'])
        if legacy_ranked_chunk_ids is not None:
            from dataclasses import replace
            trace = replace(trace, legacy_ranked_chunk_ids=tuple(legacy_ranked_chunk_ids))
        return trace
    gaps = ['pre_admission_candidates', 'per_action_chunk_provenance', 'executed_retrieval_receipts']
    provenance = result.get('evidence_source_chunk_ids')
    evidence = None
    if 'evidence' in audit:
        evidence = []
        for row in audit['evidence']:
            eid = str(row['evidence_id'])
            sources = _ids(provenance.get(eid)) if isinstance(provenance, dict) and eid in provenance else None
            if sources is None:
                gaps.append('registry_chunk_provenance')
            projections = row.get('structured_projections')
            singleton = row.get('structured_projection')
            projected = bool(projections or singleton) if 'structured_projections' in row or 'structured_projection' in row else None
            verified = _ids(row.get('verified_supports_requirement_ids'))
            if verified is None:
                gaps.append('verified_support_authority')
            if 'evaluated_requirement_ids' not in row:
                gaps.append('historical_evaluated_scope')
            if 'structured_projections' not in row:
                gaps.append('historical_multi_projection')
            evidence.append(EvidenceTrace(eid, row.get('paper_id'), sources,
                qualified=row.get('analysis_status') == 'qualified' if 'analysis_status' in row else None,
                projected=projected, verified_requirement_ids=verified,
                selected=row.get('selected'), cited=row.get('cited')))
        evidence = tuple(evidence)
    steps = None
    if 'steps' in audit:
        # A public step proves a Tool boundary was invoked, but old snapshots do
        # not expose raw candidates or exact request execution receipts.
        support_authority_known = 'evidence' in audit and all('verified_supports_requirement_ids' in r for r in audit['evidence'])
        steps = tuple(ResearchStep(int(row.get('iteration', index)), str(row.get('action_name', '')),
            True, None, None, None, row.get('new_supporting_evidence_count') if support_authority_known else None)
            for index, row in enumerate(audit['steps'], 1))
    requirements = None
    if 'requirements' in audit:
        requirements = tuple(RequirementTrace(str(row['requirement_id']), str(row.get('description', '')),
            row.get('required') is True, row.get('projection_type'), row.get('status')) for row in audit['requirements'])
        if any(r.projection_type is None for r in requirements):
            gaps.append('requirement_projection_type')
    answer = None
    if result.get('answer'):
        row = result['answer']
        has = bool(str(row['answer']).strip()) if 'answer' in row else None
        if has is None:
            gaps.append('answer_text_presence')
        answer = AnswerTrace(has, row.get('insufficient_evidence'), row.get('partial_answer'),
            _ids(row.get('citations')), row.get('status'))
    elif 'answer' in result:
        answer = AnswerTrace(False, None, None, (), None)
    counters = tuple((k, v) for k, v in audit.get('budget', {}).items() if type(v) is int)
    decision_diagnostics=[r for r in audit.get('llm_calls',()) if r.get('node')=='decide_action']
    decision_calls=(len({r['decision_iteration'] for r in decision_diagnostics}) if decision_diagnostics
        and all(type(r.get('decision_iteration')) is int for r in decision_diagnostics) else None)
    return CaseTrace(result.get('run_status', audit.get('run_status')), result.get('stop_reason', audit.get('stop_reason')),
        evidence, steps, requirements, _ids(result.get('selected_evidence_ids')),
        _ids(result.get('selected_source_chunk_ids')), answer, counters,
        tuple(str(r.get('rejection_code', 'unknown')) for r in audit['rejected_actions']) if 'rejected_actions' in audit else None,
        len(audit['validation_errors']) if 'validation_errors' in audit else None,
        tuple((str(r['node']), bool(r['success'])) for r in audit['llm_calls']) if 'llm_calls' in audit else None,
        tuple(legacy_ranked_chunk_ids) if legacy_ranked_chunk_ids is not None else None,
        audit.get('planning_mode'), setup_failure_stage=('scope' if result.get('run_status',audit.get('run_status'))=='needs_clarification'
            or result.get('stop_reason',audit.get('stop_reason'))=='scope_unresolved' else None),
        observability_gaps=tuple(dict.fromkeys(gaps)), decision_calls=decision_calls,
        steps_complete=steps is not None and dict(counters).get('tool_calls')==len(steps))


def trace_from_result(result) -> CaseTrace:
    from paperbase.agentic.grounding import verified_support_ids
    from paperbase.agentic.projection_reads import iter_projections
    selected = tuple(record.evidence_id for record in result.selected_evidence)
    cited = tuple(result.answer.citations) if result.answer else ()
    requirements = result.plan.requirements if result.plan else ()
    evidence = tuple(EvidenceTrace(record.evidence_id, record.evidence.paper_id,
        tuple(record.evidence.source_chunk_ids), tuple(record.evidence.seed_chunk_ids),
        record.analysis_status.value == 'qualified', bool(tuple(iter_projections(record))),
        verified_support_ids(record, requirements),
        tuple(p.projection_type.value for p in iter_projections(record)), record.evidence_id in selected,
        record.evidence_id in cited) for record in result.evidence)
    gaps = []
    steps = []
    for row in result.observations:
        candidates = None
        report = row.evidence_filter_report
        if report is not None and len(report.decisions) == report.total_candidates:
            candidates = tuple(RetrievalCandidate(d.paper_id, tuple(d.source_chunk_ids),
                retrieval_rank=d.retrieval_rank) for d in report.decisions)
        else:
            gaps.append('pre_admission_candidates')
        queries = None
        retrieved = None
        if row.action_name in SEARCH_TOOLS and row.request_executions:
            queries = sum(r.get('executed') is True for r in row.request_executions)
            retrieved = bool(queries)
        elif row.action_name in CONTEXT_TOOLS and row.tool_status.value in {'ok', 'empty', 'partial'}:
            # Context is a real Tool result; it contributes provenance, not Top-K.
            retrieved = True
            queries = 0
        elif row.action_name in SEARCH_TOOLS:
            gaps.append('executed_retrieval_receipts')
        steps.append(ResearchStep(row.iteration, row.action_name, True, retrieved, queries,
            candidates, row.new_supporting_evidence_count))
    states = {r.requirement_id: r.status.value for r in result.requirement_states}
    requirement_trace = tuple(RequirementTrace(r.requirement_id, r.description, r.required,
        r.projection_type.value if r.projection_type is not None else None, states.get(r.requirement_id)) for r in requirements)
    if any(r.projection_type is None for r in requirement_trace):
        gaps.append('requirement_projection_type')
    answer = (AnswerTrace(bool(result.answer.answer.strip()), result.answer.insufficient_evidence,
        result.answer.partial_answer, cited, result.answer.status) if result.answer else AnswerTrace(False, None, None, (), None))
    setup = ('scope' if result.query_context is None else 'planning' if result.plan is None else None)
    decision_diagnostics=[r for r in result.llm_diagnostics if r.node=='decide_action']
    decision_calls=(len({r.decision_iteration for r in decision_diagnostics}) if decision_diagnostics
        and all(r.decision_iteration is not None for r in decision_diagnostics) else None)
    steps_complete=len(steps)==result.budget_usage.tool_calls
    if not steps_complete:
        gaps.append('incomplete_tool_observations')
    return CaseTrace(result.status.value, result.stop_reason.value, evidence, tuple(steps), requirement_trace,
        selected, tuple(dict.fromkeys(c for r in result.selected_evidence for c in
            (*r.evidence.source_chunk_ids, *r.evidence.seed_chunk_ids))), answer,
        tuple((k, v) for k, v in json_safe(result.budget_usage).items() if type(v) is int),
        tuple(r.rejection_code for r in result.action_rejections),
        sum(len(r.validation_diagnostics) for r in result.llm_diagnostics),
        tuple((r.node, r.success) for r in result.llm_diagnostics),
        planning_mode=result.plan.planning_mode.value if result.plan else None,
        setup_failure_stage=setup, observability_gaps=tuple(dict.fromkeys(gaps)),decision_calls=decision_calls,
        steps_complete=steps_complete)


def result_artifact(result) -> dict:
    # Reuse v1's safe result envelope, adding only public answer and typed trace.
    from .run_agentic_eval import _result_payload
    payload = _result_payload(result)
    if result.answer:
        payload['answer']['answer'] = result.answer.answer
    payload['case_trace_schema_version'] = TRACE_SCHEMA_VERSION
    payload['case_trace'] = json_safe(trace_from_result(result))
    return payload
