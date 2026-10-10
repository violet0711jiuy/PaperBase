"""E1 deterministic layer metrics. No gold-fact/Requirement semantic matching."""
from __future__ import annotations

from .agentic import gold_chunk_ids, ranking_metrics
from .contracts import (AgenticCaseEvaluation, RetrievalEvaluation, ResearchEvaluation,
    RequirementEvaluation, EvidenceFunnelEvaluation, AnswerEvaluation, EfficiencyEvaluation,
    CaseTrace, DatasetIssue)
from .trace import SEARCH_TOOLS, RETRIEVAL_TOOLS


def _chunks(item):
    return set(item.source_chunk_ids or ()) | set(item.seed_chunk_ids)


def evaluate_agentic_case(golden: dict, trace: CaseTrace, *,
                          dataset_issues: tuple[DatasetIssue, ...] = ()) -> AgenticCaseEvaluation:
    gold = gold_chunk_ids(golden)
    gold_set = set(gold)
    gaps = list(trace.observability_gaps)
    records = trace.evidence
    selected_ids = trace.selected_evidence_ids
    if selected_ids is None and records is not None and all(r.selected is not None for r in records):
        selected_ids = tuple(r.evidence_id for r in records if r.selected)
    answer = trace.answer
    citations = answer.citations if answer is not None else None

    def gold_ids(known, complete):
        overlap = gold_set.intersection(known)
        # Positive observations remain valid when history is incomplete. Full
        # gold exposure is exact even if non-gold candidate provenance is missing.
        return tuple(c for c in gold if c in overlap) if complete or overlap == gold_set else None

    def stage_ids(predicate):
        known = set(); complete = records is not None
        for r in records or ():
            flag = predicate(r)
            if flag is False:
                continue
            if r.source_chunk_ids is None or (gold_set.intersection(_chunks(r)) and flag is None):
                complete = False
            if flag is True:
                known.update(_chunks(r))
        return gold_ids(known, complete)

    registered = stage_ids(lambda r: True)
    projected = stage_ids(lambda r: r.projected)
    verified = stage_ids(lambda r: bool(r.verified_requirement_ids) if r.verified_requirement_ids is not None else None)
    selected = (gold_ids(trace.selected_source_chunk_ids, True) if trace.selected_source_chunk_ids is not None
        else stage_ids(lambda r: r.evidence_id in selected_ids if selected_ids is not None else r.selected))
    def legally_cited(r):
        is_cited = r.evidence_id in citations if citations is not None else r.cited
        is_selected = r.evidence_id in selected_ids if selected_ids is not None else r.selected
        if is_cited is False or is_selected is False:
            return False
        return True if is_cited is True and is_selected is True else None
    cited = stage_ids(legally_cited)
    selected_verified = stage_ids(lambda r: (bool(r.verified_requirement_ids) and r.evidence_id in selected_ids)
        if r.verified_requirement_ids is not None and selected_ids is not None else None)
    for name, value in [('registered',registered),('projected',projected),('verified',verified),
                        ('selected',selected),('cited',cited),('verified_selected',selected_verified)]:
        if value is None:
            gaps.append(f'gold_{name}_provenance_or_authority')

    retrieved_known = set(c for r in records or () for c in _chunks(r))
    retrieval_complete = trace.steps is not None and trace.steps_complete
    first = None; first_unknown = not trace.steps_complete; query_count = 0; queries_known = True
    until_gold = None; until_possible = trace.steps_complete
    for step in trace.steps or ():
        if step.tool not in RETRIEVAL_TOOLS or step.executed is False:
            continue
        if step.retrieval_executed is None:
            retrieval_complete = False
            if step.tool in SEARCH_TOOLS and first is None:
                first_unknown = True
            until_possible = False
            continue
        if not step.retrieval_executed:
            continue
        if step.tool in SEARCH_TOOLS:
            if first is None and not first_unknown:
                first = step
            if step.executed_queries is None:
                queries_known = False
            else:
                query_count += step.executed_queries
        if step.candidates is None:
            retrieval_complete = False; until_possible = False
            continue
        step_chunks = set(c for candidate in step.candidates for c in _chunks(candidate))
        retrieved_known.update(step_chunks)
        if gold_set.intersection(step_chunks) and until_gold is None and until_possible and queries_known:
            if step.tool in SEARCH_TOOLS and step.executed_queries == 1:
                until_gold = query_count
            else:
                # A context hit or multi-request action cannot identify the
                # first gold-producing retrieval query from these receipts.
                gaps.append('queries_until_first_gold_attribution')
            until_possible = False
    retrieved = gold_ids(retrieved_known, retrieval_complete)
    if retrieved is None:
        gaps.append('cumulative_retrieval_provenance')
    initial_chunks = None
    if first is not None and first.candidates is not None:
        initial_chunks = set(c for candidate in first.candidates for c in _chunks(candidate))
    elif first_unknown or first is not None:
        gaps.append('initial_search_provenance')

    def recall(ids):
        return len(ids) / len(gold) if ids is not None and gold else None

    budget = dict(trace.counters)
    retrieval_queries = budget.get('retrieval_queries', query_count if queries_known and retrieval_complete else None)
    legacy = ranking_metrics(trace.legacy_ranked_chunk_ids, gold) if trace.legacy_ranked_chunk_ids is not None else {}
    retrieval = RetrievalEvaluation(legacy.get('hit_at_5'), legacy.get('recall_at_5'),
        bool(gold_set.intersection(initial_chunks)) if initial_chunks is not None and gold else None,
        len(gold_set.intersection(initial_chunks))/len(gold) if initial_chunks is not None and gold else None,
        recall(retrieved), until_gold, retrieval_queries)
    funnel = EvidenceFunnelEvaluation(gold, retrieved, registered, projected, verified, selected, cited,
        tuple(c for c in gold if c in retrieved_known), selected_verified)

    support_complete = records is not None and all(r.verified_requirement_ids is not None for r in records)
    requirement_rows = []
    for req in trace.requirements or ():
        ids = tuple(r.evidence_id for r in records or () if req.requirement_id in (r.verified_requirement_ids or ()))
        requirement_rows.append(RequirementEvaluation(req.requirement_id, req.description, req.projection_type,
            req.status, req.required, ids if support_complete or ids else None))
    n = len(requirement_rows) if trace.requirements is not None else None
    required_n = sum(r.required for r in requirement_rows) if n is not None else None
    supported_n = sum(bool(r.verified_supporting_evidence_ids) for r in requirement_rows) if n is not None and support_complete else None
    research = ResearchEvaluation(n, required_n, supported_n, n-supported_n if n is not None and supported_n is not None else None,
        supported_n/n if n and supported_n is not None else None, tuple(requirement_rows))
    if n is None or not support_complete:
        gaps.append('requirement_support_metrics')

    valid_ids = {r.evidence_id for r in records} if records is not None else None
    legal = valid_ids.intersection(selected_ids) if valid_ids is not None and selected_ids is not None else None
    invalid = tuple(c for c in citations if c not in legal) if citations is not None and legal is not None else None
    validity = (len(citations)-len(invalid))/len(citations) if citations and invalid is not None else None
    selected_coverage = len(set(citations).intersection(selected_ids))/len(selected_ids) if selected_ids and citations is not None else None
    insufficient = answer.insufficient_evidence if answer else None
    has_answer = answer.has_answer if answer else None
    answerable = golden.get('answerable') is True
    correct_abstention = (insufficient is True and not citations) if not answerable and insufficient is not None and citations is not None else None
    unsafe = (insufficient is False and has_answer is True) if not answerable and insufficient is not None and has_answer is not None else None
    false_abstention = insufficient if answerable else None
    answer_metrics = AnswerEvaluation(has_answer, insufficient, answer.partial_answer if answer else None,
        len(citations) if citations is not None else None, validity, invalid, selected_coverage, recall(cited),
        correct_abstention, unsafe, false_abstention)
    if answer is None or has_answer is None:
        gaps.append('answer_presence')
    if invalid is None:
        gaps.append('citation_validity')

    executed = [s for s in trace.steps or () if s.executed is True]
    execution_complete = trace.steps is not None and trace.steps_complete and all(s.executed is not None for s in trace.steps)
    deltas_known = execution_complete and all(s.new_verified_support is not None for s in executed)
    peak = 0; streak = 0
    for s in executed:
        streak = streak+1 if s.new_verified_support == 0 else 0
        peak = max(peak, streak)
    first_support = next((i for i,s in enumerate(executed,1) if (s.new_verified_support or 0)>0),None) if deltas_known else None
    rejected = trace.guard_rejection_codes
    duplicates = sum('Duplicate' in c for c in rejected) if rejected is not None else None
    decision_llm_calls = sum(node == 'decide_action' for node,_ in trace.llm_nodes) if trace.llm_nodes is not None else None
    efficiency = EfficiencyEvaluation(budget.get('tool_calls',len(executed) if execution_complete else None),retrieval_queries,
        trace.decision_calls,len(rejected) if rejected is not None else None,duplicates,trace.validation_error_count,
        budget.get('agent_iterations'),budget.get('llm_calls'),len(executed) if execution_complete else None,
        sum(s.new_verified_support == 0 for s in executed) if deltas_known else None,first_support,peak if deltas_known else None,
        decision_llm_calls)
    if not deltas_known:
        gaps.append('per_action_verified_support_delta')
    if any(getattr(efficiency,k) is None for k in ('decision_calls','guard_rejections','validation_errors','iterations','llm_calls')):
        gaps.append('control_counters')

    labels = []; primary = None
    def failure(stage, label):
        nonlocal primary
        if primary is None:
            primary = stage
        labels.append(label)
    if dataset_issues:
        primary='dataset';labels=['dataset_error']
    else:
        if trace.setup_failure_stage:
            failure(trace.setup_failure_stage, f'{trace.setup_failure_stage}_setup_failure')
        if trace.requirements == ():
            failure('planning','no_usable_plan')
        planner_diagnostics = [success for node,success in trace.llm_nodes or () if node == 'build_plan']
        if trace.planning_mode == 'literal_fallback' and planner_diagnostics and not planner_diagnostics[-1]:
            labels.append('planner_validation_fallback')
        if answerable and gold:
            if retrieved == ():
                failure('retrieval','retrieval_miss')
            elif retrieved is not None and len(retrieved)<len(gold):
                failure('retrieval','retrieval_partial')
            if retrieved and registered == ():
                failure('registry','gold_retrieved_not_registered')
            if registered and verified == ():
                failure('projection','gold_registered_not_verified')
            if verified and selected_verified == ():
                failure('selection','gold_verified_not_selected')
            if selected_verified and insufficient is True:
                failure('answer','false_abstention')
            if selected_verified and has_answer is False and insufficient is not True:
                failure('answer','missing_answer')
            if selected and cited == ():
                failure('answer','gold_selected_not_cited')
        if answerable and insufficient is True:
            failure('answer','false_abstention')
            if supported_n == 0:
                labels.append('false_no_support')
        if unsafe is True:
            failure('answer','unsafe_non_abstention')
        if answer_metrics.partial_answer is True:
            failure('answer','partial_answer')
        if invalid:
            failure('answer','invalid_citation')
        if trace.stop_reason == 'budget_exhausted':
            labels.append('budget_exhausted')
        if trace.stop_reason == 'invalid_action_limit':
            labels.append('invalid_action_stop')
        if trace.stop_reason == 'fatal_error':
            failure('control','fatal_run_stop')
        stopped_control = trace.stop_reason in {'budget_exhausted','invalid_action_limit'}
        if stopped_control and (duplicates or 0)>=2:
            labels.append('duplicate_loop')
        if trace.stop_reason == 'invalid_action_limit' and rejected is not None and len(rejected)>=2:
            labels.append('guard_rejection_loop')
        if stopped_control and deltas_known and peak>=2:
            labels.append('agent_search_stagnation')
        if primary is None and any(l in labels for l in ('budget_exhausted','invalid_action_stop','duplicate_loop','guard_rejection_loop','agent_search_stagnation')):
            primary='control'
    return AgenticCaseEvaluation(str(golden.get('id','unknown')),str(golden.get('primary_type','unknown')),answerable,
        trace.run_status,trace.stop_reason,retrieval,research,funnel,answer_metrics,efficiency,primary,
        tuple(dict.fromkeys(labels)),tuple(dict.fromkeys(gaps)),dataset_issues)
