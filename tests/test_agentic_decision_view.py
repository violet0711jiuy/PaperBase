"""Zero-LLM input projection and observability contracts (P1.3)."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    ActionRejectionContext, BudgetUsage, EvidenceAnalysis, EvidenceAnalysisStatus,
    EvidenceOwnership, EvidenceRecord, EvidenceTarget, FactItem, FactProjection,
    Intent, MechanismProjection, QueryContext, QueryScope, ResearchGoal,
    ResearchObservation, ResearchPlan, ResearchRequirement, RequirementState,
    RequirementStatus, RequirementSupportAnchor, ScopeSource,
)
from paperbase.agentic.decision import (
    DecisionState, LLMDecisionAgent, _legacy_state_prompt, _runtime_action_capabilities,
)
from paperbase.agentic.decision_view import CompactDecisionLimits, build_compact_decision_view, expand_decision_references, build_recent_research_events, compute_no_progress_signal
from paperbase.agentic.audit import _llm_call
from paperbase.llm.client import LLMCallRecord, LLMCompletion
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence, ToolStatus


def record(eid="E1", rid="R1", scope="mean branch", gaps=("variance details",), iteration=1):
    source = f"{eid}: authentic source excerpt. " + "Source-only long text. " * 100
    anchor = RequirementSupportAnchor(rid, f"{eid}: authentic source excerpt.", scope, gaps)
    return EvidenceRecord(
        eid, RetrievedEvidence(source_type=EvidenceSourceType.CONTENT, paper_id="P",
            seed_chunk_ids=(eid,), source_chunk_ids=(eid,), section_id=f"section-{eid}",
            section="Methods", content=source), ("branch structure",), iteration, iteration,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requested_requirement_ids=(rid,), assessment=EvidenceAnalysis(
            eid, EvidenceAnalysisStatus.QUALIFIED,
            projection=MechanismProjection(eid, "LONG PROJECTION " * 150,
                supported_requirement_ids=(rid,), proposed_requirement_ids=(rid,), support_anchors=(anchor,)),
        ),
    )


def observation(iteration):
    args = {"searches": [{"paper_id": "P", "query": f"focused gap {iteration}",
                          "lexical_terms": [f"term{iteration}"], "requirement_ids": ["R2"]}]}
    return ResearchObservation(iteration, "search_papers", ToolStatus.OK, (f"E{iteration}",), (),
        action_arguments=args, new_supporting_evidence_count=1,
        newly_supported_requirement_ids=("R2",), newly_exposed_section_ids=(f"section-E{iteration}",))


def state(*, evidence=(), observations=(), iteration=0):
    context = QueryContext("Explain LSTM-EMVE", "Explain LSTM-EMVE", Intent.EXPLAIN,
        QueryScope.SINGLE_PAPER, ("P",), ScopeSource.UI_SELECTION, ResearchGoal.FIND_EVIDENCE,
        frozenset({EvidenceTarget.METHOD}))
    plan = ResearchPlan("Explain the mechanism", tuple(ResearchRequirement(
        f"R{i}", description, frozenset({EvidenceTarget.METHOD}), required=i != 3)
        for i, description in enumerate(("Explain LSTM-EMVE branches", "Explain variance loss", "Report limits"), 1)))
    return DecisionState(context, plan, evidence, observations,
        BudgetUsage(agent_iterations=iteration, queries_per_paper=(("P", 2),)),
        requirement_states=(RequirementState("R1", RequirementStatus.IN_PROGRESS, ("E1",)),
                            RequirementState("R2", RequirementStatus.SATISFIED, ("E2",)),
                            RequirementState("R3", RequirementStatus.NO_SUPPORT_FOUND)),
        max_agent_iterations=8, per_paper_query_limit=4,
        overview_status_by_paper=(("P", "available"),))


def test_all_requirements_goal_status_and_hints_preserved_without_completion_inference():
    s = state(evidence=(record(),))
    p = build_compact_decision_view(s).to_dict()
    assert p["question"] == s.context.resolved_query and p["research_goal"] == s.plan.goal
    assert p["scope"] == "single_paper" and p["trusted_paper_ids"] == ("P",)
    assert [r["retrieval_state"] for r in p["requirements"]] == [
        "support_available", "not_searched", "not_searched",
    ]
    assert [r["required"] for r in p["requirements"]] == [True, True, False]
    assert len(p["requirements"]) == 3
    assert p["requirements"][0]["lexical_hints"]["critical"] == ("LSTM-EMVE",)
    assert "complete" not in p["requirements"][1]


def test_local_gaps_keep_provenance_and_are_not_merged_into_global_missing():
    s = state(evidence=(record("E1", scope="X", gaps=("A",)),
                        record("E2", scope="A", gaps=("B",), iteration=2)))
    req = build_compact_decision_view(s).payload["requirements"][0]
    supports = {e["evidence_id"]: e for e in req["local_supports"]}
    assert supports["E1"]["boundaries"] == ["A"]
    assert supports["E2"]["covers"] == ["A"]
    assert supports["E2"]["boundaries"] == ["B"]
    assert [item["evidence_id"] for item in req["local_supports"]] == ["E2", "E1"]
    assert req["available_retrieval_targets"] == ["method"]
    assert "verified_support_evidence_ids" not in req
    assert "global_missing" not in json.dumps(req) and "boundaries" not in req


def test_multiple_anchors_from_one_evidence_are_grouped_once():
    original = record(scope="branch layout", gaps=("variance output",))
    second = RequirementSupportAnchor(
        "R1",
        "Source-only long text.",
        "training objective",
        ("loss weights",),
    )
    projection = replace(
        original.projection,
        support_anchors=(*original.projection.support_anchors, second),
    )
    grouped_record = replace(
        original,
        assessment=replace(
            original.assessment,
            projection=None,
            projections=(projection,),
        ),
    )

    supports = build_compact_decision_view(
        state(evidence=(grouped_record,))
    ).payload["requirements"][0]["local_supports"]

    assert len(supports) == 1
    assert supports[0]["evidence_id"] == "E1"
    assert supports[0]["covers"] == ["branch layout", "training objective"]
    assert supports[0]["boundaries"] == ["variance output", "loss weights"]


def test_raw_excerpt_projection_claims_and_unbounded_profiles_do_not_enter_view():
    s = state(evidence=(record(),))
    view = build_compact_decision_view(s).to_json()
    old = _legacy_state_prompt(s, [])
    assert "authentic source excerpt" in old and "LONG PROJECTION" in old
    assert not any(marker in view for marker in ("source_text", "source_excerpt", "Source-only", "qualified_claims", "research_profiles"))
    # Supported semantic content appears only in the formal support ledger,
    # never again as a retrieval lead.
    assert build_compact_decision_view(s).payload["research_leads"] == []


def test_research_leads_only_exist_for_zero_support_requirement_bindings():
    mixed = replace(record(), requested_requirement_ids=("R1", "R2"))
    view = build_compact_decision_view(state(evidence=(mixed,))).payload
    assert len(view["research_leads"]) == 1
    assert view["research_leads"][0]["requirement_ids"] == ["R2"]
    assert view["research_leads"][0]["usage"] == "retrieval_planning_only"


def test_bounded_windows_preserve_latest_tool_request_outcome_and_query_binding():
    s = state(observations=tuple(observation(i) for i in range(1, 9)), iteration=8)
    p = build_compact_decision_view(s).payload
    assert len(p["recent_research_events"]) == 3
    expanded = expand_decision_references(p)
    latest = expanded["recent_research_events"][-1]
    assert latest["iteration"] == 8 and latest["added_evidence"] == 1
    assert latest["new_verified_support"] == 1 and latest["new_sections"] == ["section-E8"]
    query = latest["request"]["searches"][0]
    assert query["query"] == "focused gap 8" and query["lexical_terms"] == ["term8"]
    assert query["paper_id"] == "P" and query["requirement_ids"] == ["R2"] and latest["outcome"] == "ok"
    assert "unavailable_paper_queries" not in expanded["runtime_action_capabilities"]
    assert json.dumps(p["recent_research_events"]).count('focused gap 8') == 1
    assert not {"research_requests", "research_queries"}.intersection(p)
    assert p['runtime_action_capabilities']['search_paper_ids'] == ('P',)



def test_runtime_and_recovery_fields_match_existing_authority_exactly():
    s = state(evidence=(record(),), observations=(observation(1),), iteration=1)
    rejection = ActionRejectionContext("search_papers", observation(1).action_arguments,
        ("R2",), "DuplicatePaperQuery", "executed query", "DuplicatePaperQuery", 2)
    s = replace(s, last_action_rejection=rejection, action_rejections=(rejection,))
    from paperbase.agentic.decision import _action_recovery_constraint
    p = expand_decision_references(build_compact_decision_view(s).payload)
    canonical = lambda x: json.loads(json.dumps(x, default=list))
    runtime = _runtime_action_capabilities(s)
    assert p["runtime_action_capabilities"] == canonical({
        "available_tools": runtime.available_tools,
        "valid_requirement_ids": runtime.valid_requirement_ids,
        "search_paper_ids": runtime.search_paper_ids,
        "reference_paper_ids": runtime.reference_paper_ids,
        "overview_paper_ids": runtime.overview_paper_ids,
        "available_section_ids_by_paper": runtime.available_section_ids_by_paper,
        "tool_budget_state": {
            key: value for key, value in runtime.tool_budget_state.items()
            if key != "available_tools"
        },
        "capability_reasons": runtime.capability_reasons,
    })
    assert p["action_recovery"] == canonical(_action_recovery_constraint(s))
    assert p["iteration_budget"] == {"limit": 8, "used": 1, "remaining": 7}
    assert p["runtime_action_capabilities"]["tool_budget_state"]["per_paper"][0]["remaining"] == 2


def test_rebuild_is_pure_uncached_and_returns_fresh_data():
    s = state(evidence=(record(),), observations=(observation(1),))
    original = deepcopy(s)
    first = build_compact_decision_view(s)
    first.payload["recent_research_events"][0]["request"]["searches"][0]["query"] = "mutated view"
    assert s == original
    second = build_compact_decision_view(s)
    assert "mutated view" not in second.to_json()
    assert second.to_json() == build_compact_decision_view(s).to_json()
    changed = replace(s, evidence=(*s.evidence, record("E2", rid="R2", iteration=2)))
    assert build_compact_decision_view(changed).payload["requirements"][1]["local_supports"][0]["evidence_id"] == "E2"


def test_limits_prioritize_distinct_recent_support_and_disclose_omissions():
    s = state(evidence=tuple(record(f"E{i}", iteration=i) for i in range(1, 9)))
    view = build_compact_decision_view(s, limits=CompactDecisionLimits(support_entries_per_requirement=2, sections=2))
    req = view.payload["requirements"][0]
    assert len(req["local_supports"]) == 2 and req["local_supports"][0]["evidence_id"] == "E8"
    assert req["local_supports_omitted"] == 6
    assert len(view.payload["available_context"]["section_metadata"]["P"]) == 2
    assert len(view.payload["runtime_action_capabilities"]["available_section_ids_by_paper"][0][1]) == 8
    unique = build_compact_decision_view(state(evidence=(record(rid="R2"),)), limits=CompactDecisionLimits(support_entries_per_requirement=1))
    assert unique.payload["requirements"][1]["local_supports"][0]["evidence_id"] == "E1"


def test_invalid_excerpt_and_unknown_fact_sibling_never_become_local_support():
    r = record()
    bad = replace(r.projection.support_anchors[0], source_excerpt="fabricated source")
    invalid = replace(r, assessment=replace(r.assessment, projection=None, projections=(replace(r.projection, support_anchors=(bad,)),)))
    assert build_compact_decision_view(state(evidence=(invalid,))).payload["requirements"][0]["local_supports"] == []
    anchor = r.projection.support_anchors[0]
    facts = FactProjection("E1", tuple(FactItem("kind", "value", owner, ("R1",), ("R1",),
        (replace(anchor, support_scope=scope),)) for owner,scope in ((EvidenceOwnership.THIS_PAPER,"valid"), (EvidenceOwnership.UNKNOWN,"invalid-owner"))))
    r = replace(r, assessment=replace(r.assessment, projection=None, projections=(facts,)))
    supports = build_compact_decision_view(state(evidence=(r,))).payload["requirements"][0]["local_supports"]
    assert [s["covers"] for s in supports] == [["valid"]]


def test_caps_and_normalized_dedup_are_local_and_disclosed():
    r = record(scope="x" * 1000, gaps=("A   B", "a b", "C" * 500, "D"))
    p = build_compact_decision_view(state(evidence=(r,)), limits=CompactDecisionLimits(
        support_scope_chars=80, unresolved_aspects=2, unresolved_aspect_chars=40)).payload
    support = p["requirements"][0]["local_supports"][0]
    assert support["text_truncated"] and len(support["covers"][0]) == 81
    assert support["boundaries"] == ["A B", "C" * 40 + "…"]


class Client:
    def __init__(self): self.calls = []
    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        return LLMCompletion(json.dumps({"decision": {"action": "finish", "reason": "done"}}),
            finish_reason="stop", prompt_tokens=900, completion_tokens=10)


@pytest.mark.parametrize("mode", ["compact", "shadow"])
def test_shadow_builds_both_but_calls_one_model_and_persists_only_sizes(mode):
    client = Client()
    s = state(evidence=(record(),), iteration=2)
    result = LLMDecisionAgent(client, view_mode=mode).decide(s)
    assert len(client.calls) == result.llm_calls == 1
    prompt = json.loads(client.calls[0]["user_prompt"])
    assert (prompt.get("view_type") == "AtomicDecisionView") == (mode == "compact")
    call = result.llm_diagnostics[0]
    assert call.decision_iteration == 3 and call.prompt_tokens == 900
    metrics = call.decision_view_metrics
    assert metrics["decision_view_chars"] < metrics["full_old_payload_chars"]
    assert metrics["number_of_requirements"] == 3
    restored = load_contract(dump_contract((call,), tuple[LLMCallRecord, ...]), tuple[LLMCallRecord, ...])[0]
    assert restored.decision_view_metrics == metrics
    audit = _llm_call(call)
    assert audit.decision_iteration == 3 and audit.decision_view_metrics == metrics
    # Existing checkpoints need no migration.
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_shadow_and_compact_share_current_policy_and_candidate_coverage_name():
    from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT, LEGACY_AGENT_DECISION_SYSTEM_PROMPT
    from paperbase.agentic.contracts import CoverageState, CoverageStrategy
    s = replace(state(), coverage=CoverageState(scope_paper_ids=('P',), assessments=(),
        strategy=CoverageStrategy.EXHAUSTIVE, candidate_paper_ids=('P',)))
    for mode in ('compact','shadow'):
        client = Client()
        LLMDecisionAgent(client, view_mode=mode).decide(s)
        assert client.calls[0]['system_prompt'] == AGENT_DECISION_SYSTEM_PROMPT
        payload = json.loads(client.calls[0]['user_prompt'])
        assert 'coverage' not in payload
        assert 'screening_complete' in payload['candidate_coverage']
        assert 'complete' not in payload['candidate_coverage']
        assert payload['runtime_action_capabilities']['valid_requirement_ids'] == ['R1','R2']
    assert LEGACY_AGENT_DECISION_SYSTEM_PROMPT is AGENT_DECISION_SYSTEM_PROMPT
    # Prompt simplification removes repeated self-checks; the same Runtime
    # authority and recovery semantics still accompany the current View.
    assert 'runtime_action_capabilities' in AGENT_DECISION_SYSTEM_PROMPT
    assert 'action_recovery' in AGENT_DECISION_SYSTEM_PROMPT
    assert 'recent_research_events' in AGENT_DECISION_SYSTEM_PROMPT
    assert 'requirement-level completion / coverage' not in LEGACY_AGENT_DECISION_SYSTEM_PROMPT


def test_legacy_call_checkpoint_defaults_and_view_limits_reject_zero():
    restored = load_contract([{"node": "decide_action", "attempt": 1, "success": True}], tuple[LLMCallRecord, ...])[0]
    assert restored.decision_iteration is None and restored.decision_view_metrics is None
    with pytest.raises(ValueError): CompactDecisionLimits(recent_observations=0)


def test_builder_has_no_model_dependency(monkeypatch):
    import paperbase.agentic.decision as decision
    def forbidden(*args, **kwargs): raise AssertionError("View called an LLM")
    monkeypatch.setattr(decision, "complete_json_with_diagnostics", forbidden)
    build_compact_decision_view(state(evidence=(record(),)))

def test_streaks_reset_independently_and_rejections_are_one_event():
    s = state(observations=(observation(1), replace(observation(2),
        action_name='get_section_context', new_supporting_evidence_count=0),
        replace(observation(3), added_evidence_ids=(), new_supporting_evidence_count=0)))
    rejection = ActionRejectionContext('search_papers', observation(3).action_arguments,
        ('R2',), 'DuplicateAction', 'executed', 'DuplicateAction', 4)
    s=replace(s,action_rejections=(rejection,),last_action_rejection=rejection)
    events=build_recent_research_events(s)
    assert len(events)==4 and events[-1]['outcome']=='DuplicateAction'
    assert compute_no_progress_signal(events)==dict(consecutive_no_support_actions=3,
        consecutive_no_new_evidence_actions=2,recent_section_reads_without_support=1,
        recent_duplicate_rejections=1,last_new_support_iteration=1,last_new_evidence_iteration=2,
        consecutive_no_targeted_support_actions=1,last_targeted_support_iteration=3)
    support_only=replace(observation(5),added_evidence_ids=())
    events=build_recent_research_events(replace(s,observations=(*s.observations,support_only)))
    signal=compute_no_progress_signal(events)
    assert signal['consecutive_no_support_actions']==0
    assert signal['consecutive_no_new_evidence_actions']==3 and signal['last_new_support_iteration']==5 and signal['last_new_evidence_iteration']==2
    events=build_recent_research_events(replace(s,observations=(*s.observations,
        replace(observation(5),new_supporting_evidence_count=0))))
    signal=compute_no_progress_signal(events)
    assert signal['consecutive_no_support_actions']==4 and signal['consecutive_no_new_evidence_actions']==0
    assert compute_no_progress_signal([])['last_new_support_iteration'] is None
    assert compute_no_progress_signal([])['last_new_evidence_iteration'] is None
    assert compute_no_progress_signal([])['last_targeted_support_iteration'] is None
    assert 'last_progress_iteration' not in signal


def test_atomic_events_do_not_create_query_or_request_side_tables():
    original=observation(1)
    repeated=replace(original,iteration=2,added_evidence_ids=(),new_supporting_evidence_count=0)
    rejection=ActionRejectionContext('search_papers', original.action_arguments,
        ('R2',),'DuplicatePaperQuery','executed','DuplicatePaperQuery',3)
    s=replace(state(observations=(original,repeated)),last_action_rejection=rejection,
              action_rejections=(rejection,))
    view=build_compact_decision_view(s);p=view.payload
    assert not {'research_requests','research_queries','no_progress_signal'} & p.keys()
    assert p['recent_research_events'][0]['request']['searches'][0]['query']=='focused gap 1'
    assert p['action_recovery']['blocked_action']['action_arguments']['searches'][0]['query']=='focused gap 1'
    assert len(p['recent_research_events'])==3
    assert p['recent_research_events'][-1]['outcome']=='DuplicatePaperQuery'
    assert not set(('recent_observations','recent_queries','recent_actions','last_search_outcome')) & p.keys()
    for r in s.plan.requirements:
        assert view.to_json().count(r.description)==1


def test_latest_event_and_last_search_survive_section_window():
    search=observation(1)
    reads=tuple(replace(observation(i),action_name='get_section_context',
        action_arguments={'paper_id':'P','section_id':f'section-{i}','requirement_ids':['R2']},
        added_evidence_ids=(),new_supporting_evidence_count=0) for i in range(2,8))
    p=build_compact_decision_view(state(observations=(search,*reads))).payload
    assert [e['iteration'] for e in p['recent_research_events']]==[1,5,6,7]
    assert all('request' in event and 'outcome' in event for event in p['recent_research_events'])


def test_search_focus_matches_atomic_input_and_surface_hints_are_not_completion():
    from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT, _SEARCH_FOCUS
    # The 2026-10-07 Prompt repair is authorized to replace the old full-view
    # field name. Validate the active interface instead of the P1.3c text hash.
    payload = build_compact_decision_view(state()).to_dict()
    assert 'surface_missing' in payload['requirements'][0]['lexical_hints']
    assert 'lexical_hints.surface_missing' in AGENT_DECISION_SYSTEM_PROMPT
    assert 'missing_critical_anchors' not in AGENT_DECISION_SYSTEM_PROMPT
    assert 'recent_research_events' in AGENT_DECISION_SYSTEM_PROMPT
    assert _SEARCH_FOCUS in AGENT_DECISION_SYSTEM_PROMPT
    assert 'local_supports' in AGENT_DECISION_SYSTEM_PROMPT
    assert 'boundaries' in AGENT_DECISION_SYSTEM_PROMPT
