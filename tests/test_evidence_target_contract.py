"""PR-4A: Plan-authoritative intent, source-neutral selection, zero LLM calls."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from paperbase.agentic.audit import AgenticAuditSnapshot, _step_audit, _step_from_payload
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    AgentAction, BudgetUsage, EvidenceAnalysis, EvidenceAnalysisStatus, EvidenceOwnership,
    EvidenceRecord, EvidenceTarget, FactItem, FactProjection, GetPaperOverviewAction,
    GetSectionContextAction, GuardState, QueryContext, RegistryDelta, RegistryState,
    ResearchObservation, ResearchPlan, ResearchRequirement, RequirementStatus,
    RequirementSupportAnchor, SearchKnowledgeBaseAction, SearchPapersAction, SearchReferencesAction,
)
from paperbase.agentic.decision import DecisionState, _legacy_state_prompt, _runtime_action_capabilities
from paperbase.agentic.decision_view import build_compact_decision_view, expand_decision_references
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.agentic.graph import _register_and_observe
from paperbase.agentic.grounding import verified_support_ids
from paperbase.agentic.observation import build_observation
from paperbase.agentic.requirement_state import (
    derive_requirement_states, required_requirements_researched, required_requirements_terminal,
)
from paperbase.agentic.requirement_targets import (
    preferred_tools_for_targets, requirement_target_bindings, targets_for_requirement_ids,
)
from paperbase.agentic.research_profile import build_research_profiles
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import (
    EvidenceSourceType, KnowledgeBaseSearchPayload, PaperOverviewPayload, PaperSearchOutcome,
    PaperSearchSpec, PaperSearchStatus, ReferenceSearchPayload, RetrievedEvidence,
    SearchPapersPayload, SectionContextPayload, ToolName, ToolResult, ToolStatus,
)
from test_agentic_p0_action_control import _context


TOOLS = ("search_papers", "search_references", "get_paper_overview", "get_section_context", "search_knowledge_base")


def plan(targets=(EvidenceTarget.METHOD,)):
    return ResearchPlan("Find evidence", (
        ResearchRequirement("R1", "Research the first need", frozenset(targets)),
        ResearchRequirement("R2", "Research results", frozenset({EvidenceTarget.RESULT})),
    ))


def evidence(source=EvidenceSourceType.CONTENT, paper="A", chunk="c1"):
    return RetrievedEvidence(source_type=source, paper_id=paper, section_id="method",
        section="Method", seed_chunk_ids=(chunk,), source_chunk_ids=(chunk,),
        content="The method uses a temporal encoder.")


def tool_pair(tool, ids=("R1",), *, query="focus", wire_targets=("result",)):
    if tool == "search_papers":
        spec = PaperSearchSpec("A", query, ("encoder",), frozenset(wire_targets), ids)
        chosen = SearchPapersAction((spec,))
        payload = SearchPapersPayload((PaperSearchOutcome(spec, PaperSearchStatus.OK, (evidence(),)),))
    elif tool == "search_references":
        chosen = SearchReferencesAction("A", query, ("encoder",), ids)
        payload = ReferenceSearchPayload("A", query, (evidence(EvidenceSourceType.BIBLIOGRAPHY),))
    elif tool == "get_paper_overview":
        chosen = GetPaperOverviewAction("A", ids)
        payload = PaperOverviewPayload("A", (evidence(EvidenceSourceType.OVERVIEW),))
    elif tool == "get_section_context":
        chosen = GetSectionContextAction("A", "method", ids)
        payload = SectionContextPayload("A", "method", (evidence(EvidenceSourceType.SECTION_CONTEXT),))
    else:
        chosen = SearchKnowledgeBaseAction(query, ("encoder",), ids)
        payload = KnowledgeBaseSearchPayload(query, (evidence(),))
    return chosen, ToolResult(ToolStatus.OK, ToolName(tool), payload)


def fact_record(source=EvidenceSourceType.CONTENT, *, targets=(EvidenceTarget.METHOD,), rank=None, eid="E1"):
    item = evidence(source)
    if rank is not None:
        item = replace(item, retrieval_rank=rank)
    projection = FactProjection(eid, (FactItem("method", "temporal encoder", EvidenceOwnership.THIS_PAPER,
        supported_requirement_ids=("R1",), proposed_requirement_ids=("R1",),
        support_anchors=(RequirementSupportAnchor("R1", item.content, "Temporal encoder"),)),))
    return EvidenceRecord(eid, item, ("focus",), 1, 1,
        requested_evidence_targets=frozenset(targets), requested_requirement_ids=("R1",),
        assessment=EvidenceAnalysis(eid, EvidenceAnalysisStatus.QUALIFIED, projection=projection))


def test_plan_target_union_rejects_unknown_and_never_interprets_query():
    p = plan((EvidenceTarget.REFERENCE, EvidenceTarget.PRIOR_WORK))
    original = deepcopy(p)
    assert targets_for_requirement_ids(p, ("R1", "R2", "R1")) == {
        EvidenceTarget.REFERENCE, EvidenceTarget.PRIOR_WORK, EvidenceTarget.RESULT}
    assert targets_for_requirement_ids(p, ()) == frozenset()
    with pytest.raises(ValueError, match="not in Plan"):
        targets_for_requirement_ids(p, ("unknown",))
    assert p == original


@pytest.mark.parametrize("tool,targets", [
    ("search_papers", (EvidenceTarget.METHOD,)),
    ("search_references", (EvidenceTarget.REFERENCE, EvidenceTarget.PRIOR_WORK)),
    ("get_paper_overview", (EvidenceTarget.FIELD_OVERVIEW,)),
    ("get_section_context", (EvidenceTarget.METHOD,)),
    ("search_knowledge_base", (EvidenceTarget.CLAIM,)),
])
def test_all_tools_propagate_plan_intent_to_registry_observation_and_history(tool, targets):
    p = plan(targets)
    original = deepcopy(p)
    chosen, result = tool_pair(tool)
    registry = EvidenceRegistry()
    delta = registry.register_tool_result(result, action=chosen, plan=p, iteration=1)
    record = registry.records[0]
    assert record.requested_evidence_targets == frozenset(targets)
    assert record.requested_requirement_ids == ("R1",)
    if tool == "search_references":
        assert record.evidence_id == "REF1"
    obs = build_observation(iteration=1, result=result, delta=delta, action=chosen, plan=p)
    row = obs.requirement_target_bindings[0]
    assert row["requirement_id"] == "R1"
    assert row["evidence_targets"] == sorted(t.value for t in targets)
    assert json.loads(json.dumps(obs.requirement_target_bindings))[0] == row
    restored = load_contract(dump_contract((obs,), tuple[ResearchObservation, ...]), tuple[ResearchObservation, ...])
    assert restored == (obs,)
    step = _step_audit(obs, {})
    assert _step_from_payload(json.loads(json.dumps(asdict(step)))) == step
    s = DecisionState(_context(), p, registry.records, (obs,), BudgetUsage())
    event = expand_decision_references(build_compact_decision_view(s).payload)["recent_research_events"][-1]
    assert event["requirement_ids"] == ["R1"]
    assert "requirement_target_bindings" not in event
    if tool == "search_papers":
        assert "evidence_targets" not in event["request"]["searches"][0]
    legacy = json.loads(_legacy_state_prompt(s, []))
    assert legacy["recent_observations"][-1]["requirement_target_bindings"] == list(obs.requirement_target_bindings)
    assert p == original
    assert not verified_support_ids(record, p.requirements)  # targets establish no support


def test_search_papers_batch_targets_remain_attributable_and_wire_cannot_override_plan():
    p = plan()
    specs = (PaperSearchSpec("A", "first", evidence_targets=frozenset({"result"}), requirement_ids=("R1",)),
             PaperSearchSpec("B", "second", evidence_targets=frozenset({"method"}), requirement_ids=("R2",)))
    chosen = SearchPapersAction(specs)
    result = ToolResult(ToolStatus.OK, ToolName.SEARCH_PAPERS, SearchPapersPayload(tuple(
        PaperSearchOutcome(spec, PaperSearchStatus.OK, (evidence(paper=spec.paper_id),)) for spec in specs)))
    registry = EvidenceRegistry()
    delta = registry.register_tool_result(result, action=chosen, plan=p, iteration=1)
    assert [r.requested_evidence_targets for r in registry.records] == [
        frozenset({EvidenceTarget.METHOD}), frozenset({EvidenceTarget.RESULT})]
    obs = build_observation(iteration=1, result=result, delta=delta, action=chosen, plan=p)
    assert obs.requirement_target_bindings == (
        {"search_index": 0, "paper_id": "A", "requirement_id": "R1", "evidence_targets": ["method"]},
        {"search_index": 1, "paper_id": "B", "requirement_id": "R2", "evidence_targets": ["result"]})
    assert [attempt.evidence_targets for attempt in obs.search_attempts] == [
        frozenset({EvidenceTarget.METHOD}), frozenset({EvidenceTarget.RESULT})]
    # Raw wire hint is auditable, while effective intent is separately Plan-derived.
    assert obs.action_arguments["searches"][0]["evidence_targets"] == ["result"]


@pytest.mark.parametrize("tool", TOOLS)
def test_production_graph_passes_existing_plan_to_registry_and_observation(tool):
    p = plan()
    chosen, result = tool_pair(tool)
    graph = dict(action=dump_contract(chosen, AgentAction), tool_result=dump_contract(result, ToolResult),
        query_context=dump_contract(_context(), QueryContext), plan=dump_contract(p, ResearchPlan),
        registry_state=dump_contract(RegistryState(), RegistryState),
        guard_state=dump_contract(GuardState(usage=BudgetUsage(agent_iterations=1)), GuardState))
    runtime = SimpleNamespace(context=SimpleNamespace(settings=AgenticResearchSettings()))
    update = _register_and_observe(graph, runtime)
    registry = load_contract(update["registry_state"], RegistryState)
    obs = load_contract(update["observations"], tuple[ResearchObservation, ...])[0]
    assert registry.records[0].requested_evidence_targets == {EvidenceTarget.METHOD}
    assert obs.requirement_target_bindings[0]["evidence_targets"] == ["method"]
    assert update["next_node"] == "analyze_evidence"


def test_registry_reuse_merges_plan_intent_and_queues_reanalysis_without_reset():
    p = plan()
    chosen, result = tool_pair("search_papers")
    registry = EvidenceRegistry()
    registry.register_tool_result(result, action=chosen, plan=p, iteration=1)
    initial = fact_record()
    registry.apply_analyses((initial.assessment,))
    before = registry.records[0]
    # KB returns the same underlying CONTENT Evidence via a new Requirement.
    chosen2, result2 = tool_pair("search_knowledge_base", ("R2",), query="result focus")
    delta = registry.register_tool_result(result2, action=chosen2, plan=p, iteration=2)
    reused = registry.records[0]
    assert delta.reused_ids == ("E1",) and not delta.added_ids
    assert delta.analysis_required_ids == ("E1",)
    assert reused.requested_requirement_ids == ("R1", "R2")
    assert reused.requested_evidence_targets == {EvidenceTarget.METHOD, EvidenceTarget.RESULT}
    assert reused.supporting_queries == ("focus", "result focus")
    assert reused.originating_tools == ("search_papers", "search_knowledge_base")
    assert reused.assessment == before.assessment
    assert verified_support_ids(reused, p.requirements) == ("R1",)  # retained while PR-4B is queued


def test_unknown_plan_binding_rejects_before_mutation_even_on_empty_result():
    p = plan()
    chosen, result = tool_pair("search_references", ("unknown",))
    result = replace(result, status=ToolStatus.EMPTY, payload=replace(result.payload, evidence_candidates=()))
    registry = EvidenceRegistry()
    before = registry.snapshot()
    with pytest.raises(ValueError, match="not in Plan"):
        registry.register_tool_result(result, action=chosen, plan=p, iteration=1)
    assert registry.snapshot() == before
    with pytest.raises(ValueError, match="not in Plan"):
        build_observation(iteration=1, result=result, delta=RegistryDelta((), ()), action=chosen, plan=p)


def test_legacy_target_read_and_reuse_do_not_promote_wire_or_guessed_intent():
    p = plan()
    chosen, result = tool_pair("search_papers")
    registry = EvidenceRegistry()
    registry.register_tool_result(result, action=chosen, iteration=1)  # no Plan compatibility
    assert registry.records[0].requested_evidence_targets == {EvidenceTarget.RESULT}
    # New production reuse replaces legacy wire intent with known Plan bindings.
    registry.register_tool_result(result, action=chosen, plan=p, iteration=2)
    assert registry.records[0].requested_evidence_targets == {EvidenceTarget.METHOD}
    record = registry.records[0]
    payload = dump_contract(RegistryState(records=(record,)), RegistryState)
    payload["records"][0]["covered_targets"] = ["result", "method"]
    restored = load_contract(payload, RegistryState).records[0]
    assert restored.requested_evidence_targets == record.requested_evidence_targets
    assert restored.covered_targets == frozenset()
    old_obs = asdict(build_observation(iteration=1, result=result, delta=RegistryDelta((), ()), action=chosen))
    old_obs.pop("requirement_target_bindings")
    assert TypeAdapter(ResearchObservation).validate_python(old_obs).requirement_target_bindings == ()
    empty_binding = replace(chosen, searches=(replace(chosen.searches[0], requirement_ids=()),))
    empty_result = replace(result, payload=SearchPapersPayload((replace(result.payload.paper_results[0], search=empty_binding.searches[0]),)))
    empty_registry = EvidenceRegistry()
    empty_registry.register_tool_result(empty_result, action=empty_binding, plan=p, iteration=1)
    assert empty_registry.records[0].requested_evidence_targets == frozenset()


def test_one_verified_fact_never_covers_all_targets_or_changes_support_state():
    p = plan()
    record = fact_record(targets=(EvidenceTarget.METHOD, EvidenceTarget.RESULT, EvidenceTarget.CLAIM))
    assert record.covered_targets == frozenset()
    states = derive_requirement_states(plan=p, evidence=(record,), observations=())
    assert [item.status for item in states] == [RequirementStatus.SATISFIED, RequirementStatus.PENDING]
    assert verified_support_ids(record, p.requirements) == ("R1",)
    assert not required_requirements_researched(p, states)
    assert not required_requirements_terminal(p, states)
    changed = replace(record, requested_evidence_targets=frozenset({EvidenceTarget.REFERENCE}))
    assert derive_requirement_states(plan=p, evidence=(changed,), observations=()) == states
    invalid = replace(record, assessment=replace(record.assessment, projection=None, projections=(replace(record.projection,
        facts=(replace(record.projection.facts[0], support_anchors=(
            RequirementSupportAnchor("R1", "invented source", "unverified"),)),)),)))
    assert not verified_support_ids(invalid, p.requirements)
    assert not FinalEvidenceSelector().select(plan=p, evidence=(invalid,))


def test_available_retrieval_targets_are_routes_not_completion_state(monkeypatch):
    p = plan()
    record = fact_record(targets=(EvidenceTarget.METHOD, EvidenceTarget.RESULT, EvidenceTarget.CLAIM))
    # Any accidental consumer of covered_targets fails immediately.
    def forbidden(self):
        raise AssertionError("covered_targets used as a completion authority")
    monkeypatch.setattr(EvidenceRecord, "covered_targets", property(forbidden))
    profiles = build_research_profiles(paper_ids=("A",), plan=p, evidence=(record,), observations=(),
                                     default_targets=frozenset({EvidenceTarget.GENERAL}))
    assert profiles[0].requested_targets == {EvidenceTarget.METHOD, EvidenceTarget.RESULT}
    assert profiles[0].remaining_targets == {EvidenceTarget.RESULT}
    s = DecisionState(_context(), p, (record,), (), BudgetUsage())
    view = build_compact_decision_view(s).payload
    rows = {r["requirement_id"]: r for r in view["requirements"]}
    assert rows["R1"]["available_retrieval_targets"] == ["method"]
    assert rows["R2"]["available_retrieval_targets"] == ["result"]
    assert "paper_state" not in view
    legacy = json.loads(_legacy_state_prompt(s, []))
    assert legacy["requirements"][1]["remaining_retrieval_targets"] == ["result"]
    # Shared targets cannot cancel an unsupported Requirement's hint.
    same_target = replace(p, requirements=(p.requirements[0], replace(p.requirements[1],
        evidence_targets=frozenset({EvidenceTarget.METHOD}))))
    assert build_research_profiles(paper_ids=("A",), plan=same_target, evidence=(record,),
                                  observations=())[0].remaining_targets == {EvidenceTarget.METHOD}


def test_routing_preferences_are_legal_soft_hints_not_a_bibliography_support_rule():
    p = plan((EvidenceTarget.REFERENCE, EvidenceTarget.FIELD_OVERVIEW))
    s = DecisionState(_context(), p, (), (), BudgetUsage())
    before = _runtime_action_capabilities(s)
    row = build_compact_decision_view(s).payload["requirements"][0]
    assert set(row["preferred_tools"]) == {"search_references", "get_paper_overview"}
    no_preferred = replace(s, per_paper_query_limit=1,
        budget_usage=BudgetUsage(queries_per_paper=(("A", 1), ("B", 1))),
        overview_status_by_paper=(("A", "unavailable"), ("B", "unavailable")))
    row = build_compact_decision_view(no_preferred).payload["requirements"][0]
    assert row["preferred_tools"] == ()
    assert row["available_retrieval_targets"] == ["field_overview", "reference"]
    assert row["retrieval_state"] == "not_searched"
    assert _runtime_action_capabilities(s) == before
    # CONTENT may support a REFERENCE Requirement through the unchanged integrity path.
    record = fact_record()
    assert verified_support_ids(record, p.requirements) == ("R1",)
    assert FinalEvidenceSelector().select(plan=p, evidence=(record,)) == (record,)


@pytest.mark.parametrize("source", [EvidenceSourceType.BIBLIOGRAPHY, EvidenceSourceType.OVERVIEW])
def test_selector_has_no_unconditional_source_penalty_or_boost(source):
    p = plan()
    content = fact_record(rank=5, eid="Z-content")
    alternate = fact_record(source, rank=1, eid="A-alternate")
    assert verified_support_ids(content, p.requirements) == verified_support_ids(alternate, p.requirements) == ("R1",)
    assert FinalEvidenceSelector._quality(content) == FinalEvidenceSelector._quality(alternate)
    assert FinalEvidenceSelector(max_units=1).select(plan=p, evidence=(content, alternate)) == (alternate,)
    # Rank reversal reverses the choice: source type creates no hidden boost either.
    content = replace(content, evidence=replace(content.evidence, retrieval_rank=1))
    alternate = replace(alternate, evidence=replace(alternate.evidence, retrieval_rank=5))
    assert FinalEvidenceSelector(max_units=1).select(plan=p, evidence=(content, alternate)) == (content,)


ROOT = Path(__file__).resolve().parents[1]
HISTORY = [ROOT / "storage/evaluation" / folder / "decision_inputs.jsonl" for folder in (
    "p1_3b_hard_constraint_qwen38_20261002", "p1_3c_strict_decision_qwen38_20261002")]


def test_thirteen_historical_states_parse_roundtrip_and_preserve_support_authority(monkeypatch):
    import paperbase.agentic.decision as decision
    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic target migration called a model")
    monkeypatch.setattr(decision, "complete_json_with_diagnostics", forbidden)
    count = 0
    for path in HISTORY:
        for line in path.read_text(encoding="utf8").splitlines():
            s = TypeAdapter(DecisionState).validate_python(json.loads(line)["full_decision_state"])
            before = deepcopy(s)
            registry = RegistryState(records=s.evidence)
            assert load_contract(dump_contract(registry, RegistryState), RegistryState) == registry
            assert load_contract(dump_contract(s.observations, tuple[ResearchObservation, ...]),
                                 tuple[ResearchObservation, ...]) == s.observations
            assert all(o.requirement_target_bindings == () for o in s.observations)
            support_before = tuple(verified_support_ids(r, s.plan.requirements) for r in s.evidence)
            build_compact_decision_view(s)
            assert tuple(verified_support_ids(r, s.plan.requirements) for r in s.evidence) == support_before
            assert s == before
            count += 1
    assert count == 13
