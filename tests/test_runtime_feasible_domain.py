"""PR-3 structural domain / canonical identity separation, without LLM calls."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    ActionRejectionContext, BudgetUsage, CandidateAssessment, CandidateDisposition,
    CandidateSelection, CorpusScopeSnapshot, RegistryState, ResearchObservation,
    RequirementState, RequirementStatus, RuntimeActionCapabilities,
    RuntimeUnavailablePaperQuery, QueryScope,
)
from paperbase.agentic.decision import (
    ActionDomainValidationError, DecisionState, _decision_json_schema,
    _runtime_action_capabilities, _to_action, _validate_action_references,
)
from paperbase.agentic.decision_view import build_compact_decision_view
from paperbase.agentic.guards import BudgetAndGuardTracker
from paperbase.agentic.schemas import AGENT_DECISION_ADAPTER
from paperbase.config import AgenticResearchSettings
from paperbase.retrieval.contracts import ToolStatus
from test_agentic_decision_view import record
from test_agentic_p0_action_control import _state


def section_record(paper, section):
    value = record(eid=f"{paper}-{section}")
    return replace(value, evidence=replace(value.evidence, paper_id=paper, section_id=section))


def state():
    return _state(evidence=(section_record("A", "intro"), section_record("A", "method"),
                            section_record("B", "result")))


def wire(tool, paper="A", *, query="dynamic graph", lexical=("adaptive",), section="intro"):
    if tool in {"try_answer", "stop_research"}:
        return {"decision": {"action": tool, "reason": "No more useful research"}}
    body = {"action": tool, "requirement_ids": ["R1"]}
    if tool != "search_knowledge_base":
        body["paper_id"] = paper
    if tool.startswith("search"):
        body.update(query=query, lexical_terms=list(lexical))
    if tool == "search_papers":
        body = {"action": tool, "searches": [{k: v for k, v in body.items() if k != "action"}]}
    if tool == "get_section_context":
        body["section_id"] = section
    return {"decision": body}


def action(payload):
    return _to_action(AGENT_DECISION_ADAPTER.validate_json(json.dumps(payload)).decision)


def schema_accepts(s, payload):
    schema = _decision_json_schema(s)
    Draft202012Validator.check_schema(schema)
    return not list(Draft202012Validator(schema).iter_errors(payload))


def rejection(payload):
    return ActionRejectionContext(payload["decision"]["action"], payload["decision"],
                                  ("R1",), "DuplicatePaperQuery", "effective request duplicated",
                                  "DuplicatePaperQuery", 2)


def test_paper_changed_effective_lexical_is_schema_and_guard_legal():
    s = state()
    first = wire("search_papers")
    s = replace(s, last_action_rejection=rejection(first), action_rejections=(rejection(first),))
    tracker = BudgetAndGuardTracker(AgenticResearchSettings(max_queries_per_paper=2))
    tracker.record_tool_call(action(first))
    changed = wire("search_papers", lexical=("adjacency",))
    assert schema_accepts(s, changed)
    _validate_action_references(action(changed), s)
    assert tracker.validate_action(action(changed), context=s.context, plan=s.plan, snapshot=None).accepted
    # An identical canonical request is structurally feasible, but Guard rejects it.
    assert schema_accepts(s, first)
    assert tracker.validate_action(action(first), context=s.context, plan=s.plan, snapshot=None).code == "DuplicateAction"


def test_reference_raw_query_is_not_schema_or_guard_identity():
    s = state()
    first = wire("search_references", query="q1", lexical=("wind", "svr"))
    s = replace(s, last_action_rejection=rejection(first))
    tracker = BudgetAndGuardTracker(AgenticResearchSettings(max_queries_per_paper=2))
    tracker.record_tool_call(action(first))
    raw_edit = wire("search_references", query="q2", lexical=("SVR", "wind"))
    assert schema_accepts(s, raw_edit)
    assert tracker.validate_action(action(raw_edit), context=s.context, plan=s.plan, snapshot=None).code == "DuplicateAction"
    changed = wire("search_references", query="q1", lexical=("lstm",))
    assert schema_accepts(s, changed)
    assert tracker.validate_action(action(changed), context=s.context, plan=s.plan, snapshot=None).accepted


@pytest.mark.parametrize("tool", ["search_papers", "search_references", "search_knowledge_base"])
@pytest.mark.parametrize("code", ["DuplicateAction", "DuplicatePaperQuery"])
def test_search_recovery_never_adds_raw_field_schema_exclusions(tool, code):
    s = state()
    if tool == "search_knowledge_base":
        s = replace(s, context=replace(s.context, scope=QueryScope.KNOWLEDGE_BASE,
                                       target_paper_ids=()))
    payload = wire(tool)
    previous = replace(rejection(payload), rejection_code=code)
    s = replace(s, last_action_rejection=previous, action_rejections=(previous,))
    assert schema_accepts(s, payload)
    for name in ("SearchSpecDraft", "SearchReferencesDraft", "SearchKnowledgeBaseDraft"):
        assert "allOf" not in _decision_json_schema(s)["$defs"][name]


def assert_root_matches_domain(s):
    capabilities = _runtime_action_capabilities(s)
    schema = _decision_json_schema(s)
    root = schema["properties"]["decision"]
    assert set(root["discriminator"]["mapping"]) == set(capabilities.available_tools)
    assert {v["$ref"] for v in root["oneOf"]} == set(root["discriminator"]["mapping"].values())
    assert "stop_research" in capabilities.available_tools
    assert schema_accepts(s, wire("stop_research"))


def test_per_paper_budget_filters_search_only_and_guard_remains_defensive():
    s = replace(state(), budget_usage=BudgetUsage(queries_per_paper=(("A", 2), ("B", 1))))
    caps = _runtime_action_capabilities(s)
    assert caps.search_paper_ids == caps.reference_paper_ids == ("B",)
    assert caps.overview_paper_ids == ("A", "B")
    assert ("A", ("intro", "method")) in caps.available_section_ids_by_paper
    for tool in ("search_papers", "search_references"):
        assert not schema_accepts(s, wire(tool, "A"))
        assert schema_accepts(s, wire(tool, "B"))
        with pytest.raises(ActionDomainValidationError, match="structural feasible domain"):
            _validate_action_references(action(wire(tool, "A")), s)
    assert schema_accepts(s, wire("get_paper_overview", "A"))
    assert schema_accepts(s, wire("get_section_context", "A"))
    assert_root_matches_domain(s)
    tracker = BudgetAndGuardTracker(AgenticResearchSettings(max_queries_per_paper=2))
    tracker.queries_per_paper["A"] = 2
    for tool in ("search_papers", "search_references"):
        assert tracker.validate_action(action(wire(tool, "A")), context=s.context,
                                       plan=s.plan, snapshot=None).code == "PerPaperBudget"


def test_empty_search_and_overview_domains_remove_only_their_root_branches():
    s = replace(state(), budget_usage=BudgetUsage(queries_per_paper=(("A", 2), ("B", 2))))
    assert set(_runtime_action_capabilities(s).available_tools) == {
        "try_answer", "stop_research", "get_paper_overview", "get_section_context"}
    assert not schema_accepts(s, wire("search_papers", "arbitrary"))
    assert not schema_accepts(s, wire("search_references", "arbitrary"))
    assert_root_matches_domain(s)
    s = replace(state(), overview_status_by_paper=(("A", "unavailable"), ("B", "unavailable")))
    assert "get_paper_overview" not in _runtime_action_capabilities(s).available_tools
    assert "search_papers" in _runtime_action_capabilities(s).available_tools
    assert not schema_accepts(s, wire("get_paper_overview", "arbitrary"))
    assert_root_matches_domain(s)


def test_overview_availability_and_tool_specific_enums():
    s = state()
    s = replace(s, context=replace(s.context, target_paper_ids=("A", "B", "C")),
                overview_status_by_paper=(("A", "available"), ("B", "unavailable"), ("C", "unknown")))
    caps = _runtime_action_capabilities(s)
    schema = _decision_json_schema(s)
    assert caps.overview_paper_ids == ("A", "C")
    assert schema["$defs"]["SearchReferencesDraft"]["properties"]["paper_id"]["enum"] == ["A", "B", "C"]
    assert schema["$defs"]["GetPaperOverviewDraft"]["properties"]["paper_id"]["enum"] == ["A", "C"]
    for paper in ("A", "C"):
        assert schema_accepts(s, wire("get_paper_overview", paper))
    for tool, paper in (("get_paper_overview", "B"), ("search_references", "outside")):
        assert not schema_accepts(s, wire(tool, paper))
        with pytest.raises(ActionDomainValidationError):
            _validate_action_references(action(wire(tool, paper)), s)
    assert_root_matches_domain(s)


@pytest.mark.parametrize("paper,section,valid", [
    ("A", "intro", True), ("A", "method", True), ("B", "result", True),
    ("A", "result", False), ("B", "intro", False), ("outside", "intro", False),
])
def test_correlated_section_schema_and_local_defense(paper, section, valid):
    s = state()
    payload = wire("get_section_context", paper, section=section)
    assert schema_accepts(s, payload) is valid
    if valid:
        _validate_action_references(action(payload), s)
    else:
        with pytest.raises(ActionDomainValidationError):
            _validate_action_references(action(payload), s)


@pytest.mark.parametrize("status,consumed", [
    (ToolStatus.ERROR, False), (ToolStatus.UNAVAILABLE, False),
    (ToolStatus.OK, True), (ToolStatus.EMPTY, True),
])
def test_section_consumption_requires_completed_read(status, consumed):
    s = state()
    obs = ResearchObservation(1, "get_section_context", status, (), (),
                              action_arguments={"paper_id": "A", "section_id": "intro"})
    s = replace(s, observations=(obs,))
    assert ("intro" not in dict(_runtime_action_capabilities(s).available_section_ids_by_paper)["A"]) is consumed
    assert schema_accepts(s, wire("get_section_context")) is not consumed
    if consumed:
        with pytest.raises(ActionDomainValidationError):
            _validate_action_references(action(wire("get_section_context")), s)
    else:
        _validate_action_references(action(wire("get_section_context")), s)


def test_kb_before_after_candidate_scope_and_empty_paper_domain():
    s = state()
    s = replace(s, context=replace(s.context, scope=QueryScope.KNOWLEDGE_BASE, target_paper_ids=()))
    assert _runtime_action_capabilities(s).available_tools == ("try_answer", "stop_research", "search_knowledge_base")
    assert schema_accepts(s, wire("search_knowledge_base"))
    for tool in ("search_papers", "search_references", "get_paper_overview", "get_section_context"):
        assert not schema_accepts(s, wire(tool))
    assert_root_matches_domain(s)
    selection = CandidateSelection(("A", "B", "C"), (
        CandidateAssessment("A", CandidateDisposition.CANDIDATE, "in scope"),
        CandidateAssessment("B", CandidateDisposition.UNCERTAIN, "needs evidence"),
        CandidateAssessment("C", CandidateDisposition.EXCLUDED, "out of scope")))
    s = replace(s, candidate_selection=selection)
    assert _runtime_action_capabilities(s).search_paper_ids == ("A", "B")
    assert not schema_accepts(s, wire("search_knowledge_base"))
    assert not schema_accepts(s, wire("search_papers", "C"))
    assert schema_accepts(s, wire("search_papers", "B"))
    assert_root_matches_domain(s)
    tracker = BudgetAndGuardTracker(AgenticResearchSettings())
    assert tracker.validate_action(action(wire("search_knowledge_base")), context=s.context,
        plan=s.plan, snapshot=CorpusScopeSnapshot.create(("A", "B", "C")),
        candidate_selection=selection).code == "CandidateScopeLocked"


def test_requirement_status_semantics_unchanged_and_empty_domain_stop_only():
    s = replace(state(), requirement_states=(RequirementState("R1", RequirementStatus.SATISFIED),
                                             RequirementState("R2", RequirementStatus.UNCERTAIN)))
    assert _runtime_action_capabilities(s).valid_requirement_ids == ("R1",)
    assert schema_accepts(s, wire("search_papers"))
    s = replace(s, requirement_states=(RequirementState("R1", RequirementStatus.NO_SUPPORT_FOUND),
                                       RequirementState("R2", RequirementStatus.UNCERTAIN)))
    assert _runtime_action_capabilities(s).available_tools == ("try_answer", "stop_research")
    assert_root_matches_domain(s)
    assert not schema_accepts(s, wire("search_papers"))


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [ROOT / "storage/evaluation" / folder / "decision_inputs.jsonl" for folder in (
    "p1_3b_hard_constraint_qwen38_20261002", "p1_3c_strict_decision_qwen38_20261002")]


def test_thirteen_historical_states_checkpoint_compatibility_and_pure_builders(monkeypatch):
    import paperbase.agentic.decision as decision
    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic domain builder called an LLM")
    monkeypatch.setattr(decision, "complete_json_with_diagnostics", forbidden)
    count = 0
    for path in INPUTS:
        for line in path.read_text(encoding="utf8").splitlines():
            s = TypeAdapter(DecisionState).validate_python(json.loads(line)["full_decision_state"])
            before = deepcopy(s)
            adapter = TypeAdapter(DecisionState)
            restored = adapter.validate_json(adapter.dump_json(s))
            assert restored == s
            registry = RegistryState(records=s.evidence)
            assert load_contract(dump_contract(registry, RegistryState), RegistryState) == registry
            assert load_contract(dump_contract(s.observations, tuple[ResearchObservation, ...]),
                                 tuple[ResearchObservation, ...]) == s.observations
            schema = _decision_json_schema(s)
            Draft202012Validator.check_schema(schema)
            for name in ("SearchSpecDraft", "SearchReferencesDraft", "SearchKnowledgeBaseDraft"):
                assert "allOf" not in schema["$defs"][name]
            assert_root_matches_domain(s)
            build_compact_decision_view(s)
            assert s == before
            count += 1
    assert count == 13


def test_legacy_capability_and_observation_fields_parse_without_guessing_identity():
    old = {"valid_paper_ids": ["A"], "tool_budget_state": {"available_tools": ["search_papers"]},
           "unavailable_paper_queries": [{"paper_id": "A", "query": "q", "reason_code": "ExecutedPaperQuery"}]}
    caps = TypeAdapter(RuntimeActionCapabilities).validate_python(old)
    assert caps.unavailable_paper_queries == (RuntimeUnavailablePaperQuery("A", "q", "ExecutedPaperQuery"),)
    assert caps.available_tools == caps.search_paper_ids == ()  # no inferred migration
    adapter = TypeAdapter(RuntimeActionCapabilities)
    assert adapter.validate_json(adapter.dump_json(caps)) == caps
    observation = TypeAdapter(ResearchObservation).validate_python({"iteration": 1,
        "action_name": "search_references", "tool_status": "ok", "added_evidence_ids": [],
        "reused_evidence_ids": [], "action_arguments": {"paper_id": "A", "query": "q"}})
    assert observation.request_identities == ()
    assert observation.request_contract_version == "legacy"
    s = replace(state(), observations=(observation,))
    assert _runtime_action_capabilities(s).unavailable_paper_queries == ()
    assert schema_accepts(s, wire("search_references", query="q"))


def test_schema_construction_uses_no_canonical_compiler(monkeypatch):
    import paperbase.agentic.search_requests as requests
    def forbidden(*args, **kwargs):
        raise AssertionError("Schema duplicated Search compiler")
    monkeypatch.setattr(requests, "compile_action_requests", forbidden)
    s = state()
    assert_root_matches_domain(s)
    build_compact_decision_view(s)
