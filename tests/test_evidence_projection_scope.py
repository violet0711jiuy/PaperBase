"""PR-4B Evidence-local scope, reanalysis lifecycle and unified protocol (mocked LLM)."""
from copy import deepcopy
from dataclasses import asdict, replace
import inspect
import json
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    AgentAction, EvidenceAnalysis, EvidenceAnalysisStatus as Status, EvidenceRecord, EvidenceTarget,
    EvidenceClaim, EvidenceOwnership, EvidenceRelation,
    GuardState, RegistryDelta, RegistryState, ResearchObservation, ResearchPlan,
    ResearchRequirement, RequirementStatus,
)
from paperbase.agentic.evidence_analysis import (
    EvidenceAnalysisRequest, EvidenceAnalysisResult, EvidenceAnalyzer,
    LegacyEvidenceAnalyzerAdapter, projection_requirement_scope, requires_analysis,
)
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.agentic.evidence_projection import LLMEvidenceProjector
from paperbase.agentic.projection_semantics import build_projection_semantic_request, projection_semantic_cache_context
from paperbase.agentic.contracts import EvidenceProjectionType
from paperbase.agentic.graph import _analyze_evidence, _register_and_observe
from paperbase.agentic.grounding import verified_support_ids
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.agentic.requirement_state import derive_requirement_states
from paperbase.config import AgenticResearchSettings
from test_agentic_batch_projector import Client, CONTEXT, fact, record as batch_record, item
from test_evidence_target_contract import HISTORY, fact_record, plan, tool_pair
from paperbase.agentic.decision import DecisionState


def old_record():
    source = fact_record()
    return replace(source, assessment=replace(source.assessment, evaluated_requirement_ids=("R1",)))


def expanded_record():
    return replace(old_record(), requested_requirement_ids=("R1", "R2"))


def request(records, p=None):
    p = p or plan()
    return EvidenceAnalysisRequest("facts", tuple(records), CONTEXT, p,
        tuple((r.evidence_id, projection_requirement_scope(r, p)) for r in records))


def response(record, ids=("R1",)):
    return {"projections": [{"projection_type": "fact", "evidence_id": record.evidence_id,
        "status": "qualified", "fact_type": "method", "value": "temporal encoder",
        "ownership": "this_paper", "supported_requirement_ids": list(ids),
        "support_candidates": [{"requirement_id": rid, "source_span_ids": ["S1"],
            "support_scope": "Temporal encoder", "unresolved_aspects": []} for rid in ids]}]}


def registry_for(record):
    return EvidenceRegistry(RegistryState((record,), content_counter=1))


def analyze_node(registry, analyzer):
    p = plan()
    state = dict(registry_state=dump_contract(registry.snapshot(), RegistryState),
        plan=dump_contract(p, ResearchPlan), query_context=dump_contract(CONTEXT, type(CONTEXT)),
        guard_state=dump_contract(GuardState(), GuardState), pending_analysis_ids=["E1"],
        observations=dump_contract((), tuple[ResearchObservation, ...]))
    update = _analyze_evidence(state, SimpleNamespace(context=SimpleNamespace(
        settings=AgenticResearchSettings(), evidence_analyzer=analyzer)))
    return load_contract(update["registry_state"], RegistryState).records[0], update


@pytest.mark.parametrize("ids", [("R1", "R2"), ("R1",)])
def test_reanalysis_complete_replacement_cases_a_and_b(ids):
    source = expanded_record()
    registry = registry_for(source)
    partial = response(source, ("R2",) if "R2" in ids else ())
    client = Client([partial])
    result = LLMEvidenceProjector(client).analyze(request=request((source,)))
    assert result.llm_calls == 1
    assert result.analyses[0].evaluated_requirement_ids == ("R1", "R2")
    registry.apply_analyses(result.analyses, plan=plan())
    current = registry.records[0]
    assert verified_support_ids(current, plan().requirements) == ids
    assert not requires_analysis(current, plan())
    payload = json.loads(client.calls[0]["user_prompt"])
    assert payload["projection_requirement_ids"] == ["R2"]
    assert payload["evidence"][0]["existing_verified_requirement_ids"] == []
    assert current.projections[0] == source.projections[0]


@pytest.mark.parametrize("failure", ["schema", "provider", "regression", "irrelevant"])
def test_reanalysis_failure_preserves_old_support_scope_and_assessment(failure):
    source = expanded_record()
    if failure == "provider":
        values = [RuntimeError("provider unavailable")]
    elif failure == "regression":
        values = [response(source, ("R1",))] * 2  # only missing R2 is allowed
    elif failure == "irrelevant":
        values = [{"projections": [{"projection_type": "mechanism"}]}] * 2
    else:
        values = [{"projections": []}] * 2
    client = Client(values)
    registry = registry_for(source)
    result = LLMEvidenceProjector(client).analyze(request=request((source,)))
    assert result.analyses[0] == source.assessment  # aggregate keeps the successful sibling
    registry.apply_analyses(result.analyses, plan=plan())
    assert registry.records[0].assessment == source.assessment
    assert verified_support_ids(registry.records[0], plan().requirements) == ("R1",)
    assert requires_analysis(registry.records[0], plan())


def test_invalid_missing_scope_uses_existing_repair_without_reproducing_old_support():
    source = expanded_record()
    client = Client([response(source, ("R1",)), response(source, ("R2",))])
    result = LLMEvidenceProjector(client, max_attempts=10).analyze(request=request((source,)))
    assert result.llm_calls == 2
    assert result.analyses[0].status is Status.QUALIFIED
    assert result.analyses[0].evaluated_requirement_ids == ("R1", "R2")
    registry = registry_for(source)
    registry.apply_analyses(result.analyses, plan=plan())
    assert verified_support_ids(registry.records[0], plan().requirements) == ("R1", "R2")


@pytest.mark.parametrize("tool", ["search_papers", "search_references", "get_paper_overview",
                                    "get_section_context", "search_knowledge_base"])
def test_registry_delta_routes_added_scope_expansion_but_not_evaluated_reuse(tool):
    registry = EvidenceRegistry()
    action, result = tool_pair(tool)
    p = plan()
    added = registry.register_tool_result(result, action=action, plan=p, iteration=1)
    assert added.analysis_required_ids == added.added_ids
    source = registry.records[0]
    assessment = EvidenceAnalysis(source.evidence_id, Status.IRRELEVANT, evaluated_requirement_ids=("R1",))
    registry.apply_analyses((assessment,), plan=p)
    same = registry.register_tool_result(result, action=action, plan=p, iteration=2)
    assert same.analysis_required_ids == () and same.reused_ids == added.added_ids
    action2, result2 = tool_pair(tool, ("R2",), query="other need")
    expanded = registry.register_tool_result(result2, action=action2, plan=p, iteration=3)
    assert expanded.analysis_required_ids == added.added_ids
    assert expanded.added_ids == ()
    assert registry.records[0].assessment == assessment
    registry.apply_analyses((replace(assessment, evaluated_requirement_ids=("R1", "R2")),), plan=p)
    assert registry.register_tool_result(result2, action=action2, plan=p, iteration=4).analysis_required_ids == ()


def test_graph_register_routes_reused_qualified_evidence_without_resetting_it():
    source = old_record()
    registry = registry_for(source)
    chosen, result = tool_pair("search_knowledge_base", ("R2",))
    state = dict(registry_state=dump_contract(registry.snapshot(), RegistryState),
        plan=dump_contract(plan(), ResearchPlan), query_context=dump_contract(CONTEXT, type(CONTEXT)),
        guard_state=dump_contract(GuardState(), GuardState), observations=[], traces=[],
        action=dump_contract(chosen, AgentAction),
        tool_result=dump_contract(result, type(result)))
    update = _register_and_observe(state, SimpleNamespace(context=SimpleNamespace(settings=AgenticResearchSettings())))
    assert update["pending_analysis_ids"] == ["E1"]
    assert load_contract(update["registry_state"], RegistryState).records[0].assessment == source.assessment


def test_graph_unified_call_reanalyzes_qualified_evidence_and_exceptions_preserve_it():
    source = expanded_record()
    current, update = analyze_node(registry_for(source), LLMEvidenceProjector(Client([response(source, ("R2",))])))
    assert current.assessment.evaluated_requirement_ids == ("R1", "R2")
    assert update["pending_analysis_ids"] == []
    class Raises:
        def analyze(self, *, request):
            assert request.scope_for("E1") == ("R1", "R2")
            raise RuntimeError("failure")
    failed, _ = analyze_node(registry_for(source), Raises())
    assert failed.assessment == source.assessment


@pytest.mark.parametrize("with_plan", [False, True])
def test_registry_rejects_support_regression_without_union_or_overwrite(with_plan):
    source = expanded_record()
    registry = registry_for(source)
    candidate = replace(source.assessment, evaluated_requirement_ids=("R1", "R2"),
        projection=None, projections=(replace(source.projection, facts=(replace(source.projection.facts[0],
            supported_requirement_ids=("R2",), proposed_requirement_ids=("R2",),
            support_anchors=(replace(source.projection.facts[0].support_anchors[0], requirement_id="R2"),)),)),))
    assert registry.apply_analyses((candidate,), **({"plan": plan()} if with_plan else {})) == ()
    assert registry.records[0] == source


def test_failed_reanalysis_preserves_previous_irrelevant_evaluation():
    source = replace(expanded_record(), assessment=EvidenceAnalysis("E1", Status.IRRELEVANT,
        evaluated_requirement_ids=("R1",)))
    registry = registry_for(source)
    registry.apply_analyses((EvidenceAnalysis("E1", Status.FAILED, error_code="failed"),), plan=plan())
    assert registry.records[0].assessment == source.assessment


def test_successful_irrelevant_records_full_scope_and_prevents_reanalysis():
    source = replace(expanded_record(), assessment=None)
    payload = {"projections": [{"projection_type": "fact", "evidence_id": "E1", "status": "irrelevant",
        "fact_type": None, "value": None, "ownership": None, "supported_requirement_ids": [],
        "support_candidates": []}]}
    result = LLMEvidenceProjector(Client([payload])).analyze(request=request((source,)))
    assert result.analyses[0].evaluated_requirement_ids == ("R1", "R2")
    assert not requires_analysis(replace(source, assessment=result.analyses[0]), plan())


def test_scope_binding_not_status_content_or_plan_wide_default():
    p = plan()
    source = old_record()
    assert derive_requirement_states(plan=p, evidence=(source,), observations=())[0].status is RequirementStatus.SATISFIED
    assert projection_requirement_scope(source, p) == ("R1",)
    assert request((source,)).scope_for("E1") == ("R1",)  # SATISFIED still allowed
    source = replace(source, requested_requirement_ids=("unknown", "R2", "R1"))
    assert projection_requirement_scope(source, p) == ("R1", "R2")
    assert projection_requirement_scope(replace(source, requested_requirement_ids=()), p) == ()
    with pytest.raises(ValueError, match="nonempty"):
        request((replace(source, requested_requirement_ids=()),))


@pytest.mark.parametrize("case", ["duplicate_evidence", "duplicate_scope", "missing_scope",
                                  "unknown_evidence", "unknown_requirement", "empty", "broader_than_binding"])
def test_request_validates_unique_exact_evidence_local_scope(case):
    source = old_record()
    records = (source, source) if case == "duplicate_evidence" else (source,)
    scopes = {
        "duplicate_evidence": (("E1", ("R1",)),),
        "duplicate_scope": (("E1", ("R1",)), ("E1", ("R1",))),
        "missing_scope": (), "unknown_evidence": (("E2", ("R1",)),),
        "unknown_requirement": (("E1", ("unknown",)),), "empty": (("E1", ()),),
        "broader_than_binding": (("E1", ("R1", "R2")),),
    }[case]
    with pytest.raises(ValueError):
        EvidenceAnalysisRequest("facts", records, CONTEXT, plan(), scopes)


def test_scope_for_unknown_evidence_and_immutable_request():
    original = old_record()
    before = deepcopy(original)
    req = request((original,))
    with pytest.raises(KeyError):
        req.scope_for("unknown")
    with pytest.raises(AttributeError):
        req.question = "different"
    assert original == before


def test_batch_grouping_uses_local_scope_and_keeps_same_scope_batched():
    p = ResearchPlan("facts", tuple(ResearchRequirement(rid, rid, frozenset({EvidenceTarget.PRIOR_WORK}))
        for rid in ("R1", "R2", "R3")))
    records = tuple(replace(batch_record(n), requested_requirement_ids=scope)
        for n, scope in [(1, ("R1",)), (2, ("R1",)), (3, ("R2", "R3"))])
    third = fact(3)
    third["supported_requirement_ids"] = ["R2"]
    third["support_candidates"][0]["requirement_id"] = "R2"
    client = Client([{"items": [item(2), item(1)]}, {"projections": [third]}])
    result = LLMEvidenceProjector(client).analyze(request=request(records, p))
    assert result.llm_calls == 2
    assert [a.evidence_id for a in result.analyses] == ["E1", "E2", "E3"]
    assert [a.evaluated_requirement_ids for a in result.analyses] == [("R1",), ("R1",), ("R2", "R3")]
    payloads = [json.loads(call["user_prompt"]) for call in client.calls]
    assert [len(p["evidence"]) for p in payloads] == [2, 1]
    assert [p["projection_requirement_ids"] for p in payloads] == [["R1"], ["R2", "R3"]]
    assert all("active_requirement_ids" not in p and "active_requirements" not in p for p in payloads)


def test_batch_empty_missing_projection_keeps_old_support_and_retains_valid_peer():
    p = plan()
    first = expanded_record()
    second = replace(first, evidence_id="E2", assessment=None, fingerprint="",
        evidence=replace(first.evidence, source_chunk_ids=("other",), content="Other source has a temporal encoder."))
    second = replace(second, requested_requirement_ids=("R2",))
    valid = response(second, ("R2",))["projections"][0]
    valid["evidence_id"] = "source:" + second.fingerprint
    payload = {"items": [{"evidence_id": "source:" + first.fingerprint, "projections": []},
        {"evidence_id": "source:" + second.fingerprint, "projections": [valid]}]}
    result = LLMEvidenceProjector(Client([payload])).analyze(request=request((first, second), p))
    assert result.llm_calls == 1
    assert all(a.status is Status.QUALIFIED for a in result.analyses)
    assert result.analyses[0].evaluated_requirement_ids == ("R1", "R2")
    assert result.analyses[0].projections == first.projections


def test_cache_preserves_evaluated_scope_and_expansion_misses_without_fingerprint_redesign(tmp_path):
    source = expanded_record()
    initial = replace(source, requested_requirement_ids=("R1",), assessment=None)
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "scope.sqlite3")
    first = LLMEvidenceProjector(Client([response(initial)]), cache=cache).analyze(request=request((initial,)))
    assert first.cache_stats.writes == 1
    hit = LLMEvidenceProjector(Client([]), cache=cache).analyze(request=request((initial,)))
    assert hit.cache_stats.hits == 1 and hit.analyses[0].evaluated_requirement_ids == ("R1",)
    expanded = LLMEvidenceProjector(Client([response(source, ("R2",))]), cache=cache).analyze(request=request((source,)))
    assert expanded.llm_calls == 1 and expanded.cache_stats.misses == 1
    assert source.fingerprint == initial.fingerprint
    assert expanded.analyses[0].evaluated_requirement_ids == ("R1", "R2")


@pytest.mark.parametrize("legacy_scope", [(), ("R1", "R2")])
def test_cache_missing_group_never_erases_existing_projection(tmp_path, legacy_scope):
    source = expanded_record()
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "unsafe.sqlite3")
    candidate = EvidenceAnalysis("E1", Status.IRRELEVANT, evaluated_requirement_ids=legacy_scope)
    context = projection_semantic_cache_context(build_projection_semantic_request(
        question="facts", plan=plan(), projection_type=EvidenceProjectionType.FACT, requirement_ids=("R2",)))
    assert cache.put(question="facts", record=source, analysis=candidate, projection_context=context)
    result = LLMEvidenceProjector(Client([response(source, ("R2",))]), cache=cache).analyze(request=request((source,)))
    assert result.cache_stats.misses == 1 and result.llm_calls == 1
    assert result.analyses[0].evaluated_requirement_ids == ("R1", "R2")
    assert result.analyses[0].projections[0] == source.projections[0]


def test_legacy_adapter_is_explicit_and_never_attests_evaluated_scope():
    source = old_record()
    class Legacy:
        def analyze(self, *, question, evidence):
            assert question == "facts" and evidence == (source,)
            return EvidenceAnalysisResult((source.assessment,), 0)
    result = LegacyEvidenceAnalyzerAdapter(Legacy()).analyze(request=request((source,)))
    assert result.analyses[0].evaluated_requirement_ids == ()
    assert result.analyses[0].projection == source.projection
    assert set(inspect.signature(EvidenceAnalyzer.analyze).parameters) == {"self", "request"}
    assert set(inspect.signature(LLMEvidenceProjector.analyze).parameters) == {"self", "request"}
    assert "is_task_directed_projector" not in inspect.getsource(_analyze_evidence)


@pytest.mark.parametrize("status", [Status.PENDING, Status.FAILED])
def test_unsuccessful_analysis_cannot_attest_evaluated_scope(status):
    with pytest.raises(ValueError, match="Unsuccessful"):
        EvidenceAnalysis("E1", status, error_code="failure" if status is Status.FAILED else None,
            evaluated_requirement_ids=("R1",))


def test_old_checkpoint_and_delta_defaults_never_infer_scope_from_support():
    original = old_record()
    payload = dump_contract((original,), tuple[EvidenceRecord, ...])
    payload[0]["assessment"].pop("evaluated_requirement_ids")
    restored = load_contract(payload, tuple[EvidenceRecord, ...])[0]
    assert restored.assessment.evaluated_requirement_ids == ()
    assert verified_support_ids(restored, plan().requirements) == ("R1",)
    assert requires_analysis(restored, plan())
    assert TypeAdapter(RegistryDelta).validate_python({"added_ids": [], "reused_ids": ["E1"]}).analysis_required_ids == ()


def test_real_historical_states_parse_and_roundtrip_without_scope_inference():
    count = 0
    adapter = TypeAdapter(DecisionState)
    for path in HISTORY:
        for line in path.read_text(encoding="utf-8").splitlines():
            state = adapter.validate_python(json.loads(line)["full_decision_state"])
            assert adapter.validate_json(adapter.dump_json(state)) == state
            assert all(r.assessment.evaluated_requirement_ids == () for r in state.evidence)
            count += 1
    assert count == 13


def test_legacy_claim_adapter_preserves_claim_contract_without_fake_projection():
    source = replace(old_record(), assessment=None)
    claim = EvidenceClaim("E1:C1", "temporal encoder", EvidenceRelation.SUPPORTS,
        EvidenceOwnership.THIS_PAPER, "Source states the method", source.evidence.content)
    assessment = EvidenceAnalysis("E1", Status.QUALIFIED, digest="temporal encoder", claims=(claim,))
    class Legacy:
        def analyze(self, *, question, evidence):
            return EvidenceAnalysisResult((assessment,), 0)
    current, _ = analyze_node(registry_for(source), LegacyEvidenceAnalyzerAdapter(Legacy()))
    assert current.assessment == assessment
    assert current.assessment.evaluated_requirement_ids == ()
    assert current.projection is None


def test_unbound_legacy_evidence_fails_without_plan_fallback_or_analyzer_call():
    source = replace(old_record(), requested_requirement_ids=(), assessment=None)
    class MustNotCall:
        def analyze(self, *, request):
            raise AssertionError("An empty binding must not become an all-Plan model request")
    current, update = analyze_node(registry_for(source), MustNotCall())
    assert current.analysis_status is Status.FAILED
    assert current.assessment.error_code == "MissingEvidenceProjectionScope"
    assert current.assessment.evaluated_requirement_ids == ()
    assert not update["llm_diagnostics"]


def test_registry_validates_entire_replacement_batch_before_any_write():
    first = expanded_record()
    second = replace(first, evidence_id="E2", assessment=None, fingerprint="",
        evidence=replace(first.evidence, source_chunk_ids=("second",)))
    registry = EvidenceRegistry(RegistryState((first, second), content_counter=2))
    before = registry.snapshot()
    complete = replace(first.assessment, evaluated_requirement_ids=("R1", "R2"))
    incomplete = EvidenceAnalysis("E2", Status.IRRELEVANT, evaluated_requirement_ids=("unknown",))
    with pytest.raises(ValueError, match="Evidence-local scope"):
        registry.apply_analyses((complete, incomplete), plan=plan())
    assert registry.snapshot() == before
