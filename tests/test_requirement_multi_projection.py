"""Audit #24: Requirement-directed heterogeneous projection; fixed mock clients only."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from paperbase.agentic.contracts import (
    EvidenceAnalysis, EvidenceAnalysisStatus as Status, EvidenceProjectionType as Shape,
    EvidenceRecord, EvidenceTarget, Intent, RegistryState, ResearchPlan, ResearchRequirement,
    SearchKnowledgeBaseAction, BudgetUsage, ResearchGoal, PlanningMode,
)
from paperbase.agentic.evidence_analysis import (
    EvidenceAnalysisRequest, merge_analysis_groups, projection_requirement_scope, requires_analysis,
)
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.agentic.evidence_projection import LLMEvidenceProjector
from paperbase.agentic.evidence_registry import EvidenceRegistry
from paperbase.agentic.projection_reads import dedupe_projections
from paperbase.agentic.projection_shapes import requirement_projection_type
from paperbase.agentic.projection_semantics import build_projection_semantic_request, projection_semantic_cache_context
from paperbase.agentic.grounding import verified_support_ids
from paperbase.agentic.selector import FinalEvidenceSelector, _entity_keys, _fact_keys
from paperbase.agentic.decision import DecisionState, _projection_view, _projections_view
from paperbase.agentic.decision_view import build_compact_decision_view
from paperbase.agentic.requirement_state import derive_requirement_states
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.answering import _to_evidence_unit
from paperbase.agentic.audit import _evidence_audit, _evidence_from_payload
from paperbase.agentic.planner import ResearchPlanner
from paperbase.agentic.schemas import ResearchPlanDraft
from paperbase.llm.client import LLMCompletion
from paperbase.retrieval.contracts import KnowledgeBaseSearchPayload, ToolName, ToolResult, ToolStatus
from test_agentic_batch_projector import CONTEXT, Client, record as reference_record
from test_agentic_checkpoint_semantic_authority import (
    interrupted_workflow, config, OLD_VERSION, NEW_VERSION, historical_states,
)


def typed_plan(*shapes):
    return ResearchPlan("Research mixed answer obligations", tuple(
        ResearchRequirement(f"R{i}", f"Obligation {i}", frozenset({EvidenceTarget.METHOD}),
                            projection_type=shape) for i, shape in enumerate(shapes, 1)))


def source(ids=("R1", "R2"), eid="E1"):
    record = reference_record(1)
    return replace(record, evidence_id=eid, fingerprint="", requested_requirement_ids=ids, assessment=None,
        evidence=replace(record.evidence, content=(
            "The paper uses model Alpha. Alpha processes history with an encoder. "
            "RMSE is 1.0. Alpha differs from Beta in temporal dynamics.")))


def request(records, plan, context=CONTEXT):
    return EvidenceAnalysisRequest("Mixed obligations", tuple(records), context, plan,
        tuple((r.evidence_id, projection_requirement_scope(r, plan)) for r in records))


def projection_body(shape, ids, content, eid=None, *, irrelevant=False, bad_anchor=False):
    base = dict(projection_type=shape, status="irrelevant" if irrelevant else "qualified",
        supported_requirement_ids=[] if irrelevant else list(ids),
        support_candidates=[] if irrelevant else [dict(requirement_id=rid,
            source_span_ids=["S999" if bad_anchor else "S1"],
            support_scope=f"Source supports {rid}", unresolved_aspects=[]) for rid in ids])
    if eid is not None:
        base["evidence_id"] = eid
    if shape == "fact":
        base.update(fact_type=None if irrelevant else "model/result", value=None if irrelevant else "Alpha; RMSE 1.0",
                    ownership=None if irrelevant else "this_paper")
    elif shape == "mechanism":
        base.update(mechanism=None if irrelevant else "Alpha encoder", steps=[] if irrelevant else ["encode history"],
                    inputs=[] if irrelevant else ["history"], outputs=[] if irrelevant else ["prediction"])
    elif shape == "comparison":
        base.update(comparison_dimensions=[] if irrelevant else ["temporal dynamics versus Beta"])
    elif shape == "enumeration":
        base.update(entities=[] if irrelevant else [dict(canonical_name="Alpha", entity_type="method",
                    role="adopted", ownership="this_paper")])
    return base


class StructuredClient:
    """Emits schema-specific output from the actual group input, never a provider."""
    def __init__(self, behaviors=None):
        self.behaviors = behaviors or {}
        self.calls = []

    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["user_prompt"])
        shape = payload["projection_type"]
        behavior = self.behaviors.get(shape)
        if behavior == "provider":
            raise RuntimeError("mock provider failure")
        if behavior == "schema":
            return LLMCompletion('{}')
        parts = [projection_body(shape, payload["projection_requirement_ids"], "".join(
            span["text"] for span in evidence["source_spans"]
        ),
            eid=None if kwargs["schema_name"] == "paperbase_evidence_transport_batch" else evidence["evidence_id"],
            irrelevant=behavior == "irrelevant", bad_anchor=behavior == "integrity") for evidence in payload["evidence"]]
        result = ({"items": [{"evidence_id": evidence["evidence_id"], "projections": [part]}
                   for evidence, part in zip(payload["evidence"], parts)]}
                  if kwargs["schema_name"] == "paperbase_evidence_transport_batch" else {"projections": parts})
        return LLMCompletion(json.dumps(result), prompt_tokens=100, completion_tokens=30)


def projected(plan, record=None, *, client=None, cache=None, context=CONTEXT):
    record = record or source(tuple(r.requirement_id for r in plan.requirements))
    client = client or StructuredClient()
    result = LLMEvidenceProjector(client, max_attempts=1, cache=cache).analyze(request=request((record,), plan, context))
    return replace(record, assessment=result.analyses[0]), result, client


def test_one_invalid_support_candidate_does_not_discard_valid_sibling() -> None:
    plan = typed_plan(Shape.FACT, Shape.FACT)
    current = source(("R1", "R2"))
    body = projection_body("fact", ("R1", "R2"), current.evidence.content, eid="E1")
    body["support_candidates"][1]["source_span_ids"] = ["S999"]
    client = Client([{"projections": [body]}])

    result = LLMEvidenceProjector(client).analyze(request=request((current,), plan))

    analysis = result.analyses[0]
    assert result.llm_calls == 1
    assert analysis.status is Status.QUALIFIED
    assert analysis.projection.supported_requirement_ids == ("R1",)
    assert "support_rejected:R2:source_span_not_found" in analysis.projection_rejections


@pytest.mark.parametrize("shapes", [
    (Shape.FACT, Shape.MECHANISM), (Shape.COMPARISON, Shape.FACT),
    (Shape.ENUMERATION, Shape.MECHANISM), (Shape.FACT, Shape.COMPARISON, Shape.MECHANISM, Shape.ENUMERATION),
])
def test_same_evidence_retains_every_requirement_shape_and_verified_support(shapes, monkeypatch):
    import paperbase.agentic.projection_shapes as shape_helpers
    import paperbase.agentic.evidence_projection as projector
    def forbidden(*args):
        raise AssertionError("Typed Plan cannot consult legacy global shape")
    monkeypatch.setattr(shape_helpers, "_legacy_projection_type", forbidden)
    monkeypatch.setattr(projector, "_legacy_projection_type", forbidden)
    plan = typed_plan(*shapes)
    record, result, client = projected(plan)
    assert result.llm_calls == len(set(shapes))
    assert [p.projection_type for p in record.projections] == list(shapes)
    assert record.projection is None
    assert verified_support_ids(record, plan.requirements) == tuple(r.requirement_id for r in plan.requirements)
    assert record.assessment.evaluated_requirement_ids == record.requested_requirement_ids
    assert not requires_analysis(record, plan)
    for call in client.calls:
        payload = json.loads(call["user_prompt"])
        assert all(r["projection_type"] == payload["projection_type"] for r in payload["projection_requirements"])
        assert call["json_schema"]["additionalProperties"] is False


def test_fact_fact_is_one_group_and_mixed_fact_groups_combine_r1_r3():
    p = typed_plan(Shape.FACT, Shape.MECHANISM, Shape.FACT)
    record, result, client = projected(p)
    groups = [(json.loads(c["user_prompt"])["projection_type"], json.loads(c["user_prompt"])["projection_requirement_ids"])
              for c in client.calls]
    assert groups == [("fact", ["R1", "R3"]), ("mechanism", ["R2"])]
    assert result.llm_calls == 2 and len(record.projections) == 2
    _, same, same_client = projected(typed_plan(Shape.FACT, Shape.FACT))
    assert same.llm_calls == 1
    assert json.loads(same_client.calls[0]["user_prompt"])["projection_requirement_ids"] == ["R1", "R2"]


@pytest.mark.parametrize("context,shapes", [
    (replace(CONTEXT, resolved_query="How does the model work and what RMSE does it report?"), (Shape.MECHANISM, Shape.FACT)),
    (replace(CONTEXT, intent=Intent.COMPARE), (Shape.COMPARISON, Shape.FACT)),
])
def test_global_query_and_intent_never_override_typed_factual_obligation(context, shapes):
    record, _, client = projected(typed_plan(*shapes), context=context)
    assert [json.loads(c["user_prompt"])["projection_type"] for c in client.calls] == [s.value for s in shapes]
    assert [p.projection_type for p in record.projections] == list(shapes)


def test_shared_type_and_scope_still_batch_across_evidence():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    records = (source(), source(("R1",), "E2"), source(("R2",), "E3"))
    records = tuple(replace(r, fingerprint="", evidence=replace(r.evidence,
        source_chunk_ids=(r.evidence_id,), content=r.evidence.content + " " + r.evidence_id)) for r in records)
    client = StructuredClient()
    result = LLMEvidenceProjector(client).analyze(request=request(records, plan))
    assert result.llm_calls == 2
    assert [len(json.loads(c["user_prompt"])["evidence"]) for c in client.calls] == [2, 2]
    assert all(c["schema_name"] == "paperbase_evidence_transport_batch" for c in client.calls)
    assert [len(a.projections) for a in result.analyses] == [2, 1, 1]


@pytest.mark.parametrize("failure", ["provider", "schema", "integrity"])
def test_partial_group_failure_retains_success_and_retries_only_missing_scope(failure):
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    partial, result, _ = projected(plan, client=StructuredClient({"mechanism": failure}))
    assert partial.analysis_status is Status.QUALIFIED
    assert len(partial.projections) == 1 and partial.projections[0].projection_type is Shape.FACT
    assert partial.assessment.evaluated_requirement_ids == ("R1",)
    assert verified_support_ids(partial, plan.requirements) == ("R1",)
    assert requires_analysis(partial, plan) and result.llm_calls == 2
    repaired, next_result, client = projected(plan, partial)
    assert next_result.llm_calls == 1
    assert json.loads(client.calls[0]["user_prompt"])["projection_requirement_ids"] == ["R2"]
    assert repaired.projections[0] == partial.projections[0]
    assert repaired.assessment.evaluated_requirement_ids == ("R1", "R2")


def test_all_irrelevant_groups_attest_scope_without_retaining_projections():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    record, _, _ = projected(plan, client=StructuredClient({"fact": "irrelevant", "mechanism": "irrelevant"}))
    assert record.analysis_status is Status.IRRELEVANT and record.projections == ()
    assert record.assessment.evaluated_requirement_ids == ("R1", "R2")
    assert not requires_analysis(record, plan)


def test_all_failed_groups_have_no_evaluated_scope():
    record, _, _ = projected(typed_plan(Shape.FACT, Shape.MECHANISM),
        client=StructuredClient({"fact": "provider", "mechanism": "schema"}))
    assert record.analysis_status is Status.FAILED and record.projections == ()
    assert record.assessment.evaluated_requirement_ids == ()


def test_irrelevant_new_group_preserves_qualified_history_and_attests_only_new_scope():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    old, _, _ = projected(plan, source(("R1",)))
    expanded = replace(old, requested_requirement_ids=("R1", "R2"))
    record, result, _ = projected(plan, expanded, client=StructuredClient({"mechanism": "irrelevant"}))
    assert result.llm_calls == 1 and record.projections == old.projections
    assert record.analysis_status is Status.QUALIFIED
    assert record.assessment.evaluated_requirement_ids == ("R1", "R2")
    assert verified_support_ids(record, plan.requirements) == ("R1",)
    assert not requires_analysis(record, plan)


def test_failed_new_group_preserves_successful_irrelevant_history_without_attesting_failure():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    old, _, _ = projected(plan, source(("R1",)), client=StructuredClient({"fact": "irrelevant"}))
    expanded = replace(old, requested_requirement_ids=("R1", "R2"))
    record, _, _ = projected(plan, expanded, client=StructuredClient({"mechanism": "provider"}))
    assert record.assessment == old.assessment and record.analysis_status is Status.IRRELEVANT
    assert record.assessment.evaluated_requirement_ids == ("R1",)
    assert requires_analysis(record, plan)


@pytest.mark.parametrize("second_shape", [Shape.FACT, Shape.MECHANISM])
def test_reuse_adds_only_missing_shape_scope_and_preserves_original_projection(second_shape):
    plan = typed_plan(Shape.FACT, second_shape)
    old, _, _ = projected(plan, source(("R1",)))
    registry = EvidenceRegistry(RegistryState((old,), content_counter=1))
    action = SearchKnowledgeBaseAction("new obligation", (), ("R2",))
    delta = registry.register_tool_result(ToolResult(ToolStatus.OK, ToolName.SEARCH_KNOWLEDGE_BASE,
        KnowledgeBaseSearchPayload(action.query, (old.evidence,))), action=action, plan=plan, iteration=2)
    assert delta.analysis_required_ids == delta.reused_ids == ("E1",)
    expanded = registry.records[0]
    assert expanded.assessment == old.assessment
    new, result, client = projected(plan, expanded)
    assert result.llm_calls == 1
    assert json.loads(client.calls[0]["user_prompt"])["projection_requirement_ids"] == ["R2"]
    assert new.projections[0] == old.projections[0]
    assert len(new.projections) == 2 and new.projection is None
    assert verified_support_ids(new, plan.requirements) == ("R1", "R2")
    assert not requires_analysis(new, plan)
    registry.apply_analyses((new.assessment,), plan=plan)
    assert registry.records[0] == new


def test_projection_dedupe_uses_payload_ids_and_anchors_not_just_type():
    plan = typed_plan(Shape.FACT, Shape.FACT)
    first, _, _ = projected(plan, source(("R1",)))
    second, _, _ = projected(plan, source(("R2",)))
    assert dedupe_projections((first.projections[0], first.projections[0], second.projections[0])) == (
        first.projections[0], second.projections[0])
    expanded = replace(first, requested_requirement_ids=("R1", "R2"))
    merged = merge_analysis_groups(expanded, (second.assessment, second.assessment), plan)
    assert len(merged.projections) == 2
    assert len(merge_analysis_groups(replace(expanded, assessment=merged), (second.assessment,), plan).projections) == 2


def test_multi_projection_checkpoint_roundtrip_and_legacy_single_normalization():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    record, _, _ = projected(plan)
    payload = dump_contract((record,), tuple[EvidenceRecord, ...])
    assert "projection" not in payload[0]["assessment"]
    assert len(payload[0]["assessment"]["projections"]) == 2
    assert load_contract(payload, tuple[EvidenceRecord, ...]) == (record,)
    legacy = deepcopy(payload)
    legacy_assessment = legacy[0]["assessment"]
    legacy_assessment["projection"] = legacy_assessment.pop("projections")[0]
    legacy_assessment.pop("evaluated_requirement_ids")
    restored = load_contract(legacy, tuple[EvidenceRecord, ...])[0]
    assert restored.projections == (record.projections[0],)
    assert restored.assessment.evaluated_requirement_ids == ()
    assert restored.projection == restored.projections[0]


def test_conflicting_single_and_multi_authorities_are_rejected():
    record, _, _ = projected(typed_plan(Shape.FACT, Shape.MECHANISM))
    with pytest.raises(ValueError, match="Conflicting"):
        EvidenceAnalysis("E1", Status.QUALIFIED, projection=record.projections[0], projections=record.projections)


def test_selector_decision_research_state_audit_and_answer_input_read_every_projection():
    plan = typed_plan(Shape.ENUMERATION, Shape.FACT, Shape.MECHANISM, Shape.COMPARISON)
    record, _, _ = projected(plan)
    assert any("alpha" in key[0] for key in _entity_keys(record))
    keys = _fact_keys(record)
    assert "alpha encoder" in keys and any("model/result" in key for key in keys)
    assert any("anchor:" in key for key in keys)
    assert FinalEvidenceSelector().select(plan=plan, evidence=(record,)) == (record,)
    states = derive_requirement_states(plan=plan, evidence=(record,), observations=())
    assert all(s.status.value == "satisfied" for s in states)
    assert _projection_view(record) is None
    assert len(_projections_view(record)) == 4
    state = DecisionState(CONTEXT, plan, (record,), (), BudgetUsage())
    compact = build_compact_decision_view(state).payload
    assert [row["local_supports"][0]["evidence_id"] for row in compact["requirements"]] == [
        "E1", "E1", "E1", "E1"
    ]
    assert compact["research_leads"] == []
    audit = _evidence_audit(record, {"E1"}, set(), {}, {}, set(record.requested_requirement_ids), plan.requirements)
    assert len(audit.structured_projections) == 4 and audit.structured_projection == {}
    assert _evidence_from_payload(json.loads(json.dumps(asdict(audit)))) == audit
    unit = _to_evidence_unit(record, plan)
    assert len(unit.projection_context["projections"]) == 4
    assert unit.supported_requirement_ids == ("R1", "R2", "R3", "R4")


def test_single_view_alias_is_safe_and_array_still_present():
    plan = typed_plan(Shape.FACT)
    record, _, _ = projected(plan)
    assert _projection_view(record)["type"] == "fact"
    assert len(_projections_view(record)) == 1


def test_invalid_sibling_support_does_not_borrow_another_projection_integrity():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    record, _, _ = projected(plan)
    invalid = replace(record.projections[1], support_anchors=(replace(record.projections[1].support_anchors[0],
        source_excerpt="not in the source"),))
    bad = replace(record, assessment=replace(record.assessment, projection=None,
        projections=(record.projections[0], invalid)))
    assert verified_support_ids(bad, plan.requirements) == ("R1",)


def test_support_union_preserves_requirement_domain_with_one_shot_iterable():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    record, _, _ = projected(plan)
    # R2 is outside this caller's domain; the second projection must not treat an
    # exhausted Requirement iterator as an unconstrained empty domain.
    assert verified_support_ids(record, iter(plan.requirements[:1])) == ("R1",)


def test_cache_group_isolation_hits_and_misses_and_semantic_changes_invalidate(tmp_path):
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "groups.sqlite3")
    record, first, _ = projected(plan, cache=cache)
    assert first.cache_stats.writes == 2
    _, second, client = projected(plan, cache=cache)
    assert second.llm_calls == 0 and second.cache_stats.hits == 2 and client.calls == []
    contexts = [projection_semantic_cache_context(build_projection_semantic_request(
                    question="Mixed obligations", plan=plan, projection_type=shape, requirement_ids=(rid,)))
                for shape, rid in ((Shape.FACT, "R1"), (Shape.MECHANISM, "R2"))]
    keys = [cache._key(question="Mixed obligations", record=record, projection_context=context) for context in contexts]
    assert keys[0] != keys[1]
    cache._delete_quietly(keys[1])
    _, third, client = projected(plan, cache=cache)
    assert third.cache_stats.hits == 1 and third.cache_stats.misses == 1 and third.llm_calls == 1
    assert json.loads(client.calls[0]["user_prompt"])["projection_type"] == "mechanism"
    changed = replace(plan, requirements=tuple(replace(r, description="Changed semantic obligation") for r in plan.requirements))
    _, stale, _ = projected(changed, cache=cache)
    assert stale.cache_stats.hits == 0 and stale.cache_stats.misses == 2 and stale.llm_calls == 2
    assert not cache.put(question="Mixed obligations", record=record, analysis=record.assessment,
                         projection_context=contexts[0])  # no aggregate cache entry


class PlannerClient:
    def __init__(self, values): self.values, self.calls = list(values), []
    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        return LLMCompletion(json.dumps(self.values.pop(0)))


def planner_payload(include_shape=True):
    return {"goal": "Mixed obligations", "requirements": [dict(description=description, required=True,
        evidence_targets=["method"], **({"projection_type": shape} if include_shape else {}))
        for description, shape in (("What model is used?", "fact"), ("Why and how does it work?", "mechanism"),
                                   ("What RMSE is reported?", "fact"))]}


def test_model_planner_requires_shape_and_can_build_heterogeneous_plan():
    context = replace(CONTEXT, resolved_query="What model is used, how does it work, and what RMSE is reported?")
    client = PlannerClient([planner_payload()])
    result = ResearchPlanner(client).plan(context)
    assert not result.used_fallback
    assert [r.projection_type for r in result.plan.requirements] == [Shape.FACT, Shape.MECHANISM, Shape.FACT]
    schema = ResearchPlanDraft.model_json_schema()["$defs"]["ResearchRequirementDraft"]
    assert "projection_type" in schema["required"]


def test_missing_model_shape_repairs_without_silent_fact_default():
    context = replace(CONTEXT, resolved_query="What model is used, how does it work, and what RMSE is reported?")
    client = PlannerClient([planner_payload(False), planner_payload()])
    result = ResearchPlanner(client).plan(context)
    assert result.llm_calls == 2 and not result.used_fallback
    assert result.llm_diagnostics[0].error_stage == "schema_validation"
    assert [r.projection_type for r in result.plan.requirements] == [Shape.FACT, Shape.MECHANISM, Shape.FACT]
    with pytest.raises(ValidationError): ResearchPlanDraft.model_validate_json(json.dumps(planner_payload(False)))


def test_missing_model_shape_exhausts_existing_repair_then_uses_explicit_intent_fallback():
    context = replace(CONTEXT, resolved_query="How does the model work?", intent=Intent.EXPLAIN)
    result = ResearchPlanner(PlannerClient([planner_payload(False), planner_payload(False)])).plan(context)
    assert result.llm_calls == 2 and result.used_fallback
    assert result.plan.planning_mode is PlanningMode.LITERAL_FALLBACK
    assert result.plan.requirements[0].projection_type is Shape.MECHANISM


def test_collection_literal_fallback_preserves_question_without_guessing_entity_domain():
    context = replace(CONTEXT, retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION)
    result = ResearchPlanner().plan(context)
    assert result.plan.planning_mode is PlanningMode.LITERAL_FALLBACK
    assert len(result.plan.requirements) == 1
    assert result.plan.requirements[0].description == context.resolved_query
    assert all(r.projection_type is Shape.FACT for r in result.plan.requirements)


@pytest.mark.parametrize('question', ['列出全部数据集', 'List all authors', '列出所有方法和模型'])
def test_collection_fallback_does_not_infer_entities_or_drop_the_requested_scope(question):
    context = replace(CONTEXT,original_query=question,resolved_query=question,
        retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION)
    result = ResearchPlanner().plan(context)
    requirement, = result.plan.requirements
    assert result.llm_calls == 0 and result.used_fallback
    assert result.plan.goal == requirement.description == question
    assert requirement.required and requirement.evidence_targets == context.default_evidence_targets
    assert requirement.projection_type is Shape.FACT


@pytest.mark.parametrize("question,intent,expected", [
    ("how model works", Intent.EXPLAIN, Shape.MECHANISM), ("Compare models", Intent.COMPARE, Shape.COMPARISON),
    ("model name", Intent.SYNTHESIZE, Shape.FACT),
])
def test_deterministic_literal_fallback_assigns_explicit_shape(question, intent, expected):
    result = ResearchPlanner().plan(replace(CONTEXT, resolved_query=question, intent=intent))
    assert result.plan.requirements[0].projection_type is expected


def test_legacy_requirement_none_parses_and_only_explicit_compatibility_uses_fallback():
    legacy = ResearchRequirement("R1", "method", frozenset({EvidenceTarget.METHOD}))
    assert TypeAdapter(ResearchRequirement).validate_python({"requirement_id": "R1", "description": "method",
        "evidence_targets": ["method"]}).projection_type is None
    assert requirement_projection_type(legacy, context=replace(CONTEXT, resolved_query="how?")) is Shape.MECHANISM
    typed = replace(legacy, projection_type=Shape.FACT)
    assert requirement_projection_type(typed, context=replace(CONTEXT, resolved_query="how?")) is Shape.FACT


@pytest.mark.parametrize("old_version", [OLD_VERSION, "projector-evidence-local-requirement-scope-v3"])
@pytest.mark.parametrize("mode", ["resume", "resume_stream"])
def test_v2_v3_checkpoint_cannot_resume_under_v4(old_version, mode):
    workflow, _, _, fresh = interrupted_workflow()
    assert fresh.state["semantic_authority_version"] == NEW_VERSION == "projector-requirement-directed-multi-projection-v4"
    workflow._graph.update_state(config("authority"), {"semantic_authority_version": old_version}, as_node="execute_tool")
    before = deepcopy(workflow.inspect("authority"))
    with pytest.raises(ValueError, match="Incompatible checkpoint semantic authority"):
        workflow.resume("authority") if mode == "resume" else list(workflow.resume_stream("authority"))
    assert workflow.inspect("authority") == before


def test_thirteen_historical_states_normalize_single_projection_without_semantic_migration():
    count = 0
    for _, _, state in historical_states():
        assert all(r.assessment.evaluated_requirement_ids == () for r in state.evidence)
        assert all(r.projections == ((r.projection,) if r.projection is not None else ()) for r in state.evidence)
        assert all(req.projection_type is None for req in state.plan.requirements)
        restored = load_contract(dump_contract(RegistryState(records=state.evidence), RegistryState), RegistryState)
        assert restored.records == state.evidence
        count += 1
    assert count == 13
