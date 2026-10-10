"""Audit #27: correct semantic reuse with fixed clients and real local SQLite."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from paperbase.agentic.contracts import ResearchGoal, ResearchPlan, EvidenceTarget, PlanningMode
from paperbase.agentic.evidence_analysis import requires_analysis
from paperbase.agentic.evidence_projection import (
    LLMEvidenceProjector, ProjectionRequest, _prompt_payload, _revalidate_cached,
)
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache, _analysis_payload
from paperbase.agentic.grounding import source_contains, verified_support_ids
from paperbase.agentic.projection_semantics import (
    PROJECTION_SEMANTIC_REQUEST_VERSION, build_projection_semantic_request,
    projection_semantic_snapshot, projection_semantic_fingerprint,
    projection_semantic_cache_context,
)
from test_requirement_multi_projection import Shape, typed_plan, source, request, StructuredClient, CONTEXT


QUESTION = "Explain the method"


def baseline_plan():
    plan = typed_plan(Shape.FACT, Shape.MECHANISM)
    return replace(plan, goal="Explain the method", requirements=(
        replace(plan.requirements[0], description="What model architecture is used?"),
        replace(plan.requirements[1], description="How does the encoder work?")))


def semantic(plan, *, question=QUESTION, ids=("R1",), shape=Shape.FACT):
    return build_projection_semantic_request(question=question, plan=plan,
        projection_type=shape, requirement_ids=ids)


def run(cache, plan, *, question=QUESTION, record=None, client=None):
    record = record or source(("R1",))
    client = client or StructuredClient()
    result = LLMEvidenceProjector(client, cache=cache, max_attempts=1).analyze(
        request=replace(request((record,), plan), question=question))
    return result, client


def changed(plan, field):
    if field == "goal":
        return replace(plan, goal="Compare experimental results")
    changes = {
        "description": {"description": "What optimizer configuration is used?"},
        "targets": {"evidence_targets": frozenset({EvidenceTarget.RESULT})},
        "required": {"required": False},
        "shape": {"projection_type": Shape.MECHANISM},
    }
    return replace(plan, requirements=(replace(plan.requirements[0], **changes[field]), *plan.requirements[1:]))


@pytest.mark.parametrize("field", ["description", "targets", "required", "goal", "shape"])
def test_semantic_field_change_misses_real_group_cache(field, tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "miss.sqlite3")
    plan = baseline_plan()
    first, _ = run(cache, plan)
    assert first.cache_stats.writes == first.llm_calls == 1
    same, client = run(cache, plan)
    assert same.cache_stats.hits == 1 and same.llm_calls == 0 and not client.calls
    new_plan = changed(plan, field)
    new, client = run(cache, new_plan)
    assert new.cache_stats.misses == new.llm_calls == new.cache_stats.writes == 1
    assert new.cache_stats.errors == new.cache_stats.hits == 0
    shape = new_plan.requirements[0].projection_type
    assert projection_semantic_fingerprint(semantic(plan)) != projection_semantic_fingerprint(semantic(new_plan, shape=shape))


@pytest.mark.parametrize("field", ["exact", "targets_order", "whitespace", "nfkc", "unrelated", "planning_mode"])
def test_semantic_equivalences_hit_real_group_cache(field, tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "hit.sqlite3")
    plan = baseline_plan()
    if field == "targets_order":
        plan = replace(plan, requirements=(replace(plan.requirements[0],
            evidence_targets=frozenset((EvidenceTarget.METHOD, EvidenceTarget.RESULT))), plan.requirements[1]))
    run(cache, plan)
    question = QUESTION
    new = plan
    if field == "targets_order":
        new = replace(plan, requirements=(replace(plan.requirements[0],
            evidence_targets=frozenset((EvidenceTarget.RESULT, EvidenceTarget.METHOD))), plan.requirements[1]))
    elif field == "whitespace":
        question = "  Explain\t the\nmethod "
        new = replace(plan, goal="  Explain   the\tmethod", requirements=(
            replace(plan.requirements[0], description=" What model\narchitecture   is used? "), plan.requirements[1]))
    elif field == "nfkc":
        question = "Ｅxplain the method"
        new = replace(plan, goal="Ｅxplain the method", requirements=(
            replace(plan.requirements[0], description="Ｗhat model architecture is used?"), plan.requirements[1]))
    elif field == "unrelated":
        new = replace(plan, requirements=(plan.requirements[0], replace(plan.requirements[1],
            description="Completely different R2", required=False, projection_type=Shape.COMPARISON,
            evidence_targets=frozenset({EvidenceTarget.REFERENCE}))))
    elif field == "planning_mode":
        new = replace(plan, planning_mode=PlanningMode.MODEL)
    result, client = run(cache, new, question=question)
    assert result.cache_stats.hits == 1 and result.llm_calls == result.cache_stats.errors == 0
    assert client.calls == []
    assert projection_semantic_fingerprint(semantic(plan)) == projection_semantic_fingerprint(semantic(new, question=question))


@pytest.mark.parametrize("field", ["question", "goal", "description"])
@pytest.mark.parametrize("old,new", [
    ("US wind", "us wind"), ("95% interval", "90% interval"),
    ("24 h", "24 s"), ("LSTM-GMM", "LSTM GMM"),
])
def test_case_numbers_units_and_punctuation_are_semantic(field, old, new, tmp_path):
    plan = baseline_plan()
    question = QUESTION
    if field == "question":
        question = old
        newer = plan
    elif field == "goal":
        plan = replace(plan, goal=old)
        newer = replace(plan, goal=new)
    else:
        plan = replace(plan, requirements=(replace(plan.requirements[0], description=old), plan.requirements[1]))
        newer = replace(plan, requirements=(replace(plan.requirements[0], description=new), plan.requirements[1]))
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "symbols.sqlite3")
    run(cache, plan, question=question)
    result, _ = run(cache, newer, question=new if field == "question" else question)
    assert result.cache_stats.misses == result.llm_calls == 1 and result.cache_stats.errors == 0


def test_stale_source_anchor_is_real_but_changed_obligation_must_not_reuse(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "stale.sqlite3")
    plan = baseline_plan()
    record = source(("R1",))
    record = replace(record, fingerprint="", evidence=replace(record.evidence,
        content="We use LSTM with a temporal encoder."))
    initial, _ = run(cache, plan, record=record)
    old = initial.analyses[0]
    new_plan = changed(plan, "description")
    assert all(source_contains(record.evidence.content, anchor.source_excerpt)
               for projection in old.projections for anchor in projection.support_anchors)
    # Original Integrity can only prove provenance, even under the changed meaning.
    revalidated = _revalidate_cached(record, old, new_plan)
    assert verified_support_ids(replace(record, assessment=revalidated), new_plan.requirements) == ("R1",)
    assert revalidated == old
    result, client = run(cache, new_plan, record=record)
    assert result.cache_stats.hits == 0 and result.cache_stats.misses == 1 and len(client.calls) == 1


def test_snapshot_is_canonical_group_local_and_builder_is_pure(monkeypatch):
    import paperbase.agentic.evidence_projection as projector
    def forbidden(*args, **kwargs):
        raise AssertionError("Semantic identity must never call an LLM")
    monkeypatch.setattr(projector, "complete_json_with_diagnostics", forbidden)
    plan = baseline_plan()
    before = deepcopy(plan)
    compiled = semantic(plan)
    snapshot = projection_semantic_snapshot(compiled)
    assert snapshot == json.loads(json.dumps(snapshot, ensure_ascii=False))
    assert snapshot["version"] == PROJECTION_SEMANTIC_REQUEST_VERSION
    assert snapshot["requirements"] == [{"requirement_id": "R1", "description": plan.requirements[0].description,
        "evidence_targets": ["method"], "required": True, "projection_type": "fact"}]
    assert "planning_mode" not in snapshot and "R2" not in json.dumps(snapshot)
    assert len(projection_semantic_fingerprint(compiled)) == 64
    assert plan == before


def test_requirement_order_matches_actual_prompt_and_only_actual_order_change_misses(tmp_path):
    plan = typed_plan(Shape.FACT, Shape.FACT)
    record = source(("R1", "R2"))
    original = semantic(plan, ids=("R2", "R1"))
    identical = semantic(plan, ids=("R1", "R2"))
    payload = _prompt_payload(ProjectionRequest(QUESTION, CONTEXT, plan, ("R2", "R1"), Shape.FACT), Shape.FACT, (record,))
    assert [r.requirement_id for r in original.requirements] == payload["projection_requirement_ids"] == [
        r["requirement_id"] for r in payload["projection_requirements"]] == ["R1", "R2"]
    assert projection_semantic_fingerprint(original) == projection_semantic_fingerprint(identical)
    reordered = replace(plan, requirements=tuple(reversed(plan.requirements)))
    new = semantic(reordered, ids=("R1", "R2"))
    new_payload = _prompt_payload(ProjectionRequest(QUESTION, CONTEXT, reordered, ("R1", "R2"), Shape.FACT), Shape.FACT, (record,))
    assert [r.requirement_id for r in new.requirements] == new_payload["projection_requirement_ids"] == ["R2", "R1"]
    assert projection_semantic_fingerprint(original) != projection_semantic_fingerprint(new)
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "order.sqlite3")
    run(cache, plan, record=record)
    changed_result, client = run(cache, reordered, record=record)
    assert changed_result.cache_stats.misses == changed_result.llm_calls == 1
    assert [r["requirement_id"] for r in json.loads(client.calls[0]["user_prompt"])["projection_requirements"]] == ["R2", "R1"]


def test_requirement_shape_none_is_distinct_from_explicit_shape_at_same_group_type(tmp_path):
    plan = baseline_plan()
    legacy = replace(plan, requirements=(replace(plan.requirements[0], projection_type=None), plan.requirements[1]))
    assert projection_semantic_snapshot(semantic(legacy))["requirements"][0]["projection_type"] is None
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "typed.sqlite3")
    run(cache, plan)
    result, _ = run(cache, legacy)  # query resolves to FACT too; typed Integrity branch still differs
    assert result.cache_stats.misses == 1 and result.llm_calls == 1


def test_multi_projection_semantic_change_isolates_only_affected_group(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "multi.sqlite3")
    plan = baseline_plan()
    record = source()
    first, _ = run(cache, plan, record=record)
    assert first.llm_calls == first.cache_stats.writes == 2
    newer = replace(plan, requirements=(plan.requirements[0], replace(plan.requirements[1], description="Why causal modeling is used?")))
    result, client = run(cache, newer, record=record)
    assert result.cache_stats.hits == result.cache_stats.misses == result.llm_calls == 1
    assert result.cache_stats.errors == 0
    assert json.loads(client.calls[0]["user_prompt"])["projection_requirement_ids"] == ["R2"]
    assert len(result.analyses[0].projections) == 2


def test_incremental_reuse_preserves_old_group_cache_and_only_analyzes_missing_scope(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "reuse.sqlite3")
    plan = baseline_plan()
    first, _ = run(cache, plan)
    # Fresh DTO with expanded binding: old group is reusable; new group has no entry.
    fresh, client = run(cache, plan, record=source())
    assert fresh.cache_stats.hits == fresh.cache_stats.misses == fresh.llm_calls == 1
    assert json.loads(client.calls[0]["user_prompt"])["projection_requirement_ids"] == ["R2"]
    # Real evaluated reuse does not re-request old R1 at all; frozen #24 behavior.
    assessed = replace(source(), assessment=first.analyses[0])
    assert requires_analysis(assessed, plan)
    resumed, no_calls = run(cache, plan, record=assessed)
    assert resumed.cache_stats.hits == 1 and resumed.llm_calls == 0 and not no_calls.calls
    assert resumed.analyses[0].projections[0] == first.analyses[0].projections[0]


def test_legacy_rows_naturally_miss_without_fallback_deletion_or_error(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "legacy.sqlite3")
    plan = baseline_plan()
    record = source(("R1",))
    old, _ = run(None, plan, question="question", record=record)
    context = {"projection_type": "fact", "projection_requirement_ids": ["R1"]}
    old_key = cache._key(question="question", record=record, projection_context=context)
    payload = json.dumps(_analysis_payload(old.analyses[0]))
    with cache._connect() as connection:
        connection.execute("INSERT INTO evidence_analysis_cache VALUES (?, ?, ?, ?)",
                           (old_key, record.fingerprint, payload, "legacy"))
    result, _ = run(cache, plan, question="question", record=record)
    assert result.cache_stats.misses == result.llm_calls == 1 and result.cache_stats.errors == 0
    with cache._connect() as connection:
        assert connection.execute("SELECT payload_json FROM evidence_analysis_cache WHERE cache_key=?", (old_key,)).fetchone() == (payload,)
        assert connection.execute("SELECT COUNT(*) FROM evidence_analysis_cache").fetchone()[0] == 2


def test_missing_compiled_semantic_identity_cannot_read_or_write_old_key(tmp_path, monkeypatch):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "missing.sqlite3")
    record = source(("R1",))
    result, _ = run(None, baseline_plan(), record=record)
    def forbidden(**kwargs):
        raise AssertionError("Never reconstruct a legacy key")
    monkeypatch.setattr(cache, "_key", forbidden)
    assert cache.lookup(question=QUESTION, record=record).analysis is None
    assert not cache.lookup(question=QUESTION, record=record).error
    assert not cache.put(question=QUESTION, record=record, analysis=result.analyses[0])


def test_semantic_version_changes_both_fingerprint_and_key_without_payload_or_checkpoint_upgrade(tmp_path, monkeypatch):
    import paperbase.agentic.projection_semantics as semantics
    from paperbase.agentic.evidence_analysis_cache import EVIDENCE_ANALYSIS_SCHEMA_VERSION
    from paperbase.agentic.graph import _SEMANTIC_AUTHORITY_VERSION
    from paperbase.prompts.evidence_projector import EVIDENCE_PROJECTOR_PROMPT_VERSION
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "version.sqlite3")
    plan = baseline_plan()
    original = projection_semantic_fingerprint(semantic(plan))
    run(cache, plan)
    monkeypatch.setattr(semantics, "PROJECTION_SEMANTIC_REQUEST_VERSION", "projection-semantic-v2")
    assert projection_semantic_fingerprint(semantic(plan)) != original
    result, _ = run(cache, plan)
    assert result.cache_stats.misses == 1 and result.cache_stats.errors == 0
    assert EVIDENCE_ANALYSIS_SCHEMA_VERSION == "evidence-projection-schema-v22-bounded-span-envelope"
    assert EVIDENCE_PROJECTOR_PROMPT_VERSION == "requirement-directed-multi-projection-v20-bounded-span-envelope"
    assert _SEMANTIC_AUTHORITY_VERSION == "projector-requirement-directed-multi-projection-v4"


def test_runtime_metadata_and_evidence_ids_do_not_change_semantic_cache(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "metadata.sqlite3")
    plan = baseline_plan()
    initial, _ = run(cache, plan)
    record = replace(source(("R1",)), evidence_id="E9", assessment=None,
        supporting_queries=("other retrieval request",), first_seen_iteration=10, last_seen_iteration=20,
        originating_tools=("search_references",))
    result, client = run(cache, plan, record=record)
    assert result.cache_stats.hits == 1 and result.llm_calls == 0 and not client.calls
    assert result.analyses[0].evidence_id == "E9" and result.analyses[0].projections[0].evidence_id == "E9"
    assert "semantic_fingerprint" not in _analysis_payload(initial.analyses[0])


@pytest.mark.parametrize('old_version', ['requirement-directed-multi-projection-v12', 'requirement-directed-multi-projection-v13', 'requirement-directed-multi-projection-v14', 'requirement-directed-multi-projection-v15'])
def test_prompt_responsibility_update_cannot_reuse_old_prompt_cache(tmp_path, old_version):
    path = tmp_path / 'prompt-version.sqlite3'
    old = SQLiteEvidenceAnalysisCache(path, prompt_version=old_version)
    plan = baseline_plan()
    run(old, plan)
    reused, old_client = run(old, plan)
    assert reused.cache_stats.hits == 1 and not old_client.calls
    current = SQLiteEvidenceAnalysisCache(path)
    refreshed, client = run(current, plan)
    assert refreshed.cache_stats.misses == 1 and len(client.calls) == 1
    hit, current_client = run(current, plan)
    assert hit.cache_stats.hits == 1 and not current_client.calls
    retained, legacy_client = run(old, plan)
    assert retained.cache_stats.hits == 1 and not legacy_client.calls


def test_cache_opaque_context_ignores_unrelated_debug_material(tmp_path):
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "opaque.sqlite3")
    plan = baseline_plan()
    first, _ = run(cache, plan)
    context = projection_semantic_cache_context(semantic(plan))
    decorated = dict(context, runtime_iteration=999, semantic_snapshot={"debug": "not key material"})
    hit = cache.lookup(question=QUESTION, record=source(("R1",)), projection_context=decorated)
    assert hit.analysis == first.analyses[0] and not hit.error


@pytest.mark.parametrize("ids", [(), ("R1", "R1"), ("R9",)])
def test_semantic_builder_rejects_invalid_group_scope(ids):
    with pytest.raises(ValueError, match="scope"):
        semantic(baseline_plan(), ids=ids)
