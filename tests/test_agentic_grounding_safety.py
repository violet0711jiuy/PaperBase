"""Projector authority, Evidence integrity, and answer coverage boundaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from paperbase.agentic.answering import _to_evidence_unit, build_completion_view
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.contracts import (
    EvidenceAnalysis, EvidenceAnalysisStatus, EvidenceRecord, EvidenceTarget,
    Intent, MechanismProjection, QueryContext, QueryScope, ResearchGoal,
    ResearchPlan, ResearchRequirement, RequirementStatus,
    RequirementSupportAnchor, ScopeSource,
    EvidenceProjectionType,
)
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.agentic.projection_semantics import build_projection_semantic_request, projection_semantic_cache_context
from paperbase.agentic.evidence_analysis import EvidenceAnalysisRequest
from paperbase.agentic.evidence_projection import LLMEvidenceProjector
from paperbase.agentic.decision import _projection_view
from paperbase.agentic.grounding import verified_support_ids
from paperbase.agentic.graph import _require_current_semantic_authority, _SEMANTIC_AUTHORITY_VERSION
from paperbase.agentic.source_spans import source_span_ids_for_excerpt
from paperbase.agentic.requirement_state import finalize_requirement_states
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.generation.answer_generator import (
    AnswerDomainValidationError, AnswerSemanticDraft, _compile_semantic_answer,
)
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence
from paperbase.prompts.evidence_projector import EVIDENCE_PROJECTOR_SYSTEM_PROMPT


def _analysis_request(*, question, evidence, context, plan, projection_requirement_ids):
    """Bind pre-scope test fixtures explicitly to the scope this test exercises."""
    records = tuple(replace(record, requested_requirement_ids=tuple(projection_requirement_ids))
                    for record in evidence)
    return EvidenceAnalysisRequest(question, records, context, plan,
        tuple((record.evidence_id, tuple(projection_requirement_ids)) for record in records))


class _Client:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = 0

    def complete_json(self, **_kwargs: object) -> str:
        self.calls += 1
        return json.dumps(self.response, ensure_ascii=False)


def _plan(*descriptions: str) -> ResearchPlan:
    return ResearchPlan("explain mechanism", tuple(
        ResearchRequirement(f"R{index}", description, frozenset({EvidenceTarget.METHOD}))
        for index, description in enumerate(descriptions, 1)
    ))


def _record(content: str, evidence_id: str = "E1") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT, paper_id="P",
            section="3.2.3 Generating prediction intervals",
            seed_chunk_ids=(evidence_id,), source_chunk_ids=(evidence_id,),
            content=content,
        ),
        supporting_queries=("model outputs",), first_seen_iteration=1,
        last_seen_iteration=1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
        requested_requirement_ids=("R1", "R2"),
    )


def _response(
    excerpt: str,
    requirement_id: str = "R1",
    evidence_id: str = "E1",
    *,
    source_span_ids: tuple[str, ...] = ("S1",),
) -> dict:
    return {"projections": [{
        "projection_type": "mechanism", "evidence_id": evidence_id, "status": "qualified",
        "mechanism": "LSTM-EMVE", "steps": ["outputs mean and prediction interval"],
        "inputs": [], "outputs": ["mean", "standard deviation", "prediction interval"],
        "supported_requirement_ids": [requirement_id],
        "support_candidates": [{"requirement_id": requirement_id,
                                "source_span_ids": list(source_span_ids),
                                "support_scope": "The model outputs mean, standard deviation, and prediction interval.",
                                "unresolved_aspects": []}],
    }]}


def _project(response: dict, record: EvidenceRecord, plan: ResearchPlan, *, cache=None):
    client = _Client(response)
    context = QueryContext(
        original_query="how model output works", resolved_query="how model output works", intent=Intent.EXPLAIN,
        scope=QueryScope.SINGLE_PAPER, target_paper_ids=("P",),
        scope_source=ScopeSource.UI_SELECTION, retrieval_goal=ResearchGoal.FIND_EVIDENCE,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    result = LLMEvidenceProjector(client, max_attempts=1, cache=cache).analyze(request=_analysis_request(question="how model output works", evidence=(record,), context=context, plan=plan, projection_requirement_ids=tuple(item.requirement_id for item in plan.requirements)))
    return result.analyses[0], client


def _qualified(record: EvidenceRecord, requirement_id: str) -> EvidenceRecord:
    anchor = RequirementSupportAnchor(requirement_id, record.evidence.content,
                                      "This source describes one component.")
    projection = MechanismProjection(
        record.evidence_id, "LSTM-EMVE", supported_requirement_ids=(requirement_id,),
        proposed_requirement_ids=(requirement_id,), support_anchors=(anchor,),
    )
    return replace(record, assessment=EvidenceAnalysis(
        record.evidence_id, EvidenceAnalysisStatus.QUALIFIED, projection=projection,
    ))


def test_e12_r2_coreference_excerpt_keeps_local_support() -> None:
    source = (
        "Using the estimated mean and standard deviation, the model calculates the "
        "upper and lower bounds of the prediction intervals. The final output of the "
        "model is a concatenation of the predicted mean, standard deviation, upper bound, "
        "and lower bound. The model trains using the proposed L EMVE loss function "
        "specifically designed to predict distribution parameters and optimal prediction intervals."
    )
    excerpt = source[source.index("The final output"):source.index(" The model trains")]
    plan = _plan("LSTM-EMVE architecture", "LSTM-EMVE如何同时输出点预测、标准差和预测区间的机制")
    analysis, _ = _project(_response(
        excerpt, "R2", "E12", source_span_ids=source_span_ids_for_excerpt(source, excerpt)
    ), _record(source, "E12"), plan)
    assert analysis.projection.supported_requirement_ids == ("R2",)
    assert not analysis.projection_rejections
    record = replace(_record(source, "E12"), assessment=analysis)
    assert verified_support_ids(record, plan.requirements) == ("R2",)
    state = finalize_requirement_states(plan=plan, evidence=(record,), observations=()).states[1]
    assert state.status is RequirementStatus.SATISFIED  # legacy name: support_found
    assert FinalEvidenceSelector().select(plan=plan, evidence=(record,), requirement_states=(state,)) == (record,)


def test_forged_excerpt_and_unknown_requirement_are_rejected() -> None:
    source = "The model outputs mean and prediction intervals."
    plan = _plan("Explain output")
    forged, _ = _project(_response(
        "The model outputs an invented variance.", source_span_ids=("S999",)
    ), _record(source), plan)
    assert forged.projection.supported_requirement_ids == ()
    assert "support_rejected:R1:source_span_not_found" in forged.projection_rejections
    unknown, _ = _project(_response(source, "R99"), _record(source), plan)
    assert unknown.status is EvidenceAnalysisStatus.FAILED


def test_partial_local_support_is_selectable_and_multiple_sources_survive() -> None:
    plan = _plan("Explain components A, B, and C together")
    records = tuple(
        _qualified(_record(f"Component {part} contributes to the mechanism.", f"E{index}"), "R1")
        for index, part in enumerate("ABC", 1)
    )
    states = finalize_requirement_states(plan=plan, evidence=records, observations=()).states
    selected = FinalEvidenceSelector().select(plan=plan, evidence=records, requirement_states=states)
    assert {item.evidence_id for item in selected} == {"E1", "E2", "E3"}
    assert set(build_completion_view(plan=plan, states=states, selected=selected).support_for("R1")) == {
        "E1", "E2", "E3"
    }


def test_partial_support_boundary_survives_cache_checkpoint_decision_and_answer(tmp_path) -> None:
    source = "The encoder processes history; the decoder predicts the next wind speed."
    plan = _plan("Explain encoder, decoder, and prediction interval output")
    response = _response("The encoder processes history")
    candidate = response["projections"][0]["support_candidates"][0]
    candidate["support_scope"] = "The encoder processes history."
    candidate["unresolved_aspects"] = ["How the decoder works", "How prediction intervals are produced"]
    record = _record(source)
    analysis, _ = _project(response, record, plan)
    assessed = replace(record, assessment=analysis)
    assert verified_support_ids(assessed, plan.requirements) == ("R1",)
    anchor = assessed.projection.support_anchors[0]
    assert anchor.support_scope == "The encoder processes history."
    assert anchor.unresolved_aspects == ("How the decoder works", "How prediction intervals are produced")
    assert finalize_requirement_states(plan=plan, evidence=(assessed,), observations=()).states[0].status is RequirementStatus.SATISFIED
    assert FinalEvidenceSelector().select(plan=plan, evidence=(assessed,)) == (assessed,)

    cache = SQLiteEvidenceAnalysisCache(tmp_path / "boundary.sqlite3")
    context = projection_semantic_cache_context(build_projection_semantic_request(
        question="local boundary", plan=plan, projection_type=EvidenceProjectionType.MECHANISM,
        requirement_ids=("R1",)))
    assert cache.put(question="local boundary", record=record, analysis=analysis, projection_context=context)
    cached = cache.lookup(question="local boundary", record=record, projection_context=context).analysis
    assert cached is not None and cached.projection.support_anchors == (anchor,)
    restored = load_contract(dump_contract((assessed,), tuple[EvidenceRecord, ...]), tuple[EvidenceRecord, ...])[0]
    assert restored.projection.support_anchors == (anchor,)
    assert _projection_view(restored)["support_anchors"][0]["unresolved_aspects"] == list(anchor.unresolved_aspects)
    assert _to_evidence_unit(restored, plan).projection_context["support_anchors"][0]["support_scope"] == anchor.support_scope


def test_empty_local_unresolved_does_not_complete_requirement() -> None:
    plan = _plan("Explain encoder and decoder together")
    source = "The encoder processes historical input."
    response = _response(source)
    response["projections"][0]["support_candidates"][0]["unresolved_aspects"] = []
    record = _record(source)
    analysis, _ = _project(response, record, plan)
    assessed = replace(record, assessment=analysis)
    assert assessed.projection.support_anchors[0].unresolved_aspects == ()
    assert finalize_requirement_states(plan=plan, evidence=(assessed,), observations=()).states[0].status is RequirementStatus.SATISFIED
    partial_answer = AnswerSemanticDraft(
        direct_answer="The encoder processes historical input.[E1]",
        evidence_explanation="The encoder processes historical input.[E1]",
        reading_interpretation=None,requirement_status={"R1":"partial"},coverage_note=None,
    )
    result = _compile_semantic_answer(
        partial_answer, allowed_evidence_ids={"E1"},
        research_requirements=(("R1", plan.requirements[0].description),),
        support_by_requirement={"R1": ("E1",)},
    )
    assert result.partial_answer and not result.insufficient_evidence


def test_unsupported_projection_does_not_gain_support_to_fill_boundary() -> None:
    plan = _plan("Explain prediction interval output")
    response = _response("The encoder processes history")
    response["projections"][0]["supported_requirement_ids"] = []
    response["projections"][0]["support_candidates"] = []
    record = _record("The encoder processes history.")
    analysis, _ = _project(response, record, plan)
    assert analysis.projection.supported_requirement_ids == ()
    assert analysis.projection.support_anchors == ()
    assert verified_support_ids(replace(record, assessment=analysis), plan.requirements) == ()


def test_unresolved_aspects_are_limited_to_requirement_by_projector_instruction() -> None:
    assert "添加 Requirement 没有要求的新维度" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "把其他 Requirement 的要求放进当前项" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "不超出所选原文实际表达的信息" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "只共享方法名或实体名" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "先识别 Requirement 实际要求的" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    plan = _plan("Explain the encoder mechanism")
    source = "The encoder processes historical input."
    response = _response(source)
    response["projections"][0]["support_candidates"][0]["unresolved_aspects"] = []
    analysis, _ = _project(response, _record(source), plan)
    assert analysis.projection.support_anchors[0].unresolved_aspects == ()


def test_answer_coverage_derives_overall_flags_and_checks_citations() -> None:
    requirements = (("R1", "first part"), ("R2", "second part"))
    support = {"R1": ("E1",), "R2": ()}
    draft = AnswerSemanticDraft(direct_answer="The first part is supported.[E1]",
        evidence_explanation="The first part is supported.[E1]",reading_interpretation=None,
        requirement_status={"R1":"partial"},coverage_note=None)
    result = _compile_semantic_answer(draft,allowed_evidence_ids={"E1","E2"},
        research_requirements=requirements,support_by_requirement=support)
    assert result.partial_answer and not result.insufficient_evidence
    assert result.requirement_coverage[1].status=="unsupported"
    bad = draft.model_copy(update={"direct_answer":"Fact.[E2]","evidence_explanation":"Fact.[E2]"})
    with pytest.raises(AnswerDomainValidationError):
        _compile_semantic_answer(bad,allowed_evidence_ids={"E1","E2"},
            research_requirements=requirements,support_by_requirement=support)


@pytest.mark.parametrize("old_version", [
    "evidence-projection-schema-v15-plan-wide-support",
    "evidence-projection-schema-v16-semantic-authority",
    "evidence-projection-schema-v17-full-evidence",
    "evidence-projection-schema-v18-support-boundary",
])
def test_old_projection_cache_is_not_reused(tmp_path, old_version: str) -> None:
    path = tmp_path / "cache.sqlite3"
    record = _record("The model outputs mean and prediction intervals.")
    context = projection_semantic_cache_context(build_projection_semantic_request(
        question="model output", plan=_plan("Model output"), projection_type=EvidenceProjectionType.MECHANISM,
        requirement_ids=("R1",)))
    old = SQLiteEvidenceAnalysisCache(path, schema_version=old_version)
    assert old.put(
        question="model output", record=record,
        analysis=EvidenceAnalysis("E1", EvidenceAnalysisStatus.QUALIFIED,
            projection=MechanismProjection("E1", "LSTM-EMVE", supported_requirement_ids=())),
        projection_context=context,
    )
    current = SQLiteEvidenceAnalysisCache(path)
    assert current.lookup(question="model output", record=record, projection_context=context).analysis is None


@pytest.mark.parametrize("wrong_source_number", [False, True])
def test_two_bibliography_facts_share_one_selected_evidence_citation(wrong_source_number: bool) -> None:
    answer = "Review.[REF2] SVR.[REF3]" if wrong_source_number else "Review.[REF1] SVR.[REF1]"
    draft = AnswerSemanticDraft(direct_answer=answer,evidence_explanation=answer,
        reading_interpretation=None,requirement_status={"R1":"complete","R2":"complete"},coverage_note=None)
    arguments = dict(allowed_evidence_ids={"REF1"},
                     research_requirements=(("R1", "Review"), ("R2", "SVR")),
                     support_by_requirement={"R1": ("REF1",), "R2": ("REF1",)})
    if wrong_source_number:
        with pytest.raises(AnswerDomainValidationError) as caught:
            _compile_semantic_answer(draft, **arguments)
        assert caught.value.code == "ANSWER_UNKNOWN_CITATION"
    else:
        result = _compile_semantic_answer(draft, **arguments)
        assert result.citations == ["REF1"]
        assert all(entry.status == "complete" for entry in result.requirement_coverage)


def test_old_filtered_checkpoint_cannot_resume() -> None:
    with pytest.raises(ValueError, match="Incompatible checkpoint semantic authority"):
        _require_current_semantic_authority({})
    _require_current_semantic_authority({"semantic_authority_version": _SEMANTIC_AUTHORITY_VERSION})
