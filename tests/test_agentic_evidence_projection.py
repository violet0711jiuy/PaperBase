from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from paperbase.agentic.contracts import (
    EvidenceAnalysisStatus,
    EvidenceAnalysis,
    EvidenceClaim,
    EvidenceRelation,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceTarget,
    EnumerationProjection,
    FactProjection,
    EvidenceProjectionType,
    ProjectedEntity,
    ProjectedEntityRole,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRequirement,
    RequirementSupportAnchor,
    ScopeSource,
)
from paperbase.agentic.evidence_analysis import EvidenceAnalysisRequest
from paperbase.agentic.evidence_projection import LLMEvidenceProjector
from paperbase.agentic.projection_shapes import requirement_projection_type
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache, _analysis_payload
from paperbase.agentic.projection_semantics import build_projection_semantic_request, projection_semantic_cache_context
from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
from paperbase.agentic.schemas import EvidenceProjectionBatchDraft, FactProjectionBatchDraft
from paperbase.agentic.source_spans import source_span_ids_for_excerpt
from paperbase.prompts.evidence_projector import EVIDENCE_PROJECTOR_SYSTEM_PROMPT
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


def _analysis_request(*, question, evidence, context, plan, projection_requirement_ids):
    """Bind pre-scope test fixtures explicitly to the scope this test exercises."""
    records = tuple(replace(record, requested_requirement_ids=tuple(projection_requirement_ids))
                    for record in evidence)
    return EvidenceAnalysisRequest(question, records, context, plan,
        tuple((record.evidence_id, tuple(projection_requirement_ids)) for record in records))


class _QueuedClient:
    def __init__(self, payloads: tuple[str, ...]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        prompt = json.loads(str(kwargs["user_prompt"]))
        sources = {item["evidence_id"]: item["source_spans"] for item in prompt["evidence"]}
        # Upgrade compact enumeration fixtures to the per-support span shape.
        for item in parsed.get("projections", []):
            if item.get("projection_type") != "enumeration" or "support_candidates" in item:
                continue
            entities = item.get("entities", [])
            excerpt = entities[0]["canonical_name"] if entities else "EMVE"
            spans = sources[item["evidence_id"]]
            span_id = next(
                (span["span_id"] for span in spans if excerpt in span["text"]),
                spans[0]["span_id"],
            )
            item["support_candidates"] = [
                {"requirement_id": requirement_id, "source_span_ids": [span_id],
                 "support_scope": excerpt, "unresolved_aspects": []}
                for requirement_id in item.get("supported_requirement_ids", [])
            ]
        return json.dumps(parsed)


def _context(*, goal: ResearchGoal = ResearchGoal.EXHAUSTIVE_ENUMERATION, intent: Intent = Intent.SYNTHESIZE, question: str = "知识库中有哪些风速预测方法") -> QueryContext:
    return QueryContext(
        original_query=question,
        resolved_query=question,
        intent=intent,
        scope=QueryScope.SINGLE_PAPER,
        target_paper_ids=("P",),
        scope_source=ScopeSource.UI_SELECTION,
        retrieval_goal=goal,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD, EvidenceTarget.PRIOR_WORK}),
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="枚举风速预测方法",
        requirements=(
            ResearchRequirement("R1", "论文自身提出或采用的核心方法", required=True, evidence_targets=frozenset({EvidenceTarget.METHOD})),
            ResearchRequirement("R2", "实验 baseline/comparison methods", required=True, evidence_targets=frozenset({EvidenceTarget.METHOD})),
            ResearchRequirement("R3", "正文明确讨论的 prior-work 方法", required=True, evidence_targets=frozenset({EvidenceTarget.PRIOR_WORK})),
        ),
    )


def _e2() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="E2",
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="P",
            section_id="method",
            section="Method",
            seed_chunk_ids=("c2",),
            source_chunk_ids=("c2",),
            content=(
                "We propose EMVE and a Proposed model for wind speed forecasting. "
                "EMVE loss function trains the model. We compare it with LUBE and "
                "LUBE-ANN. Prior work includes Old family."
            ),
        ),
        supporting_queries=("proposed wind speed forecasting method",),
        first_seen_iteration=1,
        last_seen_iteration=1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD, EvidenceTarget.PRIOR_WORK}),
        requested_requirement_ids=("R1", "R2", "R3"),
    )


def test_enumeration_projection_is_primary_and_does_not_require_claims_excerpt_or_digest() -> None:
    response = json.dumps(
        {
            "projections": [
                {
                    "projection_type": "enumeration",
                    "evidence_id": "E2",
                    "status": "qualified",
                    "entities": [
                        {"canonical_name": "EMVE", "entity_type": "method", "role": "proposed", "ownership": "this_paper"},
                        {"canonical_name": "LUBE", "entity_type": "method", "role": "baseline", "ownership": "prior_work"},
                    ],
                    "supported_requirement_ids": ["R1", "R2"],
                }
            ]
        }
    )
    client = _QueuedClient((response,))
    result = LLMEvidenceProjector(client, max_attempts=1).analyze(request=_analysis_request(question="知识库中有哪些风速预测方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R1", "R2", "R3")))

    analysis = result.analyses[0]
    assert result.llm_calls == 1
    assert analysis.status is EvidenceAnalysisStatus.QUALIFIED
    assert analysis.claims == ()
    assert analysis.digest is None
    assert isinstance(analysis.projection, EnumerationProjection)
    assert [(entity.canonical_name, entity.role.value, entity.ownership) for entity in analysis.projection.entities] == [
        ("EMVE", "proposed", EvidenceOwnership.THIS_PAPER),
        ("LUBE", "baseline", EvidenceOwnership.PRIOR_WORK),
    ]
    assert analysis.projection.supported_requirement_ids == ("R1", "R2")
    prompt = json.loads(client.calls[0]["user_prompt"])
    assert prompt["projection_type"] == "enumeration"
    assert [{k:v for k,v in r.items() if k != "projection_type"} for r in prompt["projection_requirements"]] == [
        {"requirement_id": "R1", "description": "论文自身提出或采用的核心方法"},
        {"requirement_id": "R2", "description": "实验 baseline/comparison methods"},
        {"requirement_id": "R3", "description": "正文明确讨论的 prior-work 方法"},
    ]
    assert "source_excerpt" not in json.dumps(prompt)
    assert "claims" not in json.dumps(prompt)


def test_projector_prompt_is_chinese_and_states_minimal_requirement_rules() -> None:
    assert "面向 Requirement 的最小证据投影器" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "只输出 Requirement 明确要求的实体类型" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "只因为该 Evidence 是为该 Requirement 检索得到" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT
    assert "实验比较对象：baseline" in EVIDENCE_PROJECTOR_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (_context(goal=ResearchGoal.EXHAUSTIVE_ENUMERATION), "enumeration"),
        (_context(goal=ResearchGoal.COMPARISON_SUPPORT, intent=Intent.COMPARE), "comparison"),
        (_context(goal=ResearchGoal.FIND_EVIDENCE, question="这篇论文使用了什么数据集"), "fact"),
        (_context(goal=ResearchGoal.FIND_EVIDENCE, question="该方法如何构建和工作"), "mechanism"),
    ),
)
def test_untyped_requirement_fallback_uses_query_context(context: QueryContext, expected: str) -> None:
    requirement = ResearchRequirement("R1", context.resolved_query, context.default_evidence_targets)
    assert requirement.projection_type is None
    assert requirement_projection_type(requirement, context=context).value == expected


def test_projector_rejects_requirement_outside_projection_scope() -> None:
    invalid = json.dumps(
        {
            "projections": [
                {
                    "projection_type": "enumeration",
                    "evidence_id": "E2",
                    "status": "qualified",
                    "entities": [],
                    "supported_requirement_ids": ["R3"],
                }
            ]
        }
    )
    result = LLMEvidenceProjector(_QueuedClient((invalid,)), max_attempts=1).analyze(request=_analysis_request(question="知识库中有哪些风速预测方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R1", "R2")))
    assert result.analyses[0].status is EvidenceAnalysisStatus.FAILED
    assert result.analyses[0].claims == ()


def test_projector_can_support_requirement_after_binding_expansion() -> None:
    source = "Wang, Y. (2015). Short-term load forecasting using support vector regression."
    record = replace(
        _e2(),
        evidence=replace(_e2().evidence, content=source),
        requested_requirement_ids=("R1",),
    )
    plan = ResearchPlan(
        "识别参考文献",
        requirements=(
            ResearchRequirement("R1", "识别 Itakura 1975 文献", required=True, evidence_targets=frozenset({EvidenceTarget.REFERENCE})),
            ResearchRequirement("R2", "识别 Wang 2015 支持向量回归文献", required=True, evidence_targets=frozenset({EvidenceTarget.REFERENCE})),
        ),
    )
    response = json.dumps({"projections": [{
        "projection_type": "fact", "evidence_id": "E2", "status": "qualified",
        "fact_type": "reference", "value": source, "ownership": "prior_work",
        "supported_requirement_ids": ["R2"],
        "support_candidates": [{
            "requirement_id": "R2", "source_span_ids": list(source_span_ids_for_excerpt(source, source)),
            "support_scope": source, "unresolved_aspects": [],
        }],
    }]})
    client = _QueuedClient((response,))
    result = LLMEvidenceProjector(client, max_attempts=1).analyze(request=_analysis_request(question="这两篇参考文献是什么？", evidence=(record,), context=_context(goal=ResearchGoal.FIND_EVIDENCE, question="这两篇参考文献是什么？"), plan=plan, projection_requirement_ids=("R1", "R2")))

    assert result.analyses[0].projection.supported_requirement_ids == ("R2",)
    prompt = json.loads(client.calls[0]["user_prompt"])
    assert {item["requirement_id"] for item in prompt["projection_requirements"]} == {"R1", "R2"}
    assert "projection_requirement_ids" not in prompt["evidence"][0]

    with TemporaryDirectory() as directory:
        cache = SQLiteEvidenceAnalysisCache(Path(directory) / "projection.sqlite3")
        context = projection_semantic_cache_context(build_projection_semantic_request(
            question="这两篇参考文献是什么？", plan=plan, projection_type=EvidenceProjectionType.FACT,
            requirement_ids=("R1", "R2")))
        assert cache.put(question="这两篇参考文献是什么？", record=record, analysis=result.analyses[0], projection_context=context)
        restored = cache.lookup(question="这两篇参考文献是什么？", record=record, projection_context=context)
        assert restored.analysis is not None
        assert restored.analysis.projection.supported_requirement_ids == ("R2",)


def test_one_bibliography_chunk_can_hold_two_anchored_facts() -> None:
    lei = "Lei, M. (2009). A review on wind speed and wind power forecasting."
    wang = "Wang, Y. (2015). Wind speed forecasting using support vector regression."
    record = replace(
        _e2(), evidence=replace(_e2().evidence, content=lei + "\n" + wang),
        requested_requirement_ids=("R1",),
    )
    plan = ResearchPlan("识别两篇文献", requirements=(
        ResearchRequirement("R1", "识别 Lei 2009 风速预测综述", required=True, evidence_targets=frozenset({EvidenceTarget.REFERENCE})),
        ResearchRequirement("R2", "识别 Wang 2015 支持向量回归风速预测论文", required=True, evidence_targets=frozenset({EvidenceTarget.REFERENCE})),
    ))
    response = json.dumps({"projections": [
        {
            "projection_type": "fact", "evidence_id": "E2", "status": "qualified",
            "fact_type": "reference", "value": lei, "ownership": "prior_work",
            "supported_requirement_ids": ["R1"],
            "support_candidates": [{"requirement_id": "R1", "source_span_ids": list(source_span_ids_for_excerpt(lei + "\n" + wang, lei)), "support_scope": lei, "unresolved_aspects": []}],
        },
        {
            "projection_type": "fact", "evidence_id": "E2", "status": "qualified",
            "fact_type": "reference", "value": wang, "ownership": "prior_work",
            "supported_requirement_ids": ["R2"],
            "support_candidates": [{"requirement_id": "R2", "source_span_ids": list(source_span_ids_for_excerpt(lei + "\n" + wang, wang)), "support_scope": wang, "unresolved_aspects": []}],
        },
    ]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="两篇文献分别是什么", evidence=(record,), context=_context(goal=ResearchGoal.FIND_EVIDENCE, question="两篇文献分别是什么"), plan=plan, projection_requirement_ids=("R1", "R2")))
    projection = result.analyses[0].projection
    assert isinstance(projection, FactProjection)
    assert len(projection.facts) == 2
    assert projection.supported_requirement_ids == ("R1", "R2")
    assert tuple(fact.value for fact in projection.facts) == (lei, wang)

    assessed = replace(record, assessment=result.analyses[0])
    restored = load_contract(
        dump_contract((assessed,), tuple[EvidenceRecord, ...]),
        tuple[EvidenceRecord, ...],
    )[0]
    assert restored.projection.supported_requirement_ids == ("R1", "R2")
    legacy_payload = dump_contract((assessed,), tuple[EvidenceRecord, ...])
    legacy_projection = legacy_payload[0]["assessment"].pop("projections")[0]
    legacy_payload[0]["assessment"]["projection"] = legacy_projection
    first_fact = legacy_projection.pop("facts")[0]
    legacy_projection.update(first_fact)
    legacy_restored = load_contract(legacy_payload, tuple[EvidenceRecord, ...])[0]
    assert len(legacy_restored.projection.facts) == 1
    assert legacy_restored.projection.supported_requirement_ids == ("R1",)
    with TemporaryDirectory() as directory:
        cache = SQLiteEvidenceAnalysisCache(Path(directory) / "facts.sqlite3")
        context = projection_semantic_cache_context(build_projection_semantic_request(
            question="两篇文献分别是什么", plan=plan, projection_type=EvidenceProjectionType.FACT,
            requirement_ids=("R1", "R2")))
        assert cache.put(question="两篇文献分别是什么", record=record, analysis=result.analyses[0], projection_context=context)
        hit = cache.lookup(question="两篇文献分别是什么", record=record, projection_context=context)
        assert hit.analysis is not None
        assert len(hit.analysis.projection.facts) == 2
        assert hit.analysis.projection.supported_requirement_ids == ("R1", "R2")


def test_irrelevant_fact_candidate_does_not_discard_qualified_fact() -> None:
    source = "Lei, M. (2009). A review on wind speed and wind power forecasting."
    record = replace(_e2(), evidence=replace(_e2().evidence, content=source))
    response = json.dumps({"projections": [
        {"projection_type": "fact", "evidence_id": "E2", "status": "qualified",
         "fact_type": "reference", "value": source, "ownership": "prior_work",
         "supported_requirement_ids": ["R1"],
         "support_candidates": [{"requirement_id": "R1", "source_span_ids": list(source_span_ids_for_excerpt(source, source)),
                                 "support_scope": source, "unresolved_aspects": []}]},
        {"projection_type": "fact", "evidence_id": "E2", "status": "irrelevant",
         "fact_type": None, "value": None, "ownership": None,
         "supported_requirement_ids": [], "support_candidates": []},
    ]})
    plan = ResearchPlan("识别文献", requirements=(
        ResearchRequirement("R1", "识别 Lei 2009 风速预测综述", required=True,
                            evidence_targets=frozenset({EvidenceTarget.REFERENCE})),
    ))
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="Lei 2009 文献是什么", evidence=(record,), context=_context(goal=ResearchGoal.FIND_EVIDENCE, question="Lei 2009 文献是什么"), plan=plan, projection_requirement_ids=("R1",)))
    assert result.analyses[0].status is EvidenceAnalysisStatus.QUALIFIED
    assert result.analyses[0].projection.supported_requirement_ids == ("R1",)


@pytest.mark.parametrize(
    "projection",
    (
        {
            "projection_type": "comparison",
            "evidence_id": "E2",
            "status": "qualified",
            "comparison_dimensions": ["forecast horizon"],
            "supported_requirement_ids": ["R1"],
        },
        {
            "projection_type": "fact",
            "evidence_id": "E2",
            "status": "qualified",
            "fact_type": "dataset",
            "value": "NREL wind data",
            "ownership": "this_paper",
            "supported_requirement_ids": ["R1"],
            "support_candidates": [{"requirement_id": "R1", "source_span_ids": ["S1"],
                                    "support_scope": "NREL wind data", "unresolved_aspects": []}],
        },
        {
            "projection_type": "mechanism",
            "evidence_id": "E2",
            "status": "qualified",
            "mechanism": "Encode historical wind speed then forecast the next horizon.",
            "steps": ["encode history", "decode forecast"],
            "inputs": ["wind speed history"],
            "outputs": ["forecast"],
            "supported_requirement_ids": ["R1"],
        },
    ),
)
def test_non_enumeration_projection_schemas_are_tool_specific(
    projection: dict[str, object],
) -> None:
    parsed = EvidenceProjectionBatchDraft.model_validate_json(
        json.dumps({"projections": [projection]})
    )
    assert parsed.projections[0].projection_type == projection["projection_type"]


def test_projection_cache_does_not_reuse_legacy_claim_analysis_payload() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "evidence.sqlite3"
        record = _e2()
        legacy_cache = SQLiteEvidenceAnalysisCache(path, schema_version="evidence-analysis-schema-v3")
        legacy_analysis = EvidenceAnalysis(
            "E2",
            EvidenceAnalysisStatus.QUALIFIED,
            digest="legacy",
            claims=(
                EvidenceClaim(
                    "E2:C1",
                    "EMVE is proposed.",
                    EvidenceRelation.SUPPORTS,
                    EvidenceOwnership.THIS_PAPER,
                    "legacy",
                    "We propose EMVE",
                ),
            ),
        )
        legacy_key = legacy_cache._key(question="哪些方法", record=record)
        with legacy_cache._connect() as connection:
            connection.execute("INSERT INTO evidence_analysis_cache VALUES (?, ?, ?, ?)",
                (legacy_key, record.fingerprint, json.dumps(_analysis_payload(legacy_analysis)), "legacy"))

        projector_cache = SQLiteEvidenceAnalysisCache(path)
        assert projector_cache.lookup(question="哪些方法", record=record).analysis is None


def test_projection_checkpoint_round_trip_is_additive_for_existing_record_contract() -> None:
    response = json.dumps(
        {
            "projections": [
                {
                    "projection_type": "enumeration", "evidence_id": "E2",
                    "status": "qualified", "entities": [],
                    "supported_requirement_ids": ["R1"],
                }
            ]
        }
    )
    analysis = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="知识库中有哪些风速预测方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R1",))).analyses[0]
    record = _e2()
    projected = replace(record, assessment=analysis)
    payload = dump_contract((projected,), tuple[EvidenceRecord, ...])
    assert "assessment" in payload[0] and "analysis" not in payload[0]
    assert "covered_requirement_ids" not in payload[0]["assessment"]
    assert "claims" not in payload[0]["assessment"]
    assert "digest" not in payload[0]["assessment"]
    restored = load_contract(payload, tuple[EvidenceRecord, ...])
    assert isinstance(restored[0].projection, EnumerationProjection)


def test_old_checkpoint_duplicate_coverage_cannot_override_projection() -> None:
    record = replace(_e2(), assessment=EvidenceAnalysis(
        "E2", EvidenceAnalysisStatus.QUALIFIED,
        projection=EnumerationProjection(
            "E2", (ProjectedEntity("EMVE", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),), ("R1",),
            (RequirementSupportAnchor("R1", "EMVE", "EMVE"),)
        ),
    ))
    payload = dump_contract((record,), tuple[EvidenceRecord, ...])
    legacy = payload[0]
    legacy["analysis"] = legacy.pop("assessment")
    legacy["analysis"]["covered_requirement_ids"] = ["R2"]
    legacy["analysis"]["method_entities"] = ["invented method"]

    restored = load_contract([legacy], tuple[EvidenceRecord, ...])[0]

    assert restored.covered_requirement_ids == ("R1",)
    assert restored.method_entities == ("EMVE",)


def test_cached_enumeration_revalidates_entity_against_source(tmp_path) -> None:
    record = _e2()
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "analysis.sqlite3")
    assert cache.put(
        question="知识库中有哪些风速预测方法", record=record,
        analysis=EvidenceAnalysis(
            "E2", EvidenceAnalysisStatus.QUALIFIED,
            evaluated_requirement_ids=("R1",),
            projection=EnumerationProjection(
                "E2", (ProjectedEntity("invented method", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),), ("R1",),
                (RequirementSupportAnchor("R1", "EMVE", "EMVE"),)
            ),
        ),
        projection_context=projection_semantic_cache_context(build_projection_semantic_request(
            question="知识库中有哪些风速预测方法", plan=_plan(), projection_type=EvidenceProjectionType.ENUMERATION,
            requirement_ids=("R1",))),
    )
    client = _QueuedClient(())

    result = LLMEvidenceProjector(client, cache=cache).analyze(request=_analysis_request(question="知识库中有哪些风速预测方法", evidence=(record,), context=_context(), plan=_plan(), projection_requirement_ids=("R1",)))

    assert result.cache_stats.hits == 1
    assert client.calls == []
    assert result.analyses[0].projection.supported_requirement_ids == ()
    assert result.analyses[0].projection.entities == ()


@pytest.mark.parametrize(
    ("requirement_id", "entity"),
    (
        ("R1", {"canonical_name": "LUBE", "entity_type": "method", "role": "prior_work", "ownership": "prior_work"}),
        ("R2", {"canonical_name": "EMVE", "entity_type": "method", "role": "proposed", "ownership": "this_paper"}),
        ("R3", {"canonical_name": "LUBE", "entity_type": "method", "role": "baseline", "ownership": "prior_work"}),
    ),
)
def test_attributed_entity_preserves_projector_support_without_role_lexical_recheck(
    requirement_id: str, entity: dict[str, str],
) -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [entity], "supported_requirement_ids": [requirement_id],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="枚举方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=(requirement_id,)))
    analysis = result.analyses[0]
    assert analysis.projection is not None
    assert analysis.projection.supported_requirement_ids == (requirement_id,)
    assert [item.canonical_name for item in analysis.projection.entities] == [entity["canonical_name"]]
    assert not analysis.projection_rejections


def test_baseline_projection_retains_valid_entities_but_only_baseline_supports_r2() -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [
            {"canonical_name": "EMVE", "entity_type": "method", "role": "proposed", "ownership": "this_paper"},
            {"canonical_name": "LUBE-ANN", "entity_type": "model", "role": "baseline", "ownership": "prior_work"},
            {"canonical_name": "Old family", "entity_type": "method_family", "role": "prior_work", "ownership": "prior_work"},
        ], "supported_requirement_ids": ["R2"],
    }]})
    record = replace(_e2(), requested_requirement_ids=("R2",))
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="哪些 baseline", evidence=(record,), context=_context(), plan=_plan(), projection_requirement_ids=("R2",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert [entity.canonical_name for entity in projection.entities] == ["EMVE", "LUBE-ANN", "Old family"]
    assert projection.supported_requirement_ids == ("R2",)


@pytest.mark.parametrize(
    ("requirement_id", "entity"),
    (
        ("R1", {"canonical_name": "Proposed model", "entity_type": "model", "role": "proposed", "ownership": "unknown"}),
        ("R2", {"canonical_name": "LUBE-ANN", "entity_type": "model", "role": "baseline", "ownership": "unknown"}),
        ("R3", {"canonical_name": "LUBE", "entity_type": "method", "role": "prior_work", "ownership": "unknown"}),
    ),
)
def test_enumeration_does_not_invent_support_from_entity_role(
    requirement_id: str, entity: dict[str, str],
) -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [entity], "supported_requirement_ids": [],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="枚举方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=(requirement_id,)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert projection.supported_requirement_ids == ()


def test_component_is_retained_but_cannot_support_method_requirement() -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [
            {"canonical_name": "EMVE loss function", "entity_type": "method", "role": "proposed", "ownership": "this_paper"},
            {"canonical_name": "Proposed model", "entity_type": "model", "role": "proposed", "ownership": "this_paper"},
        ], "supported_requirement_ids": ["R1"],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="枚举方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R1",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert [entity.canonical_name for entity in projection.entities] == [
        "EMVE loss function", "Proposed model"
    ]
    assert projection.supported_requirement_ids == ("R1",)
    assert not result.analyses[0].projection_rejections


def test_mixed_evidence_retains_all_entities_but_only_proposed_supports_r1() -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [
            {"canonical_name": "EMVE", "entity_type": "method", "role": "proposed", "ownership": "this_paper"},
            {"canonical_name": "LUBE", "entity_type": "method", "role": "baseline", "ownership": "prior_work"},
            {"canonical_name": "MVE", "entity_type": "method_family", "role": "prior_work", "ownership": "prior_work"},
        ], "supported_requirement_ids": ["R1"],
    }]})
    record = replace(_e2(), evidence=replace(
        _e2().evidence, content="EMVE is proposed. LUBE is a baseline. MVE is prior work."
    ))
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="枚举方法", evidence=(record,), context=_context(), plan=_plan(), projection_requirement_ids=("R1",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert [entity.canonical_name for entity in projection.entities] == ["EMVE", "LUBE", "MVE"]
    assert projection.supported_requirement_ids == ("R1",)
    assert not result.analyses[0].projection_rejections


def test_real_audit_entity_shapes_are_retained_when_present_in_evidence() -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [
            {"canonical_name": "deep learning framework", "entity_type": "method_family", "role": "proposed", "ownership": "this_paper"},
            {"canonical_name": "single neural network architecture", "entity_type": "model", "role": "adopted", "ownership": "this_paper"},
            {"canonical_name": "MVE approach", "entity_type": "method", "role": "prior_work", "ownership": "prior_work"},
        ], "supported_requirement_ids": ["R1"],
    }]})
    record = replace(_e2(), evidence=replace(_e2().evidence, content=(
        "The proposed deep learning framework uses a single neural network architecture. "
        "The MVE approach is discussed as prior work."
    )))
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="枚举方法", evidence=(record,), context=_context(), plan=_plan(), projection_requirement_ids=("R1",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert [entity.canonical_name for entity in projection.entities] == [
        "deep learning framework", "single neural network architecture", "MVE approach"
    ]
    assert projection.supported_requirement_ids == ("R1",)


def test_entity_not_found_in_evidence_is_the_only_content_based_admission_rejection() -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [
            {"canonical_name": "deep learning framework", "entity_type": "method_family", "role": "proposed", "ownership": "this_paper"},
            {"canonical_name": "EMVE", "entity_type": "method", "role": "proposed", "ownership": "this_paper"},
        ], "supported_requirement_ids": ["R1"],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="枚举方法", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R1",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert [entity.canonical_name for entity in projection.entities] == ["EMVE"]
    assert any("entity_rejected:deep learning framework:reason=not_present_in_evidence" in reason
               for reason in result.analyses[0].projection_rejections)


@pytest.mark.parametrize(
    "entity",
    (
        {"canonical_name": "EMVE", "entity_type": "wind speed forecasting method", "role": "proposed", "ownership": "this_paper"},
        {"canonical_name": "EMVE", "entity_type": "method", "role": "other", "ownership": "this_paper"},
        {"canonical_name": "EMVE", "entity_type": "method", "role": "proposed", "ownership": "general_background"},
    ),
)
def test_enumeration_schema_is_closed_for_entity_type_role_and_ownership(entity: dict[str, str]) -> None:
    with pytest.raises(Exception):
        EvidenceProjectionBatchDraft.model_validate({"projections": [{
            "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
            "entities": [entity], "supported_requirement_ids": [],
        }]})


def test_unsupported_evidence_is_an_empty_projection_not_forced_support() -> None:
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [], "supported_requirement_ids": [],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="哪些 baseline", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R2",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert projection.entities == ()
    assert projection.supported_requirement_ids == ()


def test_duplicate_entities_are_deduplicated_deterministically() -> None:
    entity = {"canonical_name": "LUBE", "entity_type": "method", "role": "baseline", "ownership": "prior_work"}
    response = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [entity, entity], "supported_requirement_ids": ["R2"],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="哪些 baseline", evidence=(_e2(),), context=_context(), plan=_plan(), projection_requirement_ids=("R2",)))
    projection = result.analyses[0].projection
    assert isinstance(projection, EnumerationProjection)
    assert [entity.canonical_name for entity in projection.entities] == ["LUBE"]
    assert any("entity_rejected:LUBE:reason=duplicate" in reason
               for reason in result.analyses[0].projection_rejections)


def test_invalid_json_recovery_keeps_the_complete_evidence_text() -> None:
    valid = json.dumps({"projections": [{
        "projection_type": "enumeration", "evidence_id": "E2", "status": "qualified",
        "entities": [], "supported_requirement_ids": [],
    }]})
    client = _QueuedClient(("{not json", valid))
    full_content = "baseline evidence " * 120 + "TAIL_MARKER_AFTER_1600_CHARS"
    record = replace(_e2(), evidence=replace(_e2().evidence, content=full_content))
    result = LLMEvidenceProjector(client, max_attempts=2).analyze(request=_analysis_request(question="哪些 baseline", evidence=(record,), context=_context(), plan=_plan(), projection_requirement_ids=("R2",)))
    assert result.llm_calls == 2
    assert len(client.calls) == 2
    first = json.loads(client.calls[0]["user_prompt"])
    second = json.loads(client.calls[1]["user_prompt"])
    assert len(full_content) > 1600
    assert first["evidence"][0]["source_spans"] == second["evidence"][0]["source_spans"]
    assert "TAIL_MARKER_AFTER_1600_CHARS" in first["evidence"][0]["source_spans"][-1]["text"]
    assert result.llm_diagnostics[0].error_stage == "invalid_json"


def test_irrelevant_fact_schema_rejects_nonempty_fields() -> None:
    item = {"projection_type": "fact", "evidence_id": "E2", "status": "irrelevant",
            "fact_type": "reference", "value": None, "ownership": "prior_work",
            "supported_requirement_ids": [], "support_candidates": []}
    with pytest.raises(ValidationError) as caught:
        FactProjectionBatchDraft.model_validate({"projections": [item]})
    assert {error["loc"][-1] for error in caught.value.errors()} == {"fact_type", "ownership"}
    assert all(error["type"] == "none_required" for error in caught.value.errors())
    item.update(fact_type=None, ownership=None)
    assert FactProjectionBatchDraft.model_validate({"projections": [item]}).projections[0].status == "irrelevant"


def test_fact_repair_receives_constraints_and_preserves_full_evidence() -> None:
    item = {"projection_type": "fact", "evidence_id": "E2", "status": "irrelevant",
            "fact_type": "reference", "value": None, "ownership": "prior_work",
            "supported_requirement_ids": [], "support_candidates": []}
    bad = json.dumps({"projections": [item]})
    item.update(fact_type=None, ownership=None)
    client = _QueuedClient((bad, json.dumps({"projections": [item]})))
    source = "reference text " * 200 + "TAIL_MARKER"
    record = replace(_e2(), evidence=replace(_e2().evidence, content=source))
    result = LLMEvidenceProjector(client, max_attempts=2).analyze(request=_analysis_request(question="这篇论文使用了什么数据集", evidence=(record,), context=_context(goal=ResearchGoal.FIND_EVIDENCE, question="这篇论文使用了什么数据集"), plan=_plan(), projection_requirement_ids=("R1",)))
    assert result.analyses[0].status is EvidenceAnalysisStatus.IRRELEVANT
    first, second = [json.loads(call["user_prompt"]) for call in client.calls]
    assert "output_validation_feedback" not in first
    violations = second["output_validation_feedback"][0]["violations"]
    assert {entry["field_path"].split(".")[-1] for entry in violations} == {"fact_type", "ownership"}
    assert {entry["code"] for entry in violations} == {"none_required"}
    assert first["evidence"][0]["source_spans"] == second["evidence"][0]["source_spans"]
    assert "TAIL_MARKER" in first["evidence"][0]["source_spans"][-1]["text"]
    assert "reference" not in json.dumps(second["output_validation_feedback"])


@pytest.mark.parametrize("forged", [False, True])
def test_long_excerpt_uses_integrity_not_character_limit(forged: bool) -> None:
    source = "The proposed model uses wind speed history. " * 30
    span_ids = ["S999"] if forged else ["S1"]
    response = json.dumps({"projections": [{
        "projection_type": "fact", "evidence_id": "E2", "status": "qualified",
        "fact_type": "model input", "value": "wind speed history", "ownership": "this_paper",
        "supported_requirement_ids": ["R1"],
        "support_candidates": [{"requirement_id": "R1", "source_span_ids": span_ids,
                                "support_scope": "The model uses wind speed history.", "unresolved_aspects": []}],
    }]})
    result = LLMEvidenceProjector(_QueuedClient((response,)), max_attempts=1).analyze(request=_analysis_request(question="这篇论文使用了什么数据集", evidence=(replace(_e2(), evidence=replace(_e2().evidence, content=source)),), context=_context(goal=ResearchGoal.FIND_EVIDENCE, question="这篇论文使用了什么数据集"), plan=_plan(), projection_requirement_ids=("R1",)))
    analysis = result.analyses[0]
    assert analysis.status is EvidenceAnalysisStatus.QUALIFIED
    assert analysis.projection.supported_requirement_ids == (() if forged else ("R1",))
    if forged:
        assert any("source_span_not_found" in rejection for rejection in analysis.projection_rejections)
    else:
        assert analysis.projection.support_anchors[0].source_excerpt in source
