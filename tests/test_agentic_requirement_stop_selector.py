from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from paperbase.config import AnswerGenerationSettings
from paperbase.agentic.contracts import (
    CoverageState,
    CoverageStrategy,
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EnumerationProjection,
    EvidenceRecord,
    EvidenceRelation,
    EvidenceTarget,
    EvidenceProjectionType,
    Intent,
    PaperAssessment,
    PaperAssessmentStatus,
    QueryContext,
    QueryScope,
    ProjectedEntity,
    ProjectedEntityRole,
    ResearchGoal,
    ResearchObservation,
    PaperSearchAttempt,
    ResearchRunStatus,
    StopReason,
    ResearchPlan,
    ResearchRequirement,
    RequirementSupportAnchor,
    RequirementState,
    RequirementStatus,
    ScopeSource,
)
from paperbase.agentic.audit import build_agentic_audit_snapshot
from paperbase.agentic.research_profile import goal_is_sufficient
from paperbase.agentic.selector import FinalEvidenceSelector
from paperbase.llm.client import LLMRequestError
from paperbase.generation.answer_generator import GroundedAnswerGenerator
from paperbase.generation.section_expander import EvidenceUnit
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence, ScoreKind, ScoreDirection
from paperbase.retrieval.contracts import ToolStatus


def _context() -> QueryContext:
    return QueryContext(
        original_query="列出风速预测论文的自身方法、实验 baseline 和正文 prior work 方法。",
        resolved_query="列出风速预测论文的自身方法、实验 baseline 和正文 prior work 方法。",
        intent=Intent.RETRIEVE,
        scope=QueryScope.KNOWLEDGE_BASE,
        target_paper_ids=(),
        scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
        retrieval_goal=ResearchGoal.EXHAUSTIVE_ENUMERATION,
        default_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _plan() -> ResearchPlan:
    # These tests intentionally ask for all three roles; literal fallback must
    # no longer manufacture them for a question that only asks for core methods.
    return ResearchPlan(_context().resolved_query, (
        ResearchRequirement('R1', '识别论文自身提出或实际采用的核心方法，不包含 baseline 或 prior work',
            frozenset({EvidenceTarget.METHOD}), projection_type=EvidenceProjectionType.ENUMERATION),
        ResearchRequirement('R2', '识别实验 baseline 或 comparison methods，不包含自身方法或 prior work',
            frozenset({EvidenceTarget.METHOD}), projection_type=EvidenceProjectionType.ENUMERATION),
        ResearchRequirement('R3', '识别正文 prior work 方法或方法族，不包含自身方法或实验 baseline',
            frozenset({EvidenceTarget.PRIOR_WORK}), projection_type=EvidenceProjectionType.ENUMERATION),
    ))


def _coverage() -> CoverageState:
    return CoverageState(
        scope_paper_ids=("P",),
        assessments=(
            PaperAssessment(
                "P",
                PaperAssessmentStatus.POSITIVE,
                queries_tried=("methods",),
                evidence_ids=("E1",),
                rationale="direct",
            ),
        ),
        strategy=CoverageStrategy.EXHAUSTIVE,
        candidate_paper_ids=("P",),
    )


def _record(
    evidence_id: str,
    *,
    score: float,
    methods: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
    baselines: tuple[str, ...] = (),
    prior_work: bool = False,
    requirement_ids: tuple[str, ...] = (),
) -> EvidenceRecord:
    text = f"{evidence_id} discusses " + ", ".join((*methods, *families, *baselines))
    claims = [
        EvidenceClaim(
            f"{evidence_id}:C1",
            text,
            EvidenceRelation.SUPPORTS,
            EvidenceOwnership.THIS_PAPER,
            "direct",
            text,
        )
    ]
    if prior_work:
        claims.append(
            EvidenceClaim(
                f"{evidence_id}:C2",
                "Prior work is compared.",
                EvidenceRelation.SUPPORTS,
                EvidenceOwnership.PRIOR_WORK,
                "comparison",
                text,
            )
        )
    projection = (
        EnumerationProjection(
            evidence_id=evidence_id,
            supported_requirement_ids=requirement_ids,
            support_anchors=tuple(
                RequirementSupportAnchor(requirement_id, text, text)
                for requirement_id in requirement_ids
            ),
            entities=(
                *(ProjectedEntity(name, "method", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER) for name in methods),
                *(ProjectedEntity(name, "method_family", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER) for name in families),
                *(ProjectedEntity(name, "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK) for name in baselines),
            ),
        )
        if methods or families or baselines else None
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="P",
            seed_chunk_ids=(evidence_id,),
            source_chunk_ids=(evidence_id,),
            section_id=evidence_id,
            section=evidence_id,
            content=text,
            raw_score=score,
            score_kind=ScoreKind.RERANKER,
            score_direction=ScoreDirection.HIGHER_IS_BETTER,
        ),
        supporting_queries=("methods",),
        first_seen_iteration=1,
        last_seen_iteration=1,
        requested_evidence_targets=frozenset(
            {EvidenceTarget.METHOD, EvidenceTarget.PRIOR_WORK}
        ),
        requested_requirement_ids=("R1", "R2", "R3"),
        assessment=EvidenceAnalysis(
            evidence_id=evidence_id,
            status=EvidenceAnalysisStatus.QUALIFIED,
            digest=text if projection is None else None,
            claims=tuple(claims) if projection is None else (),
            projection=projection,
        ),
    )


def _projected_record(
    evidence_id: str,
    *,
    score: float,
    requirement_ids: tuple[str, ...],
    entities: tuple[tuple[str, str, ProjectedEntityRole, EvidenceOwnership], ...],
) -> EvidenceRecord:
    record = _record(evidence_id, score=score, requirement_ids=requirement_ids)
    return replace(
        record,
        evidence=replace(record.evidence, content=(
            record.evidence.content + " " + ", ".join(name for name, *_ in entities)
        )),
        fingerprint=None,
        assessment=replace(
            record.assessment,
            digest=None,
            claims=(),
            projection=None, projections=(EnumerationProjection(
                evidence_id=evidence_id,
                supported_requirement_ids=requirement_ids,
                support_anchors=tuple(
                    RequirementSupportAnchor(
                        requirement_id, record.evidence.content, record.evidence.content
                    )
                    for requirement_id in requirement_ids
                ),
                entities=tuple(
                    ProjectedEntity(name, entity_type, role, ownership)
                    for name, entity_type, role, ownership in entities
                ),
            ),),
        ),
    )


class _FailThenAnswerClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            raise LLMRequestError("timeout", error_stage="timeout")
        return json.dumps(
            {
                "direct_answer": "可检索到 EMVE。[E1]",
                "evidence_explanation": "证据明确给出该方法。[E1]",
                "reading_interpretation": None,
                "citations": ["E1"],
                "insufficient_evidence": False,
                "partial_answer": False,
                "coverage_note": None,
            }
        )


class RequirementStopSelectorTests(unittest.TestCase):
    def test_core_selector_excludes_exploratory_projection_and_audit_explains_it(self) -> None:
        supporting = _projected_record(
            "E1", score=0.8, requirement_ids=("R1",),
            entities=(("EMVE", "method", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER),),
        )
        exploratory = _projected_record(
            "E2", score=0.99, requirement_ids=(),
            entities=(("novel loss function", "method", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER),),
        )
        states = (RequirementState("R1", RequirementStatus.SATISFIED, ("E1",)),)
        selected = FinalEvidenceSelector(max_units=8).select(
            plan=ResearchPlan(goal="method", requirements=(_plan().requirements[0],)),
            evidence=(supporting, exploratory), requirement_states=states,
        )

        self.assertEqual([item.evidence_id for item in selected], ["E1"])
        # Selection is a view: the Registry evidence remains available for
        # research/audit even though it cannot enter Answer Context.
        self.assertEqual((supporting, exploratory)[1].evidence_id, "E2")
        result = SimpleNamespace(
            status=ResearchRunStatus.COMPLETED, stop_reason=StopReason.AGENT_FINISHED,
            thread_id="T", query_context=_context(),
            plan=ResearchPlan(goal="method", requirements=(_plan().requirements[0],)),
            evidence=(supporting, exploratory), selected_evidence=selected,
            observations=(), coverage=_coverage(), candidate_selection=None, answer=None,
            budget_usage=SimpleNamespace(), requirement_states=states,
            error_codes=(), clarification=None, llm_diagnostics=(),
        )
        audit = build_agentic_audit_snapshot(result)
        item = next(value for value in audit.evidence if value.evidence_id == "E2")
        self.assertFalse(item.selected)
        self.assertEqual(item.selection_reason, "no_required_requirement_support")

    def test_requirement_selector_covers_all_states_and_preserves_distinct_prior_work(self) -> None:
        records = (
            _projected_record("E2", score=0.99, requirement_ids=("R1",), entities=(("deep learning framework", "framework", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E3", score=0.98, requirement_ids=("R1",), entities=(("EMVE", "method", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E4", score=0.97, requirement_ids=("R1",), entities=(("EMVE", "method", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E6", score=0.88, requirement_ids=("R2",), entities=(("LUBE-ANN", "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK), ("LSTM QR", "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK))),
            _projected_record("E10", score=0.70, requirement_ids=("R3",), entities=(("direct PI generation", "method_family", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK), ("MVE approaches", "method_family", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK))),
            _projected_record("E11", score=0.69, requirement_ids=("R3",), entities=(("fuzzy logic", "method_family", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK), ("credibility-based uncertainty quantification", "method_family", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK))),
        )
        states = (
            RequirementState("R1", RequirementStatus.SATISFIED, ("E2", "E3", "E4")),
            RequirementState("R2", RequirementStatus.SATISFIED, ("E6",)),
            RequirementState("R3", RequirementStatus.SATISFIED, ("E10", "E11")),
        )
        # Fixture tuple is the Retriever's returned order, not a calibrated
        # global score domain. Preserve its explicit ordinal tie-break.
        records = tuple(replace(r, evidence=replace(r.evidence, retrieval_rank=i))
                        for i, r in enumerate(records, start=1))

        selected = FinalEvidenceSelector(max_units=4).select(
            context=_context(), plan=_plan(), coverage=_coverage(), evidence=records,
            requirement_states=states,
        )

        self.assertEqual({item.evidence_id for item in selected}, {"E2", "E6", "E10", "E11"})
        selected_requirements = {
            requirement_id for record in selected
            for requirement_id in record.projection.supported_requirement_ids
        }
        self.assertEqual(selected_requirements, {"R1", "R2", "R3"})
        self.assertNotIn("E3", {item.evidence_id for item in selected})
        self.assertNotIn("E4", {item.evidence_id for item in selected})

    def test_selector_normalizes_known_lstm_variants_and_never_budget_drops_unique_requirement(self) -> None:
        records = (
            _projected_record("E1", score=0.9, requirement_ids=("R1",), entities=(("EMVE", "method", ProjectedEntityRole.ADOPTED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E2", score=0.8, requirement_ids=("R2",), entities=(("LSTM QR", "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK),)),
            _projected_record("E3", score=0.7, requirement_ids=("R2",), entities=(("LSTM-QR", "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK),)),
            _projected_record("E4", score=0.6, requirement_ids=("R3",), entities=(("fuzzy logic", "method_family", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK),)),
        )
        states = tuple(
            RequirementState(requirement_id, RequirementStatus.SATISFIED, evidence_ids)
            for requirement_id, evidence_ids in (("R1", ("E1",)), ("R2", ("E2", "E3")), ("R3", ("E4",)))
        )
        records = tuple(replace(r, evidence=replace(r.evidence, retrieval_rank=i))
                        for i, r in enumerate(records, start=1))

        selected = FinalEvidenceSelector(max_units=1).select(
            plan=_plan(), coverage=_coverage(), evidence=records, requirement_states=states,
        )

        self.assertEqual({item.evidence_id for item in selected}, {"E1", "E2", "E4"})
        self.assertTrue(selected)
    def test_explicit_role_plan_has_only_the_requested_method_requirements(self) -> None:
        plan = _plan()

        self.assertEqual(
            [item.requirement_id for item in plan.requirements],
            ["R1", "R2", "R3"],
        )
        self.assertTrue(
            all(item.required for item in plan.requirements)
        )
        descriptions = " ".join(item.description for item in plan.requirements)
        for forbidden in ("数据集", "指标", "预处理", "超参数"):
            self.assertNotIn(forbidden, descriptions)

    def test_goal_stop_requires_method_baseline_and_prior_work_coverage(self) -> None:
        # 即使模型自报 R1/R2/R3，程序也不能把“只有本文方法”的证据
        # 误判为已经区分 baseline 与 prior work。
        emve_only = _projected_record(
            "E2", score=0.99, requirement_ids=("R1",),
            entities=(("EMVE", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),),
        )
        self.assertFalse(
            goal_is_sufficient(
                context=_context(),
                plan=_plan(),
                coverage=_coverage(),
                evidence=(emve_only,),
            )
        )

        diverse = _projected_record(
            "E7", score=0.8, requirement_ids=("R1", "R2", "R3"),
            entities=(
                ("EMVE", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),
                ("LSTM_QR", "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK),
                ("fuzzy logic", "method_family", ProjectedEntityRole.PRIOR_WORK, EvidenceOwnership.PRIOR_WORK),
            ),
        )
        self.assertTrue(
            goal_is_sufficient(
                context=_context(),
                plan=_plan(),
                coverage=_coverage(),
                evidence=(emve_only, diverse),
            )
        )

    def test_selector_prefers_new_method_entities_over_repeated_emve(self) -> None:
        records = (
            _projected_record("E2", score=0.99, requirement_ids=("R1",), entities=(("EMVE", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E3", score=0.98, requirement_ids=("R1",), entities=(("EMVE", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E4", score=0.97, requirement_ids=("R1",), entities=(("EMVE", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),)),
            _projected_record("E7", score=0.80, requirement_ids=("R3",), entities=(("fuzzy logic", "method_family", ProjectedEntityRole.PRIOR_WORK, EvidenceOwnership.PRIOR_WORK),)),
            _projected_record("E13", score=0.60, requirement_ids=("R2",), entities=(("LSTM_QR", "method", ProjectedEntityRole.BASELINE, EvidenceOwnership.PRIOR_WORK),)),
        )

        selected = FinalEvidenceSelector(max_units=4).select(
            context=_context(),
            plan=_plan(),
            coverage=_coverage(),
            evidence=records,
            requirement_states=(
                RequirementState("R1", RequirementStatus.SATISFIED, ("E2", "E3", "E4")),
                RequirementState("R2", RequirementStatus.SATISFIED, ("E13",)),
                RequirementState("R3", RequirementStatus.SATISFIED, ("E7",)),
            ),
        )
        selected_ids = {item.evidence_id for item in selected}

        self.assertIn("E7", selected_ids)
        self.assertIn("E13", selected_ids)
        self.assertLess(len(selected_ids.intersection({"E2", "E3", "E4"})), 3)


    def test_final_answer_retries_one_request_error_and_then_succeeds(self) -> None:
        client = _FailThenAnswerClient()
        generator = GroundedAnswerGenerator(
            settings=AnswerGenerationSettings(request_error_retries=1),
            client=client,
        )
        evidence = EvidenceUnit(
            evidence_id="E1",
            kind="content",
            paper_id="P",
            paper_title="Wind",
            section="Method",
            page_start=1,
            page_end=1,
            seed_chunk_ids=("c",),
            chunk_ids=("c",),
            text="EMVE is proposed.",
            token_count=10,
        )

        outcome = generator.generate(
            query="有哪些方法？",
            evidence=(evidence,),
        )

        self.assertEqual(outcome.status, "success")
        self.assertEqual(client.calls, 2)
        self.assertEqual(
            [item.success for item in outcome.llm_diagnostics], [False, True]
        )
        self.assertEqual(outcome.llm_diagnostics[0].error_stage, "timeout")

    def test_audit_uses_formal_requirement_rows_instead_of_per_paper_completion(self) -> None:
        record = _record(
            "E7",
            score=0.8,
            methods=("EMVE",),
            families=("fuzzy logic",),
            baselines=("LSTM_QR",),
            prior_work=True,
            requirement_ids=("R1", "R2", "R3"),
        )
        observation = ResearchObservation(
            iteration=1,
            action_name="search_papers",
            tool_status=ToolStatus.OK,
            added_evidence_ids=("E7",),
            reused_evidence_ids=(),
            search_attempts=(
                PaperSearchAttempt(
                    paper_id="P",
                    query="wind methods",
                    evidence_targets=frozenset({EvidenceTarget.METHOD}),
                    requirement_ids=("R1", "R2", "R3"),
                    status="ok",
                ),
            ),
        )
        result = SimpleNamespace(
            status=ResearchRunStatus.COMPLETED,
            stop_reason=StopReason.AGENT_FINISHED,
            thread_id="T",
            query_context=_context(),
            plan=_plan(),
            evidence=(record,),
            selected_evidence=(record,),
            observations=(observation,),
            coverage=_coverage(),
            candidate_selection=None,
            answer=None,
            budget_usage=SimpleNamespace(
                agent_iterations=1,
                tool_calls=1,
                retrieval_queries=1,
                llm_calls=1,
                invalid_actions=0,
                queries_per_paper=(("P", 1),),
            ),
            error_codes=(),
            clarification=None,
            llm_diagnostics=(),
        )

        snapshot = build_agentic_audit_snapshot(result)

        self.assertEqual(
            tuple(item.requirement_id for item in snapshot.requirements),
            ("R1", "R2", "R3"),
        )
        self.assertEqual(
            snapshot.evidence[0].verified_supports_requirement_ids,
            ("R1", "R2", "R3"),
        )
        self.assertEqual(snapshot.papers[0].covered_requirement_ids, ())
        self.assertNotIn("covered_requirement_ids", snapshot.to_dict()["papers"][0])
        self.assertNotIn("requested_targets", snapshot.to_dict()["papers"][0])


if __name__ == "__main__":
    unittest.main()
