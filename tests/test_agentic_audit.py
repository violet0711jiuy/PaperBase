"""Agentic 用户审计快照的契约、脱敏与往返测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from paperbase.agentic.audit import (
    AUDIT_SCHEMA_VERSION,
    AgenticAuditSnapshot,
    build_agentic_audit_snapshot,
)
from paperbase.agentic.contracts import (
    EvidenceTarget,
    RequirementState,
    RequirementStatus,
    ResearchPlan,
    ResearchRequirement,
)


def test_audit_snapshot_round_trip_and_redacts_internal_fields() -> None:
    source = SimpleNamespace(
        paper_id="paper-a",
        paper_title="论文 A",
        section="方法",
        page_start=3,
        page_end=4,
    )
    claim = SimpleNamespace(
        claim_id="E1:C1",
        statement="论文 A 使用时空网络。",
        relation=SimpleNamespace(value="supports"),
        ownership=SimpleNamespace(value="this_paper"),
        rationale="直接描述本文方法。",
        source_excerpt="We propose a spatio-temporal network.",
    )
    evidence = SimpleNamespace(
        evidence_id="E1",
        evidence=source,
        digest="论文提出时空预测网络。",
        analysis_status=SimpleNamespace(value="qualified"),
        covered_targets=(SimpleNamespace(value="method"),),
        addressed_information=("模型结构",),
        claims=(claim,),
        source_chunk_ids=("secret-chunk",),
        score=0.99,
        projection=SimpleNamespace(
            projection_type=SimpleNamespace(value="enumeration"),
            supported_requirement_ids=("R1",),
            entities=(
                SimpleNamespace(
                    canonical_name="EMVE",
                    entity_type="method",
                    role=SimpleNamespace(value="proposed"),
                    ownership=SimpleNamespace(value="this_paper"),
                ),
            ),
        ),
        projection_rejections=("entity:EMVE:incompatible_with_active_requirements",),
    )
    selection = SimpleNamespace(
        assessments=(
            SimpleNamespace(
                paper_id="paper-a",
                disposition=SimpleNamespace(value="candidate"),
                reason="标题与问题直接相关。",
            ),
            SimpleNamespace(
                paper_id="paper-b",
                disposition=SimpleNamespace(value="excluded"),
                reason="主题无关。",
            ),
        )
    )
    attempt = SimpleNamespace(
        paper_id="paper-a",
        query="paper A forecasting method",
        lexical_terms=("forecasting", "method"),
        requirement_ids=("R1",),
        status="ok",
        error_code=None,
    )
    observation = SimpleNamespace(
        iteration=1,
        action_name="search_papers",
        tool_status=SimpleNamespace(value="ok"),
        added_evidence_ids=("E1",),
        reused_evidence_ids=(),
        search_attempts=(attempt,),
        evidence_filter_report=SimpleNamespace(
            total_candidates=2,
            accepted_candidates=1,
            rejected_candidates=1,
            deprioritized_candidates=0,
            decisions=(
                SimpleNamespace(reasons=("below_relevance_threshold",)),
            ),
        ),
        evidence_analysis_cache=SimpleNamespace(hits=0, misses=1, writes=1, errors=0),
        new_supporting_evidence_count=1,
        newly_supported_requirement_ids=("R1",),
        newly_exposed_section_ids=("methods",),
        error_codes=(),
        action_arguments={
            "searches": [
                {
                    "paper_id": "paper-a",
                    "query": "paper A forecasting method",
                    "lexical_terms": ["forecasting", "method"],
                    "requirement_ids": ["R1"],
                }
            ]
        },
    )
    coverage = SimpleNamespace(
        coverage_complete=True,
        assessments=(
            SimpleNamespace(
                paper_id="paper-a",
                status=SimpleNamespace(value="positive"),
                rationale="找到方法证据。",
                queries_tried=("paper A forecasting method",),
                evidence_ids=("E1",),
            ),
        ),
    )
    result = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        stop_reason=SimpleNamespace(value="agent_finished"),
        thread_id="thread-1",
        query_context=SimpleNamespace(scope=SimpleNamespace(value="selected_papers")),
        candidate_selection=selection,
        coverage=coverage,
        plan=None,
        evidence=(evidence,),
        selected_evidence=(evidence,),
        observations=(observation,),
        budget_usage=SimpleNamespace(
            agent_iterations=1,
            tool_calls=1,
            retrieval_queries=1,
            llm_calls=2,
            invalid_actions=0,
        ),
        answer=SimpleNamespace(citations=("E1",), llm_diagnostics=()),
        llm_diagnostics=(
            SimpleNamespace(
                node="planner",
                attempt=1,
                success=True,
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=40,
                error_stage=None,
                recovery_action=None,
                validation_summary="schema_valid",
                system_prompt="禁止落盘",
                hidden_reasoning="禁止落盘",
            ),
        ),
        error_codes=(),
        clarification=None,
        action_rejections=(
            SimpleNamespace(
                iteration=2,
                action_name="search_papers",
                action_arguments={"searches": [{"paper_id": "paper-a"}]},
                requirement_ids=("R1",),
                rejection_code="PerPaperBudget",
                rejection_reason="Query budget exhausted for paper-a.",
                guard_name="PerPaperBudget",
                recovery_action="replan_action",
                rejection_stage="budget",
                blockers=({"type": "per_paper_query_limit", "paper_id": "paper-a", "used": 2, "limit": 2},),
            ),
        ),
    )

    snapshot = build_agentic_audit_snapshot(
        result,
        paper_titles={"paper-b": "无关论文 B"},
    )
    serialized = snapshot.to_json()
    restored = AgenticAuditSnapshot.from_json(serialized)

    assert restored == snapshot
    assert "requested_requirement_ids" not in snapshot.to_dict()["papers"][0]
    assert "addressed_information" not in snapshot.to_dict()["evidence"][0]
    legacy_payload = json.loads(serialized)
    legacy_payload["papers"][0]["covered_requirement_ids"] = ["R1"]
    legacy_payload["papers"][0]["addressed_information"] = ["方法"]
    legacy_payload["evidence"][0]["supports_requirement_ids"] = ["R1"]
    legacy_restored = AgenticAuditSnapshot.from_json(json.dumps(legacy_payload))
    assert legacy_restored.papers[0].covered_requirement_ids == ("R1",)
    assert legacy_restored.papers[0].addressed_information == ("方法",)
    assert legacy_restored.evidence[0].supports_requirement_ids == ("R1",)
    # The loose SimpleNamespace fixture is not a verified EvidenceRecord.
    assert snapshot.evidence[0].supports_requirement_ids == ()
    assert snapshot.evidence[0].selected_entity_or_fact_gain == ("EMVE",)
    assert snapshot.evidence[0].selection_reason == "entity_or_fact_gain:EMVE"
    assert restored.schema_version == AUDIT_SCHEMA_VERSION
    assert restored.candidates[1].title == "无关论文 B"
    assert restored.steps[0].attempts[0].query == "paper A forecasting method"
    assert restored.steps[0].attempts[0].lexical_terms == ("forecasting", "method")
    assert restored.steps[0].attempts[0].requirement_ids == ("R1",)
    assert restored.steps[0].action_arguments["searches"][0]["requirement_ids"] == ["R1"]
    assert restored.steps[0].filter_reasons[0].reason == "below_relevance_threshold"
    assert restored.steps[0].new_supporting_evidence_count == 1
    assert restored.steps[0].newly_exposed_section_ids == ("methods",)
    assert restored.rejected_actions[0].rejection_stage == "budget"
    assert restored.rejected_actions[0].blockers[0]["paper_id"] == "paper-a"
    assert restored.evidence[0].claims[0].ownership == "this_paper"
    assert restored.evidence[0].projection_type == "enumeration"
    assert restored.evidence[0].structured_projection["supported_requirement_ids"] == ["R1"]
    assert restored.evidence[0].projection_rejections == (
        "entity:EMVE:incompatible_with_active_requirements",
    )
    for forbidden in (
        "system_prompt",
        "hidden_reasoning",
        "secret-chunk",
        '"score"',
        "graph_state",
        "api_key",
    ):
        assert forbidden not in serialized


def test_audit_snapshot_rejects_unknown_schema_version() -> None:
    try:
        AgenticAuditSnapshot.from_json(
            '{"schema_version":999,"run_status":"completed","stop_reason":"agent_finished"}'
        )
    except ValueError as error:
        assert "schema version" in str(error)
    else:
        raise AssertionError("未知审计 Schema 必须被拒绝")


def test_audit_projects_requirement_definition_state_and_evidence() -> None:
    requirement = ResearchRequirement(
        requirement_id="R1",
        description="识别论文中提出或采用的风速预测方法",
        required=True,
        evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )
    result = SimpleNamespace(
        status="completed",
        stop_reason="agent_finished",
        thread_id="requirement-audit",
        query_context=None,
        candidate_selection=None,
        coverage=None,
        plan=ResearchPlan(goal="总结方法", requirements=(requirement,)),
        requirement_states=(
            RequirementState("R1", RequirementStatus.SATISFIED, ("E2",)),
        ),
        evidence=(),
        selected_evidence=(),
        observations=(),
        budget_usage=None,
        answer=None,
        llm_diagnostics=(),
        error_codes=(),
        clarification=None,
    )

    snapshot = build_agentic_audit_snapshot(result)
    restored = AgenticAuditSnapshot.from_json(snapshot.to_json())

    assert restored.requirements[0].requirement_id == "R1"
    assert restored.requirements[0].description == requirement.description
    assert restored.requirements[0].required is True
    assert restored.requirements[0].status == "satisfied"
    assert restored.requirements[0].evidence_ids == ("E2",)
