"""Agentic 用户审计快照的契约、脱敏与往返测试。"""

from __future__ import annotations

from types import SimpleNamespace

from paperbase.agentic.audit import (
    AUDIT_SCHEMA_VERSION,
    AgenticAuditSnapshot,
    build_agentic_audit_snapshot,
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
        error_codes=(),
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
    )

    snapshot = build_agentic_audit_snapshot(
        result,
        paper_titles={"paper-b": "无关论文 B"},
    )
    serialized = snapshot.to_json()
    restored = AgenticAuditSnapshot.from_json(serialized)

    assert restored == snapshot
    assert restored.schema_version == AUDIT_SCHEMA_VERSION
    assert restored.candidates[1].title == "无关论文 B"
    assert restored.steps[0].attempts[0].query == "paper A forecasting method"
    assert restored.steps[0].filter_reasons[0].reason == "below_relevance_threshold"
    assert restored.evidence[0].claims[0].ownership == "this_paper"
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
