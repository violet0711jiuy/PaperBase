from __future__ import annotations

import json
import sqlite3

from paperbase.agentic.contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    EvidenceTarget,
    EnumerationProjection,
    ProjectedEntity,
    ProjectedEntityRole,
    EvidenceProjectionType, ResearchPlan, ResearchRequirement,
)
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.agentic.projection_semantics import build_projection_semantic_request, projection_semantic_cache_context
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


def _record(evidence_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence=RetrievedEvidence(
            source_type=EvidenceSourceType.CONTENT,
            paper_id="A",
            seed_chunk_ids=("c1",),
            source_chunk_ids=("c1",),
            content="We propose a temporal encoder.",
        ),
        supporting_queries=("temporal encoder",),
        first_seen_iteration=1,
        last_seen_iteration=1,
        requested_evidence_targets=frozenset({EvidenceTarget.METHOD}),
    )


def _analysis(evidence_id: str) -> EvidenceAnalysis:
    return EvidenceAnalysis(
        evidence_id=evidence_id,
        status=EvidenceAnalysisStatus.QUALIFIED,
        digest="本文提出时间编码器。",
        claims=(EvidenceClaim(
            f"{evidence_id}:C1",
            "本文提出时间编码器。",
            EvidenceRelation.SUPPORTS,
            EvidenceOwnership.THIS_PAPER,
            "原文直接陈述。",
            "We propose a temporal encoder.",
        ),),
    )


def _context():
    plan = ResearchPlan("Method obligation", (ResearchRequirement(
        "R1", "Identify the method", frozenset({EvidenceTarget.METHOD}),
        projection_type=EvidenceProjectionType.ENUMERATION),))
    return projection_semantic_cache_context(build_projection_semantic_request(
        question="方法是什么？", plan=plan, projection_type=EvidenceProjectionType.ENUMERATION,
        requirement_ids=("R1",)))


def test_same_context_hits_cache_and_rebinds_claim_ids(tmp_path) -> None:
    cache = SQLiteEvidenceAnalysisCache(tmp_path / "analysis.sqlite3")
    assert cache.put(question="方法是什么？", record=_record("E1"), analysis=_analysis("E1"), projection_context=_context())
    hit = cache.lookup(question="方法是什么？", record=_record("E9"), projection_context=_context())
    assert hit.analysis is not None
    assert hit.analysis.evidence_id == "E9"
    assert hit.analysis.claims[0].claim_id == "E9:C1"


def test_question_and_prompt_version_change_cache_key(tmp_path) -> None:
    path = tmp_path / "analysis.sqlite3"
    cache = SQLiteEvidenceAnalysisCache(path)
    assert cache.put(question="方法是什么？", record=_record("E1"), analysis=_analysis("E1"), projection_context=_context())
    assert cache.lookup(question="方法有什么限制？", record=_record("E1"), projection_context=_context()).analysis is None
    changed_version = SQLiteEvidenceAnalysisCache(path, prompt_version="new-policy")
    assert changed_version.lookup(question="方法是什么？", record=_record("E1"), projection_context=_context()).analysis is None


def test_corrupted_entry_is_deleted(tmp_path) -> None:
    path = tmp_path / "analysis.sqlite3"
    cache = SQLiteEvidenceAnalysisCache(path)
    assert cache.put(question="方法是什么？", record=_record("E1"), analysis=_analysis("E1"), projection_context=_context())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE evidence_analysis_cache SET payload_json = '{bad-json'")
    miss = cache.lookup(question="方法是什么？", record=_record("E1"), projection_context=_context())
    assert miss.analysis is None and miss.error
    with sqlite3.connect(path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM evidence_analysis_cache").fetchone()[0]
        assert count == 0


def test_cache_stores_projection_once_and_ignores_old_duplicate_support(tmp_path) -> None:
    path = tmp_path / "analysis.sqlite3"
    cache = SQLiteEvidenceAnalysisCache(path)
    record = _record("E1")
    projection = EnumerationProjection(
        "E1",
        (ProjectedEntity("temporal encoder", "method", ProjectedEntityRole.PROPOSED, EvidenceOwnership.THIS_PAPER),),
    )
    assert cache.put(
        question="方法是什么？", record=record,
        analysis=EvidenceAnalysis("E1", EvidenceAnalysisStatus.QUALIFIED, projection=projection),
        projection_context=_context(),
    )
    with sqlite3.connect(path) as connection:
        payload = json.loads(connection.execute("SELECT payload_json FROM evidence_analysis_cache").fetchone()[0])
        assert "covered_requirement_ids" not in payload
        assert "method_entities" not in payload
        assert "claims" not in payload
        payload["covered_requirement_ids"] = ["R1"]
        connection.execute(
            "UPDATE evidence_analysis_cache SET payload_json = ?",
            (json.dumps(payload),),
        )

    restored = cache.lookup(question="方法是什么？", record=record, projection_context=_context()).analysis
    assert restored is not None
    assert restored.projection.supported_requirement_ids == ()
