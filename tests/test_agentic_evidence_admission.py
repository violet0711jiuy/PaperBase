from __future__ import annotations

import unittest
from dataclasses import replace

from paperbase.agentic.evidence_admission import EvidenceAdmissionFilter
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchOutcome,
    PaperSearchSpec,
    PaperSearchStatus,
    RetrievedEvidence,
    ScoreKind,
    ScoreDirection,
    SearchPapersPayload,
    ToolName,
    ToolResult,
    ToolStatus,
)


def _evidence(
    chunk_ids: tuple[str, ...],
    content: str,
    score: float,
    *,
    reranking_status: str = "success",
    rank: int = 1,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        source_type=EvidenceSourceType.CONTENT,
        paper_id="A",
        seed_chunk_ids=(chunk_ids[0],),
        source_chunk_ids=chunk_ids,
        content=content,
        raw_score=score,
        score_kind=ScoreKind.RERANKER if reranking_status == "success" else ScoreKind.RRF,
        score_direction=ScoreDirection.HIGHER_IS_BETTER,
        retrieval_rank=rank,
        retrieval_metadata={"reranking_status": reranking_status,
                            "rerank_score_normalized": reranking_status == "success"},
    )


def _result(
    evidence: tuple[RetrievedEvidence, ...], *, targets: frozenset[str] = frozenset()
) -> ToolResult:
    return ToolResult(
        status=ToolStatus.OK,
        tool_name=ToolName.SEARCH_PAPERS,
        payload=SearchPapersPayload(
            (
                PaperSearchOutcome(
                    search=PaperSearchSpec(paper_id="A", query="query", evidence_targets=targets),
                    status=PaperSearchStatus.OK,
                    evidence_candidates=evidence,
                ),
            )
        ),
    )


class EvidenceAdmissionTests(unittest.TestCase):
    def test_low_score_is_rejected_only_after_successful_reranking(self) -> None:
        low = _evidence(("c1",), "low relevance prose", 0.01)
        fallback = _evidence(
            ("c2",), "fallback score is not calibrated", 0.01,
            reranking_status="fallback",
        )

        admission = EvidenceAdmissionFilter(min_rerank_score=0.05).apply(
            _result((low, fallback))
        )

        kept = admission.tool_result.payload.paper_results[0].evidence_candidates
        self.assertEqual(tuple(item.source_chunk_ids for item in kept), (("c2",),))
        self.assertEqual(admission.report.rejected_candidates, 1)
        self.assertEqual(
            admission.report.decisions[0].reasons,
            ("below_rerank_threshold",),
        )

    def test_overlapping_windows_keep_stronger_candidate(self) -> None:
        weaker = _evidence(("c1", "c2", "c3"), "weaker window", 0.3, rank=2)
        stronger = _evidence(("c1", "c2", "c3", "c4"), "stronger window", 0.8)

        admission = EvidenceAdmissionFilter(
            source_overlap_threshold=0.7
        ).apply(_result((weaker, stronger)))

        kept = admission.tool_result.payload.paper_results[0].evidence_candidates
        self.assertEqual(kept, (stronger,))
        reasons = {
            item.source_chunk_ids: item.reasons
            for item in admission.report.decisions
        }
        self.assertEqual(reasons[weaker.source_chunk_ids], ("source_overlap",))

    def test_result_target_rescues_best_table_when_all_scores_are_low(self) -> None:
        table = _evidence(
            ("table-1",),
            "Model, PICP=0.95, PINRW=0.41, MSE=5.8",
            0.01,
        )

        admission = EvidenceAdmissionFilter(min_rerank_score=0.05).apply(
            _result((table,), targets=frozenset({"result"}))
        )

        kept = admission.tool_result.payload.paper_results[0].evidence_candidates
        self.assertEqual(kept, (table,))
        self.assertEqual(
            admission.report.decisions[0].reasons,
            ("result_table_rescued",),
        )

    def test_overlap_does_not_discard_lower_ranked_candidate_with_new_chunks(self) -> None:
        stronger = _evidence(("c1", "c2", "c3"), "overview", 0.8)
        expanded = _evidence(("c1", "c2", "c3", "c4"), "overview plus mechanism", 0.3, rank=2)
        shifted = _evidence(("c2", "c3", "c4"), "additional mechanism", 0.2, rank=3)
        admission = EvidenceAdmissionFilter(source_overlap_threshold=0.5).apply(
            _result((stronger, expanded, shifted))
        )
        kept = admission.tool_result.payload.paper_results[0].evidence_candidates
        self.assertEqual(kept, (stronger, expanded))
        self.assertEqual(
            set().union(*(set(item.source_chunk_ids) for item in kept)),
            {"c1", "c2", "c3", "c4"},
        )

    def test_table_for_non_result_target_is_kept_but_deprioritized(self) -> None:
        table = _evidence(
            ("table-1",),
            "Model, PICP=0.95, PINRW=0.41, MSE=5.8",
            0.8,
        )

        admission = EvidenceAdmissionFilter().apply(
            _result((table,), targets=frozenset({"method"}))
        )

        self.assertEqual(admission.report.accepted_candidates, 1)
        self.assertEqual(admission.report.deprioritized_candidates, 1)
        self.assertEqual(admission.report.decisions[0].decision, "deprioritized")

    def test_negative_bm25_is_not_a_low_rerank_score_and_best_rank_survives_overlap(self):
        # Even a historical/global reranking_status cannot reinterpret BM25.
        best = replace(_evidence(("r1", "r2", "r3", "r4"), "best reference", -12.0),
                       source_type=EvidenceSourceType.BIBLIOGRAPHY,
                       score_kind=ScoreKind.BM25_FTS5, score_direction=ScoreDirection.LOWER_IS_BETTER)
        worse = replace(best, content="worse reference", raw_score=-0.1,
                        source_chunk_ids=("r1", "r2", "r3"), retrieval_rank=2)
        result = EvidenceAdmissionFilter(source_overlap_threshold=0.7).apply(_result((worse, best)))
        self.assertEqual(result.tool_result.payload.paper_results[0].evidence_candidates, (best,))
        self.assertEqual(result.report.accepted_candidates, 1)
        self.assertFalse(any('below_rerank_threshold' in d.reasons for d in result.report.decisions))
        decision = result.report.decisions[1]
        self.assertEqual((decision.raw_score, decision.score_kind, decision.score_direction, decision.retrieval_rank),
                         (-12.0, ScoreKind.BM25_FTS5, ScoreDirection.LOWER_IS_BETTER, 1))

    def test_unscaled_or_unknown_reranker_does_not_use_sigmoid_threshold(self):
        for normalized in (False, None):
            with self.subTest(normalized=normalized):
                logits = replace(_evidence(("c1",), "unscaled relevance", -4.0),
                                 retrieval_metadata={"reranking_status": "success", "rerank_score_normalized": normalized})
                result = EvidenceAdmissionFilter().apply(_result((logits,)))
                self.assertEqual(result.report.accepted_candidates, 1)

    def test_legacy_unknown_score_is_not_guessed_from_status_or_numeric_range(self):
        legacy = RetrievedEvidence(EvidenceSourceType.CONTENT, 'A', ('c1',), ('c1',), 'legacy prose',
            score=0.001, retrieval_metadata={'reranking_status':'success'})
        result = EvidenceAdmissionFilter().apply(_result((legacy,)))
        self.assertEqual(result.report.accepted_candidates, 1)
        self.assertEqual(legacy.score_kind, ScoreKind.LEGACY_UNKNOWN)

    def test_table_rescue_uses_rank_even_when_raw_values_disagree(self):
        first = _evidence(('t1',), 'PICP=0.9 MSE=0.2', 0.001, rank=1)
        second = _evidence(('t2',), 'PICP=0.8 MSE=0.3', 0.04, rank=2)
        result = EvidenceAdmissionFilter().apply(_result((second,first), targets=frozenset({'result'})))
        self.assertEqual(result.tool_result.payload.paper_results[0].evidence_candidates, (first,))


if __name__ == "__main__":
    unittest.main()
