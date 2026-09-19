from __future__ import annotations

import unittest

from paperbase.agentic.evidence_admission import EvidenceAdmissionFilter
from paperbase.retrieval.contracts import (
    EvidenceSourceType,
    PaperSearchOutcome,
    PaperSearchStatus,
    RetrievedEvidence,
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
) -> RetrievedEvidence:
    return RetrievedEvidence(
        source_type=EvidenceSourceType.CONTENT,
        paper_id="A",
        seed_chunk_ids=(chunk_ids[0],),
        source_chunk_ids=chunk_ids,
        content=content,
        score=score,
        retrieval_metadata={"reranking_status": reranking_status},
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
                    paper_id="A",
                    status=PaperSearchStatus.OK,
                    query="query",
                    lexical_terms=(),
                    evidence_targets=targets,
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
        weaker = _evidence(("c1", "c2", "c3"), "weaker window", 0.3)
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


if __name__ == "__main__":
    unittest.main()
