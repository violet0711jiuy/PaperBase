"""Deterministic tests for the standalone Candidate Golden generator."""

from __future__ import annotations

import unittest

from eval.scripts.generate_candidates import (
    ChunkRecord,
    PaperRecord,
    _expand_evidence,
    _validate_batch,
    allocate_quotas,
    CandidateBatch,
    TARGET_TYPE_COUNTS,
)


class CandidateGenerationTests(unittest.TestCase):
    def test_default_quotas_are_exact_and_balanced_for_four_papers(self) -> None:
        paper_ids = [f"paper_{index}" for index in range(4)]
        quotas = allocate_quotas(paper_ids)
        self.assertEqual(
            {
                kind: sum(quotas[paper_id][kind] for paper_id in paper_ids)
                for kind in TARGET_TYPE_COUNTS
            },
            TARGET_TYPE_COUNTS,
        )
        self.assertEqual([sum(quotas[paper_id].values()) for paper_id in paper_ids], [14] * 4)

    def test_evidence_locations_are_derived_from_sqlite_chunks(self) -> None:
        chunks = [
            _chunk("chunk_1", 1, "Method", 3, 3),
            _chunk("chunk_2", 2, "Method", 4, 4),
            _chunk("chunk_3", 3, "Results", 7, 8),
        ]
        self.assertEqual(
            _expand_evidence(["chunk_2", "chunk_1", "chunk_3"], chunks),
            [
                {
                    "paper_id": "paper_test",
                    "section": "Method",
                    "page_start": 3,
                    "page_end": 4,
                    "chunk_ids": ["chunk_1", "chunk_2"],
                },
                {
                    "paper_id": "paper_test",
                    "section": "Results",
                    "page_start": 7,
                    "page_end": 8,
                    "chunk_ids": ["chunk_3"],
                },
            ],
        )

    def test_unanswerable_contract_is_enforced(self) -> None:
        paper = PaperRecord("paper_test", "Test")
        chunks = [_chunk("chunk_1", 1, "Method", 3, 3)]
        invalid = CandidateBatch.model_validate(
            {
                "candidates": [
                    {
                        "question": "作者是否报告了未给出的能耗？",
                        "primary_type": "unanswerable",
                        "tags": [
                            "zh_query_en_doc",
                            "semantic_paraphrase",
                            "single_hop",
                            "hard",
                        ],
                        "answerable": False,
                        "reference_answer": "没有报告。",
                        "required_facts": [],
                        "evidence_chunk_ids": [],
                        "expected_bibliography_intent": False,
                    }
                ]
            }
        )
        with self.assertRaisesRegex(Exception, "unanswerable draft"):
            _validate_batch(
                invalid,
                paper=paper,
                chunks=chunks,
                visible_ids={"chunk_1"},
                requested_counts={"unanswerable": 1},
                english_question_count=0,
            )


def _chunk(
    chunk_id: str,
    index: int,
    section: str,
    page_start: int,
    page_end: int,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        paper_id="paper_test",
        chunk_index=index,
        section=section,
        section_type="content",
        page_start=page_start,
        page_end=page_end,
        raw_text="Source text",
    )


if __name__ == "__main__":
    unittest.main()
