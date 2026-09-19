from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from paperbase.agentic.contracts import EvidenceRecord, EvidenceTarget
from paperbase.agentic.evidence_analysis import LLMEvidenceAnalyzer
from paperbase.agentic.evidence_analysis_cache import SQLiteEvidenceAnalysisCache
from paperbase.retrieval.contracts import EvidenceSourceType, RetrievedEvidence


class _QueuedClient:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, **kwargs) -> str:
        self.calls += 1
        return self.responses.pop(0)


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
        requested_missing_information=("核心机制",),
    )


def _response(evidence_id: str) -> str:
    return json.dumps(
        {
            "analyses": [
                {
                    "evidence_id": evidence_id,
                    "status": "qualified",
                    "digest": "本文提出时间编码器。",
                    "covered_targets": ["method"],
                    "addressed_information": ["核心机制"],
                    "claims": [
                        {
                            "statement": "本文提出时间编码器。",
                            "relation": "supports",
                            "ownership": "this_paper",
                            "rationale": "原文直接陈述。",
                            "source_excerpt": "We propose a temporal encoder.",
                        }
                    ],
                }
            ]
        }
    )


class EvidenceAnalysisCacheTests(unittest.TestCase):
    def test_same_context_hits_cache_without_llm_and_rebinds_claim_ids(self) -> None:
        with TemporaryDirectory() as directory:
            cache = SQLiteEvidenceAnalysisCache(Path(directory) / "analysis.sqlite3")
            client = _QueuedClient((_response("E1"),))
            analyzer = LLMEvidenceAnalyzer(client, cache=cache, max_attempts=1)

            first = analyzer.analyze(question="方法是什么？", evidence=(_record("E1"),))
            second = analyzer.analyze(question="方法是什么？", evidence=(_record("E9"),))

            self.assertEqual(first.llm_calls, 1)
            self.assertEqual(first.cache_stats.misses, 1)
            self.assertEqual(first.cache_stats.writes, 1)
            self.assertEqual(second.llm_calls, 0)
            self.assertEqual(second.cache_stats.hits, 1)
            self.assertEqual(second.analyses[0].evidence_id, "E9")
            self.assertEqual(second.analyses[0].claims[0].claim_id, "E9:C1")
            self.assertEqual(client.calls, 1)

    def test_question_and_prompt_version_changes_are_cache_misses(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.sqlite3"
            cache = SQLiteEvidenceAnalysisCache(path)
            first_client = _QueuedClient((_response("E1"), _response("E1")))
            analyzer = LLMEvidenceAnalyzer(first_client, cache=cache, max_attempts=1)
            analyzer.analyze(question="方法是什么？", evidence=(_record("E1"),))

            changed_question = analyzer.analyze(
                question="方法有什么限制？", evidence=(_record("E1"),)
            )
            changed_version = LLMEvidenceAnalyzer(
                _QueuedClient((_response("E1"),)),
                cache=SQLiteEvidenceAnalysisCache(path, prompt_version="new-policy"),
                max_attempts=1,
            ).analyze(question="方法是什么？", evidence=(_record("E1"),))

            self.assertEqual(changed_question.cache_stats.misses, 1)
            self.assertEqual(changed_question.llm_calls, 1)
            self.assertEqual(changed_version.cache_stats.misses, 1)
            self.assertEqual(changed_version.llm_calls, 1)

    def test_corrupted_entry_is_deleted_and_falls_back_to_llm(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.sqlite3"
            cache = SQLiteEvidenceAnalysisCache(path)
            LLMEvidenceAnalyzer(
                _QueuedClient((_response("E1"),)), cache=cache, max_attempts=1
            ).analyze(question="方法是什么？", evidence=(_record("E1"),))
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "UPDATE evidence_analysis_cache SET payload_json = '{bad-json'"
                )
                connection.commit()
            finally:
                connection.close()

            client = _QueuedClient((_response("E2"),))
            result = LLMEvidenceAnalyzer(
                client, cache=cache, max_attempts=1
            ).analyze(question="方法是什么？", evidence=(_record("E2"),))

            self.assertEqual(result.llm_calls, 1)
            self.assertEqual(result.cache_stats.misses, 1)
            self.assertEqual(result.cache_stats.errors, 1)
            self.assertEqual(result.cache_stats.writes, 1)


if __name__ == "__main__":
    unittest.main()
