from __future__ import annotations

import json
import unittest

from paperbase.agentic.candidate_selection import LLMCandidateSelector, PaperMetadata
from paperbase.agentic.contracts import CandidateDisposition, CorpusScopeSnapshot


class _MetadataSource:
    def get_many(self, paper_ids):
        values = {
            "A": PaperMetadata(
                "A",
                "Wind speed forecasting with transformers",
                "Wind forecasting",
            ),
            "B": PaperMetadata("B", "Air pollution prediction", "PM2.5 forecasting"),
            "C": PaperMetadata("C", "Forecasting study"),
        }
        return tuple(values[paper_id] for paper_id in paper_ids)


class _QueuedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(*assessments):
    return json.dumps({"assessments": list(assessments)}, ensure_ascii=False)


class CandidateSelectionTests(unittest.TestCase):
    def test_selector_returns_complete_mutually_exclusive_partition(self) -> None:
        client = _QueuedClient(
            (
                _response(
                    {
                        "paper_id": "A",
                        "disposition": "candidate",
                        "reason": "直接研究风速预测",
                    },
                    {
                        "paper_id": "B",
                        "disposition": "excluded",
                        "reason": "研究空气污染",
                    },
                    {
                        "paper_id": "C",
                        "disposition": "uncertain",
                        "reason": "元数据不足",
                    },
                ),
            )
        )
        result = LLMCandidateSelector(client, _MetadataSource()).select(
            question="目前有哪些风速预测方法？",
            snapshot=CorpusScopeSnapshot.create(("A", "B", "C")),
        )
        self.assertEqual(result.selection.candidate_paper_ids, ("A",))
        self.assertEqual(result.selection.uncertain_paper_ids, ("C",))
        self.assertEqual(result.selection.excluded_paper_ids, ("B",))
        self.assertFalse(result.used_fallback)
        prompt = json.loads(client.calls[0]["user_prompt"])
        self.assertEqual(
            [item["paper_id"] for item in prompt["papers"]], ["A", "B", "C"]
        )

    def test_incomplete_response_retries_then_falls_back_to_uncertain(self) -> None:
        incomplete = _response(
            {"paper_id": "A", "disposition": "candidate", "reason": "相关"}
        )
        result = LLMCandidateSelector(
            _QueuedClient((incomplete, incomplete)),
            _MetadataSource(),
            max_attempts=2,
        ).select(
            question="风速预测",
            snapshot=CorpusScopeSnapshot.create(("A", "B", "C")),
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(
            result.selection.paper_ids_for(CandidateDisposition.UNCERTAIN),
            ("A", "B", "C"),
        )

    def test_duplicate_or_out_of_scope_ids_cannot_enter_selection(self) -> None:
        invalid = _response(
            {"paper_id": "A", "disposition": "candidate", "reason": "相关"},
            {"paper_id": "A", "disposition": "excluded", "reason": "重复"},
            {"paper_id": "X", "disposition": "candidate", "reason": "越界"},
        )
        result = LLMCandidateSelector(
            _QueuedClient((invalid,)), _MetadataSource(), max_attempts=1
        ).select(
            question="风速预测",
            snapshot=CorpusScopeSnapshot.create(("A", "B", "C")),
        )
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.selection.excluded_paper_ids, ())


if __name__ == "__main__":
    unittest.main()
