from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paperbase.llm.answer_output_capture import (
    capture_answer_outputs,
    record_answer_response,
)


@pytest.mark.parametrize("schema_name", ["paperbase_grounded_answer_v2", "paperbase_grounded_coverage_repair_v2"])
def test_current_answer_and_coverage_schema_are_captured(tmp_path, schema_name) -> None:
    path = tmp_path / "answer_outputs.jsonl"
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"requirements":[]}'),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )
    request = {
        "model": "test-model",
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"strict": True},
        },
    }

    with capture_answer_outputs(path, case_id="case_1"):
        record_answer_response(
            schema_name=schema_name,
            request=request,
            response=response,
        )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["case_id"] == "case_1"
    assert row["raw_content"] == '{"requirements":[]}'
    assert row["final_content_present"] is True
