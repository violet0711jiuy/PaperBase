from __future__ import annotations

import json
from types import SimpleNamespace

from paperbase.llm.answer_output_capture import (
    capture_answer_outputs,
    record_answer_response,
)


def test_current_requirement_answer_schema_is_captured(tmp_path) -> None:
    path = tmp_path / "answer_outputs.jsonl"
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"answers":[]}'),
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
            schema_name="paperbase_grounded_requirement_answer_v1",
            request=request,
            response=response,
        )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["case_id"] == "case_1"
    assert row["raw_content"] == '{"answers":[]}'
    assert row["final_content_present"] is True
