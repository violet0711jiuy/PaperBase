"""Case-scoped local capture of final Answer text before extraction/validation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock


_active_capture: ContextVar[object | None] = ContextVar("answer_output_capture", default=None)
_ANSWER_SCHEMAS = {"paperbase_grounded_answer", "paperbase_grounded_answer_semantic_v2",
                   "paperbase_grounded_answer_semantic_v3",
                   "paperbase_grounded_requirement_answer_v1"}


class _Capture:
    def __init__(self, path: Path, case_id: str):
        self.path = path
        self.case_id = case_id
        self.lock = Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        self.ordinal = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())

    def record(self, *, schema_name, request, response):
        choices = getattr(response, "choices", None)
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        usage = getattr(response, "usage", None)
        response_format = request.get("response_format", {})
        # Only final content and allowlisted metadata. Never serialize SDK
        # objects, request messages, credentials, or hidden reasoning fields.
        record = dict(
            capture_version="answer-final-output-v1", case_id=self.case_id,
            captured_at=datetime.now(timezone.utc).isoformat(), schema_name=schema_name,
            model=request.get("model"), temperature=request.get("temperature"),
            response_mode=response_format.get("type"),
            strict=response_format.get("json_schema", {}).get("strict"),
            requested_max_tokens=request.get("max_tokens"),
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            raw_content=content if isinstance(content, str) else None,
            content_chars=len(content) if isinstance(content, str) else 0,
            final_content_present=isinstance(content, str),
        )
        with self.lock:
            self.ordinal += 1
            record["response_ordinal"] = self.ordinal
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def capture_answer_outputs(path: Path, *, case_id: str):
    """All Answer responses in this case, including invalid/empty/repair output.

    An empty file means no Answer response was returned, not a missing capture.
    ContextVar keeps concurrent cases isolated and propagates with Graph tasks.
    """
    token = _active_capture.set(_Capture(path, case_id))
    try:
        yield
    finally:
        _active_capture.reset(token)


def record_answer_response(*, schema_name: str | None, request: dict, response):
    capture = _active_capture.get()
    if capture is not None and schema_name in _ANSWER_SCHEMAS:
        capture.record(schema_name=schema_name, request=request, response=response)
