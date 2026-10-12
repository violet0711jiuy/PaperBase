from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from paperbase.agentic.factory import create_agentic_research_service
from paperbase.agentic.schemas import (
    AGENT_DECISION_ADAPTER,
    CandidateSelectionDraft,
    GroundedAnswerDraft,
    QueryUnderstandingDraft,
    ResearchPlanDraft,
)
from paperbase.config import load_settings
from paperbase.prompts.agentic_decision import AGENT_DECISION_SYSTEM_PROMPT
from paperbase.prompts.grounded_answer import GROUNDED_ANSWER_SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]


def _configured_settings():
    settings = load_settings(ROOT / "config.example.yaml")
    agentic = settings.agentic_research.model_copy(update={
        "llm_sampling_profile": "configured",
        "decision_thinking_enabled": False,
        "decision_temperature": 0,
        "planner_temperature": 0,
        "answer_temperature": 0,
        "schema_critical_output_mode": "json_object",
    })
    return settings.model_copy(update={"agentic_research": agentic})


class _Client:
    def complete_json(self, **kwargs):
        raise AssertionError("factory contract test must not call the model")


class _BudgetClient(_Client):
    def __init__(self, *, calls=None, timeout=None, thinking=None, temperature=None):
        self.calls = calls if calls is not None else []
        self.timeout = timeout
        self.thinking = thinking
        self.temperature = temperature

    def _view(self, **changes):
        values = dict(calls=self.calls, timeout=self.timeout, thinking=self.thinking,
                      temperature=self.temperature)
        values.update(changes)
        return _BudgetClient(**values)

    def with_max_tokens(self, value):
        self.calls.append(("max_tokens", value)); return self._view()

    def with_timeout_seconds(self, value):
        self.calls.append(("timeout", value)); return self._view(timeout=value)

    def with_transport_max_retries(self, value):
        self.calls.append(("retries", value)); return self._view()

    def with_thinking_disabled(self):
        self.calls.append(("thinking", False)); return self._view(thinking=False)

    def with_thinking_enabled(self, value):
        self.calls.append(("thinking", value)); return self._view(thinking=value)

    def with_temperature(self, value):
        self.calls.append(("temperature", value)); return self._view(temperature=value)

    def with_structured_output_mode(self, value):
        self.calls.append(("mode", value)); return self._view()

    def with_schema_prompt_injection(self, value):
        self.calls.append(("schema_prompt", value)); return self._view()

    def with_sampling_parameters(self, **kwargs):
        self.calls.append(("sampling", kwargs)); return self._view()


class _Database:
    def __init__(self): self.initialized = False
    def initialize(self): self.initialized = True
    def list_documents(self): return ()
    def get_document(self, paper_id): return None


class _Retriever:
    pass


def _assert_all_properties_required(schema: dict) -> None:
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert set(node["properties"]) == set(node.get("required", ()))
            for value in node.values(): walk(value)
        elif isinstance(node, list):
            for value in node: walk(value)
    walk(schema)


def test_current_structured_schemas_are_strict_and_complete() -> None:
    for schema in (
        QueryUnderstandingDraft.model_json_schema(), ResearchPlanDraft.model_json_schema(),
        AGENT_DECISION_ADAPTER.json_schema(), CandidateSelectionDraft.model_json_schema(),
        GroundedAnswerDraft.model_json_schema(),
    ):
        _assert_all_properties_required(schema)
    with __import__("pytest").raises(ValidationError):
        AGENT_DECISION_ADAPTER.validate_json(json.dumps({
            "decision": {"action": "try_answer", "reason": "ready", "top_k": 100}
        }))


def test_prompts_expose_try_answer_and_one_grounded_semantic_authority() -> None:
    assert "try_answer" in AGENT_DECISION_SYSTEM_PROMPT
    assert "stop_research" in AGENT_DECISION_SYSTEM_PROMPT
    assert "finish" not in AGENT_DECISION_SYSTEM_PROMPT.casefold()
    assert "Evidence.content" in GROUNDED_ANSWER_SYSTEM_PROMPT
    assert "source_span_ids" not in GROUNDED_ANSWER_SYSTEM_PROMPT
    assert "Projector" not in GROUNDED_ANSWER_SYSTEM_PROMPT


def test_factory_wires_grounded_answerer_and_disables_answer_thinking() -> None:
    client = _BudgetClient()
    with patch("paperbase.agentic.factory.load_settings", return_value=_configured_settings()):
        service = create_agentic_research_service(
            config_path=ROOT / "config.example.yaml", client=client,
            database=_Database(), hybrid_retriever=_Retriever(), checkpointer=InMemorySaver(),
        )
    context = service._workflow._context
    assert context.grounded_answerer is not None
    assert not hasattr(context, "evidence_analyzer")
    assert not hasattr(context, "answer_generator")
    assert ("max_tokens", None) in client.calls
    assert ("thinking", False) in client.calls
