from __future__ import annotations

from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from paperbase.agentic.factory import AgenticResearchConfigurationError, create_agentic_research_service
from paperbase.config import load_settings
from test_agentic_llm_contracts import ROOT, _BudgetClient, _Client, _Database, _Retriever


def _build(client, *, thinking):
    settings = load_settings(ROOT / "config.example.yaml")
    settings = settings.model_copy(update={"agentic_research": settings.agentic_research.model_copy(update={
        "llm_sampling_profile": "qwen38_recommended",
        "decision_thinking_enabled": thinking,
        "schema_critical_output_mode": "json_schema",
    })})
    with patch("paperbase.agentic.factory.load_settings", return_value=settings):
        return create_agentic_research_service(
            config_path=ROOT / "config.example.yaml", client=client,
            database=_Database(), hybrid_retriever=_Retriever(), checkpointer=InMemorySaver(),
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_qwen_profile_applies_thinking_only_to_decision(thinking) -> None:
    client = _BudgetClient()
    service = _build(client, thinking=thinking)
    assert ("thinking", thinking) in client.calls
    assert ("sampling", {
        "top_p": 0.95 if thinking else 0.8,
        "presence_penalty": 0.0 if thinking else 1.5,
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
    }) in client.calls
    assert service._workflow._context.grounded_answerer._client.thinking is False


def test_qwen_profile_is_not_silently_ignored() -> None:
    with pytest.raises(AgenticResearchConfigurationError, match="cannot apply"):
        _build(_Client(), thinking=False)
