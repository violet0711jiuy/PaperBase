from __future__ import annotations

import json

import pytest

from paperbase.agentic.decision import DecisionError, LLMDecisionAgent
from test_agentic_p0_action_control import _state


class TimeoutClient:
    def __init__(self): self.calls = []
    def complete_json_diagnostic(self, **kwargs):
        self.calls.append(kwargs)
        raise TimeoutError("simulated timeout")


def test_timeout_retry_rebuilds_a_smaller_decision_view() -> None:
    client = TimeoutClient()
    with pytest.raises(DecisionError) as caught:
        LLMDecisionAgent(client, max_attempts=2).decide(_state())
    assert caught.value.llm_calls == 2
    assert len(client.calls) == 2
    first = json.loads(client.calls[0]["user_prompt"])
    second = json.loads(client.calls[1]["user_prompt"])
    assert len(client.calls[1]["user_prompt"]) <= len(client.calls[0]["user_prompt"])
    assert second["runtime_action_capabilities"] == first["runtime_action_capabilities"]
