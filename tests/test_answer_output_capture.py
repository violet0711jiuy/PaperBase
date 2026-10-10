"""No-network capture of actual SDK final text, before parsing or stripping."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from paperbase.config import AnswerGenerationSettings
from paperbase.generation.answer_generator import GroundedAnswerGenerator
from paperbase.llm.answer_output_capture import capture_answer_outputs, record_answer_response
from paperbase.llm.client import LLMRequestError, ZhipuAIChatClient
from test_agentic_p1_3c_structured_decision import real_client
from test_answer_semantic_contract import draft, units


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("provider", ["openai", "zhipu"])
@pytest.mark.parametrize("content", ['  {"answer":"中文\\ntext"} \n', r'{"answer":"\alpha"}', None, ""])
def test_provider_records_exact_final_content_before_extraction(provider, content, tmp_path):
    client, calls = real_client([content])
    if provider == "zhipu":
        zhipu = object.__new__(ZhipuAIChatClient)
        zhipu._settings = replace(client._settings, provider="zhipuai")
        zhipu._client = client._client
        zhipu._thinking = {"type": "disabled"}
        client = zhipu
    path = tmp_path / "case.answer_llm_outputs.jsonl"
    with capture_answer_outputs(path, case_id="method_002"):
        if content:
            completion = client.complete_json_diagnostic(system_prompt="secret prompt", user_prompt="source text",
                json_schema={"type": "object"}, schema_name="paperbase_grounded_answer_semantic_v2")
            assert completion.content == content.strip()
        else:
            with pytest.raises(LLMRequestError):
                client.complete_json_diagnostic(system_prompt="secret prompt", user_prompt="source text",
                    json_schema={"type": "object"}, schema_name="paperbase_grounded_answer_semantic_v2")
    saved = rows(path)
    assert len(saved) == len(calls) == 1
    assert saved[0]["raw_content"] == content
    assert saved[0]["case_id"] == "method_002"
    assert saved[0]["response_ordinal"] == 1
    serialized = path.read_text(encoding="utf-8")
    assert "secret prompt" not in serialized and "source text" not in serialized
    assert "reasoning_content" not in serialized and "api_key" not in serialized


@pytest.mark.parametrize("kind", ["invalid_json", "success", "repair"])
def test_answer_validation_and_repair_preserve_every_original_response(kind, tmp_path):
    valid = draft(statuses={"R1": "complete"}, direct="Fact. [E1]", explanation="Fact. [E1]")
    outputs = ([r'{"direct_answer":"\alpha"}'] if kind == "invalid_json" else
        [json.dumps(draft(statuses={"R1":"complete"}, direct="Fact. [E9]", explanation="Fact. [E1]")),
         json.dumps(valid)] if kind == "repair" else [json.dumps(valid)])
    client, calls = real_client(outputs)
    path = tmp_path / "answer.jsonl"
    with capture_answer_outputs(path, case_id="case"):
        outcome = GroundedAnswerGenerator(settings=AnswerGenerationSettings(), client=client).generate(
            query="Fact?", evidence=units(("E1",)), research_requirements=(("R1", "Fact"),),
            support_by_requirement={"R1": ("E1",)})
    assert [r["raw_content"] for r in rows(path)] == outputs
    assert len(calls) == len(outputs)
    assert outcome.status == ("fallback" if kind == "invalid_json" else "success")
    if kind == "invalid_json":
        assert outcome.llm_diagnostics[0].validation_diagnostics[0].error_code == "ANSWER_JSON_INVALID"
    assert not hasattr(outcome.llm_diagnostics[0], "raw_content")


def test_capture_is_case_scoped_filters_non_answer_and_resets(tmp_path):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="final", reasoning_content="hidden secret"), finish_reason="stop")])
    def record(schema="paperbase_grounded_answer"):
        record_answer_response(schema_name=schema, request={"model":"test"}, response=response)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    with capture_answer_outputs(first, case_id="first"):
        record("paperbase_agent_action")
        with capture_answer_outputs(second, case_id="second"):
            record()
        record()
    record()
    assert [r["case_id"] for r in rows(first)] == ["first"]
    assert [r["case_id"] for r in rows(second)] == ["second"]
    assert "hidden secret" not in first.read_text(encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    with capture_answer_outputs(empty, case_id="no_answer"):
        record("paperbase_evidence_projection")
    assert empty.read_text(encoding="utf-8") == ""


def test_concurrent_cases_do_not_mix_outputs(tmp_path):
    def run(case_id):
        path = tmp_path / f"{case_id}.jsonl"
        with capture_answer_outputs(path, case_id=case_id):
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=case_id))])
            record_answer_response(schema_name="paperbase_grounded_answer", request={}, response=response)
        return rows(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("a", "b")))
    assert [r[0]["raw_content"] for r in results] == ["a", "b"]


def test_capture_context_reaches_langgraph_provider_task(tmp_path):
    from langgraph.graph import StateGraph, START, END
    client, calls = real_client(['{"answer":"graph final"}'])
    def answer(state):
        client.complete_json_diagnostic(system_prompt="s", user_prompt="u",
            json_schema={"type":"object"}, schema_name="paperbase_grounded_answer_semantic_v2")
        return {}
    builder = StateGraph(dict)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    path = tmp_path / "graph.jsonl"
    with capture_answer_outputs(path, case_id="graph"):
        builder.compile().invoke({})
    assert rows(path)[0]["raw_content"] == '{"answer":"graph final"}'
    assert len(calls) == 1
