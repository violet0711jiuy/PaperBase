from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch
from paperbase.config import load_settings

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from paperbase.agentic.factory import create_agentic_research_service
from paperbase.agentic.schemas import (
    AGENT_DECISION_ADAPTER,
    CandidateSelectionDraft,
    EnumerationProjectionBatchDraft,
    FactProjectionBatchDraft,
    QueryUnderstandingDraft,
    ResearchPlanDraft,
)
from paperbase.prompts.agentic_decision import (
    AGENT_DECISION_SYSTEM_PROMPT,
    build_agent_decision_user_prompt,
)
from paperbase.prompts.agentic_planner import (
    RESEARCH_PLANNER_SYSTEM_PROMPT,
    build_research_planner_user_prompt,
)
from paperbase.prompts.agentic_query_understanding import (
    QUERY_UNDERSTANDING_SYSTEM_PROMPT,
    build_query_understanding_user_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def _configured_settings():
    """Keep lightweight-client compatibility tests independent of the Qwen template."""
    settings = load_settings(ROOT / "config.example.yaml")
    return settings.model_copy(update={"agentic_research": settings.agentic_research.model_copy(
        update={"llm_sampling_profile": "configured", "decision_thinking_enabled": False,
                "decision_temperature": 0, "planner_temperature": 0,
                "projector_temperature": 0, "answer_temperature": 0,
                "schema_critical_output_mode": "json_object"})})


class _Client:
    def complete_json(self, **kwargs):
        raise AssertionError("装配测试不应真正调用模型")


class _BudgetClient(_Client):
    def __init__(self, budget: int | None = None, calls=None, timeout=None, thinking_disabled=False) -> None:
        self.budget = budget
        self.calls = calls if calls is not None else []
        self.timeout = timeout
        self.thinking_disabled = thinking_disabled

    def with_max_tokens(self, max_tokens: int | None):
        self.calls.append(max_tokens)
        return _BudgetClient(max_tokens, self.calls, self.timeout, self.thinking_disabled)

    def with_timeout_seconds(self, timeout_seconds: float):
        self.calls.append(("timeout", timeout_seconds))
        return _BudgetClient(self.budget, self.calls, timeout_seconds, self.thinking_disabled)

    def with_thinking_disabled(self):
        self.calls.append(("enable_thinking", False))
        return _BudgetClient(self.budget, self.calls, self.timeout, True)


class _Database:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def list_documents(self):
        return ()

    def get_document(self, paper_id):
        return None


class _Retriever:
    pass


class AgenticLLMContractTests(unittest.TestCase):
    def test_prompts_are_dedicated_and_serialize_untrusted_text(self) -> None:
        understanding = json.loads(
            build_query_understanding_user_prompt(
                query="A 与 B 的方法区别？",
                scope="selected_papers",
                trusted_paper_ids=("A", "B"),
            )
        )
        planner = json.loads(
            build_research_planner_user_prompt(
                question="compare",
                intent="compare",
                scope="selected_papers",
                trusted_paper_ids=("A", "B"),
                evidence_targets=("method",),
            )
        )
        decision = json.loads(build_agent_decision_user_prompt({"question": "compare"}))
        self.assertEqual(understanding["trusted_paper_ids"], ["A", "B"])
        self.assertEqual(planner["default_evidence_targets"], ["method"])
        self.assertEqual(decision, {"question": "compare"})
        self.assertIn("绝不能增加、删除", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("不要选择 Tool", RESEARCH_PLANNER_SYSTEM_PROMPT)
        self.assertIn("Requirement 只能来自用户问题", RESEARCH_PLANNER_SYSTEM_PROMPT)
        self.assertIn("不要提前枚举用户没有给出的具体答案", RESEARCH_PLANNER_SYSTEM_PROMPT)
        for action_name in (
            "search_papers",
            "search_knowledge_base",
            "search_references",
            "get_section_context",
            "finish",
        ):
            self.assertIn(action_name, AGENT_DECISION_SYSTEM_PROMPT)

    def test_structured_output_schemas_require_all_object_fields(self) -> None:
        for schema in (
            QueryUnderstandingDraft.model_json_schema(),
            ResearchPlanDraft.model_json_schema(),
            AGENT_DECISION_ADAPTER.json_schema(),
            CandidateSelectionDraft.model_json_schema(),
            EnumerationProjectionBatchDraft.model_json_schema(),
            FactProjectionBatchDraft.model_json_schema(),
        ):
            _assert_all_properties_required(self, schema)
        decision_schema = AGENT_DECISION_ADAPTER.json_schema()
        self.assertEqual(decision_schema["type"], "object")
        self.assertEqual(decision_schema["required"], ["decision"])

    def test_local_schema_validation_rejects_missing_and_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            QueryUnderstandingDraft.model_validate_json(
                json.dumps(
                    {
                        "resolved_query": "question",
                        "semantic_query_en": None,
                        # 故意缺少 lexical_keywords_en，验证必填字段检查。
                        "intent": "retrieve",
                        "retrieval_goal": "find_evidence",
                        "evidence_targets": ["general"],
                    }
                )
            )
        with self.assertRaises(ValidationError):
            AGENT_DECISION_ADAPTER.validate_json(
                json.dumps(
                    {
                        "decision": {
                            "action": "finish",
                            "reason": "done",
                            "top_k": 100,
                        }
                    }
                )
            )

    def test_production_factory_shares_one_real_client_across_llm_stages(self) -> None:
        database = _Database()
        shared_client = _Client()
        runtime_settings = object()
        with (
            patch("paperbase.agentic.factory.load_settings", return_value=_configured_settings()),
            patch(
                "paperbase.agentic.factory.load_llm_runtime_settings",
                return_value=runtime_settings,
            ) as load_runtime,
            patch(
                "paperbase.agentic.factory.create_chat_completion_client",
                return_value=shared_client,
            ) as client_factory,
        ):
            service = create_agentic_research_service(
                config_path=ROOT / "config.example.yaml",
                database=database,
                hybrid_retriever=_Retriever(),
            )

        self.assertTrue(database.initialized)
        load_runtime.assert_called_once_with(ROOT / ".env")
        client_factory.assert_called_once_with(runtime_settings)
        workflow = service._workflow
        self.assertIsNotNone(workflow)
        context = workflow._context
        self.assertIs(context.query_understanding._client, shared_client)
        self.assertIs(context.planner._client, shared_client)
        self.assertIs(context.decision_agent._client, shared_client)
        self.assertIs(context.candidate_selector._client, shared_client)
        self.assertIs(context.evidence_analyzer._client, shared_client)
        self.assertIs(context.answer_generator._client, shared_client)

    def test_production_factory_uses_uncapped_views_for_all_agentic_nodes(self) -> None:
        client = _BudgetClient(500)
        with patch("paperbase.agentic.factory.load_settings", return_value=_configured_settings()):
            service = create_agentic_research_service(
                config_path=ROOT / "config.example.yaml",
                client=client,
                database=_Database(),
                hybrid_retriever=_Retriever(),
                checkpointer=InMemorySaver(),
            )

        context = service._workflow._context
        clients = (
            context.query_understanding._client,
            context.planner._client,
            context.candidate_selector._client,
            context.decision_agent._client,
            context.evidence_analyzer._client,
            context.answer_generator._client,
        )
        self.assertIs(clients[2], clients[0])
        self.assertIsNot(clients[3], clients[0])
        self.assertIsNot(clients[1], clients[0])
        self.assertIsNone(clients[1].budget)
        self.assertIsNot(clients[4], clients[0])
        self.assertIsNot(clients[5], clients[0])
        self.assertTrue(all(view.budget is None for view in clients))
        self.assertEqual(client.budget, 500)
        self.assertIsNone(clients[4].budget)
        self.assertEqual(clients[3].timeout, 60.0)
        self.assertEqual(clients[4].timeout, 60.0)
        self.assertIsNone(clients[5].budget)
        self.assertEqual(clients[5].timeout, 60.0)
        self.assertTrue(clients[5].thinking_disabled)
        self.assertEqual(
            client.calls,
            [
                None,
                ("timeout", 60.0),
                None,
                None,
                ("timeout", 60.0),
                None,
                ("enable_thinking", False),
                ("timeout", 60.0),
                None,
                None,
                None,
            ],
        )
        self.assertEqual(context.evidence_analyzer._batch_size, 4)
        self.assertIsNone(context.evidence_analyzer.output_budget(4))
        self.assertIsNone(context.evidence_analyzer.output_budget(32))
        self.assertEqual(context.evidence_analyzer._max_attempts, 2)


def _assert_all_properties_required(
    case: unittest.TestCase, schema: dict[str, object]
) -> None:
    candidates = [schema, *schema.get("$defs", {}).values()]
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("type") != "object":
            continue
        properties = candidate.get("properties", {})
        case.assertEqual(
            set(candidate.get("required", [])),
            set(properties),
            msg=f"存在未进入 required 的属性：{candidate.get('title', 'root')}",
        )
        case.assertFalse(candidate.get("additionalProperties", True))


if __name__ == "__main__":
    unittest.main()
