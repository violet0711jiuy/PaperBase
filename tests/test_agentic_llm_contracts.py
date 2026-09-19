from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from paperbase.agentic.factory import create_agentic_research_service
from paperbase.agentic.schemas import (
    AGENT_DECISION_ADAPTER,
    CandidateSelectionDraft,
    EvidenceAnalysisBatchDraft,
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


class _Client:
    def complete_json(self, **kwargs):
        raise AssertionError("装配测试不应真正调用模型")


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
            EvidenceAnalysisBatchDraft.model_json_schema(),
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
            patch(
                "paperbase.agentic.factory.load_llm_runtime_settings",
                return_value=runtime_settings,
            ) as load_runtime,
            patch(
                "paperbase.agentic.factory.OpenAICompatibleChatClient",
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
        self.assertIs(context.evidence_analyzer._client, shared_client)
        self.assertIs(context.answer_generator._client, shared_client)


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
