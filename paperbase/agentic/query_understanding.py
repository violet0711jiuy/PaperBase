"""在已解析 Scope 上理解研究意图；失败时使用确定性保守降级。"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from paperbase.llm.client import (
    ChatCompletionClient,
    LLMCompletion,
    LLMRequestError,
    complete_json_with_diagnostics,
)
from paperbase.prompts.agentic_query_understanding import (
    QUERY_UNDERSTANDING_SYSTEM_PROMPT,
    build_query_understanding_user_prompt,
)

from .contracts import (
    EvidenceTarget,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ScopeResolution,
    ScopeResolutionStatus,
    LLMCallRecord,
)
from paperbase.llm.diagnostics import failure_record, success_record
from .schemas import QueryUnderstandingDraft


@dataclass(frozen=True)
class QueryUnderstandingResult:
    context: QueryContext
    used_fallback: bool
    llm_calls: int
    llm_diagnostics: tuple[LLMCallRecord, ...] = ()


class QueryUnderstandingService:
    def __init__(self, client: ChatCompletionClient | None = None) -> None:
        self._client = client

    def understand(
        self, query: str, resolution: ScopeResolution
    ) -> QueryUnderstandingResult:
        if resolution.status is not ScopeResolutionStatus.RESOLVED:
            raise ValueError("Query understanding requires a resolved scope.")
        fallback = _deterministic_draft(query, resolution.scope)
        draft = fallback
        used_fallback = self._client is None
        llm_calls = 0
        diagnostics: list[LLMCallRecord] = []
        if self._client is not None:
            llm_calls = 1
            completion: LLMCompletion | None = None
            try:
                completion = complete_json_with_diagnostics(
                    self._client,
                    system_prompt=QUERY_UNDERSTANDING_SYSTEM_PROMPT,
                    user_prompt=build_query_understanding_user_prompt(
                        query=query,
                        scope=resolution.scope.value,
                        trusted_paper_ids=resolution.target_paper_ids,
                    ),
                    json_schema=QueryUnderstandingDraft.model_json_schema(),
                    schema_name="paperbase_query_understanding",
                )
                draft = QueryUnderstandingDraft.model_validate_json(completion.content)
                diagnostics.append(
                    success_record(
                        node="understand_query", attempt=1, completion=completion
                    )
                )
                used_fallback = False
            except (LLMRequestError, ValidationError, ValueError, TypeError) as error:
                diagnostics.append(
                    failure_record(
                        node="understand_query",
                        attempt=1,
                        error=error,
                        completion=completion,
                        can_retry=False,
                    )
                )
                draft = fallback
                used_fallback = True
        # The validated draft owns the advisory hints. Do not apply the Search
        # compiler's independent quota or inject original-language query terms.
        lexical = tuple(dict.fromkeys(term.strip() for term in draft.lexical_keywords_en if term.strip()))
        targets = frozenset(draft.evidence_targets)
        context = QueryContext(
            original_query=query,
            resolved_query=draft.resolved_query,
            semantic_query_en=(draft.semantic_query_en or None),
            lexical_keywords_en=lexical,
            intent=draft.intent,
            scope=resolution.scope,
            target_paper_ids=resolution.target_paper_ids,
            scope_source=resolution.scope_source,
            retrieval_goal=draft.retrieval_goal,
            default_evidence_targets=targets,
        )
        return QueryUnderstandingResult(
            context, used_fallback, llm_calls, tuple(diagnostics)
        )


def _deterministic_draft(
    query: str, scope: QueryScope | None
) -> QueryUnderstandingDraft:
    folded = query.casefold()
    compare = scope is QueryScope.SELECTED_PAPERS or any(
        term in folded for term in ("区别", "比较", "差异", "compare", "versus", " vs ")
    )
    exhaustive = any(
        term in folded for term in ("所有", "全部", "哪些", "逐一", "all ", "every", "which papers")
    )
    bibliography = any(
        term in folded for term in ("参考文献", "引用", "cite", "citation", "reference")
    )
    if compare:
        intent = Intent.COMPARE
        goal = ResearchGoal.COMPARISON_SUPPORT
    elif scope is QueryScope.KNOWLEDGE_BASE:
        intent = Intent.SYNTHESIZE
        goal = ResearchGoal.EXHAUSTIVE_ENUMERATION if exhaustive else ResearchGoal.SYNTHESIS_SUPPORT
    elif any(term in folded for term in ("为什么", "如何", "解释", "why", "how")):
        intent = Intent.EXPLAIN
        goal = ResearchGoal.FIND_EVIDENCE
    else:
        intent = Intent.RETRIEVE
        goal = ResearchGoal.EXHAUSTIVE_ENUMERATION if exhaustive else ResearchGoal.FIND_EVIDENCE
    if bibliography:
        targets = [EvidenceTarget.REFERENCE]
    elif any(term in folded for term in ("方法", "模型", "架构", "method", "architecture")):
        targets = [EvidenceTarget.METHOD]
    elif any(term in folded for term in ("结果", "性能", "实验", "result", "performance")):
        targets = [EvidenceTarget.RESULT]
    else:
        targets = [EvidenceTarget.GENERAL]
    return QueryUnderstandingDraft(
        resolved_query=query,
        semantic_query_en=None,
        lexical_keywords_en=[],
        intent=intent,
        retrieval_goal=goal,
        evidence_targets=targets,
    )
