"""Compile chosen Actions; Requirement bindings remain telemetry, not filters."""
from __future__ import annotations

from collections.abc import Mapping
from paperbase.retrieval.requests import (
    CanonicalSearchRequest, compile_paper_request, compile_reference_request, compile_kb_request,
)
from .contracts import (AgentAction, QueryContext, CorpusScopeSnapshot, ScopeSource,
                        SearchPapersAction, SearchReferencesAction, SearchKnowledgeBaseAction)


def compile_action_requests(action: AgentAction, *, context: QueryContext | None = None,
                            snapshot: CorpusScopeSnapshot | None = None, top_k: int = 5,
                            workspace_ids: Mapping[str, str] | None = None,
                            parameters: Mapping[str, dict[str, object]] | None = None,
                            kb_parameters: dict[str, object] | None = None) -> tuple[CanonicalSearchRequest, ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if isinstance(action, SearchPapersAction):
        source = ("workspace" if context and context.scope_source is ScopeSource.TRUSTED_WORKSPACE
                  else "formal_knowledge_base")
        return tuple(compile_paper_request(paper_id=search.paper_id, query=search.query,
                    lexical_terms=search.lexical_terms, top_k=top_k, source=source,
                    workspace_id=(workspace_ids or {}).get(search.paper_id),
                    retrieval_parameters=(parameters or {}).get(search.paper_id))
                    for search in action.searches)
    if isinstance(action, SearchReferencesAction):
        return (compile_reference_request(paper_id=action.paper_id, query=action.query,
                lexical_terms=action.lexical_terms, top_k=top_k),)
    if isinstance(action, SearchKnowledgeBaseAction):
        return (compile_kb_request(paper_ids=snapshot.paper_ids if snapshot else (),
                query=action.query, lexical_terms=action.lexical_terms, top_k=top_k,
                retrieval_parameters=kb_parameters),)
    return ()


def canonical_action_identity(requests):
    import hashlib, json
    value = json.dumps(sorted(request.identity for request in requests), separators=(",", ":"))
    return "search-action-v1:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
