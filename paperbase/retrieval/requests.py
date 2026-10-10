"""The sole compiler of Agent-supplied effective search inputs.

Lexical hints are independent OR tokens, never implicit exact phrases. Dense
queries preserve case/punctuation (embedders need not treat them as equivalent).
No Query Understanding hints are implicitly added here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Literal, Sequence, TypeAlias

from .lexical_terms import LEXICAL_STOPWORDS, extract_lexical_terms


def normalize_semantic_query(query: str) -> str:
    value = " ".join(query.split())
    if not value:
        raise ValueError("semantic_query must not be empty")
    return value


def compile_lexical_terms(query: str, hints: Sequence[str], *, max_fallback_hints: int = 5) -> tuple[str, ...]:
    # Explicit nonempty hints replace fallback; even all-stopword hints are an
    # intentional empty lexical route, not permission to inject query terms.
    inputs = hints if any(value.strip() for value in hints) else extract_lexical_terms(query, max_terms=max_fallback_hints)
    tokens = set()
    for value in inputs:
        value = unicodedata.normalize("NFKC", value).casefold()
        tokens.update(token for token in re.findall(r"[^\W_]+", value)
                      if token not in LEXICAL_STOPWORDS)
    return tuple(sorted(tokens))


def compile_fts_query(terms: Sequence[str]) -> str:
    # Tokens contain only letters/numbers. Quoting each SINGLE token protects
    # reserved operators; it does not impose a multiword phrase constraint.
    return " OR ".join('"' + term + '"' for term in terms)


class _RequestIdentity:
    def __post_init__(self):
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if any(not paper_id or paper_id != paper_id.strip() for paper_id in self.paper_ids):
            raise ValueError("paper IDs must be explicit canonical IDs")
        expected = (" ".join(self.lexical_terms) if getattr(self, "source", None) == "workspace"
                    else compile_fts_query(self.lexical_terms))
        if self.lexical_query != expected:
            raise ValueError("lexical_query does not match compiler tokens")
        if tuple(sorted(set(self.lexical_terms))) != self.lexical_terms:
            raise ValueError("lexical terms must be canonical unique sorted tokens")
        if getattr(self, "scope", None) == "all" and self.paper_ids:
            raise ValueError("all-scope request must not carry a paper filter")
        if self.kind != "search_knowledge_base" and len(self.paper_ids) != 1:
            raise ValueError("paper-scoped search requires one paper ID")
        if hasattr(self, "semantic_query") and self.semantic_query != normalize_semantic_query(self.semantic_query):
            raise ValueError("semantic query is not compiler-normalized")

    @property
    def identity(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"))
        return "search-v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalPaperSearchRequest(_RequestIdentity):
    paper_ids: tuple[str, ...]
    semantic_query: str
    lexical_terms: tuple[str, ...]
    lexical_query: str
    top_k: int
    source: str
    workspace_id: str | None = None
    retrieval_parameters: dict[str, object] = field(default_factory=dict)
    kind: Literal["search_papers"] = "search_papers"


@dataclass(frozen=True)
class CanonicalReferenceSearchRequest(_RequestIdentity):
    paper_ids: tuple[str, ...]
    lexical_terms: tuple[str, ...]
    lexical_query: str
    top_k: int
    kind: Literal["search_references"] = "search_references"


@dataclass(frozen=True)
class CanonicalKBSearchRequest(_RequestIdentity):
    paper_ids: tuple[str, ...]
    semantic_query: str
    lexical_terms: tuple[str, ...]
    lexical_query: str
    top_k: int
    retrieval_parameters: dict[str, object] = field(default_factory=dict)
    scope: Literal["paper_ids", "all"] = "paper_ids"
    kind: Literal["search_knowledge_base"] = "search_knowledge_base"


CanonicalSearchRequest: TypeAlias = (
    CanonicalPaperSearchRequest | CanonicalReferenceSearchRequest | CanonicalKBSearchRequest
)


@dataclass(frozen=True)
class EffectiveRequestRejection:
    code: str
    detail: str


def validate_effective_request(request: CanonicalSearchRequest) -> EffectiveRequestRejection | None:
    """Shared executability authority; empty lexical is legal for semantic search."""
    if isinstance(request, CanonicalReferenceSearchRequest) and not request.lexical_terms:
        return EffectiveRequestRejection(
            "NoEffectiveLexicalQuery",
            "search_references requires at least one effective lexical token after canonical normalization. "
            "Revise lexical_terms rather than only changing the audit-only raw query.",
        )
    return None


def compile_paper_request(*, paper_id: str, query: str, lexical_terms: Sequence[str],
                          top_k: int, source: str, workspace_id: str | None = None,
                          retrieval_parameters: dict[str, object] | None = None) -> CanonicalPaperSearchRequest:
    terms = compile_lexical_terms(query, lexical_terms)
    if source == "workspace":
        from paperbase.staging.bm25 import _tokenize
        terms = tuple(sorted(set(_tokenize(" ".join(terms)))))
    return CanonicalPaperSearchRequest((paper_id.strip(),), normalize_semantic_query(query),
                                      terms, " ".join(terms) if source == "workspace" else compile_fts_query(terms), top_k, source,
                                      workspace_id, retrieval_parameters or {})


def compile_reference_request(*, paper_id: str, query: str,
                              lexical_terms: Sequence[str], top_k: int) -> CanonicalReferenceSearchRequest:
    terms = compile_lexical_terms(query, lexical_terms)
    return CanonicalReferenceSearchRequest((paper_id.strip(),), terms, compile_fts_query(terms), top_k)


def compile_kb_request(*, paper_ids: Sequence[str] | None, query: str, lexical_terms: Sequence[str],
                       top_k: int, retrieval_parameters: dict[str, object] | None = None) -> CanonicalKBSearchRequest:
    terms = compile_lexical_terms(query, lexical_terms)
    return CanonicalKBSearchRequest(tuple(sorted({paper_id.strip() for paper_id in paper_ids or ()})), normalize_semantic_query(query),
                                   terms, compile_fts_query(terms), top_k, retrieval_parameters or {},
                                   "all" if paper_ids is None else "paper_ids")


def request_trace(requests: Sequence[CanonicalSearchRequest]) -> list[dict[str, object]]:
    return [json.loads(json.dumps({**asdict(request), "request_identity": request.identity},
                                  ensure_ascii=False)) for request in requests]


def retriever_parameters(retriever: object) -> dict[str, object]:
    """Snapshot real live retrieval knobs, excluding diagnostic/LLM settings."""
    result: dict[str, object] = {}
    settings = getattr(retriever, "_settings", None)
    if settings is not None:
        result["retrieval"] = {key: value for key, value in (settings.model_dump(mode="json") if hasattr(settings, "model_dump") else asdict(settings)).items()
                               if key not in {"query_rewrite", "dense_semantic_weight"}}
    reranking = getattr(retriever, "_reranking_settings", None)
    if reranking is not None and getattr(retriever, "_reranker", None) is not None and reranking.enabled:
        fields = reranking.model_dump(mode="json") if hasattr(reranking, "model_dump") else asdict(reranking)
        # Device/batch size are execution placement, not retrieval intent. Formal
        # final_top_k only supplies the default before top_k is made explicit.
        ignored = {"device", "batch_size"}
        if retriever.__class__.__name__ == "HybridRetriever":
            ignored.add("final_top_k")
        result["reranking"] = {key:value for key,value in fields.items() if key not in ignored}
    else:
        result["reranking"] = None
    return result
