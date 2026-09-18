"""把用户问题解析为可信论文范围；LLM 不得发明 paper_id。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from paperbase.database import MetadataDatabase

from .contracts import (
    CorpusScopeSnapshot,
    QueryScope,
    ScopeResolution,
    ScopeResolutionStatus,
    ScopeSource,
)


@dataclass(frozen=True)
class TrustedWorkspaceScope:
    paper_id: str
    paper_title: str


class ScopeResolver:
    """按可信上下文、UI 选择、显式 ID、标题的顺序解析范围。"""

    def __init__(self, database: MetadataDatabase) -> None:
        self._database = database

    def resolve(
        self,
        query: str,
        *,
        trusted_workspace: TrustedWorkspaceScope | None = None,
        selected_paper_ids: tuple[str, ...] = (),
        knowledge_base_all: bool = False,
    ) -> ScopeResolution:
        if trusted_workspace is not None:
            # Workspace 是独立生命周期，可信 paper_id 不要求已经晋升到正式知识库。
            return ScopeResolution(
                status=ScopeResolutionStatus.RESOLVED,
                scope=QueryScope.SINGLE_PAPER,
                scope_source=ScopeSource.TRUSTED_WORKSPACE,
                target_paper_ids=(trusted_workspace.paper_id,),
            )
        if selected_paper_ids:
            return self._resolved_ids(selected_paper_ids, ScopeSource.UI_SELECTION)

        documents = self._database.list_documents()
        if knowledge_base_all:
            paper_ids = tuple(str(row["paper_id"]) for row in documents)
            return ScopeResolution(
                status=ScopeResolutionStatus.RESOLVED,
                scope=QueryScope.KNOWLEDGE_BASE,
                scope_source=ScopeSource.KNOWLEDGE_BASE_ALL,
                corpus_snapshot=CorpusScopeSnapshot.create(paper_ids),
            )

        query_folded = query.casefold()
        explicit_ids = tuple(
            str(row["paper_id"])
            for row in documents
            if _contains_token(query_folded, str(row["paper_id"]).casefold())
        )
        if explicit_ids:
            return self._resolved_ids(explicit_ids, ScopeSource.RESOLVED_FROM_QUERY)

        normalized_query = _normalize_identity(query)
        matched_identities: dict[str, list[str]] = {}
        for row in documents:
            title = _normalize_identity(str(row["paper_title"] or ""))
            filename = _normalize_identity(str(row["source_filename"] or ""))
            paper_id = str(row["paper_id"])
            if title and len(title) >= 8 and title in normalized_query:
                matched_identities.setdefault(title, []).append(paper_id)
            elif filename and len(filename) >= 8 and filename in normalized_query:
                matched_identities.setdefault(filename, []).append(paper_id)
            elif title and normalized_query == title:
                matched_identities.setdefault(title, []).append(paper_id)
        ambiguous = tuple(
            dict.fromkeys(
                paper_id
                for paper_ids in matched_identities.values()
                if len(set(paper_ids)) > 1
                for paper_id in paper_ids
            )
        )
        if ambiguous:
            return ScopeResolution(
                status=ScopeResolutionStatus.AMBIGUOUS,
                candidate_paper_ids=ambiguous,
                clarification="同一标题匹配到多篇论文，请通过 UI 或 paper_id 明确选择。",
            )
        matches = tuple(
            dict.fromkeys(
                paper_id
                for paper_ids in matched_identities.values()
                for paper_id in paper_ids
            )
        )
        if matches:
            return self._resolved_ids(matches, ScopeSource.RESOLVED_FROM_QUERY)
        return ScopeResolution(
            status=ScopeResolutionStatus.NOT_FOUND,
            clarification="未能从当前问题中确定论文。请选中论文，或提供准确标题 / paper_id。",
        )

    def _resolved_ids(
        self, paper_ids: tuple[str, ...], source: ScopeSource
    ) -> ScopeResolution:
        normalized = tuple(dict.fromkeys(" ".join(value.split()) for value in paper_ids))
        missing = tuple(
            paper_id
            for paper_id in normalized
            if self._database.get_document(paper_id) is None
        )
        if missing:
            return ScopeResolution(
                status=ScopeResolutionStatus.NOT_FOUND,
                candidate_paper_ids=missing,
                clarification="以下论文不在正式知识库中：" + ", ".join(missing),
            )
        return ScopeResolution(
            status=ScopeResolutionStatus.RESOLVED,
            scope=(
                QueryScope.SINGLE_PAPER
                if len(normalized) == 1
                else QueryScope.SELECTED_PAPERS
            ),
            scope_source=source,
            target_paper_ids=normalized,
        )


def _normalize_identity(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", folded).split())


def _contains_token(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?<![\w-]){re.escape(needle)}(?![\w-])", haystack) is not None
