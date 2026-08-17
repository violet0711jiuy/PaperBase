"""Ask This Paper 的 section_id 边界证据扩展。"""

from __future__ import annotations

from dataclasses import dataclass

from paperbase.config import ContextExpansionSettings
from paperbase.generation.section_expander import EvidenceUnit, ExpansionResult
from paperbase.retrieval.hybrid_retriever import RetrievalResult
from paperbase.staging.sections import WorkspaceChunk, WorkspaceSectionSnapshot


@dataclass(frozen=True)
class _EvidenceGroup:
    """同一直属 Section 中可连通的命中窗口。"""

    chunks: tuple[WorkspaceChunk, ...]
    seed_chunk_ids: tuple[str, ...]
    priority: float


class WorkspaceSectionEvidenceExpander:
    """使用 ``section_id`` 取相邻块，绝不通过 section 展示文本猜测章节归属。"""

    def __init__(self, *, snapshot: WorkspaceSectionSnapshot, settings: ContextExpansionSettings) -> None:
        self._snapshot = snapshot
        self._settings = settings
        self._content_by_id = {
            chunk.chunk_id: chunk for chunk in snapshot.chunks if chunk.section_type == "content"
        }

    def expand(self, result: RetrievalResult) -> ExpansionResult:
        """正文按同一 section_id 连续扩展；bibliography 始终保持单条 R 证据。"""
        content_seeds = tuple(chunk for chunk in result.chunks if chunk.section_type == "content")
        bibliography = tuple(chunk for chunk in result.chunks if chunk.section_type == "bibliography")
        groups = self._groups(content_seeds)
        remaining = self._settings.max_total_tokens
        content: list[EvidenceUnit] = []
        for group in groups:
            tokens = sum(chunk.raw_token_count for chunk in group.chunks)
            if tokens > remaining:
                continue
            remaining -= tokens
            first = group.chunks[0]
            content.append(EvidenceUnit(
                evidence_id=f"E{len(content) + 1}", kind="content", paper_id=self._snapshot.paper_id,
                paper_title=first.paper_title or self._title, section=first.section or "",
                page_start=_minimum_page(group.chunks), page_end=_maximum_page(group.chunks),
                seed_chunk_ids=group.seed_chunk_ids, chunk_ids=tuple(chunk.chunk_id for chunk in group.chunks),
                text="\n\n".join(chunk.raw_text.strip() for chunk in group.chunks).strip(), token_count=tokens,
            ))
        references = tuple(
            EvidenceUnit(
                evidence_id=f"R{index}", kind="bibliography", paper_id=self._snapshot.paper_id,
                paper_title=chunk.paper_title or self._title, section=chunk.section or "References",
                page_start=chunk.page_start, page_end=chunk.page_end, seed_chunk_ids=(chunk.chunk_id,),
                chunk_ids=(chunk.chunk_id,), text=chunk.raw_text,
                token_count=self._snapshot_chunk_token_count(chunk.chunk_id),
            )
            for index, chunk in enumerate(bibliography, start=1)
        )
        return ExpansionResult(content_evidence=tuple(content), bibliography_evidence=references)

    def _groups(self, seeds: tuple[object, ...]) -> tuple[_EvidenceGroup, ...]:
        grouped: dict[str, list[object]] = {}
        for seed in seeds:
            chunk = self._content_by_id.get(seed.chunk_id)  # type: ignore[attr-defined]
            if chunk is None:
                continue
            # NULL section_id 无可靠的层级边界，因此只保留命中块，不向任何相邻块扩展。
            key = chunk.section_id or f"__chunk__{chunk.chunk_id}"
            grouped.setdefault(key, []).append(seed)
        groups: list[_EvidenceGroup] = []
        for section_id, section_seeds in grouped.items():
            rows = (
                tuple(chunk for chunk in self._content_by_id.values() if chunk.section_id == section_id)
                if not section_id.startswith("__chunk__")
                else (self._content_by_id[section_id.removeprefix("__chunk__")],)
            )
            by_index = {chunk.chunk_index: chunk for chunk in rows}
            windows: list[tuple[object, set[int]]] = []
            for seed in section_seeds:
                center = self._content_by_id[seed.chunk_id].chunk_index  # type: ignore[attr-defined]
                indices = {center}
                if self._settings.enabled:
                    for offset in range(1, self._settings.neighbor_window + 1):
                        if center - offset in by_index:
                            indices.add(center - offset)
                        else:
                            break
                    for offset in range(1, self._settings.neighbor_window + 1):
                        if center + offset in by_index:
                            indices.add(center + offset)
                        else:
                            break
                windows.append((seed, indices))
            components: list[tuple[list[object], set[int]]] = []
            for seed, indices in windows:
                matches = [index for index, (_, current) in enumerate(components) if current.intersection(indices)]
                if not matches:
                    components.append(([seed], set(indices)))
                    continue
                target_seeds, target_indices = components[matches[0]]
                target_seeds.append(seed)
                target_indices.update(indices)
                for index in reversed(matches[1:]):
                    extra_seeds, extra_indices = components.pop(index)
                    target_seeds.extend(extra_seeds)
                    target_indices.update(extra_indices)
            for component_seeds, indices in components:
                ordered_seeds = sorted(component_seeds, key=_seed_priority, reverse=True)
                groups.append(_EvidenceGroup(
                    chunks=tuple(by_index[index] for index in sorted(indices)),
                    seed_chunk_ids=tuple(seed.chunk_id for seed in ordered_seeds),  # type: ignore[attr-defined]
                    priority=_seed_priority(ordered_seeds[0]),
                ))
        return tuple(sorted(groups, key=lambda group: (-group.priority, group.chunks[0].chunk_index)))

    def _snapshot_chunk_token_count(self, chunk_id: str) -> int:
        """RetrievedChunk 没有 token 字段，统一回查已持久化 workspace chunk。"""
        for chunk in self._snapshot.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.raw_token_count
        return 1

    @property
    def _title(self) -> str:
        return self._snapshot.paper_title or "论文标题未识别"


def _seed_priority(seed: object) -> float:
    score = getattr(seed, "rerank_score", None)
    return float(score) if score is not None else -float(getattr(seed, "rank"))


def _minimum_page(chunks: tuple[WorkspaceChunk, ...]) -> int | None:
    pages = [chunk.page_start for chunk in chunks if chunk.page_start is not None]
    return min(pages) if pages else None


def _maximum_page(chunks: tuple[WorkspaceChunk, ...]) -> int | None:
    pages = [chunk.page_end for chunk in chunks if chunk.page_end is not None]
    return max(pages) if pages else None
