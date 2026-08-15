"""按论文和章节边界补齐 Reranker 命中的上下文，不跨越参考文献边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from paperbase.config import ContextExpansionSettings
from paperbase.database import MetadataDatabase
from paperbase.retrieval.hybrid_retriever import RetrievedChunk, RetrievalResult


@dataclass(frozen=True)
class EvidenceUnit:
    """可直接交给回答模型的一组可溯源证据，而不是未经边界控制的长文本。"""

    evidence_id: str
    kind: str
    paper_id: str
    paper_title: str
    section: str
    page_start: int | None
    page_end: int | None
    seed_chunk_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    text: str
    token_count: int


@dataclass(frozen=True)
class ExpansionResult:
    """一次检索结果经 Section-aware Neighbor Expansion 后的证据快照。"""

    content_evidence: tuple[EvidenceUnit, ...]
    bibliography_evidence: tuple[EvidenceUnit, ...]

    @property
    def evidence(self) -> tuple[EvidenceUnit, ...]:
        """正文证据在前、参考文献证据在后，保持回答提示词中的稳定编号顺序。"""
        return (*self.content_evidence, *self.bibliography_evidence)


@dataclass(frozen=True)
class _CandidateGroup:
    """同一章节中的一个连续 chunk 组，尚未分配给最终 E 编号。"""

    paper_id: str
    paper_title: str
    section: str
    rows: tuple[Any, ...]
    seed_chunk_ids: tuple[str, ...]
    priority: float


class SectionAwareNeighborExpander:
    """将正文种子块扩展到同论文、同章节且连续的相邻块。

    References/Bibliography 已在 Step 6 标为 ``section_type='bibliography'``，此处
    永不查询其相邻块；它们只能作为独立的 R 证据单元保留。
    """

    def __init__(
        self,
        *,
        database: MetadataDatabase,
        settings: ContextExpansionSettings,
    ) -> None:
        self._database = database
        self._settings = settings

    def expand(self, result: RetrievalResult) -> ExpansionResult:
        """将 Step 7 的最终结果转换为正文 E 证据和参考文献 R 证据。"""
        content_seeds = tuple(
            chunk for chunk in result.chunks if chunk.section_type == "content"
        )
        bibliography = tuple(
            chunk for chunk in result.chunks if chunk.section_type == "bibliography"
        )
        content_groups = self._build_content_groups(content_seeds)
        content_evidence = self._apply_content_token_budget(content_groups)
        bibliography_evidence = tuple(
            _bibliography_to_evidence(chunk, index=index + 1)
            for index, chunk in enumerate(bibliography)
        )
        return ExpansionResult(
            content_evidence=content_evidence,
            bibliography_evidence=bibliography_evidence,
        )

    def _build_content_groups(
        self, seeds: tuple[RetrievedChunk, ...]
    ) -> tuple[_CandidateGroup, ...]:
        """为每个章节生成重叠窗口合并后的连续证据组。"""
        if not seeds:
            return ()
        grouped_seeds: dict[tuple[str, str], list[RetrievedChunk]] = {}
        for seed in seeds:
            # 空 section 不做邻居扩展：没有可靠的章节边界时，保留命中块本身最安全。
            section = seed.section.strip()
            if not section:
                key = (seed.paper_id, f"__chunk__{seed.chunk_id}")
            else:
                key = (seed.paper_id, section)
            grouped_seeds.setdefault(key, []).append(seed)

        groups: list[_CandidateGroup] = []
        for (paper_id, section_key), section_seeds in grouped_seeds.items():
            section = "" if section_key.startswith("__chunk__") else section_key
            rows = (
                self._database.list_content_chunks_in_section(
                    paper_id=paper_id,
                    section=section,
                )
                if section
                else ()
            )
            groups.extend(
                _expand_section_seed_windows(
                    seeds=tuple(section_seeds),
                    rows=rows,
                    settings=self._settings,
                )
            )
        # 先以最强 seed 的 rerank 分数排序；分数相同时保留检索结果中更靠前的组。
        return tuple(
            sorted(
                groups,
                key=lambda item: (
                    -item.priority,
                    item.paper_id,
                    item.section,
                    tuple(str(row["chunk_id"]) for row in item.rows),
                ),
            )
        )

    def _apply_content_token_budget(
        self, groups: tuple[_CandidateGroup, ...]
    ) -> tuple[EvidenceUnit, ...]:
        """按优先级装入完整组，绝不从 raw_text 中间截断以凑 token 预算。"""
        if not groups:
            return ()
        remaining = self._settings.max_total_tokens
        evidence: list[EvidenceUnit] = []
        for group in groups:
            token_count = sum(int(row["raw_token_count"]) for row in group.rows)
            if token_count > remaining:
                continue
            remaining -= token_count
            evidence.append(
                EvidenceUnit(
                    evidence_id=f"E{len(evidence) + 1}",
                    kind="content",
                    paper_id=group.paper_id,
                    paper_title=group.paper_title,
                    section=group.section,
                    page_start=_minimum_page(group.rows),
                    page_end=_maximum_page(group.rows),
                    seed_chunk_ids=group.seed_chunk_ids,
                    chunk_ids=tuple(str(row["chunk_id"]) for row in group.rows),
                    text=_join_rows(group.rows),
                    token_count=token_count,
                )
            )
        return tuple(evidence)


def _expand_section_seed_windows(
    *,
    seeds: tuple[RetrievedChunk, ...],
    rows: Iterable[Any],
    settings: ContextExpansionSettings,
) -> tuple[_CandidateGroup, ...]:
    """在一个章节内构造连续窗口，并只合并实际重叠的窗口。"""
    row_by_id = {str(row["chunk_id"]): row for row in rows}
    row_by_index = {int(row["chunk_index"]): row for row in row_by_id.values()}
    windows: list[tuple[RetrievedChunk, set[int]]] = []
    for seed in seeds:
        seed_row = row_by_id.get(seed.chunk_id)
        if seed_row is None:
            # 测试替身、旧库或异常数据缺少同节记录时安全退化为种子本身。
            fallback = _seed_as_row(seed)
            windows.append((seed, {int(fallback["chunk_index"])}))
            row_by_index[int(fallback["chunk_index"])] = fallback
            continue
        center = int(seed_row["chunk_index"])
        indices = {center}
        if settings.enabled:
            for offset in range(1, settings.neighbor_window + 1):
                previous = center - offset
                if previous in row_by_index:
                    indices.add(previous)
                else:
                    break
            for offset in range(1, settings.neighbor_window + 1):
                following = center + offset
                if following in row_by_index:
                    indices.add(following)
                else:
                    break
        windows.append((seed, indices))

    components: list[tuple[list[RetrievedChunk], set[int]]] = []
    for seed, indices in windows:
        overlapping = [
            index
            for index, (_, component_indices) in enumerate(components)
            if component_indices.intersection(indices)
        ]
        if not overlapping:
            components.append(([seed], set(indices)))
            continue
        first = overlapping[0]
        merged_seeds, merged_indices = components[first]
        merged_seeds.append(seed)
        merged_indices.update(indices)
        for index in reversed(overlapping[1:]):
            extra_seeds, extra_indices = components.pop(index)
            merged_seeds.extend(extra_seeds)
            merged_indices.update(extra_indices)

    candidates: list[_CandidateGroup] = []
    for component_seeds, indices in components:
        selected_rows = _select_complete_group_rows(
            rows_by_index=row_by_index,
            indices=indices,
        )
        if not selected_rows:
            continue
        best_seed = max(component_seeds, key=_seed_priority)
        candidates.append(
            _CandidateGroup(
                paper_id=best_seed.paper_id,
                paper_title=best_seed.paper_title,
                section=best_seed.section,
                rows=selected_rows,
                seed_chunk_ids=tuple(
                    seed.chunk_id
                    for seed in sorted(component_seeds, key=_seed_priority, reverse=True)
                ),
                priority=_seed_priority(best_seed),
            )
        )
    return tuple(candidates)


def _select_complete_group_rows(
    *,
    rows_by_index: dict[int, Any],
    indices: set[int],
) -> tuple[Any, ...]:
    """保留合并窗口中的全部完整块，避免因条数上限静默删除 Reranker 种子。

    窗口规模由 ``neighbor_window`` 自然约束；跨种子合并后如总长度变大，统一由
    ``max_total_tokens`` 在组级别裁决，而不是从一个已经确认相关的证据组内丢弃块。
    """
    return tuple(rows_by_index[index] for index in sorted(indices))


def _seed_priority(seed: RetrievedChunk) -> float:
    """优先使用 Cross-Encoder 分数；没有重排分数时用最终 rank 保持稳定顺序。"""
    return seed.rerank_score if seed.rerank_score is not None else -float(seed.rank)


def _seed_as_row(seed: RetrievedChunk) -> dict[str, Any]:
    """在不完整数据库记录下构造只含 seed 的安全退化行。"""
    return {
        "chunk_id": seed.chunk_id,
        "chunk_index": 0,
        "raw_text": seed.raw_text,
        "raw_token_count": max(1, len(seed.raw_text.split())),
        "page_start": seed.page_start,
        "page_end": seed.page_end,
    }


def _bibliography_to_evidence(chunk: RetrievedChunk, *, index: int) -> EvidenceUnit:
    """参考文献不扩展相邻块，每条检索命中单独形成一个可标注的 R 证据。"""
    return EvidenceUnit(
        evidence_id=f"R{index}",
        kind="bibliography",
        paper_id=chunk.paper_id,
        paper_title=chunk.paper_title,
        section=chunk.section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        seed_chunk_ids=(chunk.chunk_id,),
        chunk_ids=(chunk.chunk_id,),
        text=chunk.raw_text,
        token_count=max(1, len(chunk.raw_text.split())),
    )


def _join_rows(rows: tuple[Any, ...]) -> str:
    """保留块边界，便于人工追溯，不以任意字符串规则重新拼接或改写原文。"""
    return "\n\n".join(str(row["raw_text"]).strip() for row in rows).strip()


def _minimum_page(rows: tuple[Any, ...]) -> int | None:
    pages = [int(row["page_start"]) for row in rows if row["page_start"] is not None]
    return min(pages) if pages else None


def _maximum_page(rows: tuple[Any, ...]) -> int | None:
    pages = [int(row["page_end"]) for row in rows if row["page_end"] is not None]
    return max(pages) if pages else None
