"""Temporary Workspace 的纯内存 BM25 索引。

正式知识库的 BM25 由 SQLite FTS5 提供；临时工作区刻意没有 SQLite，因而在读取
``chunks.jsonl`` 后只在进程内建立一个小型、可丢弃的倒排统计结构。它绝不读取正式库。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping

from .sections import WorkspaceChunk, WorkspaceSectionSnapshot


class WorkspaceBM25Error(ValueError):
    """临时论文的 BM25 输入不符合检索契约时抛出。"""


@dataclass(frozen=True)
class WorkspaceBM25Match:
    """一条工作区 BM25 命中；分数仅供调试，跨路融合仍应使用 rank。"""

    chunk: WorkspaceChunk
    rank: int
    bm25_score: float


@dataclass(frozen=True)
class WorkspaceBM25Index:
    """单个工作区正文 chunk 的轻量 BM25F 风格索引。

    ``term_frequencies`` 保留章节和正文的加权词频；这样不需要 staging SQLite，
    仍可保持 v0.1 FTS5 中 section 0.8、正文 1.0 的字段优先级。单论文的标题对每个
    chunk 都相同，放入 BM25 反而会制造无意义的全量命中，因此刻意不索引它。
    """

    workspace_id: str
    paper_id: str
    source_fingerprint: str
    section_type: str
    chunks: tuple[WorkspaceChunk, ...]
    term_frequencies: tuple[Mapping[str, float], ...]
    document_frequencies: Mapping[str, int]
    average_document_length: float

    # 与正式 SQLite FTS5 的字段权重语义对齐，但 BM25 的具体实现不依赖 SQLite。
    section_weight = 0.8
    raw_text_weight = 1.0
    k1 = 1.2
    b = 0.75

    @classmethod
    def build(
        cls, snapshot: WorkspaceSectionSnapshot, *, section_type: str = "content"
    ) -> "WorkspaceBM25Index":
        """从当前快照构建指定类型索引；默认正文，参考文献只允许显式单独调用。"""
        if section_type not in {"content", "bibliography"}:
            raise WorkspaceBM25Error("Workspace BM25 supports content or bibliography only.")
        chunks = tuple(
            chunk for chunk in snapshot.chunks if chunk.section_type == section_type
        )
        if not chunks:
            raise WorkspaceBM25Error(
                f"Temporary workspace has no {section_type} chunks for BM25."
            )

        term_frequencies: list[Mapping[str, float]] = []
        document_frequencies: dict[str, int] = {}
        document_lengths: list[float] = []
        for chunk in chunks:
            frequencies: dict[str, float] = {}
            _add_weighted_terms(frequencies, _tokenize(chunk.section or ""), cls.section_weight)
            _add_weighted_terms(frequencies, _tokenize(chunk.raw_text), cls.raw_text_weight)
            if not frequencies:
                # chunks.jsonl 已要求 raw_text 非空；此保护避免异常数据造成除零。
                continue
            term_frequencies.append(MappingProxyType(frequencies))
            document_lengths.append(sum(frequencies.values()))
            for term in frequencies:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1

        # 所有可检索 chunk 都必须产生至少一个词元，否则索引没有可解释的语义。
        if len(term_frequencies) != len(chunks):
            raise WorkspaceBM25Error("A content chunk cannot be tokenized for workspace BM25.")
        return cls(
            workspace_id=snapshot.workspace_id,
            paper_id=snapshot.paper_id,
            source_fingerprint=workspace_bm25_fingerprint(snapshot, section_type=section_type),
            section_type=section_type,
            chunks=chunks,
            term_frequencies=tuple(term_frequencies),
            document_frequencies=MappingProxyType(document_frequencies),
            average_document_length=sum(document_lengths) / len(document_lengths),
        )

    def search(self, query: str, *, top_k: int) -> tuple[WorkspaceBM25Match, ...]:
        """以当前论文为唯一语料进行 BM25 词法检索。"""
        if top_k < 1:
            raise WorkspaceBM25Error("BM25 top_k must be positive.")
        query_terms = _tokenize(" ".join(query.split()))
        if not query_terms:
            raise WorkspaceBM25Error("BM25 query must not be empty.")

        total_documents = len(self.chunks)
        scored: list[tuple[WorkspaceChunk, float]] = []
        for chunk, frequencies in zip(self.chunks, self.term_frequencies, strict=True):
            document_length = sum(frequencies.values())
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0.0)
                if term_frequency <= 0:
                    continue
                document_frequency = self.document_frequencies.get(term, 0)
                # 使用常见的非负 IDF，避免高频词给出负分而干扰 Top-K 的稳定排序。
                inverse_document_frequency = math.log(
                    1.0
                    + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = self.k1 * (
                    1.0 - self.b + self.b * document_length / self.average_document_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0) / (term_frequency + normalization)
                )
            if score > 0:
                scored.append((chunk, score))

        ordered = sorted(scored, key=lambda item: (-item[1], item[0].chunk_index, item[0].chunk_id))
        return tuple(
            WorkspaceBM25Match(chunk=chunk, rank=rank, bm25_score=score)
            for rank, (chunk, score) in enumerate(ordered[:top_k], start=1)
        )


class WorkspaceBM25IndexCache:
    """按 workspace 内容指纹缓存内存索引；不向磁盘或 SQLite 写入任何数据。"""

    def __init__(self, *, max_entries: int = 16) -> None:
        if max_entries < 1:
            raise ValueError("Workspace BM25 cache max_entries must be positive.")
        self._max_entries = max_entries
        self._indexes: OrderedDict[str, WorkspaceBM25Index] = OrderedDict()

    def get_or_build(
        self, snapshot: WorkspaceSectionSnapshot, *, section_type: str = "content"
    ) -> WorkspaceBM25Index:
        """内容未变时复用索引；同 ID 工作区数据变更后自动换用新索引。"""
        if section_type not in {"content", "bibliography"}:
            raise WorkspaceBM25Error("Workspace BM25 supports content or bibliography only.")
        cache_key = f"{snapshot.workspace_id}:{section_type}"
        fingerprint = workspace_bm25_fingerprint(snapshot, section_type=section_type)
        existing = self._indexes.get(cache_key)
        if existing is not None and existing.source_fingerprint == fingerprint:
            self._indexes.move_to_end(cache_key)
            return existing

        index = WorkspaceBM25Index.build(snapshot, section_type=section_type)
        self._indexes[cache_key] = index
        self._indexes.move_to_end(cache_key)
        while len(self._indexes) > self._max_entries:
            self._indexes.popitem(last=False)
        return index

    def invalidate(self, workspace_id: str) -> None:
        """删除工作区时可主动释放对应索引；未知 ID 是安全的空操作。"""
        for cache_key in tuple(self._indexes):
            if cache_key.startswith(f"{workspace_id}:"):
                self._indexes.pop(cache_key)


def workspace_bm25_fingerprint(
    snapshot: WorkspaceSectionSnapshot, *, section_type: str = "content"
) -> str:
    """只对影响词法检索的持久化字段计算哈希，确保缓存不会使用旧内容。"""
    digest = hashlib.sha256()
    for value in (snapshot.workspace_id, snapshot.paper_id, section_type):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for chunk in snapshot.chunks:
        if chunk.section_type != section_type:
            continue
        for value in (chunk.chunk_id, str(chunk.chunk_index), chunk.section or "", chunk.raw_text):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


_CJK_RUN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> tuple[str, ...]:
    """轻量、确定性的中英文分词：英文按词，中文保留单字和双字词元。"""
    normalized = " ".join(text.casefold().split())
    if not normalized:
        return ()
    terms = _LATIN_WORD_PATTERN.findall(normalized)
    for run in _CJK_RUN_PATTERN.findall(normalized):
        terms.extend(run)
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(terms)


def _add_weighted_terms(
    frequencies: dict[str, float], terms: tuple[str, ...], weight: float
) -> None:
    """将一个字段的词元按字段权重加入单个 BM25 文档。"""
    for term in terms:
        frequencies[term] = frequencies.get(term, 0.0) + weight
