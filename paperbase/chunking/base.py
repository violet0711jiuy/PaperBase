"""PaperBase 分块结果的统一契约。
分块数据结构"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from paperbase.parsing.base import ParsedPaper


@dataclass(frozen=True)
class PaperChunk:
    """一段可检索、可引用、可连接相邻上下文的论文内容。

    ``raw_text`` 保持结构分块后的原文，后续 Generation 直接使用它；
    ``embedding_text`` 在原文前添加论文标题和章节路径，专门服务跨语言向量检索。
    二者必须分开保存，不能让为检索添加的提示性元数据混入最终给 LLM 的原文证据。

    ``vector_id`` 在 Step 5 由统一 FAISS 索引分配，因此在当前 Step 2 为 ``None``。
    ``prev_chunk_id`` / ``next_chunk_id`` 已在此阶段建立，Step 8 可据此实现同章节的
    邻居扩展，而无需依赖导入时的偶然排序。
    """

    chunk_id: str
    vector_id: int | None
    paper_id: str
    paper_title: str | None
    source: Path
    chunk_index: int
    raw_text: str
    embedding_text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    raw_token_count: int
    embedding_token_count: int
    prev_chunk_id: str | None
    next_chunk_id: str | None
    # 全文文本只保存一次。前置元数据也作为普通可检索 chunk 写入，靠这两个字段区分，
    # 以便后续统一分配 vector_id 并通过 FAISS 回查，不再维护一份重复的 metadata 正文。
    content_kind: str = "body"
    front_matter_type: str | None = None


@dataclass(frozen=True)
class ChunkingResult:
    """一次论文分块的统一输出及可审计诊断信息。"""

    parsed_paper: ParsedPaper
    chunks: tuple[PaperChunk, ...]
    diagnostics: dict[str, int | str]


@runtime_checkable
class PaperChunker(Protocol):
    """可替换的论文分块器接口。

    分块器只消费解析器无关的 ``ParsedPaper``，并输出 ``ChunkingResult``。未来替换
    Docling 的分块方法时，SQLite、Embedding、FAISS 与 Retrieval 层都不必修改。
    """

    chunker_id: str

    def chunk(self, parsed_paper: ParsedPaper) -> ChunkingResult:
        """把一次已完成的论文解析结果转换为统一的 chunk 列表。"""
