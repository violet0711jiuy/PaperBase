"""使用 Docling HybridChunker 的结构感知论文分块器。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from paperbase.parsing.base import FrontMatterBlock, ParsedPaper

from .base import ChunkingResult, PaperChunk, PaperChunker


@dataclass(frozen=True)
class ChunkerSettings:
    """当前 Docling HybridChunker 的全部非敏感、可配置参数。"""

    tokenizer_path: Path
    max_tokens: int = 512
    embedding_metadata_reserve_tokens: int = 64
    repeat_table_header: bool = True
    merge_peers: bool = True


class DoclingHybridPaperChunker(PaperChunker):
    """以 Qwen tokenizer 计数的 Docling HybridChunker 适配器。"""

    chunker_id = "docling_hybrid"

    def __init__(self, settings: ChunkerSettings) -> None:
        self._settings = settings
        self._tokenizer = self._load_local_tokenizer()

    def chunk(self, parsed_paper: ParsedPaper) -> ChunkingResult:
        """将一个 ``ParsedPaper`` 分成保留章节和页码来源的统一 chunk。

        HybridChunker 本身根据文档树、标题、表格和 token 预算进行切分。这里不再按
        字符数或页面进行二次切割，以保留 Step 1 已验证过的论文结构与阅读顺序。
        """
        document = _require_docling_document(parsed_paper)
        paper_id = _paper_id(parsed_paper.source)
        effective_content_tokens = (
            self._settings.max_tokens - self._settings.embedding_metadata_reserve_tokens
        )
        if effective_content_tokens < 1:
            raise ValueError(
                "embedding_metadata_reserve_tokens must be smaller than max_tokens."
            )

        # 标题、章节路径和字段标签会在后面加入 embedding_text。预留它们的 token 空间，
        # 才能让最终喂给 Qwen embedding 的文本尽量不超过配置的总上限。
        docling_tokenizer = HuggingFaceTokenizer.from_pretrained(
            self._settings.tokenizer_path,
            max_tokens=effective_content_tokens,
            local_files_only=True,
        )
        hybrid_chunker = HybridChunker(
            tokenizer=docling_tokenizer,
            repeat_table_header=self._settings.repeat_table_header,
            merge_peers=self._settings.merge_peers,
        )

        draft_chunks: list[_DraftChunk] = []
        skipped_noise_chunk_count = 0
        for docling_chunk in hybrid_chunker.chunk(document):
            raw_text = (docling_chunk.text or "").strip()
            if not raw_text:
                # 空对象对向量检索没有意义，也不能提供有效证据，因此不建立空 chunk。
                continue
            section = _section_from_headings(docling_chunk.meta.headings)
            page_start, page_end = _page_range(docling_chunk)
            if _is_layout_noise_chunk(
                raw_text=raw_text,
                section=section,
                raw_token_count=self._tokenizer.count_tokens(raw_text),
            ):
                # 例如 MDPI 首页孤立的 ``Article`` 标签没有论文学术语义。这里仅过滤
                # 精确的出版类型标签；短标题、短公式、图注和关键词均不会匹配。
                skipped_noise_chunk_count += 1
                continue
            embedding_text = _build_embedding_text(
                paper_title=parsed_paper.paper_title,
                section=section,
                raw_text=raw_text,
            )
            front_matter_type = _front_matter_type_for_chunk(
                section=section,
                page_start=page_start,
                page_end=page_end,
                front_matter_blocks=parsed_paper.front_matter,
            )
            # 基于 Docling 已恢复的标题层级分类，而非在段落正文中搜索 “references”。
            # 因此 “Related Work” 即使讨论引用，也仍然是正文 content。
            section_type = _section_type_from_headings(docling_chunk.meta.headings)
            draft_chunks.append(
                _DraftChunk(
                    raw_text=raw_text,
                    embedding_text=embedding_text,
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                    raw_token_count=self._tokenizer.count_tokens(raw_text),
                    embedding_token_count=self._tokenizer.count_tokens(embedding_text),
                    content_kind=("front_matter" if front_matter_type else "body"),
                    front_matter_type=front_matter_type,
                    section_type=section_type,
                )
            )

        chunks = tuple(
            _to_paper_chunk(
                draft=draft,
                paper_id=paper_id,
                parsed_paper=parsed_paper,
                chunk_index=index,
                total_chunks=len(draft_chunks),
            )
            for index, draft in enumerate(draft_chunks)
        )
        over_limit_count = sum(
            chunk.embedding_token_count > self._settings.max_tokens for chunk in chunks
        )
        return ChunkingResult(
            parsed_paper=parsed_paper,
            chunks=chunks,
            diagnostics={
                "chunking.chunker_id": self.chunker_id,
                "chunking.tokenizer_path": str(self._settings.tokenizer_path),
                "chunking.max_tokens": self._settings.max_tokens,
                "chunking.effective_content_tokens": effective_content_tokens,
                "chunking.chunk_count": len(chunks),
                "chunking.skipped_layout_noise_chunk_count": skipped_noise_chunk_count,
                "chunking.max_embedding_token_count": max(
                    (chunk.embedding_token_count for chunk in chunks), default=0
                ),
                "chunking.over_limit_chunk_count": over_limit_count,
            },
        )

    def _load_local_tokenizer(self) -> HuggingFaceTokenizer:
        """只从 D 盘本地模型目录加载 Qwen tokenizer，不下载默认 MiniLM。"""
        if not self._settings.tokenizer_path.is_dir():
            raise FileNotFoundError(
                "Chunking tokenizer directory not found: "
                f"{self._settings.tokenizer_path}"
            )
        return HuggingFaceTokenizer.from_pretrained(
            self._settings.tokenizer_path,
            max_tokens=self._settings.max_tokens,
            local_files_only=True,
        )


@dataclass(frozen=True)
class _DraftChunk:
    """在生成前后邻居 ID 前暂存的 Chunk 数据。"""

    raw_text: str
    embedding_text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    raw_token_count: int
    embedding_token_count: int
    content_kind: str
    front_matter_type: str | None
    section_type: str


def _require_docling_document(parsed_paper: ParsedPaper) -> DoclingDocument:
    """取得 Docling 原生文档，阻止该适配器误用于其他解析器的结果。"""
    if not isinstance(parsed_paper.native_document, DoclingDocument):
        raise TypeError(
            "Docling hybrid chunking requires a ParsedPaper produced by DoclingParser."
        )
    return parsed_paper.native_document


def _paper_id(source: Path) -> str:
    """从 PDF 字节生成稳定 paper_id，避免依赖中文文件名或当前目录。"""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return f"paper_{digest}"


def _chunk_id(paper_id: str, chunk_index: int) -> str:
    """生成可排序、可读且跨重复运行稳定的 chunk ID。"""
    return f"{paper_id}_chunk_{chunk_index:04d}"


def _to_paper_chunk(
    *,
    draft: _DraftChunk,
    paper_id: str,
    parsed_paper: ParsedPaper,
    chunk_index: int,
    total_chunks: int,
) -> PaperChunk:
    """将暂存数据补齐稳定 ID 与同论文的双向邻居引用。"""
    chunk_id = _chunk_id(paper_id, chunk_index)
    previous_id = _chunk_id(paper_id, chunk_index - 1) if chunk_index else None
    next_id = (
        _chunk_id(paper_id, chunk_index + 1)
        if chunk_index + 1 < total_chunks
        else None
    )
    return PaperChunk(
        chunk_id=chunk_id,
        vector_id=None,
        paper_id=paper_id,
        paper_title=parsed_paper.paper_title,
        source=parsed_paper.source,
        chunk_index=chunk_index,
        raw_text=draft.raw_text,
        embedding_text=draft.embedding_text,
        section=draft.section,
        page_start=draft.page_start,
        page_end=draft.page_end,
        raw_token_count=draft.raw_token_count,
        embedding_token_count=draft.embedding_token_count,
        prev_chunk_id=previous_id,
        next_chunk_id=next_id,
        content_kind=draft.content_kind,
        front_matter_type=draft.front_matter_type,
        section_type=draft.section_type,
    )


def _section_from_headings(headings: list[str] | None) -> str | None:
    """将 Docling 的标题层级序列转为稳定、易读的章节路径。"""
    normalized_headings = [
        _normalize_whitespace(heading) for heading in (headings or []) if heading.strip()
    ]
    return " > ".join(normalized_headings) if normalized_headings else None


def _section_type_from_headings(headings: list[str] | None) -> str:
    """依据 Docling 标题树的末级标题识别参考文献章节。

    仅接受受控章节标题及其可选编号，例如 ``6. References``；不会在正文、图注或
    ``Related Work`` 中搜索单词，避免把讨论既有工作的正文误判为 bibliography。
    """
    if not headings:
        return "content"
    heading = _normalize_whitespace(headings[-1])
    # 删除章节编号、罗马数字或括号编号，保留真正的标题语义。
    leaf = re.sub(
        r"^(?:\d+(?:\.\d+)*(?:[.)]|\s+)|[IVXLC]+(?:[.)]|\s+))",
        "",
        heading,
        flags=re.IGNORECASE,
    )
    normalized = _normalize_whitespace(leaf).casefold().rstrip(":")
    if normalized in {"references", "bibliography", "works cited", "literature cited"}:
        return "bibliography"
    return "content"


def _front_matter_type_for_chunk(
    *,
    section: str | None,
    page_start: int | None,
    page_end: int | None,
    front_matter_blocks: tuple[FrontMatterBlock, ...],
) -> str | None:
    """依据 Step 1 的语义块给 chunk 标记前置元数据类型。

    这里不从某篇论文的正文关键词猜测，也不改动切块边界。只有标题末级名称与 Step 1
    识别出的标准 section 一致，并且页码范围相交（任一方缺失页码时不阻止匹配）时，
    才将该 chunk 标记为 front_matter。这使不同出版社的 Authors、Abstract、Keywords
    等已规范化标题都走同一条规则。
    """
    if section is None:
        return None

    section_leaf = _normalized_section_leaf(section)
    for block in front_matter_blocks:
        if section_leaf != _normalized_section_leaf(block.canonical_section):
            continue
        if _page_ranges_overlap(
            page_start,
            page_end,
            block.page_start,
            block.page_end,
        ):
            return block.block_type
    return None


def _normalized_section_leaf(section: str) -> str:
    """比较标题时只使用末级名称，兼容是否显式保留 ``Front matter >`` 父级。"""
    normalized = _normalize_whitespace(section.rsplit(">", maxsplit=1)[-1]).casefold()
    # 某些 PDF 会把 ``ABSTRACT`` 输出成 ``A B S T R A C T``。仅在每个空格分隔单元
    # 都是单个英文字母时去掉其中空格，避免误把正常多词标题（例如 Data Availability）混淆。
    pieces = normalized.split()
    if pieces and all(len(piece) == 1 and piece.isalpha() for piece in pieces):
        return "".join(pieces)
    return normalized


def _page_ranges_overlap(
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
) -> bool:
    """判断两个闭区间页码是否相交；缺页码时宁可保留语义标题匹配。"""
    if None in (left_start, left_end, right_start, right_end):
        return True
    return max(left_start, right_start) <= min(left_end, right_end)


def _page_range(docling_chunk: object) -> tuple[int | None, int | None]:
    """从 chunk 的全部来源对象计算最小、最大页码，不猜测缺失页码。"""
    pages = sorted(
        {
            provenance.page_no
            for item in getattr(getattr(docling_chunk, "meta", None), "doc_items", [])
            for provenance in getattr(item, "prov", [])
        }
    )
    return (pages[0], pages[-1]) if pages else (None, None)


def _build_embedding_text(
    *, paper_title: str | None,
    section: str | None,
    raw_text: str,
) -> str:
    """为向量检索拼接少量、可解释的论文上下文。

    用户会用中文查询、论文原文主要为英文。显式加入论文标题与章节，使例如“数据集”、
    “实验设置”这样的短 query 能同时得到论文级和章节级语义提示；不加入页码和内部 ID，
    避免让检索向量学习无语义的实现细节。
    """
    title = _normalize_whitespace(paper_title or "Untitled paper")
    section_label = _normalize_whitespace(section or "Unsectioned content")
    return f"Paper title: {title}\nSection: {section_label}\nContent:\n{raw_text}"


def _normalize_whitespace(text: str) -> str:
    """收紧元数据中的排版空白，不改写 raw_text 的换行或表格结构。"""
    return re.sub(r"\s+", " ", text).strip()


def _is_layout_noise_chunk(
    *, raw_text: str,
    section: str | None,
    raw_token_count: int,
) -> bool:
    """识别无章节、孤立的出版类型标签，避免其污染向量检索。

    这不是按 token 长度笼统删除短块。只有文本规范化后精确等于常见出版类型标签，
    同时没有章节路径且不超过 4 tokens 时才过滤，因而不会丢弃短公式、图注、关键词
    或正文中有实际意义的简短段落。
    """
    normalized = _normalize_whitespace(raw_text).casefold()
    return (
        section is None
        and raw_token_count <= 4
        and normalized in {"article", "research article", "review article"}
    )
