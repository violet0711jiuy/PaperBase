"""PDF解析器所共用的接口。"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PageFurniture:
    """从正文中分离出的页眉或页脚信息。

    页眉、页脚通常不应参与 Markdown、分块和检索，否则每一页重复的期刊名、卷期或
    页码会放大无关词频。但其中的期刊名、卷期、文章号仍可用于来源追溯，因此以独立
    结构保留，而不是直接丢弃。纯分页号（例如 ``3 of 11``）不写入此字段。
    """

    page_no: int
    location: str
    text: str


@dataclass(frozen=True)
class FrontMatterBlock:
    """论文前置元数据的标准化语义块。

    PDF 出版社的首页版式差异很大：作者和单位通常没有显式标题，摘要可能写成
    ``Abstract:`` 行内标签，引用格式或代码可用性又可能被错误标为普通章节。这里不把
    它们伪装成正文目录，而是使用稳定的 ``block_type`` 与 ``canonical_section`` 表达
    业务语义。后续分块器、SQLite 和检索器只依赖该契约，不依赖 Docling、MinerU 等
    具体工具的标签名称。

    ``text`` 始终保存解析得到的原始可读文本，不由模型补全或改写；
    ``detection_method`` 和 ``confidence`` 用于人工审计误识别案例。
    """

    block_type: str
    canonical_section: str
    text: str
    page_start: int | None
    page_end: int | None
    source_item_count: int
    detection_method: str
    confidence: str


@dataclass(frozen=True)
class ParsedPaper:
    """PaperBase 对一次论文解析的解析器无关结果。

    后续 Chunk、检索与存储层只消费 ``markdown``、标题元数据、``front_matter`` 和
    ``diagnostics``，不依赖任何第三方解析库的对象类型。这样未来接入 MinerU 或其他
    解析器时，只需新增适配器并保持这些字段的语义不变。

    ``native_document`` 仅供当前解析器对应的检查工具和调试代码使用。例如 Docling
    适配器会在此保存 ``DoclingDocument``。业务层不得依赖其类型，否则会重新形成
    对某一种解析工具的耦合。
    """

    source: Path
    parser_id: str
    markdown: str
    page_furniture: tuple[PageFurniture, ...]
    front_matter: tuple[FrontMatterBlock, ...]
    paper_title: str | None
    title_source: str
    title_candidates: tuple[str, ...]
    diagnostics: Mapping[str, int | str]
    native_document: object


@runtime_checkable
class PaperParser(Protocol):
    """PaperBase 论文解析器的可替换接口。

    每个解析器使用自身所需的模型、OCR 或版面分析方案，但必须产出 ``ParsedPaper``。
    ``parser_id`` 用于写入 SQLite 元数据和解析报告，保证不同工具的结果可以并存、
    对比并回溯来源。
    """

    parser_id: str

    def parse(self, source: Path) -> ParsedPaper:
        """解析一个本地 PDF，并返回符合统一契约的结果。"""
