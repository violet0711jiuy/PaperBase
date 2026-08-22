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
class FrontMatterHeading:
    """被 Parser 确认属于前置元数据的 heading provenance。

    ``FrontMatterBlock`` 保存可检索的文本内容；这个轻量记录保存标题自身在
    Docling reading order 中的位置、页码和稳定语义类型。Section Tree 与 Chunker
    只消费这份 Parser 决议，不需要各自重新猜测出版信息标题。
    """

    heading_text: str
    block_type: str
    canonical_section: str
    page_start: int | None
    page_end: int | None
    reading_order: int


@dataclass(frozen=True)
class SectionRecord:
    """论文章节树中的一个真实 heading 节点。

    该模型只保存解析器已经识别出的章节事实：标题文本、编号、层级、父子关系、阅读
    顺序与页码。Step 1 不负责从 Docling 构造这些节点；后续 Parser 实现只需填充本
    契约即可。即使父节点没有直属正文 chunk，也必须保留该节点，才能表达完整目录树。
    """

    # ``section_id`` 由后续构建器按论文和真实 heading 稳定生成，并在整个项目中唯一。
    section_id: str
    # ``paper_id`` 与 documents/chunks 的论文身份保持一致，方便后续 SQLite 建立外键。
    paper_id: str
    # 真实解析到的 heading 文本；不能由程序根据编号或路径拼造。
    section_title: str
    # 例如 ``3``、``3.1``、``3.1.2``；没有可靠编号时保留 None。
    section_number: str | None
    # 根章节为 1，子章节递增；不以 Markdown 的 # 数量作为唯一事实来源。
    section_level: int
    # 根节点为 None；子节点只指向同一 paper 的直属父章节。
    parent_section_id: str | None
    # heading 在该论文原始阅读顺序中的从零开始索引。
    section_index: int
    # 来源页码可缺失，且后续应由真实 heading provenance 填充。
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class ParsedPaper:
    """PaperBase 对一次论文解析的解析器无关结果。

    后续 Chunk、检索与存储层只消费 ``markdown``、标题元数据、``front_matter``、
    ``sections`` 和 ``diagnostics``，不依赖任何第三方解析库的对象类型。这样未来接入
    MinerU 或其他解析器时，只需新增适配器并保持这些字段的语义不变。

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
    # 默认空元组保持旧 Parser 与测试夹具兼容；DoclingParser 在 Step 2 会实际填充章节树。
    sections: tuple[SectionRecord, ...] = ()
    # 保留被 Front Matter Resolver 确认的 heading provenance，供正文树和 Chunker 消费。
    front_matter_headings: tuple[FrontMatterHeading, ...] = ()


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
