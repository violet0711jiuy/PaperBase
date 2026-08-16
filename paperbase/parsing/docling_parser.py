"""基于Docling的原生数字研究类PDF解析器。

该解析器在最小可行产品版本中特意关闭了光学字符识别（OCR）与图片理解功能。
此阶段提供的研究论文已具备可选中的文本；布局、表格、图注、标题以及来源信息依旧由Docling提供。
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    PdfPipelineOptions,
)
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.io import DocumentStream

from .base import FrontMatterBlock, PageFurniture, PaperParser, ParsedPaper


# 标题 resolver 只识别“出版栏模式”，绝不维护具体期刊名白名单。键已去除大小写、
# 空白和标点，因而可覆盖 ``Contents lists available ...``、``DOI: ...`` 等常见首页
# 版式噪声。期刊名本身不会因出现在本表而被排除，而会由其后方的 journal-homepage
# 结构信号降分。
_TITLE_NON_TITLE_PREFIXES = (
    "contentslistsavailable",
    "journalhomepage",
    "articleinfo",
    "articleinformation",
    "abstract",
    "summary",
    "keywords",
    "keyword",
    "indexterms",
    "received",
    "revised",
    "accepted",
    "published",
    "doi",
    "issn",
    "copyright",
    "license",
    "creativecommons",
    "referenceformat",
    "publicationinformation",
    "citation",
    "artifactavailability",
    "codeavailability",
    "dataavailability",
    "availability",
    "rightsandlicense",
    "authorsandaffiliations",
    "authorinformation",
    "correspondence",
)

# DOI、ISSN、URL、版权等即使不处于标准行内标签位置，也属于出版元数据而非论文标题。
_TITLE_PUBLICATION_METADATA = re.compile(
    r"(?:\bdoi\s*[:/]|\bissn\b|\b(?:copyright|license|licence)\b|"
    r"\bcreative\s+commons\b|https?://|www\.)",
    re.IGNORECASE,
)

# 作者行的识别仅用于标题候选的下游结构加分，不修改 authors / affiliations 的既有
# 识别逻辑。要求至少两个逗号分隔的人名，避免把普通 Title Case 论文题目误判为作者。
_TITLE_AUTHOR_NAME = re.compile(
    r"\b[A-Z][A-Za-z'’.-]{1,}(?:\s+[A-Z][A-Za-z'’.-]{1,})+\b"
)

# Granite-Docling 在带编号的显示公式后偶尔会继续生成相邻正文。编号在科研论文中
# 位于公式末尾，因此只在已经识别出这种末尾编号时截断；无编号公式绝不猜测边界。
_TRAILING_TEXT_AFTER_EQUATION_NUMBER = re.compile(
    r"(\\quad\s*\(\s*(?:\d\s*)+\)).*$", re.DOTALL
)

# 公式编号通常由纯数字括号组成，例如 ``(2)``。该正则故意不接受字母或章节号，
# 以免把普通表格的文本列错误转换为公式。
_EQUATION_NUMBER = re.compile(r"\(\s*(\d+)\s*\)")

# 出版社首页常把全大写栏目名称按字母拉开，例如 ``A R T I C L E  I N F O``。
# 此正则只识别「两个及以上、以空白分隔的大写字母」，不处理普通正文中的单词。
_SPACED_CAPITAL_WORD = re.compile(r"[A-Z](?: [A-Z])+")

# 含数字的章节标题在不同出版社 PDF 中可能写成 ``1. Introduction`` 或
# ``1 Introduction``；这里仅用于识别首页的 Introduction 锚点，不用于改写标题。
_INTRODUCTION_HEADING = re.compile(
    r"^(?:(?:\d+|[ivxlcdm]+)\.?\s+)?introduction$", re.IGNORECASE
)

# 摘要标题左侧与其正文左侧通常对齐。允许少量 PDF 坐标误差，但不把右栏的正文
# 续写（其左边界通常更靠右）误归入摘要。
_ABSTRACT_COLUMN_ALIGNMENT_TOLERANCE = 36.0

# U+00AD 是 PDF 为自动换行而插入的 soft hyphen（软连字符）。它不是单词本身的
# 连字符；Docling 提取文本时可能保留为不可见字符，或在其后带一个普通空格。
_SOFT_HYPHEN_AND_FOLLOWING_SPACE = re.compile(r"\u00ad\s*")

# 有些 PDF 不使用 U+00AD，而是直接在行尾保留普通 ASCII 连字符。例如
# ``humid-\nity``、``low-pres-\nsure``。Markdown 导出时它们会变成看似正常、实际
# 被断开的单词，进而同时影响 raw_text、FTS5 词法检索和 embedding。该正则只处理
# "英文字母 + 连字符 + 真实换行 + 小写英文字母" 这一种明确的版面断行形态：公式中的
# ``n-1``、负数、URL、Markdown 表格列和普通行内连字符均不会命中。
_HARD_LINE_BREAK_HYPHEN = re.compile(
    r"(?P<left>[A-Za-z]+(?:-[A-Za-z]+)*)-"
    r"[ \t]*\r?\n(?:[ \t]*\r?\n)?[ \t]*"
    r"(?P<right>[a-z][A-Za-z]*)"
)

# 同一自然段有时被 PDF 文本层拆成两个相邻 Docling TextItem：前一项以 ``humid-``
# 结束，后一项从 ``ity`` 开始。这种跨对象断词不会命中上面的“同一字符串内换行”规则，
# 因而需在保留 Docling 阅读顺序的前提下单独拼接。
_TRAILING_HARD_WORD_BREAK = re.compile(
    r"(?P<left>[A-Za-z]+(?:-[A-Za-z]+)*)-$"
)
_LEADING_LOWERCASE_WORD = re.compile(r"^\s*(?P<right>[a-z][A-Za-z]*)")

# 仅凭 PDF 文本层无法 100% 分辨 ``forecast-\ning``（应去掉连字符）和
# ``model-\nbased``（应保留连字符）。为避免用激进规则损坏原有术语，以下集合列出在
# 学术英语中高度稳定、通常构成真正复合词的后半部分；遇到它们时仅移除排版换行并保留
# 连字符。其余候选会再经过 `_is_likely_word_split` 的高置信度判断。
_COMMON_HYPHENATED_CONTINUATIONS = frozenset(
    {
        "based",
        "dependent",
        "driven",
        "related",
        "aware",
        "specific",
        "level",
        "term",
        "scale",
        "state",
        "time",
        "free",
        "less",
        "like",
        "wise",
        "pressure",
        "speed",
        "resolution",
        "dimensional",
        "linear",
        "linearized",
        "end",
    }
)

# 这些后缀出现在换行后的单词开头时，前半部分与后半部分拼接为一个单词的置信度很高。
# 例如 ``humid-\nity``、``regulari-\nzation``、``forecast-\ning``。这是通用英语形态规则，
# 不依赖某篇论文、作者或数据集名称。
_WORD_CONTINUATION_SUFFIXES = (
    "ability",
    "ation",
    "ations",
    "ibility",
    "ically",
    "ication",
    "ications",
    "ified",
    "ification",
    "ifications",
    "ifying",
    "ing",
    "ities",
    "ity",
    "ization",
    "izations",
    "ized",
    "izing",
    "ment",
    "ments",
    "ness",
    "sion",
    "sions",
    "tion",
    "tions",
)

# 当行前半部分正好以这些常见的英语构词前缀结束时，换行后的长词通常是同一单词的
# 延续。例如 ``fore-\ncasting``、``inter-\npretability``。此列表只放入独立前缀，
# 不把任意短词都视为可拼接片段，从而降低误改复合词的风险。
_WORD_SPLIT_PREFIXES = frozenset(
    {
        "fore",
        "inter",
        "intra",
        "macro",
        "micro",
        "multi",
        "semi",
    }
)

# 审稿或预印本 PDF 有时保留了不可见的旧文本层。该标记并非正式论文内容；但只有与
# 同页、同坐标带的碎片共同出现时，才将它们视为可删除的叠加层，不能凭普通重复文本猜测。
_PEER_REVIEW_MARKER = re.compile(r"\bFOR\s+PEER\s+REVIEW\b", re.IGNORECASE)

# 纯分页号没有检索或元数据价值。例如 ``3 of 11``、``Page 3`` 和 ``3 / 11`` 都不保留。
_PAGE_COUNTER = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE
)

# 少数 PDF 会将两个相邻的编号标题合为一个 section_header。只有第二个位置明确以
# 三级以内的数字编号和大写字母标题起始时才切分，避免把正文中的数字表达式拆坏。
_MERGED_NUMBERED_HEADING_BOUNDARY = re.compile(
    r"\s+(?=(?:\d+\.){1,3}\s+[A-Z])"
)
_NUMBERED_HEADING = re.compile(r"^(?:\d+\.){1,3}\s+\S.*$")

# 图表标题通常以 ``Figure 1.`` 或 ``Table 2.`` 开始。只有同一 caption 内第二次出现
# 与开头完全相同的编号标记时，才进一步检查是否为文本层重复，而不处理正文中的交叉引用。
_CAPTION_START = re.compile(r"^(?P<kind>Figure|Table)\s*(?P<number>\d+)\.", re.IGNORECASE)

# 当跨页表格与下一页页眉重叠时，页码有时只剩 ``8 of`` 并附着在单元格末尾。它缺少
# 分母和后续名词，不是完整的学术短语；只删除这种悬空残片，绝不据此重排表格行列。
_DANGLING_PAGE_COUNTER_IN_TABLE = re.compile(r"\s+\d+\s+of\s*$", re.IGNORECASE)

# 论文贡献、优点或步骤常以 ``1) ...``、``2) ...`` 形式列在正文中。版面模型偶尔会
# 把第一条误判为章节；仅凭括号编号不足以改写，因此还必须在后续正文找到连续的下一项。
_LIST_STYLE_ITEM = re.compile(r"^(?P<number>\d+)\)\s+(?P<content>.+)$")

# Docling 偶尔会把 running header/footer 标为普通 text。只有同一文本跨至少两页、坐标
# 稳定且落在页面边缘时才将其视为页眉/页脚，避免把跨页重复的正文或表格表头误删。
_RUNNING_FURNITURE_EDGE_RATIO = 0.12
_RUNNING_FURNITURE_POSITION_TOLERANCE = 2.0
_RUNNING_FURNITURE_MIN_PAGES = 2

# 前置元数据并非正文目录。无论不同出版社把它写成全大写标题、带冒号的行内标签，
# 还是完全不加标题，PaperBase 都使用这组稳定类型表达其业务语义。映射键会先删除
# 空白、标点和大小写差异，因此可同时识别 ``PVLDBReference Format:`` 与
# ``PVLDB Reference Format`` 等版式写法。
_FRONT_MATTER_HEADING_TYPES = {
    "abstract": "abstract",
    "summary": "abstract",
    "keywords": "keywords",
    "keyword": "keywords",
    "keywordsandphrases": "keywords",
    "indexterms": "keywords",
    "articleinfo": "article_info",
    "articleinformation": "article_info",
    "authorinformation": "authors_affiliations",
    "authorsandaffiliations": "authors_affiliations",
    "pvdblreferenceformat": "publication_info",
    "pvldbreferenceformat": "publication_info",
    "referenceformat": "publication_info",
    "publicationinformation": "publication_info",
    "citation": "publication_info",
    "citationinformation": "publication_info",
    "pvdblartifactavailability": "availability",
    "pvldbartifactavailability": "availability",
    "artifactavailability": "availability",
    "codeavailability": "availability",
    "dataavailability": "availability",
    "availability": "availability",
    "rightsandlicense": "rights",
}

# 这些名称既会写入 Markdown 的可读标题，也会成为未来分块和 SQLite 使用的统一 section。
# 中文查询由 embedding 模型处理；这里保留英文论文的标准学术标签，避免混入模型生成内容。
_FRONT_MATTER_SECTION_NAMES = {
    "authors_affiliations": "Front matter > Authors and affiliations",
    "correspondence": "Front matter > Correspondence",
    "abstract": "Front matter > Abstract",
    "keywords": "Front matter > Keywords",
    "article_info": "Front matter > Article information",
    "publication_info": "Front matter > Publication information",
    "availability": "Front matter > Availability",
    "rights": "Front matter > Rights and license",
}

_FRONT_MATTER_HEADING_TEXT = {
    "authors_affiliations": "Authors and affiliations",
    "correspondence": "Correspondence",
    "abstract": "Abstract",
    "keywords": "Keywords",
    "article_info": "Article information",
    "publication_info": "Publication information",
    "availability": "Availability",
    "rights": "Rights and license",
}

# ``Abstract: ...`` 和 ``Keywords: ...`` 在不少单栏 PDF 中是普通文本而不是标题。
# 这里仅识别首页常见的两种强标签；不通过内容长度或语言模型猜测未标注摘要。
_INLINE_CONTENT_LABEL = re.compile(
    r"^(?P<label>abstract|keywords?|key\s+words?|index\s+terms?)\s*:\s*(?P<content>\S.+)$",
    re.IGNORECASE | re.DOTALL,
)

# 出版信息可能在首页侧栏中被 Docling 排到 Introduction 之后。它们不安全地移动回
# Markdown 顶部，而是提取为独立 front_matter 块，后续由 metadata chunk 单独消费。
_INLINE_PUBLICATION_LABELS = {
    "citation": "publication_info",
    "academiceditor": "publication_info",
    "received": "publication_info",
    "revised": "publication_info",
    "accepted": "publication_info",
    "published": "publication_info",
    "copyright": "rights",
    "license": "rights",
}

# 作者单位块的判定不能依赖具体人名；只在标题后的有限区间中，同时看到机构/通讯标识
# 时才创建该语义块。这样可避免把没有明确边界的长摘要或正文误归为作者信息。
_AFFILIATION_SIGNAL = re.compile(
    r"\b(?:academy|college|department|faculty|institute|laboratory|school|university|"
    r"correspondence|e-?mail)\b|@",
    re.IGNORECASE,
)
_MAX_AUTHORSHIP_BLOCK_CHARS = 1800


@dataclass(frozen=True)
class _TitleResolverWeights:
    """标题候选的可审计评分参数；集中定义，避免散落的 magic numbers。"""

    title_label: int = 42
    section_header_label: int = 20
    text_label: int = 3
    first_page: int = 2
    early_reading_order: int = 4
    title_like_word_shape: int = 3
    overly_long_text_shape_penalty: int = 3
    preceding_contents_penalty: int = 4
    author_text_candidate_penalty: int = 34
    affiliation_text_candidate_penalty: int = 18
    downstream_publication_furniture_penalty: int = 38
    downstream_author_signal: int = 12
    downstream_affiliation_signal: int = 14
    downstream_abstract_or_keywords: int = 10
    complete_author_front_matter_chain: int = 16
    minimum_confidence: int = 45
    minimum_margin: int = 6


_TITLE_RESOLVER_WEIGHTS = _TitleResolverWeights()


@dataclass(frozen=True)
class _TitleCandidate:
    """一个未改写的 Docling 原始 item 在首页阅读顺序中的标题候选视图。"""

    item: object
    text: str
    label: object
    reading_order: int
    pages: tuple[int, ...]


@dataclass(frozen=True)
class _ScoredTitleCandidate:
    """候选的确定性评分结果；拒绝原因用于测试和人工排查。"""

    candidate: _TitleCandidate
    score: int
    rejection_reason: str | None


@dataclass(frozen=True)
class _TitleResolution:
    """标题 resolver 的内部结果；不向下游暴露或修改 Docling 原始节点。"""

    paper_title: str | None
    title_source: str
    title_candidates: tuple[str, ...]
    ranked_candidates: tuple[_ScoredTitleCandidate, ...]


def _normalize_pdf_word_breaks_in_text(text: str) -> str:
    """修复一个非公式文本项内的软/硬连字符断行。

    这个函数是纯文本函数，单元测试可以不加载 Docling 模型即可覆盖。返回值只会修复
    明确存在的排版换行，不会重写正常句子、增加模型生成内容，亦不会把表格或公式中的
    数值表达式作为英文单词处理。
    """
    # U+00AD 的含义就是“此处允许断行”，不属于词语本身，因此可以无条件删除。
    normalized = _SOFT_HYPHEN_AND_FOLLOWING_SPACE.sub("", text)

    def replace_hard_line_break(match: re.Match[str]) -> str:
        left = match.group("left")
        right = match.group("right")
        if _is_likely_word_split(left=left, right=right):
            # 例如 humid-\nity -> humidity；低置信度情形不会走到这里。
            return f"{left}{right}"
        # 例如 model-\nbased -> model-based。虽然换行被移除，但原始复合词语义保留。
        return f"{left}-{right}"

    return _HARD_LINE_BREAK_HYPHEN.sub(replace_hard_line_break, normalized)


def _is_likely_word_split(*, left: str, right: str) -> bool:
    """判断 ASCII 行末连字符是否高置信度地属于同一英文单词。

    这不是词典或 LLM 猜测：只有明显的构词后缀，或“一个已开始的连字符复合词中的
    第二个词又被排版拆开”时才删除连字符。其余模糊情况保留连字符，宁可得到
    ``fore-casting``，也不把原本正确的 ``model-based`` 破坏成 ``modelbased``。
    """
    normalized_right = right.casefold()
    if normalized_right in _COMMON_HYPHENATED_CONTINUATIONS:
        return False

    # ``state-of-the-\nart`` 已有两个真实连字符，应保留最后一个；
    # ``low-pres-\nsure`` 则只有一个已有连字符，且最后的 ``pres`` 显然是
    # ``pressure`` 的半个词，可安全拼回 ``low-pressure``。
    hyphen_count = left.count("-")
    trailing_fragment = left.rsplit("-", maxsplit=1)[-1]
    if hyphen_count == 1 and len(trailing_fragment) >= 3 and len(right) <= 4:
        return True
    if hyphen_count >= 2:
        return False

    if trailing_fragment.casefold() in _WORD_SPLIT_PREFIXES:
        return True

    # ``ity``、``tion``、``ing`` 等是学术英语里极稳定的词尾，能避免将
    # humid-\nity、forecast-\ning、normaliza-\ntion 留成两个检索词。
    return normalized_right.startswith(_WORD_CONTINUATION_SUFFIXES)


def _join_adjacent_hard_word_break(
    *, left_text: str, right_text: str
) -> str | None:
    """尝试连接两个相邻文本项之间被 PDF 拆开的一个单词。

    返回 ``None`` 表示两项之间不存在“行末连字符 + 下一项小写字母”的强证据；调用方
    必须保持两个原对象完全不动。成功时返回合并后的完整文本，右侧文本项可安全删除。
    这里不会猜测两个普通句子是否相关，也不会跨标题、公式、表格或列表项连接。
    """
    left_match = _TRAILING_HARD_WORD_BREAK.search(left_text)
    right_match = _LEADING_LOWERCASE_WORD.match(right_text)
    if left_match is None or right_match is None:
        return None

    left = left_match.group("left")
    right = right_match.group("right")
    connector = "" if _is_likely_word_split(left=left, right=right) else "-"
    # right_match.start("right") 之后保留右侧所有原始字符，包括逗号、空格和句子余量。
    # 因此只修复边界的一个词，不会吞掉后续内容或重排句子。
    return (
        left_text[: left_match.start("left")]
        + left
        + connector
        + right_text[right_match.start("right") :]
    )


@dataclass(frozen=True)
class ParserSettings:
    """运行本地Docling模型所需的非机密设置。"""

    artifacts_path: Path
    device: str = "cuda"
    enable_ocr: bool = False
    enable_formula_enrichment: bool = True
    formula_preset: str = "granite_docling"
    compile_layout_model: bool = False
    compile_formula_model: bool = False
    remove_page_furniture: bool = True
    remove_peer_review_artifacts: bool = True
    list_style_heading_min_chars: int = 80
    enable_front_matter_recognition: bool = True
    front_matter_max_pages: int = 2


class DoclingParser(PaperParser):
    """Docling 对 ``PaperParser`` 接口的当前实现。"""

    # 这是写入统一结果的稳定解析器标识，而不是 Docling 版本号或模型名。模型预设
    # 等可变参数放入 diagnostics，便于同一解析器的不同配置结果并排比较。
    parser_id = "docling"

    def __init__(self, settings: ParserSettings) -> None:
        self._settings = settings
        self._converter = self._build_converter()

    def parse(self, source: Path) -> ParsedPaper:
        """返回 ``source`` 的结构化文档及标题元数据。

        采用字节流而非直接传入路径，是因为当前的 docling-parse 后端无法可靠打开
        包含中文字符的 Windows 路径。原始 PDF 文件不会被复制或修改。
        """
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"PDF not found: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {source.name}")

        stream = DocumentStream(name="paper.pdf", stream=BytesIO(source.read_bytes()))
        document = self._converter.convert(stream).document
        corrected_formula_table_count = self._convert_formula_like_tables(document)
        self._clean_enriched_formula_text(document)
        normalized_hyphenation_count = self._normalize_pdf_word_breaks(document)
        normalized_heading_count = self._normalize_heading_spacing(document)
        split_merged_heading_count = self._split_merged_numbered_headings(document)
        converted_list_style_heading_count = (
            self._convert_misclassified_list_style_headings(
                document,
                min_chars=self._settings.list_style_heading_min_chars,
            )
        )
        corrected_reading_order_count = self._repair_first_page_abstract_order(document)
        if self._settings.remove_page_furniture:
            # 必须在删除审稿叠加层之前提取。某些异常 PDF 会让两个文本层共用容器，
            # 先保存已可靠标注的页眉/页脚，才能避免元数据随叠加层一起成为孤立节点。
            page_furniture, removed_page_furniture_count = self._extract_page_furniture(
                document
            )
        else:
            page_furniture = ()
            removed_page_furniture_count = 0

        if self._settings.remove_peer_review_artifacts:
            (
                removed_peer_review_artifact_count,
                normalized_page_counter_prefix_count,
            ) = self._remove_peer_review_artifacts(document)
        else:
            removed_peer_review_artifact_count = 0
            normalized_page_counter_prefix_count = 0
        removed_table_page_counter_fragment_count = (
            self._remove_dangling_table_page_counter_fragments(document)
        )
        deduplicated_caption_count = self._deduplicate_repeated_captions(document)
        paper_title, title_source, title_candidates = self._extract_title(
            document,
            max_pages=self._settings.front_matter_max_pages,
        )
        if self._settings.enable_front_matter_recognition:
            front_matter = self._normalize_and_extract_front_matter(
                document,
                paper_title=paper_title,
                max_pages=self._settings.front_matter_max_pages,
            )
        else:
            front_matter = ()

        diagnostics: dict[str, int | str] = {
            # 诊断字段全部采用解析器前缀，避免未来不同工具使用同名指标时语义混淆。
            "docling.corrected_formula_table_count": corrected_formula_table_count,
            "docling.normalized_hyphenation_count": normalized_hyphenation_count,
            "docling.normalized_heading_count": normalized_heading_count,
            "docling.split_merged_heading_count": split_merged_heading_count,
            "docling.converted_list_style_heading_count": (
                converted_list_style_heading_count
            ),
            "docling.corrected_reading_order_count": corrected_reading_order_count,
            "docling.removed_peer_review_artifact_count": (
                removed_peer_review_artifact_count
            ),
            "docling.normalized_page_counter_prefix_count": (
                normalized_page_counter_prefix_count
            ),
            "docling.removed_table_page_counter_fragment_count": (
                removed_table_page_counter_fragment_count
            ),
            "docling.deduplicated_caption_count": deduplicated_caption_count,
            "docling.removed_page_furniture_count": removed_page_furniture_count,
            "docling.extracted_page_furniture_count": len(page_furniture),
            "docling.front_matter_block_count": len(front_matter),
            "docling.formula_preset": self._settings.formula_preset,
        }
        for block_type in _FRONT_MATTER_SECTION_NAMES:
            diagnostics[f"docling.front_matter_{block_type}_count"] = sum(
                block.block_type == block_type for block in front_matter
            )
        return ParsedPaper(
            source=source,
            parser_id=self.parser_id,
            markdown=document.export_to_markdown(),
            page_furniture=page_furniture,
            front_matter=front_matter,
            paper_title=paper_title,
            title_source=title_source,
            title_candidates=title_candidates,
            diagnostics=diagnostics,
            native_document=document,
        )

    def _build_converter(self) -> DocumentConverter:
        options = PdfPipelineOptions()
        options.artifacts_path = self._settings.artifacts_path
        options.do_ocr = self._settings.enable_ocr
        # CodeFormulaV2 会从公式所在的页面图像恢复 LaTeX 结构，避免 PDF 文本层
        # 将上下标、分式和希腊字母压扁为普通字符。该功能只处理代码/公式对象，
        # 不会开启图片描述或多模态 RAG。
        options.do_formula_enrichment = self._settings.enable_formula_enrichment
        # 根据预设选择公式模型。当前只暴露 Docling 官方的两个公式预设，便于用
        # 同一篇论文对比识别质量；不把模型仓库名称硬编码在业务调用方。
        formula_options = CodeFormulaVlmOptions.from_preset(
            self._settings.formula_preset
        )
        formula_options.extract_code = False

        # AutoInline 引擎会在 Windows 上隐式启用 torch.compile，继而触发 GBK
        # 编码错误。这里显式选用 Transformers 引擎，并将编译开关单独配置，
        # 不影响布局模型及其他 Docling 模块的推理方式。
        formula_options.engine_options = TransformersVlmEngineOptions(
            device=self._settings.device,
            compile_model=self._settings.compile_formula_model,
        )
        options.code_formula_options = formula_options

        options.do_picture_classification = False
        options.do_picture_description = False
        options.do_chart_extraction = False
        options.generate_page_images = False
        options.generate_picture_images = False
        options.layout_options.engine_options.compile_model = (
            self._settings.compile_layout_model
        )
        options.accelerator_options.device = self._settings.device

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    @staticmethod
    def _extract_title(
        document: DoclingDocument,
        *,
        max_pages: int = 2,
    ) -> tuple[str | None, str, tuple[str, ...]]:
        """从首页候选中选择可信标题，而不改写 Docling 原始 item 或阅读顺序。"""
        resolution = _resolve_title_from_document(document, max_pages=max_pages)
        return (
            resolution.paper_title,
            resolution.title_source,
            resolution.title_candidates,
        )

    @staticmethod
    def _clean_enriched_formula_text(document: DoclingDocument) -> None:
        """清理公式模型可确定为非公式的生成尾部。

        Docling 的 ``orig`` 字段始终保留 PDF 文本层原文；这里只清理供 Markdown、
        Chunking 与后续 RAG 使用的富化 LaTeX ``text``。规则刻意保守，避免把
        无编号长公式的有效内容误删。
        """
        for item, _level in document.iterate_items():
            if item.label != DocItemLabel.FORMULA:
                continue

            formula_text = (getattr(item, "text", "") or "").strip()
            if not formula_text:
                continue

            # 模型偶尔会把内部结束标记直接写入文本；标记及其后的内容均不是 LaTeX。
            formula_text = formula_text.split("</formula", maxsplit=1)[0].strip()

            # 仅当末尾存在论文常用的 "\\quad (12)" 编号时，才移除编号后的内容。
            # 这样能够处理公式区域与相邻段落发生轻微重叠的情况。
            formula_text = _TRAILING_TEXT_AFTER_EQUATION_NUMBER.sub(
                r"\1", formula_text
            ).strip()
            item.text = formula_text

    @staticmethod
    def _convert_formula_like_tables(document: DoclingDocument) -> int:
        """将被误判为两列表格的连续编号公式转换回 FormulaItem。

        科研论文常把显示公式左对齐、编号右对齐。版面模型可能因此识别为两列表格。
        本规则必须同时满足严格的六项结构条件，才会执行转换；不能确认的表格一律保持
        原样，优先避免损坏真实实验结果表。
        """
        converted_count = 0
        # 先物化列表，再删除项目，避免迭代过程中改变 Docling 文档树。
        tables = [
            item
            for item, _level in document.iterate_items()
            if item.label == DocItemLabel.TABLE
        ]

        for table in tables:
            formula_rows = _formula_rows_from_pseudo_table(table)
            if not formula_rows:
                continue

            for formula_text, provenance in formula_rows:
                # 逐行插入到原表格之前，使阅读顺序与 PDF 中由上至下的公式顺序一致。
                # ``orig`` 保留来自表格单元格的原始文本，方便后续人工追溯。
                document.insert_formula(
                    sibling=table,
                    text=formula_text,
                    orig=formula_text,
                    prov=provenance,
                    after=False,
                )
            document.delete_items(node_items=[table])
            converted_count += 1

        return converted_count

    @staticmethod
    def _normalize_heading_spacing(document: DoclingDocument) -> int:
        """修复栏目标题的字母间距，保留 ``orig`` 中可审计的 PDF 原文。

        此修复限定在标题和章节标题：例如 ``A B S T R A C T`` 变为 ``ABSTRACT``、
        ``A R T I C L E  I N F O`` 变为 ``ARTICLE INFO``。正文、作者姓名和公式不
        会经过该规则，从而避免把真正有语义的空格删除。
        """
        normalized_count = 0
        for item, _level in document.iterate_items():
            if item.label not in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}:
                continue

            original_text = (getattr(item, "text", "") or "").strip()
            normalized_text = _normalize_spaced_capital_heading(original_text)
            if normalized_text != original_text:
                item.text = normalized_text
                normalized_count += 1

        return normalized_count

    @staticmethod
    def _normalize_pdf_word_breaks(document: DoclingDocument) -> int:
        """在创建 Markdown 和 Chunk 前修复 PDF 自动换行造成的断词。

        PDF 的文本层常将同一单词拆成两行。常见的软连字符 ``U+00AD`` 可以无条件
        删除；普通 ASCII 连字符则存在歧义，因此交由
        `_is_likely_word_split` 采用保守的英语形态规则判断。无论是否
        去掉连字符，换行本身都会被消除，避免词法检索把一个词分成两段。

        此处修改的是 Docling 原生对象的 ``text``，所以 Markdown、Step 2 的
        ``raw_text``、SQLite FTS5 以及 embedding 会同步使用同一份干净文本；不会在
        各下游环节复制一套容易不一致的补丁。``orig`` 仍保留 Docling 的原始抽取内容
        以便审计。公式对象由独立的 LaTex 富化模型产生，绝不在这里改写。
        """
        normalized_count = 0
        for item, _level in document.iterate_items():
            if item.label == DocItemLabel.FORMULA:
                continue

            original_text = getattr(item, "text", "") or ""
            normalized_text = _normalize_pdf_word_breaks_in_text(original_text)
            if normalized_text != original_text:
                item.text = normalized_text
                normalized_count += 1

        # 文本层还可能把一个自然段拆为两个相邻 TextItem。这里仅接受前项以连字符结束、
        # 后项从小写英文词开始的强证据；标题、公式、表格或列表项会中断候选链，绝不会
        # 被跨越。合并后把右项 provenance 一并附到左项，页码追溯仍覆盖原始两端来源。
        previous_text_item: object | None = None
        items_to_delete: list[object] = []
        for item, _level in document.iterate_items():
            if getattr(item, "label", None) != DocItemLabel.TEXT:
                previous_text_item = None
                continue
            if previous_text_item is None:
                previous_text_item = item
                continue

            merged_text = _join_adjacent_hard_word_break(
                left_text=getattr(previous_text_item, "text", "") or "",
                right_text=getattr(item, "text", "") or "",
            )
            if merged_text is None:
                previous_text_item = item
                continue

            previous_text_item.text = merged_text
            previous_provenance = getattr(previous_text_item, "prov", None)
            if isinstance(previous_provenance, list):
                # provenance 是来源页与坐标的列表。把右项来源附给合并后的左项，确保
                # Step 2 计算 page_start/page_end 时仍能看到跨页句子的完整来源范围。
                previous_provenance.extend(getattr(item, "prov", []) or [])
            items_to_delete.append(item)
            normalized_count += 1

        if items_to_delete:
            document.delete_items(node_items=items_to_delete)
        return normalized_count


    @staticmethod
    def _split_merged_numbered_headings(document: DoclingDocument) -> int:
        """拆分一个对象中被拼接的两个明确编号章节标题。

        该问题来自版面分区边界，而不是 Markdown 导出。规则只处理
        ``SECTION_HEADER``，并要求文本中恰好有一个“空格 + 新编号标题”的边界；例如
        ``3.3. Result 3.3.1. Detail`` 会被拆成两个标题。正文、表格单元格与公式完全
        不经过此函数，避免把诸如 ``O(1.2)`` 之类的数学/统计表达式误拆。
        """
        split_count = 0
        headings = [
            item
            for item, _level in document.iterate_items()
            if item.label == DocItemLabel.SECTION_HEADER
        ]

        for heading in headings:
            heading_text = (getattr(heading, "text", "") or "").strip()
            split_text = _split_merged_numbered_heading(heading_text)
            if split_text is None:
                continue

            first_heading, second_heading = split_text
            provenance = getattr(heading, "prov", [])
            # Docling 的插入 API 一次只接受一条 provenance。合并标题在当前样本中
            # 来自同一页；若未来出现跨页标题，仍保留第一条来源以便可回溯。
            first_provenance = provenance[0] if provenance else None
            heading_level = getattr(heading, "level", 1)
            content_layer = getattr(heading, "content_layer", None)
            first_item = document.insert_heading(
                sibling=heading,
                text=first_heading,
                orig=first_heading,
                level=heading_level,
                prov=first_provenance,
                content_layer=content_layer,
                after=False,
            )
            document.insert_heading(
                sibling=first_item,
                text=second_heading,
                orig=second_heading,
                level=heading_level,
                prov=first_provenance,
                content_layer=content_layer,
                after=True,
            )
            document.delete_items(node_items=[heading])
            split_count += 1

        return split_count

    @staticmethod
    def _convert_misclassified_list_style_headings(
        document: DoclingDocument,
        *,
        min_chars: int,
    ) -> int:
        """将有连续列表证据的超长 ``N)`` 伪章节恢复为真正的 ListItem。

        部分 PDF 的贡献列表会跨页，第一项被布局模型标为 ``SECTION_HEADER``，后续项却
        已作为正文列表抽出。若直接相信该标签，分块会错误创建子章节，第一项还会从
        Markdown 正文中消失。转换需同时满足：

        1. 原对象确为 ``SECTION_HEADER``；
        2. 标题匹配 ``N) 内容``，且内容足够长，不像正常简短章节名；
        3. 在下一个真实章节之前、相邻的有限正文范围内出现 ``(N+1)`` 项。

        因此短的 ``1) Study area``、孤立的长标题或真正的 ``2)`` 章节不会被自动改写。
        原始 provenance、marker 与内容层均保留，使 Markdown、Chunking 和 Citation 都从
        统一的正确文档树读取，而不是在下游临时打补丁。
        """
        body_children = document.body.children
        root_items = [reference.resolve(document) for reference in body_children]
        replacements: list[tuple[object, str, str, str, object, object]] = []

        for index, item in enumerate(root_items):
            if getattr(item, "label", None) != DocItemLabel.SECTION_HEADER:
                continue
            text = _normalize_inline_whitespace(getattr(item, "text", "") or "")
            list_item = _LIST_STYLE_ITEM.fullmatch(text)
            if list_item is None or len(text) < min_chars:
                continue
            number = int(list_item.group("number"))
            if not _has_next_list_item(
                document=document,
                root_items=root_items,
                start_index=index,
                expected_number=number + 1,
            ):
                continue

            provenance = getattr(item, "prov", [])
            if not provenance:
                # 没有来源页码时无法为恢复后的列表提供可靠 Citation，因此保持原标签。
                continue
            original_text = (getattr(item, "orig", "") or text).strip()
            original_item = _LIST_STYLE_ITEM.fullmatch(
                _normalize_inline_whitespace(original_text)
            )
            original_content = (
                original_item.group("content") if original_item else list_item.group("content")
            )
            replacements.append(
                (
                    item,
                    list_item.group("number") + ")",
                    list_item.group("content"),
                    original_content,
                    provenance[0],
                    getattr(item, "content_layer", None),
                )
            )

        for item, marker, content, original_content, provenance, content_layer in replacements:
            document.insert_list_item(
                sibling=item,
                text=content,
                enumerated=True,
                marker=marker,
                orig=original_content,
                prov=provenance,
                content_layer=content_layer,
                after=False,
            )
            document.delete_items(node_items=[item])
        return len(replacements)

    @staticmethod
    def _normalize_and_extract_front_matter(
        document: DoclingDocument,
        *,
        paper_title: str | None,
        max_pages: int,
    ) -> tuple[FrontMatterBlock, ...]:
        """在不猜测正文含义的前提下，识别并标准化论文首页元数据。

        此处的目标不是把所有首页内容强行移动到同一位置，而是建立可审计的语义边界：

        - 已有明确标题的 ``ABSTRACT``、``PVLDBReference Format`` 等会统一命名；
        - ``Abstract:``、``Keywords:`` 这样的行内强标签会被提升为真正标题；
        - 作者与单位没有标题时，只从论文标题到下一个明确元数据/正文边界之间提取；
        - 因侧栏阅读顺序而出现在 Introduction 后的 Citation、Received 等信息仅作为独立
          front_matter 块记录，不在此阶段重排正文节点。

        后一条尤其重要：错误地移动节点会破坏 Citation 对应的 provenance 和正文续写顺序。
        """
        if not document.body.children:
            return ()

        heading_detection_methods: dict[int, str] = {}
        root_items = _root_body_items(document)

        # 先处理 Docling 已经识别为 section_header 的强标签。只扫描首页窗口，避免把
        # 论文结尾的 "Data availability" 等普通正文后记误认成首页前置元数据。
        for item in root_items:
            if (
                getattr(item, "label", None) != DocItemLabel.SECTION_HEADER
                or not _item_is_in_front_matter_window(document, item, max_pages)
            ):
                continue
            block_type = _front_matter_heading_type(getattr(item, "text", "") or "")
            if block_type is None:
                continue
            item.text = _FRONT_MATTER_HEADING_TEXT[block_type]
            heading_detection_methods[id(item)] = "explicit_heading"

        # Elsevier 的 ``ARTICLE INFO`` 常只是 Keywords 的容器。若其第一段确实以
        # ``Keywords:`` 开始，则提升为 Keywords，而不是留下一个语义过宽、且会和关键词
        # 重复的 Article information block。
        DoclingParser._promote_article_info_keyword_heading(
            document,
            heading_detection_methods=heading_detection_methods,
            max_pages=max_pages,
        )

        # 单栏 PDF 经常把 Abstract/Keywords 作为普通文本或 key-value area 的第一个子项。
        # 仅接受明确的行首标签，并保留原对象内容与 provenance；没有标签的长首段不猜测。
        for root_item in _root_body_items(document):
            if not _item_is_in_front_matter_window(document, root_item, max_pages):
                continue
            if getattr(root_item, "label", None) == DocItemLabel.SECTION_HEADER:
                continue
            content_item = _first_textual_descendant(document, root_item)
            if content_item is None:
                continue
            split_label = _split_inline_content_label(
                getattr(content_item, "text", "") or ""
            )
            if split_label is None:
                continue
            block_type, content = split_label
            provenance = _first_provenance(content_item) or _first_provenance(root_item)
            heading = document.insert_heading(
                sibling=root_item,
                text=_FRONT_MATTER_HEADING_TEXT[block_type],
                orig=_FRONT_MATTER_HEADING_TEXT[block_type],
                level=1,
                prov=provenance,
                content_layer=getattr(content_item, "content_layer", None),
                after=False,
            )
            # 只剥离已被精确识别的标签和冒号，正文的其余字词保持原样。
            content_item.text = content
            heading_detection_methods[id(heading)] = "inline_label"

        authors_heading = DoclingParser._insert_authorship_heading(
            document,
            paper_title=paper_title,
            max_pages=max_pages,
        )
        if authors_heading is not None:
            heading_detection_methods[id(authors_heading)] = "title_anchor_span"

        return DoclingParser._collect_front_matter_blocks(
            document,
            max_pages=max_pages,
            heading_detection_methods=heading_detection_methods,
        )

    @staticmethod
    def _promote_article_info_keyword_heading(
        document: DoclingDocument,
        *,
        heading_detection_methods: dict[int, str],
        max_pages: int,
    ) -> None:
        """把只承载 Keywords 的 Article information 容器收敛为精确关键词标题。"""
        root_items = _root_body_items(document)
        for index, item in enumerate(root_items):
            if (
                getattr(item, "label", None) != DocItemLabel.SECTION_HEADER
                or _front_matter_heading_type(getattr(item, "text", "") or "")
                != "article_info"
            ):
                continue
            for candidate in root_items[index + 1 :]:
                if getattr(candidate, "label", None) == DocItemLabel.SECTION_HEADER:
                    break
                if not _item_is_in_front_matter_window(document, candidate, max_pages):
                    continue
                content_item = _first_textual_descendant(document, candidate)
                if content_item is None:
                    continue
                split_label = _split_inline_content_label(
                    getattr(content_item, "text", "") or ""
                )
                if split_label is None:
                    # Article information 中若先出现了其他实质文本，就不再假定整个容器
                    # 只等于关键词，保留它的宽泛语义即可。
                    break
                block_type, content = split_label
                if block_type != "keywords":
                    break
                item.text = _FRONT_MATTER_HEADING_TEXT["keywords"]
                content_item.text = content
                heading_detection_methods[id(item)] = "article_info_keyword_label"
                break

    @staticmethod
    def _insert_authorship_heading(
        document: DoclingDocument,
        *,
        paper_title: str | None,
        max_pages: int,
    ) -> object | None:
        """在标题与下一强边界之间为无标题的作者/单位块添加标准标题。

        这里不尝试从人名中拆出作者实体或猜测上标映射；这些需要更强的版面与命名实体
        证据，留给后续受控增强。当前仅把作者、单位、邮箱/通讯方式作为不可分离的原始
        语义块，确保其中的单位编号仍能与作者上标一起被保留。
        """
        if not paper_title:
            return None
        root_items = _root_body_items(document)
        title_index = _find_root_title_index(root_items, paper_title)
        if title_index is None:
            return None

        candidate_roots: list[object] = []
        for candidate in root_items[title_index + 1 :]:
            if not _item_is_in_front_matter_window(document, candidate, max_pages):
                break
            if getattr(candidate, "label", None) == DocItemLabel.SECTION_HEADER:
                # 任意已有标题都构成作者块的右边界：它可能是 Abstract、Keywords，
                # 也可能是论文的第一个正文标题。这样不会把正文吞入作者信息。
                break
            content_item = _first_textual_descendant(document, candidate)
            if content_item is not None and _has_inline_front_matter_label(
                getattr(content_item, "text", "") or ""
            ):
                break
            candidate_roots.append(candidate)

        text, _pages, _item_count = _collect_root_content(document, candidate_roots)
        if (
            not text
            or len(text) > _MAX_AUTHORSHIP_BLOCK_CHARS
            or _AFFILIATION_SIGNAL.search(text) is None
        ):
            return None

        anchor = next(
            (
                candidate
                for candidate in candidate_roots
                if _first_textual_descendant(document, candidate) is not None
            ),
            None,
        )
        if anchor is None:
            return None
        content_item = _first_textual_descendant(document, anchor)
        return document.insert_heading(
            sibling=anchor,
            text=_FRONT_MATTER_HEADING_TEXT["authors_affiliations"],
            orig=_FRONT_MATTER_HEADING_TEXT["authors_affiliations"],
            level=1,
            prov=_first_provenance(content_item) or _first_provenance(anchor),
            content_layer=getattr(content_item, "content_layer", None),
            after=False,
        )

    @staticmethod
    def _collect_front_matter_blocks(
        document: DoclingDocument,
        *,
        max_pages: int,
        heading_detection_methods: dict[int, str],
    ) -> tuple[FrontMatterBlock, ...]:
        """从已标准化的标题和行内出版标签生成解析器无关的语义块。"""
        root_items = _root_body_items(document)
        records: list[tuple[int, FrontMatterBlock]] = []
        covered_indices: set[int] = set()

        for index, item in enumerate(root_items):
            if (
                getattr(item, "label", None) != DocItemLabel.SECTION_HEADER
                or not _item_is_in_front_matter_window(document, item, max_pages)
            ):
                continue
            block_type = _front_matter_heading_type(getattr(item, "text", "") or "")
            if block_type is None:
                continue

            # 下一个任意标题都是当前语义块的结束边界。使用任意标题而非只使用已知
            # front_matter 标题，可阻止 Abstract 误吸收 Introduction 的首段。
            end_index = next(
                (
                    next_index
                    for next_index in range(index + 1, len(root_items))
                    if getattr(root_items[next_index], "label", None)
                    == DocItemLabel.SECTION_HEADER
                ),
                len(root_items),
            )
            content_roots = root_items[index + 1 : end_index]
            covered_indices.update(range(index, end_index))
            if block_type != "availability":
                text, pages, item_count = _collect_root_content(document, content_roots)
                if not text:
                    continue
                records.append(
                    (
                        index,
                        FrontMatterBlock(
                            block_type=block_type,
                            canonical_section=_FRONT_MATTER_SECTION_NAMES[block_type],
                            text=text,
                            page_start=pages[0] if pages else None,
                            page_end=pages[-1] if pages else None,
                            source_item_count=item_count,
                            detection_method=heading_detection_methods.get(
                                id(item), "explicit_heading"
                            ),
                            confidence="high",
                        ),
                    )
                )
                continue

            # ``Artifact availability`` 等标题有时是出版社的宽泛容器：代码链接、
            # 通讯作者、许可证和期刊脚注连续出现。按可解释的内容标签再分一次，避免
            # 未来用户问“代码在哪里”时召回一大段无关版权文本。
            roots_by_type: dict[str, list[tuple[int, object]]] = {}
            for content_index, content_root in enumerate(content_roots, start=index + 1):
                text, _pages, _item_count = _collect_root_content(
                    document, [content_root]
                )
                if not text:
                    continue
                refined_type = _availability_content_block_type(text)
                roots_by_type.setdefault(refined_type, []).append(
                    (content_index, content_root)
                )
            for refined_type, indexed_roots in roots_by_type.items():
                text, pages, item_count = _collect_root_content(
                    document, [root for _root_index, root in indexed_roots]
                )
                records.append(
                    (
                        indexed_roots[0][0],
                        FrontMatterBlock(
                            block_type=refined_type,
                            canonical_section=_FRONT_MATTER_SECTION_NAMES[refined_type],
                            text=text,
                            page_start=pages[0] if pages else None,
                            page_end=pages[-1] if pages else None,
                            source_item_count=item_count,
                            detection_method=(
                                heading_detection_methods.get(id(item), "explicit_heading")
                                if refined_type == "availability"
                                else "container_content_pattern"
                            ),
                            confidence="high",
                        ),
                    )
                )

        # 对于首页侧栏的 Citation、Received、Copyright 等，Docling 的阅读顺序可能将其
        # 放在 Introduction 后。它们不是正文，也不应继承 Introduction section；这里以
        # 独立块保留，且跳过已包含在显式标题范围内的内容，避免重复。
        inline_roots_by_type: dict[str, list[tuple[int, object]]] = {}
        for index, item in enumerate(root_items):
            if index in covered_indices or not _item_is_in_front_matter_window(
                document, item, max_pages
            ):
                continue
            content_item = _first_textual_descendant(document, item)
            if content_item is None:
                continue
            block_type = _inline_publication_block_type(
                getattr(content_item, "text", "") or ""
            )
            if block_type is None:
                continue
            inline_roots_by_type.setdefault(block_type, []).append((index, item))

        for block_type, indexed_roots in inline_roots_by_type.items():
            text, pages, item_count = _collect_root_content(
                document, [item for _index, item in indexed_roots]
            )
            if not text:
                continue
            records.append(
                (
                    indexed_roots[0][0],
                    FrontMatterBlock(
                        block_type=block_type,
                        canonical_section=_FRONT_MATTER_SECTION_NAMES[block_type],
                        text=text,
                        page_start=pages[0] if pages else None,
                        page_end=pages[-1] if pages else None,
                        source_item_count=item_count,
                        detection_method="inline_label",
                        confidence="high",
                    ),
                )
            )

        # 解析报告和未来检索应遵循 PDF 的源对象顺序，而不是按 block_type 字母序排列。
        return tuple(block for _index, block in sorted(records, key=lambda record: record[0]))

    @staticmethod
    def _remove_peer_review_artifacts(document: DoclingDocument) -> tuple[int, int]:
        """删除有明确审稿标识支撑的不可见重叠文本层。

        部分期刊 PDF 的文本层同时含正式版本和审稿版本。正式渲染不可见的审稿层会被
        Docling 正常读取，造成 ``FOR PEER REVIEW``、卷期碎片以及旧正文重复。这里的
        删除条件刻意很严：必须先出现审稿标记，再只删除该标记同页、bbox 顶边几乎完全
        相同的文本碎片。这样不会因为两个合法段落字面相似而删除正文。

        对同一类页面中以分页号开头的正文片段，仅移除可确定的 ``N of M`` 前缀；其余
        正文仍保留。函数返回“删除的叠加文本项数”和“移除分页前缀的项数”。
        """
        if not document.body.children:
            return 0, 0

        root_items = [reference.resolve(document) for reference in document.body.children]
        marker_indices = [
            index
            for index, item in enumerate(root_items)
            if _PEER_REVIEW_MARKER.search((getattr(item, "text", "") or ""))
        ]
        if not marker_indices:
            return 0, 0

        removable_items: list[object] = []
        review_pages: set[int] = set()
        for marker_index in marker_indices:
            marker = root_items[marker_index]
            page_numbers = _page_numbers(marker)
            marker_top = _single_page_top(marker)
            if len(page_numbers) != 1 or marker_top is None:
                # 没有单一页码或 bbox 时，证据不足，不对该标记附近内容做扩展删除。
                continue
            page_no = next(iter(page_numbers))
            review_pages.add(page_no)

            for candidate in root_items:
                if _page_numbers(candidate) != {page_no}:
                    continue
                candidate_top = _single_page_top(candidate)
                if candidate_top is None or abs(candidate_top - marker_top) > 1.0:
                    continue
                # 同一坐标带中的页眉标签会在随后统一提取；此处只清理被错误识别为
                # 正文的碎片，防止因标签差异影响页眉元数据的完整性。
                if getattr(candidate, "label", None) != DocItemLabel.TEXT:
                    continue
                removable_items.append(candidate)

        # Docling 的删除接口接受节点列表。按对象 id 去重，避免多个 marker 指向同一
        # 叠加层时重复删除同一个节点。
        unique_removable_items = list(
            {id(item): item for item in removable_items}.values()
        )
        if unique_removable_items:
            document.delete_items(node_items=unique_removable_items)

        # 审稿层还可能把 ``5 of 11`` 这类分页号粘到同页正文开头。限定在已经命中
        # 审稿标识的页面，并且只删除行首纯分页号，不重写其后的任意学术文本。
        normalized_prefix_count = 0
        for item, _level in document.iterate_items():
            if (
                getattr(item, "label", None) != DocItemLabel.TEXT
                or _page_numbers(item).isdisjoint(review_pages)
            ):
                continue
            original_text = getattr(item, "text", "") or ""
            normalized_text = _strip_page_counter_prefix(original_text)
            if normalized_text != original_text:
                item.text = normalized_text
                normalized_prefix_count += 1

        return len(unique_removable_items), normalized_prefix_count

    @staticmethod
    def _extract_page_furniture(
        document: DoclingDocument,
    ) -> tuple[tuple[PageFurniture, ...], int]:
        """将标注明确的页眉/页脚从正文树移出，并保留有信息价值的元数据。

        Docling 已经把可靠识别出的页眉、页脚标为专用标签，因此不使用坐标阈值去猜测。
        若节点位于正文树中，就将其移出，确保当前 Markdown 与未来分块不会重复摄入；
        若节点仅存在于 Docling 的原生对象列表中，它本来不会导出 Markdown，但仍会被
        采集为元数据。非纯页码的内容写入 ``ParsedPaper.page_furniture``，供来源展示或
        后续 SQLite 字段使用。
        """
        # ``iterate_items()`` 只遍历正文树；但 Docling 会把一部分页眉/页脚保留在
        # ``document.texts`` 的原生对象列表中而不挂到正文树。这类对象已经不会导出
        # Markdown，却仍是有价值的来源元数据，因此此处必须从完整文本列表读取。
        all_text_items = list(getattr(document, "texts", []))
        body_item_ids = {
            id(item) for item, _level in document.iterate_items()
        }
        # 显式标注的页眉/页脚最可靠；未标注的候选还必须通过重复页数、稳定坐标和
        # 页面边缘位置三重验证，具体判定见 `_unlabeled_running_furniture_locations`。
        furniture_locations = {
            id(item): "header"
            for item in all_text_items
            if getattr(item, "label", None) == DocItemLabel.PAGE_HEADER
        }
        furniture_locations.update(
            {
                id(item): "footer"
                for item in all_text_items
                if getattr(item, "label", None) == DocItemLabel.PAGE_FOOTER
            }
        )
        furniture_locations.update(
            _unlabeled_running_furniture_locations(document, all_text_items)
        )

        furniture_items_in_body: dict[int, object] = {}
        extracted_furniture: list[PageFurniture] = []
        for item in all_text_items:
            location = furniture_locations.get(id(item))
            if location is None:
                continue

            if id(item) in body_item_ids:
                furniture_items_in_body[id(item)] = item
            text = _normalize_inline_whitespace(getattr(item, "text", "") or "")
            page_numbers = sorted(_page_numbers(item))
            if not text or _PAGE_COUNTER.fullmatch(text) or not page_numbers:
                continue
            for page_no in page_numbers:
                extracted_furniture.append(
                    PageFurniture(page_no=page_no, location=location, text=text)
                )

        # 只有实际挂在正文树中的项目才需要删除。未挂树的页眉/页脚本来就不进入
        # Markdown；对它们调用 delete_items 反而可能影响 Docling 的对象注册表。
        if furniture_items_in_body:
            document.delete_items(node_items=list(furniture_items_in_body.values()))
        return tuple(extracted_furniture), len(furniture_items_in_body)

    @staticmethod
    def _deduplicate_repeated_captions(document: DoclingDocument) -> int:
        """删除同一个 caption 对象中可验证的整段重复文本。

        这不是通用的文本去重：只接受 caption 以 ``Figure/Table + 同一编号`` 再次开头，
        且两段经空白规范化后的相似度至少为 0.90。它覆盖 MDPI 审稿层把图注粘连为
        ``Figure 1 ... Figure 1 ...`` 的情况，同时避免删除正文对其他图表的正常引用。
        """
        deduplicated_count = 0
        for item, _level in document.iterate_items():
            if getattr(item, "label", None) != DocItemLabel.CAPTION:
                continue

            original_text = getattr(item, "text", "") or ""
            normalized_text = _normalize_inline_whitespace(original_text)
            split_caption = _split_repeated_caption(normalized_text)
            if split_caption is None:
                continue
            item.text = split_caption
            deduplicated_count += 1

        return deduplicated_count

    @staticmethod
    def _remove_dangling_table_page_counter_fragments(document: DoclingDocument) -> int:
        """删除表格单元格尾部可确定为分页残片的 ``N of``。

        规则只处理真实 ``TABLE`` 的单元格，且要求分页残片位于文本末尾。像
        ``8 of 11 samples`` 仍包含分母和名词，不会匹配；因此不会误删正常实验描述。
        这项清理只移除已证实无语义的碎片，不尝试修复合并单元格造成的行列错位。
        """
        removed_count = 0
        for table, _level in document.iterate_items():
            if getattr(table, "label", None) != DocItemLabel.TABLE:
                continue
            for row in getattr(getattr(table, "data", None), "grid", []):
                for cell in row:
                    original_text = getattr(cell, "text", "") or ""
                    cleaned_text = _DANGLING_PAGE_COUNTER_IN_TABLE.sub(
                        "", original_text
                    ).rstrip()
                    if cleaned_text != original_text:
                        cell.text = cleaned_text
                        removed_count += 1
        return removed_count

    @staticmethod
    def _repair_first_page_abstract_order(document: DoclingDocument) -> int:
        """修复首页双栏论文中摘要被 Introduction 左栏插入的阅读顺序。

        许多 Elsevier 风格论文的首页同时包含左侧 Article info、右侧 Abstract，及其
        下方双栏 Introduction。布局模型有时按左栏优先输出，导致 Introduction 出现在
        摘要正文之前。只有在下列证据都成立时才重排：摘要和 Introduction 均在首页、
        摘要当前确实排在 Introduction 之后、摘要标题与紧随正文位于同一左边界。

        该操作仅调整 ``body.children`` 内已有引用的顺序；不重写文本、页码、父节点或
        原始内容。无法证明是这种特定版式问题的 PDF 会保持 Docling 的原始顺序。
        """
        body_children = document.body.children
        root_items = [reference.resolve(document) for reference in body_children]

        abstract_index = _find_first_page_heading_index(root_items, "abstract")
        introduction_index = _find_first_page_introduction_index(root_items)
        if (
            abstract_index is None
            or introduction_index is None
            or abstract_index < introduction_index
        ):
            return 0

        abstract_heading = root_items[abstract_index]
        abstract_left = _first_page_left_edge(abstract_heading)
        if abstract_left is None:
            return 0

        # 收集摘要标题及其紧随、且仍在摘要同一栏中的正文。遍历一旦遇到其他页、
        # 其他栏目或另一个章节标题即停止，避免移动 Introduction 的右栏续写。
        moving_references = [body_children[abstract_index]]
        for position in range(abstract_index + 1, len(root_items)):
            candidate = root_items[position]
            if candidate.label == DocItemLabel.SECTION_HEADER:
                break
            if _page_numbers(candidate) != {1}:
                break
            candidate_left = _first_page_left_edge(candidate)
            if (
                candidate_left is None
                or abs(candidate_left - abstract_left)
                > _ABSTRACT_COLUMN_ALIGNMENT_TOLERANCE
            ):
                break
            moving_references.append(body_children[position])

        # 只有标题之外至少有一段摘要正文才执行，防止移动空标题或损坏异常文档。
        if len(moving_references) == 1:
            return 0

        for reference in moving_references:
            body_children.remove(reference)
        insertion_index = body_children.index(root_items[introduction_index].get_ref())
        body_children[insertion_index:insertion_index] = moving_references
        return 1


def _root_body_items(document: DoclingDocument) -> list[object]:
    """按 PDF 阅读顺序解析 Docling body 的根节点引用。"""
    return [reference.resolve(document) for reference in document.body.children]


def _iter_subtree_items(document: DoclingDocument, item: object) -> list[object]:
    """返回根节点及其子节点，供 list/key-value 容器提取可读文本。

    Docling 会把单位列表和 Keywords 包装进 ``ListGroup``、``KeyValueArea`` 等 group。
    group 自身通常没有 text，不能只看根节点；这里显式向下展开，且用对象 id 去重，
    防止未来某种复杂容器重复引用同一子项时造成文本重复。
    """
    items: list[object] = []
    visited: set[int] = set()

    def visit(candidate: object) -> None:
        if id(candidate) in visited:
            return
        visited.add(id(candidate))
        items.append(candidate)
        for reference in getattr(candidate, "children", []):
            visit(reference.resolve(document))

    visit(item)
    return items


def _first_textual_descendant(
    document: DoclingDocument,
    item: object,
) -> object | None:
    """取得一个根节点中阅读顺序最早、实际含文本的后代项目。"""
    for candidate in _iter_subtree_items(document, item):
        text = (getattr(candidate, "text", "") or "").strip()
        if text:
            return candidate
    return None


def _first_provenance(item: object | None) -> object | None:
    """安全取得第一条 provenance；插入标准标题时只作为页码来源，不猜测 bbox。"""
    provenance = getattr(item, "prov", []) if item is not None else []
    return provenance[0] if provenance else None


def _item_pages(document: DoclingDocument, item: object) -> tuple[int, ...]:
    """汇总根节点及其子节点的来源页码，结果按升序去重。"""
    return tuple(
        sorted(
            {
                provenance.page_no
                for candidate in _iter_subtree_items(document, item)
                for provenance in getattr(candidate, "prov", [])
            }
        )
    )


def _item_is_in_front_matter_window(
    document: DoclingDocument,
    item: object,
    max_pages: int,
) -> bool:
    """仅接受完全位于首页扫描窗口内的节点，防止跨到正文的长段落被误分类。"""
    pages = _item_pages(document, item)
    return bool(pages) and max(pages) <= max_pages


def _collect_root_content(
    document: DoclingDocument,
    roots: list[object],
) -> tuple[str, tuple[int, ...], int]:
    """收集多个根节点的可读正文、页码和实际文本项目数量。

    标题、图像和 page furniture 不是某一前置元数据块的内容；列表项、普通文本、
    footnote 等则保留。换行表示原始结构边界，不用空格强行拼接地址、关键词或邮箱。
    """
    texts: list[str] = []
    pages: set[int] = set()
    seen_items: set[int] = set()
    for root in roots:
        for item in _iter_subtree_items(document, root):
            if id(item) in seen_items:
                continue
            seen_items.add(id(item))
            if getattr(item, "label", None) in {
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.PAGE_HEADER,
                DocItemLabel.PAGE_FOOTER,
            }:
                continue
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            texts.append(text)
            pages.update(provenance.page_no for provenance in getattr(item, "prov", []))
    return "\n".join(texts).strip(), tuple(sorted(pages)), len(texts)


def _normalized_front_matter_label(text: str) -> str:
    """忽略标题中的大小写、空白与标点，得到可跨出版社匹配的别名键。"""
    return re.sub(r"[^a-z0-9]+", "", _normalize_inline_whitespace(text).casefold())


def _front_matter_heading_type(text: str) -> str | None:
    """将已识别标题映射为稳定语义类型；未知标题一律保持原状。"""
    return _FRONT_MATTER_HEADING_TYPES.get(_normalized_front_matter_label(text))


def _split_inline_content_label(text: str) -> tuple[str, str] | None:
    """从 ``Abstract: 正文`` / ``Keywords: 内容`` 中分离强标签与内容。"""
    match = _INLINE_CONTENT_LABEL.match(text.strip())
    if match is None:
        return None
    normalized_label = _normalized_front_matter_label(match.group("label"))
    block_type = _FRONT_MATTER_HEADING_TYPES.get(normalized_label)
    if block_type not in {"abstract", "keywords"}:
        return None
    return block_type, match.group("content").strip()


def _has_inline_front_matter_label(text: str) -> bool:
    """判断文本是否以任何已支持的前置元数据强标签开始。"""
    return _split_inline_content_label(text) is not None or _inline_publication_block_type(
        text
    ) is not None


def _inline_publication_block_type(text: str) -> str | None:
    """识别首页侧栏常见的出版/版权行内标签，不把普通正文中的词汇当标签。"""
    match = re.match(r"^\s*(?P<label>[A-Za-z][A-Za-z\s-]{1,40})\s*:", text)
    if match is None:
        return None
    return _INLINE_PUBLICATION_LABELS.get(
        _normalized_front_matter_label(match.group("label"))
    )


def _availability_content_block_type(text: str) -> str:
    """细分 Availability 容器内的通讯、许可证与出版脚注。

    该函数只处理已有 Availability 标题框定的内容，因而可以使用较宽的许可证和
    DOI 词汇，而不会在普通正文中误分类。无法确认的内容仍保留为 availability。
    """
    normalized = _normalize_inline_whitespace(text).casefold()
    if normalized.startswith("* corresponding author") or normalized.startswith(
        "correspondence:"
    ):
        return "correspondence"
    if any(
        marker in normalized
        for marker in ("creative commons", "copyright", "license", "licence")
    ):
        return "rights"
    if "proceedings of" in normalized or " issn " in normalized or "doi:" in normalized:
        return "publication_info"
    return "availability"


def _find_root_title_index(root_items: list[object], paper_title: str) -> int | None:
    """定位首页论文标题根节点，不因 journal、栏目标题或相似正文造成误命中。"""
    expected = _normalize_inline_whitespace(paper_title).casefold()
    for index, item in enumerate(root_items):
        if (
            getattr(item, "label", None)
            in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
            and _normalize_inline_whitespace(getattr(item, "text", "") or "").casefold()
            == expected
        ):
            return index
    return None


def _resolve_title_from_document(
    document: DoclingDocument,
    *,
    max_pages: int,
) -> _TitleResolution:
    """收集首页候选并用标题后的结构一致性选择标题。

    该函数只读取 item 的 label、文本和 provenance。不会修改 ``item.text``、不会调整
    文档树，也不会为了选标题而移动作者、摘要或出版信息。
    """
    candidates = _collect_title_candidates(document, max_pages=max_pages)
    return _resolve_title_candidates(candidates)


def _collect_title_candidates(
    document: DoclingDocument,
    *,
    max_pages: int,
) -> tuple[_TitleCandidate, ...]:
    """保留首页窗口内 TITLE、SECTION_HEADER 与 TEXT 的原始阅读顺序和 provenance。"""
    candidates: list[_TitleCandidate] = []
    for reading_order, (item, _level) in enumerate(document.iterate_items()):
        label = getattr(item, "label", None)
        if label not in {
            DocItemLabel.TITLE,
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.TEXT,
        }:
            continue
        text = (getattr(item, "text", "") or "").strip()
        pages = tuple(
            sorted(
                {
                    provenance.page_no
                    for provenance in getattr(item, "prov", [])
                    if getattr(provenance, "page_no", None) is not None
                }
            )
        )
        # 没有 provenance 或跨出首页窗口的对象无法证明其来源位置，宁可不把它当标题。
        if not text or not pages or max(pages) > max_pages:
            continue
        candidates.append(
            _TitleCandidate(
                item=item,
                text=text,
                label=label,
                reading_order=reading_order,
                pages=pages,
            )
        )
    return tuple(candidates)


def _resolve_title_candidates(
    candidates: tuple[_TitleCandidate, ...],
) -> _TitleResolution:
    """以多信号评分解析候选；低分或近似并列时显式返回 unresolved。"""
    scored = tuple(
        _score_title_candidate(candidate, candidates)
        for candidate in candidates
    )
    ranked = tuple(
        sorted(
            (item for item in scored if item.rejection_reason is None),
            key=lambda item: (-item.score, item.candidate.reading_order),
        )
    )
    # ``title_candidates`` 保留可参与竞争的标题状 item。普通 TEXT 只有达到可信线时
    # 才展示，避免把整段摘要或作者行扩散到 ParsedPaper 的元数据中。
    visible_candidates = tuple(
        item.candidate.text
        for item in ranked
        if item.candidate.label != DocItemLabel.TEXT
        or item.score >= _TITLE_RESOLVER_WEIGHTS.minimum_confidence
    )
    if not ranked:
        return _TitleResolution(None, "unresolved", visible_candidates, ranked)

    best = ranked[0]
    if best.score < _TITLE_RESOLVER_WEIGHTS.minimum_confidence:
        return _TitleResolution(None, "unresolved", visible_candidates, ranked)
    if (
        len(ranked) > 1
        and best.score - ranked[1].score < _TITLE_RESOLVER_WEIGHTS.minimum_margin
    ):
        return _TitleResolution(None, "unresolved", visible_candidates, ranked)
    return _TitleResolution(
        paper_title=best.candidate.text,
        title_source="front_matter_coherence_scoring",
        title_candidates=visible_candidates,
        ranked_candidates=ranked,
    )


def _score_title_candidate(
    candidate: _TitleCandidate,
    all_candidates: tuple[_TitleCandidate, ...],
) -> _ScoredTitleCandidate:
    """对单个候选计算可解释分数；文本长度只贡献很小的形态分。"""
    rejection_reason = _title_candidate_rejection_reason(candidate.text)
    if rejection_reason is not None:
        return _ScoredTitleCandidate(candidate, score=-10_000, rejection_reason=rejection_reason)

    weights = _TITLE_RESOLVER_WEIGHTS
    score = _title_label_score(candidate.label, weights)
    score += weights.first_page if candidate.pages[0] == 1 else 0
    # 靠前只能略微加分，不能像旧版一样决定性地选择“第一个够长的标题”。
    score += weights.early_reading_order if candidate.reading_order <= 6 else 0
    score += _title_text_shape_score(candidate.text, weights)
    # 普通 TEXT 允许作为标题兜底，但作者行和单位行不能借用自己后方的结构链反客为主。
    if candidate.label == DocItemLabel.TEXT:
        if _looks_like_title_author_line(candidate.text):
            score -= weights.author_text_candidate_penalty
        if _AFFILIATION_SIGNAL.search(candidate.text) is not None:
            score -= weights.affiliation_text_candidate_penalty
    score += _title_preceding_context_score(candidate, all_candidates, weights)
    score += _title_downstream_coherence_score(candidate, all_candidates, weights)
    return _ScoredTitleCandidate(candidate, score=score, rejection_reason=None)


def _title_label_score(label: object, weights: _TitleResolverWeights) -> int:
    """Docling 的 TITLE 标签最强，SECTION_HEADER 次之，普通 TEXT 仅作受控兜底。"""
    if label == DocItemLabel.TITLE:
        return weights.title_label
    if label == DocItemLabel.SECTION_HEADER:
        return weights.section_header_label
    return weights.text_label


def _title_text_shape_score(text: str, weights: _TitleResolverWeights) -> int:
    """文本形态只做弱提示，明确避免以最小字符数作为标题判定规则。"""
    word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text))
    score = weights.title_like_word_shape if 2 <= word_count <= 30 else 0
    if word_count > 60:
        score -= weights.overly_long_text_shape_penalty
    return score


def _title_preceding_context_score(
    candidate: _TitleCandidate,
    all_candidates: tuple[_TitleCandidate, ...],
    weights: _TitleResolverWeights,
) -> int:
    """Contents 紧邻期刊名是常见出版栏形态，故只对该局部结构做小幅降分。"""
    preceding = [
        item
        for item in all_candidates
        if item.reading_order < candidate.reading_order
    ]
    if not preceding:
        return 0
    previous_text = preceding[-1].text
    normalized = _normalized_front_matter_label(previous_text)
    return (
        -weights.preceding_contents_penalty
        if normalized.startswith("contentslistsavailable")
        else 0
    )


def _title_downstream_coherence_score(
    candidate: _TitleCandidate,
    all_candidates: tuple[_TitleCandidate, ...],
    weights: _TitleResolverWeights,
) -> int:
    """评估候选后方是否自然形成“标题→作者/单位→摘要或关键词”结构。"""
    author_signal = False
    affiliation_signal = False
    strong_boundary = False
    publication_furniture_seen = False

    for following in all_candidates:
        if following.reading_order <= candidate.reading_order:
            continue
        heading_type = _front_matter_heading_type(following.text)
        inline_content_type = _split_inline_content_label(following.text)
        if heading_type in {"abstract", "keywords"} or (
            inline_content_type is not None
            and inline_content_type[0] in {"abstract", "keywords"}
        ):
            strong_boundary = True
            break
        # 下一个未被识别为出版/前置元数据的标题可能是另一篇题候选。此时不把其后的
        # 作者和摘要错误归因给当前候选，例如 “Journal Name → Real Paper Title”。
        if (
            following.label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
            and _title_candidate_rejection_reason(following.text) is None
        ):
            break
        if _is_title_publication_furniture(following.text):
            publication_furniture_seen = True
            continue
        if _looks_like_title_author_line(following.text):
            author_signal = True
        if _AFFILIATION_SIGNAL.search(following.text) is not None:
            affiliation_signal = True

    score = 0
    if publication_furniture_seen:
        score -= weights.downstream_publication_furniture_penalty
    if author_signal:
        score += weights.downstream_author_signal
    if affiliation_signal:
        score += weights.downstream_affiliation_signal
    if strong_boundary:
        score += weights.downstream_abstract_or_keywords
    if author_signal and affiliation_signal and strong_boundary:
        score += weights.complete_author_front_matter_chain
    return score


def _title_candidate_rejection_reason(text: str) -> str | None:
    """仅拒绝有明确出版/前置元数据证据的候选，不按具体期刊名过滤。"""
    normalized = _normalized_front_matter_label(text)
    if not normalized:
        return "empty_text"
    if any(normalized.startswith(prefix) for prefix in _TITLE_NON_TITLE_PREFIXES):
        return "publication_or_front_matter_label"
    if _TITLE_PUBLICATION_METADATA.search(text) is not None:
        return "publication_metadata"
    return None


def _is_title_publication_furniture(text: str) -> bool:
    """判断紧邻候选的文本是否是期刊主页、DOI 等出版栏，而非论文正文。"""
    normalized = _normalized_front_matter_label(text)
    if _TITLE_PUBLICATION_METADATA.search(text) is not None:
        return True
    # Authors、Abstract、Keywords 等是标题后的合理前置边界，不是“期刊名后紧跟主页”
    # 这种反证；它们已经由调用方单独处理，不能在这里施加出版栏惩罚。
    return normalized.startswith(
        (
            "contentslistsavailable",
            "journalhomepage",
            "received",
            "revised",
            "accepted",
            "published",
            "referenceformat",
            "publicationinformation",
            "citation",
            "doi",
            "issn",
            "copyright",
            "license",
            "creativecommons",
        )
    )


def _looks_like_title_author_line(text: str) -> bool:
    """以“多个逗号分隔人名”作为作者行证据，避免依赖特定姓名或作者数量。"""
    names = _TITLE_AUTHOR_NAME.findall(text)
    return len(names) >= 2 and "," in text


def _is_fallback_title_candidate(text: str) -> bool:
    """兼容旧内部调用：现在只过滤明确出版栏，不再使用长度阈值决定标题。"""
    return _title_candidate_rejection_reason(text) is None


def _normalize_spaced_capital_heading(text: str) -> str:
    """合并标题中逐字拉开的英文，并把词与词之间的版式空白收为一个空格。"""

    def merge_word(match: re.Match[str]) -> str:
        return re.sub(r"\s+", "", match.group(0))

    # 连续的两个空白是出版社用来分隔单词的视觉间距；逐字空格已经由上面的替换移除。
    return re.sub(r"\s{2,}", " ", _SPACED_CAPITAL_WORD.sub(merge_word, text)).strip()


def _split_merged_numbered_heading(text: str) -> tuple[str, str] | None:
    """若文本恰好含两个连续编号标题，返回拆分后的两个标题，否则返回 ``None``。"""
    boundaries = list(_MERGED_NUMBERED_HEADING_BOUNDARY.finditer(text))
    if len(boundaries) != 1:
        return None
    boundary = boundaries[0]
    first_heading = text[: boundary.start()].strip()
    second_heading = text[boundary.end() :].strip()
    if not (
        _NUMBERED_HEADING.fullmatch(first_heading)
        and _NUMBERED_HEADING.fullmatch(second_heading)
    ):
        return None
    return first_heading, second_heading


def _split_repeated_caption(text: str) -> str | None:
    """若 caption 自身由两份高度相似、同编号的文本拼接而成，返回第一份。"""
    opening = _CAPTION_START.match(text)
    if opening is None:
        return None

    repeated_marker = re.compile(
        rf"{re.escape(opening.group('kind'))}\s*{re.escape(opening.group('number'))}\.",
        re.IGNORECASE,
    )
    repeated = repeated_marker.search(text, opening.end())
    if repeated is None:
        return None

    first_caption = text[: repeated.start()].strip()
    second_caption = text[repeated.start() :].strip()
    # 极短标题或只包含相同编号的正常交叉引用不作为证据。相似度的输入已经规范化空白，
    # 可容忍 PDF 文本层中 ``patterns`` / ``pa tt erns`` 这类少量断字差异。
    if (
        len(first_caption) < 80
        or len(second_caption) < 80
        or SequenceMatcher(
            None, first_caption.casefold(), second_caption.casefold()
        ).ratio()
        < 0.90
    ):
        return None
    return first_caption


def _normalize_inline_whitespace(text: str) -> str:
    """将仅用于展示的连续空白收为一个空格，便于比较页眉/页脚内容。"""
    return re.sub(r"\s+", " ", text).strip()


def _has_next_list_item(
    *,
    document: DoclingDocument,
    root_items: list[object],
    start_index: int,
    expected_number: int,
) -> bool:
    """在紧邻正文窗口中确认存在连续的下一编号列表项。"""
    next_item_pattern = re.compile(
        rf"(?:^|\n)\s*(?:[-*•]\s*)?{expected_number}\)\s+"
    )
    # 限制为八个根对象：允许跨页时经过页眉、页脚和 running header，又不会把远处一段
    # 无关正文中的数字误当作当前标题的连续列表证据。
    for candidate in root_items[start_index + 1 : start_index + 9]:
        if getattr(candidate, "label", None) == DocItemLabel.SECTION_HEADER:
            break
        if _contains_enumerated_list_item(
            document=document,
            item=candidate,
            expected_number=expected_number,
        ):
            return True
        candidate_text = getattr(candidate, "text", "") or ""
        if next_item_pattern.search(candidate_text):
            return True
    return False


def _contains_enumerated_list_item(
    *,
    document: DoclingDocument,
    item: object,
    expected_number: int,
) -> bool:
    """检查根节点或其一层 ListGroup 子项是否含有期望编号的 enumerated ListItem。"""
    candidates = [item]
    if str(getattr(item, "label", "")) == "list":
        candidates.extend(
            reference.resolve(document) for reference in getattr(item, "children", [])
        )
    expected_marker = f"{expected_number})"
    return any(
        getattr(candidate, "label", None) == DocItemLabel.LIST_ITEM
        and getattr(candidate, "enumerated", False)
        and (getattr(candidate, "marker", "") or "").strip() == expected_marker
        for candidate in candidates
    )


def _single_page_top(item: object) -> float | None:
    """返回单页项目的 bbox 顶边；跨页或没有坐标时不提供猜测值。"""
    provenance = getattr(item, "prov", [])
    if len(provenance) != 1:
        return None
    return getattr(getattr(provenance[0], "bbox", None), "t", None)


def _unlabeled_running_furniture_locations(
    document: DoclingDocument,
    all_text_items: list[object],
) -> dict[int, str]:
    """识别被标为普通文本、但在固定页边重复出现的 running header/footer。

    以规范化文本和候选位置分组后，要求至少跨两页，并且每页 bbox 的左、上、右、下
    坐标都处于 2 pt 容差内。最后还要求所有候选都落在各自页面最上或最下 12% 区域。
    这比只按重复文字删除严格得多，可避免清理跨页重复段落或表格标题。
    """
    grouped_items: dict[tuple[str, str], list[object]] = {}
    for item in all_text_items:
        if getattr(item, "label", None) != DocItemLabel.TEXT:
            continue
        original_text = getattr(item, "text", "") or ""
        text = _normalize_inline_whitespace(original_text)
        provenance = getattr(item, "prov", [])
        if not text or "\n" in original_text or len(provenance) != 1:
            continue
        location = _page_edge_location(document, provenance[0])
        if location is None:
            continue
        grouped_items.setdefault((text, location), []).append(item)

    furniture_locations: dict[int, str] = {}
    for (_text, location), items in grouped_items.items():
        page_numbers = {item.prov[0].page_no for item in items}
        if len(page_numbers) < _RUNNING_FURNITURE_MIN_PAGES:
            continue
        if not _have_stable_provenance_positions(items):
            continue
        furniture_locations.update({id(item): location for item in items})
    return furniture_locations


def _page_edge_location(document: DoclingDocument, provenance: object) -> str | None:
    """返回 provenance 是否处于页面顶部/底部边缘；中间正文区域返回 ``None``。"""
    page = getattr(document, "pages", {}).get(getattr(provenance, "page_no", None))
    page_height = getattr(getattr(page, "size", None), "height", None)
    bbox = getattr(provenance, "bbox", None)
    if page_height is None or bbox is None:
        return None
    if bbox.t >= page_height * (1 - _RUNNING_FURNITURE_EDGE_RATIO):
        return "header"
    if bbox.b <= page_height * _RUNNING_FURNITURE_EDGE_RATIO:
        return "footer"
    return None


def _have_stable_provenance_positions(items: list[object]) -> bool:
    """确认多个候选的 bbox 位置近似一致，而非仅碰巧重复了同一段文字。"""
    coordinates = [
        (
            item.prov[0].bbox.l,
            item.prov[0].bbox.t,
            item.prov[0].bbox.r,
            item.prov[0].bbox.b,
        )
        for item in items
    ]
    return all(
        max(values) - min(values) <= _RUNNING_FURNITURE_POSITION_TOLERANCE
        for values in zip(*coordinates)
    )


def _strip_page_counter_prefix(text: str) -> str:
    """仅剥离行首 ``N of M`` 分页号，保留其后的全部正式文本。"""
    return re.sub(r"^\s*(?:page\s*)?\d+\s+of\s+\d+\s+", "", text, flags=re.IGNORECASE)


def _page_numbers(item: object) -> set[int]:
    """取得项目的来源页集合；没有来源信息时返回空集合。"""
    return {provenance.page_no for provenance in getattr(item, "prov", [])}


def _first_page_left_edge(item: object) -> float | None:
    """返回项目在首页的左边界，供首页双栏阅读顺序修复使用。"""
    for provenance in getattr(item, "prov", []):
        if provenance.page_no == 1:
            return provenance.bbox.l
    return None


def _find_first_page_heading_index(items: list[object], expected: str) -> int | None:
    """在根节点中定位指定的首页栏目标题。"""
    for index, item in enumerate(items):
        if (
            getattr(item, "label", None) == DocItemLabel.SECTION_HEADER
            and _page_numbers(item) == {1}
            and (getattr(item, "text", "") or "").strip().casefold() == expected
        ):
            return index
    return None


def _find_first_page_introduction_index(items: list[object]) -> int | None:
    """定位首页的 Introduction 标题，兼容阿拉伯数字、罗马数字或无编号写法。"""
    for index, item in enumerate(items):
        if (
            getattr(item, "label", None) == DocItemLabel.SECTION_HEADER
            and _page_numbers(item) == {1}
            and _INTRODUCTION_HEADING.fullmatch((getattr(item, "text", "") or "").strip())
        ):
            return index
    return None


def _formula_rows_from_pseudo_table(table: object) -> list[tuple[str, object]]:
    """在严格条件下，从伪表格中提取 ``公式文本 + 公式编号`` 行。

    返回空列表代表证据不足，调用方应保持该项目为真实表格。这里不尝试从扁平化的
    PDF 文本猜测 LaTeX；对应的原始文本会作为 FormulaItem 保存，以保留语义和页码。
    """
    data = getattr(table, "data", None)
    captions = getattr(table, "captions", None)
    table_provenance = getattr(table, "prov", None)
    if (
        data is None
        or getattr(data, "num_cols", None) != 2
        or getattr(data, "num_rows", 0) < 2
        or captions
        or not table_provenance
    ):
        return []

    rows = getattr(data, "grid", [])
    extracted_rows: list[tuple[str, object]] = []
    equation_numbers: list[int] = []
    for row in rows:
        if len(row) != 2:
            return []
        expression_cell, number_cell = row
        expression = (getattr(expression_cell, "text", "") or "").strip()
        number_text = (getattr(number_cell, "text", "") or "").strip()
        number_match = _EQUATION_NUMBER.fullmatch(number_text)

        # 真表格往往有表头、字段名或非连续的数值列；公式块则满足每行均为等式 + 编号。
        if (
            not expression
            or "=" not in expression
            or number_match is None
            or getattr(expression_cell, "column_header", False)
            or getattr(number_cell, "column_header", False)
        ):
            return []

        equation_numbers.append(int(number_match.group(1)))
        # 单元格的边界框比整块表格更精确，因此用其替换 provenance 的 bbox，
        # 后续引用仍能回到同一 PDF 页，并且位置更接近实际公式行。
        provenance = table_provenance[0].model_copy(
            update={"bbox": getattr(expression_cell, "bbox", table_provenance[0].bbox)}
        )
        extracted_rows.append((f"{expression} \\quad {number_text}", provenance))

    expected_numbers = list(range(equation_numbers[0], equation_numbers[0] + len(rows)))
    if equation_numbers != expected_numbers:
        return []
    return extracted_rows
