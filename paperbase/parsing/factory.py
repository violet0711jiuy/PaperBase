"""依据集中配置创建可替换的论文解析器。"""

from __future__ import annotations

from paperbase.config import AppSettings

from .base import PaperParser
from .docling_parser import DoclingParser, ParserSettings


class UnsupportedParserBackendError(ValueError):
    """配置选择了尚未接入 PaperBase 的解析器时抛出的明确错误。"""


def create_parser(settings: AppSettings) -> PaperParser:
    """从 ``config.yaml`` 的 ``parsing.backend`` 创建当前解析器。

    当前只注册了 ``docling``。未来增加 MinerU 等实现时，在这里添加一个分支并复用
    同一 ``PaperParser`` 接口；Chunk、检索、SQLite 和界面层无需修改。
    """
    backend = settings.parsing.backend.casefold()
    if backend == "docling":
        docling = settings.parsing.docling
        return DoclingParser(
            ParserSettings(
                artifacts_path=docling.artifacts_path,
                device=docling.device,
                enable_ocr=docling.enable_ocr,
                enable_formula_enrichment=docling.enable_formula_enrichment,
                formula_preset=docling.formula_preset,
                compile_layout_model=docling.compile_layout_model,
                compile_formula_model=docling.compile_formula_model,
                remove_page_furniture=docling.remove_page_furniture,
                remove_peer_review_artifacts=docling.remove_peer_review_artifacts,
                list_style_heading_min_chars=docling.list_style_heading_min_chars,
                enable_heading_hierarchy=docling.enable_heading_hierarchy,
                heading_hierarchy_use_bookmarks=docling.heading_hierarchy_use_bookmarks,
                heading_hierarchy_use_numbering=docling.heading_hierarchy_use_numbering,
                heading_hierarchy_use_style=docling.heading_hierarchy_use_style,
                heading_hierarchy_max_level=docling.heading_hierarchy_max_level,
                enable_front_matter_recognition=settings.parsing.front_matter.enabled,
                front_matter_max_pages=settings.parsing.front_matter.max_pages,
            )
        )

    raise UnsupportedParserBackendError(
        f"Unsupported parsing backend: {settings.parsing.backend!r}. "
        "Currently registered backends: docling."
    )
