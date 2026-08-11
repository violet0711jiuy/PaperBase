"""为 Docling 适配器生成步骤1的可审核输出产物。

针对一个或多个PDF文件运行该模块。该模块会输出Docling完整的JSON与Markdown导出文件，
同时生成一份简明报告，涵盖标题、章节标题、条目顺序、页面来源、表格以及图注。

该模块刻意是 Docling 专用的检查适配器；未来的解析器复用 ``ParsedPaper`` 的 Markdown
与元数据契约，并可各自提供对应的原生结构检查器，避免让通用业务层依赖 Docling。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from docling_core.types.doc import DocItemLabel, DoclingDocument

from .base import ParsedPaper
from .factory import create_parser
from paperbase.config import default_config_path, load_settings


def _pages_for(item: Any) -> list[int]:
    return sorted({provenance.page_no for provenance in getattr(item, "prov", [])})


def _item_record(item: Any, position: int) -> dict[str, Any]:
    return {
        "position": position,
        "label": str(getattr(item, "label", "group")),
        "pages": _pages_for(item),
        "text": (getattr(item, "text", "") or "").strip(),
    }


def _require_docling_document(parsed_paper: ParsedPaper) -> DoclingDocument:
    """取得 Docling 原生文档，阻止专用检查代码误用于其他解析器结果。"""
    if not isinstance(parsed_paper.native_document, DoclingDocument):
        raise TypeError(
            "Docling inspection requires a ParsedPaper produced by DoclingParser."
        )
    return parsed_paper.native_document


def build_inspection_report(parsed_paper: ParsedPaper) -> dict[str, Any]:
    """汇总 Step 1 必须人工核验的字段。"""
    document = _require_docling_document(parsed_paper)
    source = parsed_paper.source
    item_records = [
        _item_record(item, position)
        for position, (item, _level) in enumerate(document.iterate_items(), start=1)
    ]
    labels = Counter(record["label"] for record in item_records)

    def records_for(label: DocItemLabel) -> list[dict[str, Any]]:
        return [record for record in item_records if record["label"] == str(label)]

    first_page_headers = [
        record
        for record in records_for(DocItemLabel.SECTION_HEADER)
        if 1 in record["pages"]
    ]
    return {
        "source_pdf": source.name,
        "parser_id": parsed_paper.parser_id,
        "parser_diagnostics": dict(parsed_paper.diagnostics),
        "page_furniture": [
            {
                "page_no": furniture.page_no,
                "location": furniture.location,
                "text": furniture.text,
            }
            for furniture in parsed_paper.page_furniture
        ],
        "front_matter": [
            {
                "block_type": block.block_type,
                "canonical_section": block.canonical_section,
                "text": block.text,
                "page_start": block.page_start,
                "page_end": block.page_end,
                "source_item_count": block.source_item_count,
                "detection_method": block.detection_method,
                "confidence": block.confidence,
            }
            for block in parsed_paper.front_matter
        ],
        "page_count": len(document.pages),
        "item_count": len(item_records),
        "label_counts": dict(sorted(labels.items())),
        "paper_title": parsed_paper.paper_title,
        "title_source": parsed_paper.title_source,
        "title_candidates": list(parsed_paper.title_candidates),
        "first_page_section_headers": first_page_headers,
        "first_page_reading_order": [
            record for record in item_records if 1 in record["pages"]
        ],
        "section_headers": records_for(DocItemLabel.SECTION_HEADER),
        "tables": records_for(DocItemLabel.TABLE),
        "captions": records_for(DocItemLabel.CAPTION),
        "pictures": records_for(DocItemLabel.PICTURE),
    }


def _artifact_stem(source: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", source.stem).strip("-").lower()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
    return f"{slug[:60] or 'paper'}-{digest}"


def write_docling_inspection_artifacts(
    parsed_paper: ParsedPaper,
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """写入 Docling 原生 JSON、统一 Markdown 与检查报告。"""
    document = _require_docling_document(parsed_paper)
    source = parsed_paper.source
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(source)
    document_json = output_dir / f"{stem}.docling.json"
    document_markdown = output_dir / f"{stem}.md"
    report_json = output_dir / f"{stem}.inspection.json"

    document_json.write_text(
        json.dumps(document.export_to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Markdown 从统一结果读取，而非再次直接调用 Docling；这验证后续消费层不会绑定
    # DoclingDocument。其他解析器只要实现同一字段，即可复用后续流程。
    document_markdown.write_text(parsed_paper.markdown, encoding="utf-8")
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return document_json, document_markdown, report_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Docling PDF parser output.")
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="Optional PDF files to inspect; omit to use config.yaml's storage.papers_dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Inspect every PDF in this directory; useful for non-ASCII filenames.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Non-sensitive PaperBase YAML configuration file.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)

    sources = args.pdfs
    if args.input_dir:
        if args.pdfs:
            parser.error("Use either PDF paths or --input-dir, not both.")
        sources = sorted(args.input_dir.glob("*.pdf"))
    if not sources:
        sources = sorted(settings.storage.papers_dir.glob("*.pdf"))
    if not sources:
        parser.error(
            "No PDF files found. Add PDFs to storage.papers_dir in config.yaml, "
            "or provide PDF paths / --input-dir."
        )

    paper_parser = create_parser(settings)
    for source in sources:
        parsed_paper = paper_parser.parse(source)
        report = build_inspection_report(parsed_paper)
        outputs = write_docling_inspection_artifacts(
            parsed_paper, report, settings.parsing.inspection_output_dir
        )
        print(
            f"{source.name}: {report['page_count']} pages, "
            f"{report['item_count']} items, outputs={', '.join(map(str, outputs))}"
        )


if __name__ == "__main__":
    main()
