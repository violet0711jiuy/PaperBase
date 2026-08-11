"""生成 Step 2 分块检查产物的命令行入口。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paperbase.config import default_config_path, load_settings
from paperbase.parsing.factory import create_parser

from .base import ChunkingResult, PaperChunk
from .factory import create_chunker


def _artifact_stem(source: Path) -> str:
    """为检查产物创建与解析阶段一致的可读、稳定文件名。"""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", source.stem).strip("-").lower()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
    return f"{slug[:60] or 'paper'}-{digest}"


def _chunk_record(chunk: PaperChunk) -> dict[str, Any]:
    """将 dataclass 转换为 JSONL 可直接写入的基础类型。"""
    record = asdict(chunk)
    record["source"] = str(chunk.source)
    return record


def build_chunking_report(result: ChunkingResult) -> dict[str, Any]:
    """生成供人工检查的简明分块统计，不代替原始 JSONL。"""
    chunks = result.chunks
    section_counts: dict[str, int] = {}
    content_kind_counts: dict[str, int] = {}
    front_matter_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        section = chunk.section or "<unsectioned>"
        section_counts[section] = section_counts.get(section, 0) + 1
        content_kind_counts[chunk.content_kind] = (
            content_kind_counts.get(chunk.content_kind, 0) + 1
        )
        if chunk.front_matter_type:
            front_matter_chunk_counts[chunk.front_matter_type] = (
                front_matter_chunk_counts.get(chunk.front_matter_type, 0) + 1
            )
    return {
        "source_pdf": result.parsed_paper.source.name,
        "paper_id": chunks[0].paper_id if chunks else None,
        "paper_title": result.parsed_paper.paper_title,
        "chunker_id": result.diagnostics["chunking.chunker_id"],
        "diagnostics": result.diagnostics,
        "chunk_count": len(chunks),
        "page_spans": [
            {
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "content_kind": chunk.content_kind,
                "front_matter_type": chunk.front_matter_type,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "raw_token_count": chunk.raw_token_count,
                "embedding_token_count": chunk.embedding_token_count,
            }
            for chunk in chunks
        ],
        "section_chunk_counts": dict(sorted(section_counts.items())),
        "content_kind_counts": dict(sorted(content_kind_counts.items())),
        "front_matter_chunk_counts": dict(sorted(front_matter_chunk_counts.items())),
    }


def write_chunking_artifacts(
    result: ChunkingResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    """写入逐 chunk JSONL 与汇总检查报告；不写 SQLite 或向量索引。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(result.parsed_paper.source)
    chunks_jsonl = output_dir / f"{stem}.chunks.jsonl"
    report_json = output_dir / f"{stem}.chunking.inspection.json"
    chunks_jsonl.write_text(
        "".join(
            json.dumps(_chunk_record(chunk), ensure_ascii=False) + "\n"
            for chunk in result.chunks
        ),
        encoding="utf-8",
    )
    report_json.write_text(
        json.dumps(build_chunking_report(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunks_jsonl, report_json


def main() -> None:
    """解析指定论文、立即分块，并写入仅供 Step 2 人工检查的产物。"""
    argument_parser = argparse.ArgumentParser(
        description="Inspect PaperBase structure-aware chunking output."
    )
    argument_parser.add_argument("pdfs", nargs="*", type=Path)
    argument_parser.add_argument("--input-dir", type=Path)
    argument_parser.add_argument(
        "--config", type=Path, default=default_config_path()
    )
    args = argument_parser.parse_args()

    settings = load_settings(args.config)
    sources = args.pdfs
    if args.input_dir:
        if args.pdfs:
            argument_parser.error("Use either PDF paths or --input-dir, not both.")
        sources = sorted(args.input_dir.glob("*.pdf"))
    if not sources:
        sources = sorted(settings.storage.papers_dir.glob("*.pdf"))
    if not sources:
        argument_parser.error("No PDF files found to parse and chunk.")

    paper_parser = create_parser(settings)
    paper_chunker = create_chunker(settings)
    for source in sources:
        # 这是 Step 2 的验证入口：在同一进程中把刚解析的原生结构直接交给分块器，
        # 而不是从 Markdown 重新推断章节或页码。正式导入服务在后续 Step 11 也会复用
        # 这个 "parse once -> chunk once" 的方式。
        parsed_paper = paper_parser.parse(source)
        result = paper_chunker.chunk(parsed_paper)
        outputs = write_chunking_artifacts(
            result, settings.chunking.inspection_output_dir
        )
        print(
            f"{source.name}: {len(result.chunks)} chunks, "
            f"max_embedding_tokens="
            f"{result.diagnostics['chunking.max_embedding_token_count']}, "
            f"outputs={', '.join(map(str, outputs))}"
        )


if __name__ == "__main__":
    main()
