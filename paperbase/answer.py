"""Step 8 命令行入口：输出最终回答及其可审计的扩展证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from paperbase.config import default_config_path
from paperbase.generation.service import AnswerServiceResult, create_answer_service


def main(argv: Sequence[str] | None = None) -> None:
    """运行 PaperBase 证据问答；任何 LLM 生成失败都会保留检索与证据输出。"""
    # Windows PowerShell 可能仍使用 GBK 代码页；论文原文常含数学 Unicode 字符，
    # 因此在实际写入终端前明确采用 UTF-8，避免结果已经生成却在 print 阶段失败。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run PaperBase Step 8 grounded paper QA.")
    parser.add_argument("query", help="用户问题，可使用中文或英文。")
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="TEXT",
        help="可重复传入近期问答上下文，用于 Query Rewrite 的指代消解和回答衔接；不能替代论文证据。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="config.yaml 路径。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="本次检索输出总条数；不修改正式配置。",
    )
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive.")
    result = create_answer_service(config_path=args.config).answer_query(
        args.query,
        conversation_context=args.context,
        retrieval_result_limit=args.top_k,
    )
    print(json.dumps(_result_to_json(result), ensure_ascii=False, indent=2))


def _result_to_json(result: AnswerServiceResult) -> dict[str, object]:
    """统一定义 CLI JSON 字段含义，确保答案、检索与证据能独立检查。"""
    return {
        # 用户原始问题：供 UI 与会话记录；第一条 Dense 使用 resolved_query。
        "query": result.retrieval.query,
        # Query Rewrite 结果：解释本次是否有英文语义改写、英文关键词及参考文献检索意图。
        "rewrite_plan": {
            "resolution_status": result.retrieval.rewrite_plan.resolution_status,
            "resolved_query": result.retrieval.rewrite_plan.resolved_query,
            "semantic_query_en": result.retrieval.rewrite_plan.semantic_query_en,
            "lexical_keywords_en": list(result.retrieval.rewrite_plan.lexical_keywords_en),
            "rewrite_status": result.retrieval.rewrite_plan.rewrite_status,
            "search_bibliography": result.retrieval.rewrite_plan.search_bibliography,
        },
        # 重排序状态：success 表示 BGE reranker 生效；fallback 时仍保留 RRF 结果。
        "reranking_status": result.retrieval.reranking_status,
        # 最终生成状态：success/disabled/insufficient_evidence/ambiguous_paper/needs_clarification/fallback。
        # 没有 repaired：结构化输出不合法时，程序不会再让 LLM 进行 JSON 格式修复，
        # 而是直接保留检索到的证据并安全降级，避免第二次生成改变答案含义。
        "answer_status": result.answer.status,
        # 回答正文：即使生成失败也会返回明确的程序提示，避免前端处理 null。
        "answer": result.answer.answer,
        # 被回答引用的 E#/R# 标识，均已经过程序验证，绝不会引用未提供的证据。
        "citations": list(result.answer.citations),
        # 证据不足标志：true 时答案不应被当作论文事实结论。
        "insufficient_evidence": result.answer.insufficient_evidence,
        # 部分回答标志：核心回答有依据，但覆盖范围或细节仍不完整。
        "partial_answer": result.answer.partial_answer,
        # 覆盖边界由 LLM 提出、程序校验后展示；完全无法回答时为 null。
        "coverage_note": result.answer.coverage_note,
        # 检索命中摘要：保留每一块的原始定位，扩展前后可逐层审计。
        "retrieved_chunks": [
            {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "paper_title": chunk.paper_title,
                "section": chunk.section,
                "section_type": chunk.section_type,
                "pre_rerank_rank": chunk.pre_rerank_rank,
                "rerank_score": chunk.rerank_score,
            }
            for chunk in result.retrieval.chunks
        ],
        # 扩展后的可用证据：E# 是正文同节连续块组合，R# 是不扩展的参考文献条目。
        "evidence": [_evidence_to_json(item) for item in result.expansion.evidence],
    }


def _evidence_to_json(item: object) -> dict[str, object]:
    """将 EvidenceUnit 显式转为 JSON，避免 dataclass 内部结构成为对外契约。"""
    return {
        # 证据编号：回答文本只允许使用这里列出的 E#/R#。
        "evidence_id": item.evidence_id,
        # 证据类别：content 是正文，bibliography 是参考文献条目。
        "kind": item.kind,
        # 论文与章节定位：支持用户回溯到具体来源。
        "paper_id": item.paper_id,
        "paper_title": item.paper_title,
        "section": item.section,
        # PDF 页码范围：解析器未能提供时为 null。
        "page_start": item.page_start,
        "page_end": item.page_end,
        # Reranker 原始命中块：邻居扩展也不会丢失最初为何选中此证据的线索。
        "seed_chunk_ids": list(item.seed_chunk_ids),
        # 实际拼入本证据单元的连续 chunk 列表。
        "chunk_ids": list(item.chunk_ids),
        # 预算统计：正文按 SQLite raw_token_count 计数，参考文献为估算值。
        "token_count": item.token_count,
        # 完整证据正文：不从中间截断，供人工核验回答中的每个 [E#]/[R#]。
        "text": item.text,
    }


if __name__ == "__main__":
    main()
