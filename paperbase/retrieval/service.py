"""Step 6 的命令行入口：加载正式知识库并展示混合召回证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from paperbase.config import default_config_path, load_settings
from paperbase.database import MetadataDatabase
from paperbase.embedding import QueryEmbedder, create_document_embedder
from paperbase.indexing import FaissIndexStore
from paperbase.reranking import create_reranker

from .hybrid_retriever import HybridRetriever, RetrievedChunk
from .query_rewriter import create_query_planner


def create_hybrid_retriever(*, config_path: Path | str | None = None) -> HybridRetriever:
    """构造正式在线检索器。

    该函数是后续 Web/API 层唯一需要调用的装配点：它从 SQLite 读取 chunk 映射，
    从正式 FAISS 文件读取向量索引，从项目根目录 ``.env`` 读取 LLM 凭证。
    """
    settings = load_settings(config_path)
    database = MetadataDatabase(settings.database.path)
    database.initialize()
    embedder = create_document_embedder(settings.embedding)
    if not isinstance(embedder, QueryEmbedder):
        raise TypeError(
            "Configured embedding backend does not implement query-side embedding."
        )
    # 索引存储适配器直接接收完整 indexing 配置，避免在线侧重复维护路径参数。
    index_store = FaissIndexStore(settings.indexing)
    rewriter = create_query_planner(
        settings=settings.retrieval.query_rewrite,
        env_path=settings.config_path.parent / ".env",
    )
    reranker = (
        create_reranker(settings.reranking) if settings.reranking.enabled else None
    )
    return HybridRetriever(
        database=database,
        query_embedder=embedder,
        index_store=index_store,
        settings=settings.retrieval,
        query_planner=rewriter,
        reranker=reranker,
        reranking_settings=settings.reranking,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """输出适合人工检查的 JSON；不输出 API Key、完整 embedding 或 LLM 思考内容。"""
    parser = argparse.ArgumentParser(description="Run PaperBase Step 6 hybrid retrieval.")
    parser.add_argument("query", help="用户问题，可使用中文或英文。")
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="TEXT",
        help="可重复传入近期检索上下文；仅用于 Query Rewrite 的指代消解。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="config.yaml 路径（默认项目根目录的配置）。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="仅改变本次展示的结果数，不修改 config.yaml 的正式融合 Top-K。",
    )
    args = parser.parse_args(argv)
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive.")

    result = create_hybrid_retriever(config_path=args.config).retrieve(
        args.query,
        conversation_context=args.context,
        # --top-k 在检索阶段分配正文与参考文献名额，而不是在最终 JSON 中简单截断。
        result_limit=args.top_k,
    )
    chunks = result.chunks
    payload = {
        # 原始问题：空白规范化后的用户输入，始终参与基础检索。
        "query": result.query,
        # 改写计划：记录 LLM 是否成功，以及实际新增了哪些检索查询。
        "rewrite_plan": {
            # 指代消解状态：unresolved 时检索器不会执行模糊原问题的 Dense/BM25。
            "resolution_status": result.rewrite_plan.resolution_status,
            # 实际进入第一条 Dense 的规范问题；原始用户问题仍在 query 字段。
            "resolved_query": result.rewrite_plan.resolved_query,
            # 语义改写：一条完整问句，进入 Rewritten Dense Top-20 通道；无改写时为 null。
            "semantic_query_en": result.rewrite_plan.semantic_query_en,
            # 英文关键词组：共同构成一条 Rewritten BM25 Top-20 的 OR 查询。
            "lexical_keywords_en": list(result.rewrite_plan.lexical_keywords_en),
            # 两个通道分开标记，便于判断完整成功、单通道可用或完全降级。
            "semantic_status": result.rewrite_plan.semantic_status,
            "lexical_status": result.rewrite_plan.lexical_status,
            "validation_diagnostics": list(result.rewrite_plan.validation_diagnostics),
            # Retrieval Rewrite 状态：success / partial / degraded / not_run。
            "rewrite_status": result.rewrite_plan.rewrite_status,
            # 仅明确 citation/reference intent 时为 true，才会额外查询 bibliography FTS5。
            "search_bibliography": result.rewrite_plan.search_bibliography,
        },
        # 正文重排序状态：success 表示 Cross-Encoder 已生效；fallback 表示安全沿用 RRF；disabled 表示未启用。
        "reranking_status": result.reranking_status,
        # 展示条数：受命令行 --top-k 限制后的结果数量。
        "result_count": len(chunks),
        # 召回片段列表：每一项均含可追溯的论文定位信息和各通道命中证据。
        "chunks": [
            _chunk_to_json(chunk) for chunk in chunks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _chunk_to_json(chunk: RetrievedChunk) -> dict[str, object]:
    """将检索结果明确转换为 JSON 字典；每个对外字段均在此集中标注中文语义。"""
    return {
        # 融合名次：RRF 去重、排序后的最终位置，从 1 开始。
        "rank": chunk.rank,
        # Chunk 唯一标识：SQLite 和引用链路使用的稳定 ID。
        "chunk_id": chunk.chunk_id,
        # 向量标识：FAISS 返回并映射回 SQLite 的全局 ID。
        "vector_id": chunk.vector_id,
        # 论文标识：由源 PDF 内容哈希派生的稳定 ID。
        "paper_id": chunk.paper_id,
        # 论文标题：用于在命令行中快速识别来源。
        "paper_title": chunk.paper_title,
        # 章节路径：该 chunk 在论文正文或前置元数据中的位置。
        "section": chunk.section,
        # 内容类别：body 表示正文，front_matter 表示作者、摘要、关键词等前置块。
        "content_kind": chunk.content_kind,
        # 前置元数据类型：仅 content_kind 为 front_matter 时有值。
        "front_matter_type": chunk.front_matter_type,
        # 章节类型：content 是正文证据；bibliography 仅用于明确引用问题的辅助证据。
        "section_type": chunk.section_type,
        # 起始页码：来自 PDF 解析 provenance；缺失时为 null。
        "page_start": chunk.page_start,
        # 结束页码：来自 PDF 解析 provenance；缺失时为 null。
        "page_end": chunk.page_end,
        # 原始文本预览：只输出前 500 字符，完整原文仍保存于 SQLite。
        "raw_text": chunk.raw_text[:500],
        # 重排序前名次：正文在 Step 6 RRF 队列中的位置；参考文献直出候选为 null。
        "pre_rerank_rank": chunk.pre_rerank_rank,
        # 重排序相关性分数：正文由 Cross-Encoder 产生；未重排或参考文献直出候选为 null。
        "rerank_score": chunk.rerank_score,
        # 融合得分：各命中通道加权 RRF 分数之和，只用于本次排序解释。
        "fused_score": chunk.fused_score,
        # 命中证据：同一 chunk 在每条检索通道中的排名、原始分数和实际权重。
        "source_matches": [
            {
                # 召回路径：如 dense_resolved、dense_semantic、bm25_keywords。
                "route": source.route,
                # 该路径实际使用的查询文本。
                "query": source.query,
                # 该路径内的候选排名，从 1 开始。
                "rank": source.rank,
                # 该路径原始分数：FAISS 为相似度，BM25 为词法得分，仅供诊断。
                "raw_score": source.raw_score,
                # 实际融合权重：已按同类改写查询数量均分。
                "effective_weight": source.effective_weight,
            }
            for source in chunk.source_matches
        ],
    }


if __name__ == "__main__":
    main()
