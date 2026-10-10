# 检索与生成：Classic 与 Agentic

核对日期：2026-10-10。两条路径共享正式知识库、Retriever、原文 provenance 和 LLM client，但编排与答案契约不同。

## Classic RAG

```text
问题与会话上下文
→ Query Resolution / Retrieval Rewrite
→ Dense + FTS5
→ Weighted RRF → Reranker → Top-K anchors
→ Section-aware expansion
→ Classic Grounded Answer
```

Classic Answer 使用 `answer_generation.max_evidence_units` 与 `max_output_tokens`。正文和 bibliography 使用独立检索入口，引用使用 E#/REF#。

## Agentic Research

```text
Question → Requirements → Decision
→ canonical search / context tools
→ Raw Evidence Registry
→ Grounded Answerer reads the complete Registry
→ deterministic source-span validation
→ complete answer, or actual gaps back to Decision
```

| 工具 | 有效输入 |
|---|---|
| `search_papers` | 每篇一个 semantic query、1–5 个 lexical terms、一个 Requirement |
| `search_knowledge_base` | 可信 corpus scope 内的 semantic + lexical request |
| `search_references` | bibliography lexical-only FTS |
| `get_section_context` | 已暴露 paper/section 的连续来源文本 |
| `get_paper_overview` | 正式库中已有的只读 Overview |

Decision 的原始 JSON 不可直接执行。Canonical compiler 生成唯一 effective request 和 identity，Guard 与 executor 共用它来校验范围、预算与重复请求。Requirement binding 表示检索目的，不限制 Evidence 最终可以回答哪些义务。

Agentic 不再对 Registry 进行语义投影或最终截断。所有去重后的 Evidence 都进入 Grounded Answerer；只有实际引用的 Evidence 成为 `selected_evidence`。完整输入快照由 `answer_input_evidence_ids` 单独记录，以便评测区分“已召回但未给 Answerer”和“给了但未引用”。

## Score 语义

`raw_score` 只能在同一 Retriever 内解释。`score_kind`、`score_direction` 和 `retrieval_rank` 明确其含义：FTS5 BM25 越低越好；Dense、reranker、RRF 和 workspace BM25 越高越好；non-ranked Evidence 没有伪造分数。没有跨 Retriever 的统一 calibration。

更多细节见[当前 Agentic 架构](../architecture/agentic-research.md)、[字段事实源](../field-source-of-truth.md)和[配置](../configuration.md)。
