# Retrieval、Evidence Expansion 与回答生成

## 正文检索链路

```text
User Query
→ Query Resolution / Retrieval Rewrite
├── dense_resolved
├── dense_semantic
└── bm25_keywords
→ Weighted RRF
→ BGE Cross-Encoder Reranker
→ Final Top-5 anchors
→ Section-aware Evidence Expansion
→ Evidence-grounded Answer
```

当前正文是三路召回，不包含 Original BM25。

## Query Planner

Planner 分为两个边界：

1. Query Resolution：利用受信任的当前论文范围或有限会话上下文消解“本文、它、这个模型”等指代；
2. Retrieval Rewrite：产生英文 `semantic_query_en` 和最多 5 个 `lexical_keywords_en`。

Lexical keywords 会与确定性实体提取结果合并。Schema、Parser 和 Validator 会阻止字段缺失、实体破坏、污染文本和明显 meaning drift。失败时记录 `partial/degraded` 状态，并保留可安全使用的路径。

Context-free Retrieval Benchmark 不向 Retriever 传入 Golden `paper_id`；`paper_id` 只用于离线 Ground Truth。

## 三路正文召回

| Route | 输入 | 后端 |
|---|---|---|
| `dense_resolved` | 原问题或已消解问题 | Qwen Embedding + FAISS |
| `dense_semantic` | 英文 semantic rewrite | Qwen Embedding + FAISS |
| `bm25_keywords` | 英文关键词 OR 查询 | SQLite `chunks_fts` |

候选按 `chunk_id` 去重，并使用：

```text
score += route_weight / (rrf_k + rank)
```

各路保留 query、rank、原始分数和有效权重，便于 Evaluation 复现。

## Bibliography 路由

明确询问引用、参考文献或条目编号时，规则优先设置 bibliography intent。系统直接查询独立 `bibliography_fts`，而不是让 References 参与正文 Dense/RRF。

Bibliography 结果：

- 不进入正文 FAISS；
- 不进入正文 RRF；
- 不交给 Cross-Encoder；
- 以独立 `R#` Evidence 输出。

普通问题即使出现外部论文名或模型名，也不会自动成为 bibliography intent。

## Reranker

正文 RRF 候选最多取 `candidate_top_k=40` 进入本地 BGE Cross-Encoder，默认输出 Final Top-5。每条结果保留 `pre_rerank_rank`、`rerank_score` 和 `fused_score`。模型不可用时回退到 RRF 顺序，并显式记录状态。

## Section-aware Evidence Expansion

Final Top-5 正文 chunks 是 anchors。生产 Expander 仅在相同 `paper_id + section` 内，按连续 `chunk_index` 向左右扩展；当前 `neighbor_window=1`。重叠窗口合并后按 reranker 优先级装入全局 token 预算。

Expansion 不跨 section，也不扩展 bibliography。最终正文 Evidence 使用 `E#`，参考文献使用 `R#`。

## Answer Generation

回答模型只能使用本次 Evidence 作为事实来源。输出包含：

- `answer`
- `citations`
- `insufficient_evidence`

程序验证 JSON Schema、引用编号和 Evidence 可用性。证据不足时拒绝补全外部知识；LLM 请求或结构校验失败时安全降级为可审阅的 Retrieval/Expansion Evidence。

```powershell
.\.venv\Scripts\python.exe -m paperbase.retrieval "D²STGNN如何学习动态空间依赖？"
.\.venv\Scripts\python.exe -m paperbase.answer "D²STGNN如何学习动态空间依赖？"
```
