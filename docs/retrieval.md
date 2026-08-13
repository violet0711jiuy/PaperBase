# Step 6：混合检索

本阶段只完成“把问题可靠地召回为带证据的 chunks”，不生成最终答案、不做 rerank，也不修改论文、解析结果或既有 FAISS 向量。正式在线链路不依赖 Step 4 的 `vectors.npy`、`records.jsonl` 等 staging 工件。

```text
用户问题
  ├─ 原始问题 → Qwen Query embedding → FAISS Top-20
  ├─ 原始问题 → SQLite FTS5 / BM25 Top-10
  └─ 一次 LLM 改写
       ├─ 一条语义改写问题 → Qwen Query embedding → FAISS Top-20
       └─ 一组英文关键词 → SQLite FTS5 / BM25 Top-20
                                      ↓
                         按 chunk_id 去重并加权 RRF 融合
                                      ↓
                           最多输出 40 条可追溯 chunk
```

## 为什么是四路召回

原始问题永远会进入稠密与 BM25 两个通道，不会被 LLM 改写覆盖。一次 LLM 调用只补充一条完整语义问题和一组英文关键词：语义问题用于跨语言 Dense 召回，英文术语、缩写和模型名用于英文论文的精确 BM25 召回。

当前语料以英文论文为主，因此不执行中文关键词 BM25。中文用户问题仍能通过 Qwen 跨语言向量和 LLM 生成的英文关键词召回英文论文。

英文关键词组逻辑上是一条 `bm25_rewrite` 路径。它会安全编译为 FTS5 OR 查询，例如 `LSTM OR "wind speed prediction" OR author`，而不是把三词当作一个必须连续出现的短语，也不是拆成三次独立 BM25 检索。因此关键词数量不会放大 RRF 权重。

## RRF 融合

各通道的原始分数不可直接相加：FAISS 返回的是余弦相似度，SQLite 返回的是 BM25 量纲。系统只使用排名进行融合：

```text
chunk_score += effective_weight / (rrf_k + rank)
```

默认 `rrf_k=60`。初始权重为 `dense_original=1.0`、`bm25_original=0.7`、`dense_rewrite=1.0`、`bm25_rewrite=1.0`。每个结果保留 `route`、该路 query、rank、原始分数与实际权重，便于评估和下一步接入 BGE reranker。

## SQLite FTS5 与正式索引

SQLite 从 schema V3 起增加 `chunks_fts`，其中仅包含 `chunk_id`、论文标题、section、`raw_text` 的可重建倒排索引；正式原文和元数据仍以 `documents`、`chunks` 为唯一事实源。触发器会随 chunk 的写入、删除和可检索字段更新而维护 FTS 索引。

查询启动时会校验：FAISS 文件校验和、manifest、SQLite 内所有 `vector_id → chunk_id → embedding_text` 指纹是否一致。通过后才加载 FAISS；因此线上检索不读取 staging 目录。

## LLM 配置与降级

非敏感参数位于 `config.yaml > retrieval`，密钥和部署信息只放项目根目录 `.env`。请从 `.env.example` 创建自己的 `.env`：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=500
```

LLM 的改写 prompt 集中在 `paperbase/prompts/query_rewrite.py`；同一种用途的 system prompt、user 模板和后续 JSON 修复 prompt 保存在同一文件，业务检索代码不拼接 prompt。若网络、模型服务或 JSON 输出异常，且 `fallback_to_original: true`，改写器会返回空补充查询，系统继续用“原始稠密 + 原始 BM25”检索，不中断用户请求。

## 运行与检查

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.retrieval "LSTM 风速预测论文的作者是谁？" --top-k 5
```

输出为 JSON，包含改写状态、各类补充查询、每个 chunk 的论文标题、章节、页码、原文前 500 字以及全部 `source_matches`。`--top-k` 仅截断显示，不改写 `config.yaml` 中的正式融合参数。

当前实测中，SQLite 已由 V2 升级至 V3：`documents=3`、`chunks=265`，迁移前备份为 `storage/paperbase.sqlite3.before_fts_v3.bak`。当前 Qwen 模型已验证可以非流式输出合约 JSON；本次四路召回改动完成后，将用作者、摘要、数据集和方法缩写问题进行下一轮召回质量检查。
