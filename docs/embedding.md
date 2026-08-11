# Step 4：本地 Embedding 向量生成

Step 4 将 SQLite 中已确认的 `chunks.embedding_text` 编码为向量工件，供下一步建立统一 FAISS
索引使用。当前使用本地的 `Qwen/Qwen3-Embedding-0.6B`，不访问网络，也不下载模型到 C 盘。

```text
SQLite chunks.embedding_text
→ Qwen3-Embedding-0.6B（文档侧、无 query instruction）
→ storage/staging/embeddings/.../
```

本阶段不会重新 Parse PDF、不会重新 Chunk、不会创建 FAISS，也不会更新 `chunks.vector_id`。
`vector_id` 只会由 Step 5 在 `FAISS add_with_ids` 成功后分配并回写。

## 为什么从 SQLite 读取

Step 3 已经确定了当前正式知识库使用哪一套 chunks。Step 4 直接读取其 `embedding_text`，能独立重跑，
不会重复消耗 Docling 解析资源，也不会把仅供人工检查的 Markdown / JSONL 当作正式输入。

读取顺序固定为 `paper_id, chunk_index`。同样的顺序会写入 records，因此向量矩阵第 `i` 行始终可回查到
第 `i` 条 `chunk_id`。若数据库中已有非空 `vector_id`，Step 4 会拒绝运行，防止新向量与已存在的 FAISS
索引混用。

## 模型使用方式

Qwen3-Embedding 支持跨语言检索：未来中文 Query 可以带 retrieval instruction，英文论文 document
则**不添加 query instruction**。当前模型配置中的 `document` prompt 为空，代码显式选择它以保持这一语义。

向量以 `float32` 输出并在 CPU 的 `float32` 上再次 L2 归一化。这样 Step 5 使用 `IndexFlatIP` 时，内积即可
作为余弦相似度；不会受到 GPU 半精度计算中极小范数偏差的影响。

## 配置

```yaml
embedding:
  backend: qwen_sentence_transformers
  model_id: Qwen/Qwen3-Embedding-0.6B
  model_path: D:/AI_Workspace/AI_Models/hf_models/Qwen3-Embedding-0.6B
  device: cuda
  batch_size: 16
  normalize_embeddings: true
  output_dir: storage/staging/embeddings/qwen3_embedding_0_6b
```

- `backend`：模型适配器选择项；未来替换 embedding 模型时新增实现，不修改 SQLite/索引业务代码。
- `model_id`：写入 manifest 的可读模型标识。
- `model_path`：本地模型目录，必须在 D 盘模型缓存中。
- `batch_size`：一次送入 GPU 的文档数。当前 8GB 显存保守使用 16，可在实际测量后调大。
- `normalize_embeddings`：必须保持 `true` 才能让未来的内积检索等价于余弦相似度。
- `output_dir`：可重建 staging 工件目录，不作为 Git 源码提交。

## 产物

`output_dir` 中固定包含三个文件：

| 文件 | 内容 |
| --- | --- |
| `vectors.npy` | 形状为 `(N, D)` 的 `float32` L2 单位向量矩阵。第 `i` 行不是 `vector_id`，只是 staging 行号。 |
| `records.jsonl` | 每行一个映射：`row_index`、`chunk_id`、`paper_id`、`chunk_index`、`embedding_text_sha256`。不重复存储全文。 |
| `manifest.json` | 模型、维度、数量、归一化状态、输入快照哈希，以及两个数据文件的 SHA-256。 |

载入器会验证行数、维度、连续 `row_index`、唯一 `chunk_id`、单位范数和文件哈希。若某次写入异常或文件被手动
修改，Step 5 应拒绝该工件，而不是把混合批次写入 FAISS。

## 运行

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.embedding --config .\config.yaml
```

当前三篇样本生成了 `265 × 1024` 的 `float32` 向量。所有 records 与 SQLite 当前 chunks 的稳定排序一致，
这些工件已被 Step 5 用于建立正式索引；线上检索不会读取它们，而是直接查询 FAISS 后按 `vector_id`
回查 SQLite。工件仍保留为可审计、可恢复的输入快照。
