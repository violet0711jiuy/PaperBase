# SQLite、Embedding、FAISS 与增量入库

## 正式知识库

PaperBase 的正式知识库由两部分共同组成：

```text
SQLite
├── documents / sections / chunks
├── chunks_fts
└── bibliography_fts

FAISS
└── content chunk vector_id
```

SQLite 是文本和 metadata 的事实源；FAISS 只保存正文向量及其整数 ID。两者必须通过 `chunks.vector_id` 一致映射。

## SQLite schema v5

| 对象 | 职责 |
|---|---|
| `schema_info` | 显式 schema 版本。 |
| `documents` | 论文身份、来源哈希、标题和处理 diagnostics。 |
| `sections` | 每篇论文的章节树、父子关系、层级与顺序。 |
| `chunks` | 原文、检索文本、section/page、邻居、类型和 vector ID。 |
| `chunks_fts` | 仅正文的可重建 FTS5 索引。 |
| `bibliography_fts` | 仅 References/Bibliography 的可重建 FTS5 索引。 |

`chunks.section_type` 只能为 `content` 或 `bibliography`。Bibliography chunk 不进入正文 Embedding 和 FAISS，`vector_id` 保持为空。

## Embedding staging

Step 4 从 SQLite 中读取全部 `section_type=content` 的 `embedding_text`，使用本地 Qwen 模型生成：

- `vectors.npy`
- `records.jsonl`
- `manifest.json`

这些文件只用于建库和审计，不是线上检索依赖。载入时会检查维度、数量、哈希、单位范数、chunk 唯一性和 SQLite 文本指纹。

```powershell
.\.venv\Scripts\python.exe -m paperbase.embedding --config .\config.yaml
```

## FAISS 发布与恢复

当前索引为 `IndexIDMap2(IndexFlatIP)`。发布过程使用 pending journal 协调 SQLite 事务与 FAISS 文件原子替换：

1. 验证 staging manifest 和 SQLite 快照；
2. 构建候选 FAISS 并记录 `vector_id → chunk_id`；
3. 在 SQLite 事务中回写 vector ID；
4. 发布 FAISS 与 manifest；
5. 删除 pending journal。

中断后会依据 journal 和文件哈希完成发布或停止，不允许 SQLite 映射连接到错误索引。

```powershell
.\.venv\Scripts\python.exe -m paperbase.indexing --config .\config.yaml
.\.venv\Scripts\python.exe -m paperbase.indexing --config .\config.yaml --verify
```

## Clean rebuild

Parser、Chunker、section hierarchy 或正文类型规则发生改变时，需要 clean rebuild，使 SQLite、Embedding 和 FAISS 来自同一版文本结构。

```powershell
.\.venv\Scripts\python.exe -m paperbase.rebuild --config .\config.yaml
```

会话数据库独立于正式知识库，不属于 rebuild 输入。

## 增量 Promotion

临时论文先在 staging workspace 中完成解析、分块和向量生成。Promotion 在验证 workspace、chunk 集、模型配置和索引快照后，将论文原子加入正式 SQLite/FAISS。

```powershell
.\.venv\Scripts\python.exe -m paperbase.promotion <workspace_id> --config .\config.yaml
```

Promotion 复用同一 `vector_id`、manifest 和恢复契约，不维护每篇论文独立的 FAISS 子索引。
