# Step 3：SQLite 元数据层

SQLite 是 PaperBase 的正式元数据存储；它是 Python 自带的嵌入式数据库，不需要另行下载、安装或启动服务。
数据库文件由 `config.yaml > database.path` 控制，默认位置为：

```text
storage/paperbase.sqlite3
```

它不保存 PDF 二进制、不生成 embedding，也不建立 FAISS。原始 PDF 始终位于 `storage/papers/`；
Step 4 只生成 staging 向量工件；Step 5 才会为 `chunks.vector_id` 分配 FAISS 向量索引 ID。

## 数据关系与唯一文本源

```text
documents（1 篇论文）
└── chunks（0 到多个：正文、作者单位、摘要、关键词等全部可检索文本）
```

`paper_id` 来自 PDF 完整 SHA-256 的前 16 位，例如 `paper_5b6a1007fa7514bf`；完整哈希存于
`documents.source_sha256` 用于校验。`chunk_id` 的形式为
`paper_<hash>_chunk_<四位序号>`。

从 schema V2 开始，**`chunks` 是唯一保存可检索文本的表**。Step 1 仍会在内存中的
`ParsedPaper.front_matter` 中产出作者、摘要、关键词等语义识别结果，但导入 SQLite 时不会再创建
重复的 `front_matter` 文本表。对应 chunk 使用 `content_kind` 和 `front_matter_type` 标记。

这样做有两个直接收益：

- 每段文本只保存一份 `raw_text` / `embedding_text`，避免重复 embedding 与重复召回；
- 所有可回答内容都能在 Step 5 获得 `vector_id`，FAISS 命中后可以直接按该 ID 回查同一行。

## 表结构

### `documents`

每篇已导入论文一行，保存：

- 稳定身份：`paper_id`、完整 `source_sha256`；
- 可追溯来源：`source_path`、`source_filename`，只保存路径，不保存 PDF 内容；
- 论文信息：`paper_title`、`title_source`；
- 处理线索：`parser_id`、`chunker_id`、解析/分块 diagnostics JSON；
- 导入时间：`ingested_at`、`updated_at`（UTC）。

### `chunks`

每个 Step 2 的原始结构 chunk 一行，保存：

- 身份与顺序：`chunk_id`、可空 `vector_id`、`paper_id`、`chunk_index`；
- 文本：`raw_text`（LLM 回答的原文证据）和 `embedding_text`（带论文标题、section 的检索文本）；
- 结构定位：`section`、`page_start` / `page_end`、两种 token 数、`prev_chunk_id` / `next_chunk_id`；
- 内容类型：`content_kind` 为 `body` 或 `front_matter`；当为 `front_matter` 时，
  `front_matter_type` 保存 `authors_affiliations`、`abstract`、`keywords`、`publication_info`
  等 Step 1 已识别的语义类型。

`(paper_id, chunk_index)` 与非空 `vector_id` 都受唯一约束。数据库同时约束：正文 chunk 不得携带
`front_matter_type`，前置元数据 chunk 必须携带该类型。

## 从 V1 迁移到 V2

项目旧库的 V1 曾有独立 `front_matter` 表，因此会与 `chunks` 重复存储文本。首次使用 V2 代码打开旧库时，
程序会在一个 SQLite 事务中：

1. 根据标准 section 末级标题和页码相交关系，为已有 chunks 回填 `content_kind` / `front_matter_type`；
2. 删除已无必要的重复 `front_matter` 表；
3. 将 `schema_version` 从 `1` 更新为 `2`。

若中途失败，SQLite 会回滚，旧库保持原样。迁移后可重新执行一次完整导入，让当前 Parser/Chunker
按最新规则重新生成所有 chunk 类型。

## 导入命令

导入 `storage/papers/` 中全部 PDF：

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.database --config .\config.yaml
```

只导入一篇论文：

```powershell
.\.venv\Scripts\python.exe -m paperbase.database --config .\config.yaml -- "D:\AI_Workspace\projects\PaperBase\storage\papers\你的论文.pdf"
```

命令使用当前正确的数据链路：

```text
PDF → Step 1 Parser → Step 2 HybridChunker → SQLite
```

不会从 Markdown 或 JSONL 反推结构；它们是人工检查产物，缺少完整 Docling provenance。正式导入直接使用
同一次解析产生的 `ParsedPaper` 与 `ChunkingResult`，避免版本不一致。

## 重复导入与安全边界

同一 PDF 重跑会得到同一 `paper_id`。在单个 SQLite 事务中，导入器会删除该论文的旧 `chunks`，再写入当前
结果：

- 成功时完整替换，不追加重复 chunk；
- 校验或写入失败时完整回滚；
- 一旦 Step 5 写入任意 `vector_id`，此替换入口会拒绝执行，避免 SQLite 与 FAISS 脱节。

导入前还会验证 PDF 哈希与 `paper_id` 一致、chunk ID 唯一、`chunk_index` 连续为 `0..N-1`，以及所有邻居
ID 均属于同一导入批次。
