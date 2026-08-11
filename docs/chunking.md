# Step 2 分块产物说明

本文件说明当前 `PaperChunk` 的分块逻辑、每个元数据字段的来源，以及 `storage/parsed/chunks/`
下的检查产物。它描述的是 Step 2 的检查阶段；SQLite 是 Step 3 的正式元数据存储，不应把 JSONL
当作长期数据库。

## 分块逻辑

```text
PDF
→ Step 1 Docling 解析与清理
→ DoclingDocument（保留标题、章节、表格、图注、页码来源）
→ Docling HybridChunker
→ PaperChunk
→ JSONL 与检查报告
```

1. **结构优先**：`HybridChunker` 使用 Docling 文档树切分，优先维护段落、章节、表格和图注的完整性；
   没有采用“按页固定字符数”或“300 tokens + overlap”。
2. **Qwen token 计数**：使用本地 `Qwen3-Embedding-0.6B` tokenizer，而非 Docling 默认 MiniLM。
   `max_tokens=512` 表示最终 `embedding_text` 的上限。
3. **预留元数据预算**：论文标题、章节路径和字段名会加入 `embedding_text`，因此先给正文保留
   `512 - 64 = 448` tokens，再重新计数最终文本。检查报告会记录是否出现超限。
4. **跨页不强拆**：一个结构块跨页时保留为一个 chunk，并写为 `page_start=3`、`page_end=4`；这比
   人为按页切开更能保留语义，但最终 Citation 会显示页码范围。
5. **短块保守处理**：不会因文本短就删除。仅过滤无章节、最多 4 tokens 且精确为 `Article`、
   `Research article`、`Review article` 的出版类型标签。
6. **继承 Step 1 的结构修复**：若 Parser 以连续编号列表的结构证据，将被 Docling 错标的超长
   `1) ...` 从 `section_header` 恢复为 `ListItem`，HybridChunker 直接消费修复后的文档树。
   因此 `section`、`raw_text` 和页码来源始终来自同一份上游结构，不在分块阶段做论文特例回退。

## 每行 JSONL 的字段

`.chunks.jsonl` 每一行就是一个完整 `PaperChunk` JSON 对象，可逐行读取，不需要一次加载整篇论文。

| 字段 | 当前含义与来源 |
| --- | --- |
| `chunk_id` | `paper_<PDF SHA-256 前16位>_chunk_<四位序号>`。例如 `paper_5b6a1007fa7514bf_chunk_0015`；同一 PDF 重跑不变。 |
| `vector_id` | 当前恒为 `null`。Step 5 写入统一 FAISS index 时才会分配整数 ID。 |
| `paper_id` | `paper_<PDF SHA-256 前16位>`；以文件字节哈希产生，不依赖中文文件名或目录。 |
| `paper_title` | Step 1 从 Docling 显式标题或首页标题回退规则得到的正式论文题名。 |
| `source` | 本地 PDF 的绝对路径，仅供当前检查和溯源；SQLite 会在 Step 3 保存受控的文件信息。 |
| `chunk_index` | 本论文内从 0 开始的阅读顺序序号。 |
| `raw_text` | HybridChunker 输出的正文、表格或图注原文。后续 LLM 回答只应以它作为证据，不含人为添加的检索标签。 |
| `embedding_text` | `Paper title + Section + raw_text`。后续仅用于 Qwen embedding，使中文 query 能获得论文与章节上下文。 |
| `section` | 当前有效章节路径，例如 `1. Introduction`。来自 Step 1 修复后的 DoclingDocument 及 HybridChunker 标题链。 |
| `page_start` / `page_end` | chunk 内所有 Docling provenance 的最小/最大页码；跨页 chunk 显示为范围。 |
| `raw_token_count` | 仅 `raw_text` 的 Qwen tokenizer token 数。 |
| `embedding_token_count` | `embedding_text` 的 Qwen tokenizer token 数；应不超过 `chunking.max_tokens`。 |
| `prev_chunk_id` / `next_chunk_id` | 同一论文中阅读顺序的相邻 chunk；Step 8 还会要求邻居属于同一 `section` 才扩展。 |

## 同目录中的文件

当前目录由 `config.yaml > chunking.inspection_output_dir` 控制：

```text
storage/parsed/chunks/qwen3_embedding_0_6b/
├── <论文标识>.chunks.jsonl
└── <论文标识>.chunking.inspection.json
```

### `*.chunks.jsonl`

逐 chunk 的完整检查数据。它包含上表全部字段，以及正文和 embedding 版本文本，是人工检查“某个
chunk 到底写了什么”的主文件。

### `*.chunking.inspection.json`

面向快速检查的汇总报告：

| 字段 | 含义 |
| --- | --- |
| `source_pdf` / `paper_id` / `paper_title` | 当前论文的来源与稳定标识。 |
| `chunker_id` | 当前为 `docling_hybrid`，为未来替换分块方案预留。 |
| `diagnostics` | 本次运行参数与结果，例如总 chunk 数、最大 token、超限数和被过滤的版面噪声数。解析结构修复的计数记录在 Step 1 的解析检查报告中。 |
| `chunk_count` | 有效 chunk 总数。 |
| `page_spans` | 每个 chunk 的 ID、章节、页码范围和两种 token 数；适合快速定位超长或跨页块。 |
| `section_chunk_counts` | 每个章节包含多少 chunk；用于发现伪章节、章节缺失或异常拆分。 |

## 如何检查

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.chunking.inspect --config .\config.yaml
```

重点检查：

1. `diagnostics.chunking.over_limit_chunk_count` 是否为 `0`；
2. `section_chunk_counts` 是否出现明显的正文句子而非章节名；
3. 表格、图注和显示公式是否仍在合理 chunk 中；
4. `page_spans` 中跨页范围是否与 PDF 对照一致；
5. `raw_text` 是否没有页眉、纯页码或审稿残留；
6. `embedding_text` 是否只比 `raw_text` 多出论文标题和章节上下文。

## 前置元数据的 chunk 类型（2026-08-13 补充）

Step 1 已识别的作者与单位、摘要、关键词等内容不会在 Step 2 被重新按特殊规则切分；它们仍是正常的
结构 chunk。Chunker 只依据 Step 1 的标准 section 末级标题和页码范围，为它们补充以下字段：

| 字段 | 含义 |
| --- | --- |
| `content_kind` | `body` 表示普通正文；`front_matter` 表示论文前置元数据。 |
| `front_matter_type` | 前置类型，例如 `authors_affiliations`、`abstract`、`keywords`；正文为 `null`。 |

这两个字段会同时进入 JSONL 和 Step 3 的 `chunks` 表。它们不改变 `raw_text`、chunk 边界或
`embedding_text`；它们的作用是保证每一段可检索文本之后都能获得统一的 `vector_id`，而不是在
`front_matter` 表中保存第二份无法由 FAISS 直接回查的全文。
