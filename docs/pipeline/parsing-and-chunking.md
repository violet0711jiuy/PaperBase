# PDF 解析与结构感知分块

## 数据流

```text
PDF
→ Docling Parser
→ ParsedPaper + Section Tree
→ Docling HybridChunker
→ PaperChunk
→ SQLite / inspection artifacts
```

解析与分块只负责构造可追溯文本和结构，不生成向量、不修改 FAISS，也不调用回答模型。

## Parser

当前注册的解析后端为 Docling。解析阶段负责：

- 从首页结构中识别论文标题、作者单位、摘要、关键词和出版信息；
- 保留正文、表格、图注、公式和页码 provenance；
- 清理有明确结构证据的页眉页脚、匿名审稿叠加层和重复图注；
- 使用书签、标题编号和字体样式推断章节层级；
- 将 `References`、`Bibliography` 等末尾根章节识别为独立章节，不继承前一个正文 section。

解析器不会按论文标题白名单修复内容，也不会猜测复杂表格中的数值。

## Section hierarchy

章节结构以显式父子关系保存。完整 section path 例如：

```text
4 THE DECOUPLED FRAMEWORK > 4.1 Residual Decomposition Mechanism
```

References/Bibliography 必须成为独立根章节。其 chunk 使用：

```text
section_type = bibliography
```

普通正文和前置元数据使用：

```text
section_type = content
```

这两个类型决定后续文本进入正文 FAISS/FTS5，还是独立 bibliography FTS5。

## Chunk 契约

每个 `PaperChunk` 至少包含：

- 稳定身份：`paper_id`、`chunk_id`、`chunk_index`；
- 原始证据：`raw_text`；
- 检索文本：`embedding_text`；
- 结构定位：`section`、`page_start`、`page_end`；
- 邻居关系：`prev_chunk_id`、`next_chunk_id`；
- 类型：`content_kind`、`front_matter_type`、`section_type`；
- token 统计与 parser/chunker provenance。

`raw_text` 是回答模型可以引用的真实文本。`embedding_text` 额外加入论文标题和 section 上下文，只用于检索。

## 分块规则

- 结构优先，不使用固定字符窗口；
- 当前 `max_tokens=512`，并为检索 metadata 预留 token；
- 同级相邻块仅在结构和预算允许时合并；
- 跨页结构块可保留一个 `page_start/page_end` 范围；
- 表格只在后续块重复表头，不猜测或重排无法确认的单元格；
- Front matter 是可检索 chunk，通过 `content_kind/front_matter_type` 区分，不重复保存全文。

## 检查命令

```powershell
.\.venv\Scripts\python.exe -m paperbase.parsing.inspect --config .\config.yaml
.\.venv\Scripts\python.exe -m paperbase.chunking.inspect --config .\config.yaml
```

检查重点包括章节树、References 根节点、token 超限、页码范围、页面噪声、表格和 `embedding_text` metadata。
