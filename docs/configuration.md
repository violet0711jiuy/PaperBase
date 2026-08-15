# 配置说明

`config.yaml` 是 PaperBase 的非敏感运行参数入口。当前包含存储目录、解析器选择和
Docling 参数、Chunking、Embedding 和 SQLite 参数；后续 FAISS、Reranker、LLM 等模块会继续在同一文件
中增加独立小节。

不要将 API Key、Token、密码或个人访问地址写入 `config.yaml`。这类敏感值将来放入未提交的
`.env` 文件。

## 读取规则

- 相对路径以 `config.yaml` 所在目录为准，与 PowerShell 当前所在目录无关。
- 配置加载时会严格校验字段名和类型；拼写错误会直接报错，不会悄悄忽略。
- 本机模型缓存必须继续指向 `D:/AI_Workspace/AI_Models/...`。

## 目前如何切换解析器

当前只注册了 `docling`：

```yaml
parsing:
  backend: docling
```

未来接入 MinerU 时，会新增一个 `MineruParser` 适配器及其配置小节；随后只需把
`backend` 改为 `mineru`。后续 Chunk、检索、SQLite 与界面层仍读取统一的 `ParsedPaper`，
不需要跟随修改。

## 当前解析命令

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.parsing.inspect --config .\config.yaml
```

默认会解析 `storage.papers_dir` 中的 PDF，并将检查产物写到
`parsing.inspection_output_dir`。

## 论文前置元数据

`parsing.front_matter` 是解析器无关的语义识别开关：

```yaml
parsing:
  front_matter:
    enabled: true
    max_pages: 2
```

- `enabled`：要求 Parser 在统一 `ParsedPaper.front_matter` 中输出作者与单位、摘要、
  关键词、出版信息、代码/数据可用性、版权许可等可确认的语义块。该字段是未来替换
  MinerU 等解析工具后也必须遵守的契约，不是 Docling 的私有输出。
- `max_pages`：只把完全位于前 N 页的内容视为“前置元数据候选”，默认 `2`。这避免把
  论文末尾的 `Data availability` 等后记误当成首页信息；首页侧栏中错排到 Introduction
  之后的 `Citation:`、`Received:`、`Copyright:` 会作为独立块保留，但不会移动正文。

当前标准 `block_type` 包括 `authors_affiliations`、`correspondence`、`abstract`、
`keywords`、`publication_info`、`availability` 和 `rights`。每个块均包含原始文本、页码、
来源项目数、识别方式与置信度，可在 `*.inspection.json > front_matter` 中审核。

该阶段不解析作者姓名与单位编号之间的一对一映射，也不生成 doc2query；它只建立可审计的
原始语义块。Step 2 的后续实验再决定如何将这些块做成 metadata chunk、检索提示和 short2big
关联，避免同时混入解析与召回变量。

## 页眉、页脚与审稿残留文本层

Docling 专用参数位于 `parsing.docling`：

```yaml
remove_page_furniture: true
remove_peer_review_artifacts: true
```

- `remove_page_furniture`：将被 Docling 明确标为 `page_header` / `page_footer`、且位于
  正文树的内容移出，使其不进入 Markdown 和后续分块。Docling 中仅存于原生对象列表的
  页眉/页脚本来就不会导出，但仍会被读取。期刊名、卷期、文章号等非纯页码信息保留在
  统一结果的 `page_furniture` 字段；`3 of 11` 之类的纯分页号直接忽略。
- `remove_peer_review_artifacts`：仅清理由 `FOR PEER REVIEW` 标记与同页同坐标带共同
  证实的不可见审稿层碎片。它不是通用去重器，无法确认的重复正文会保留。
- `list_style_heading_min_chars`：控制 Parser 对被错误标记为章节的超长 `1) ...` 正文
  列表项的最小长度，默认 `80`。只有其后在有限范围内、下一个真实章节之前，能找到连续的
  `2) ...` 列表项时，才会在 Step 1 将它恢复为 `ListItem`；短的 `1) Study area`、孤立
  的长标题和真正的括号编号章节都不会被改写。

## Step 2 分块参数

`chunking` 小节控制结构感知分块：

```yaml
chunking:
  backend: docling_hybrid
  max_tokens: 512
  embedding_metadata_reserve_tokens: 64
  tokenizer_path: D:/AI_Workspace/AI_Models/hf_models/Qwen3-Embedding-0.6B
```

- `max_tokens` 是 **最终 `embedding_text`** 的 Qwen token 上限，而不是字符数。
- `embedding_metadata_reserve_tokens` 为论文标题、章节路径和字段标签预留 token；因此
  HybridChunker 的正文预算为 `max_tokens - reserve`，默认 `448`。
- `tokenizer_path` 必须指向 D 盘下载完成的 `Qwen3-Embedding-0.6B`。Step 2 只读取其
  tokenizer，不加载约 0.6B 参数的模型权重；Step 4 才会用同一目录实际生成 embedding。
- `inspection_output_dir` 的 JSONL 与检查报告只用于人工核验。运行 Step 2 的检查命令
  不会写 SQLite 或 FAISS；正式 SQLite 导入由 Step 3 的独立命令完成。

## Step 3 SQLite 参数

```yaml
database:
  path: storage/paperbase.sqlite3
  busy_timeout_ms: 5000
```

- `path`：SQLite 单文件数据库的位置。它保存论文、前置元数据和 chunks 的正式关联记录，
  不保存 PDF 二进制或 embedding 向量。
- `busy_timeout_ms`：SQLite 暂时被其他短任务占用时的最大等待时间。当前单进程导入下通常
  不会触发；保留该参数是为将来 Streamlit 查询与上传并发做准备。

完整 schema、幂等导入语义与检查方式见 [database.md](database.md)。

## Step 4 Embedding 参数

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

- `backend`：当前实现为本地 `sentence-transformers` Qwen 适配器；未来可在保留调用接口的前提下替换。
- `model_path`：必须是已下载到 D 盘的本地模型目录，运行时会强制离线加载。
- `batch_size`：一次进入 GPU 的文档数；应基于显存实测逐步调整。
- `normalize_embeddings`：应保持 `true`，以配合下一步 `IndexFlatIP` 的余弦相似度检索。
- `output_dir`：存储 Step 4 可重建工件，不写入 SQLite，也不会分配 `vector_id`。

详细数据契约、运行命令和验证方式见 [embedding.md](embedding.md)。

## Step 5 索引参数

```yaml
indexing:
  backend: faiss_flat_ip
  index_path: storage/paperbase.faiss
  manifest_path: storage/paperbase.faiss.manifest.json
```

- `backend`：当前为全局 `IndexIDMap2(IndexFlatIP)`；后续替换索引实现时保持 SQLite 的 `vector_id`
  映射契约不变。
- `index_path`：唯一正式 FAISS 二进制文件，包含向量与整数 `vector_id`，不包含论文原文或 metadata。
- `manifest_path`：与索引配套的完整性清单；索引构建和后续查询前都可据此检查文件、维度和映射快照。

完整的 FAISS—SQLite 关系、恢复机制和命令见 [indexing.md](indexing.md)。

## Step 6 混合检索参数

```yaml
retrieval:
  backend: hybrid_rrf
  dense_top_k_per_query: 20
  bm25_original_top_k: 10
  bm25_rewrite_top_k: 20
  fused_top_k: 40
  rrf_k: 60
  dense_original_weight: 1.0
  dense_rewrite_weight: 1.0
  bm25_original_weight: 0.7
  bm25_rewrite_weight: 1.0
  query_instruction: >
    Given a Chinese or English academic question, retrieve relevant passages
    from a collection of scientific papers that answer the question.
  query_rewrite:
    enabled: true
    max_lexical_keywords_en: 3
    max_context_turns: 4
    fallback_to_original: true
```

- 四个 `*_weight` 分别对应原始 Dense、原始 BM25、改写 Dense、改写 BM25 四条路径。英文关键词组只执行一次改写 BM25，不能靠增加关键词数量放大分数。
- `rrf_k` 是 RRF 的平滑常数。检索器只融合名次，绝不直接混加 FAISS 余弦相似度和 BM25 分数。
- `query_instruction` 只用于 Qwen 的 Query 编码，不写入论文、不修改文档侧向量，也不发送给生成 LLM。
- `query_rewrite` 控制 LLM 产生一条英文 `semantic_query`、一组英文 `lexical_keywords_en` 的上限，以及 `max_context_turns` 条最近上下文。上下文只用于消解追问指代；关闭 `enabled` 后仍可运行原始稠密 + 原始 BM25。
- API Key、模型地址、模型名、超时等不属于该文件，统一由根目录 `.env` 提供；模板见 `.env.example`。完整链路见 [retrieval.md](retrieval.md)。

## Step 7 重排序参数

```yaml
reranking:
  backend: bge_cross_encoder
  enabled: true
  model_id: BAAI/bge-reranker-v2-m3
  model_path: D:/AI_Workspace/AI_Models/hf_models/bge-reranker-v2-m3
  device: cuda
  batch_size: 8
  candidate_top_k: 40
  final_top_k: 5
  max_length: 1024
  normalize_scores: true
```

- `candidate_top_k`：正文 RRF 候选中实际交给 Cross-Encoder 打分的最大数量；参考文献不参与重排序。
- `final_top_k`：未传入命令行 `--top-k` 时默认输出的正文证据数量。
- `max_length`：单个“问题 + 标题/章节/正文”配对的最大 token 数，过长正文在尾部截断。
- `normalize_scores`：将模型原始 logit 转为 0 到 1 的相关性分数，方便人工检查；排序方向不受影响。
- 重排序模型只从 `model_path` 离线加载。若模型缺失、显存不足或推理失败，系统会保留 Step 6 的 RRF 排序，并在 JSON 中标记 `reranking_status: "fallback"`。

## Step 8 上下文扩展与回答参数

```yaml
context_expansion:
  enabled: true
  neighbor_window: 1
  max_total_tokens: 6000

answer_generation:
  enabled: true
  max_evidence_units: 8
```

- `neighbor_window`：命中 chunk 左右可纳入的连续同节邻居数量；`1` 对应“前一块 + 命中块 + 后一块”。
- `max_total_tokens`：所有正文证据组总预算；重叠窗口合并后仍保留全部命中种子，预算不足时跳过整个完整组，不截断块中间的文本。
- `max_evidence_units`：实际交给回答 LLM 的 `E#` 正文证据与 `R#` 参考文献证据总数上限。
- LLM、JSON Schema 或引用编号校验失败后，命令固定安全降级为检索和扩展证据，不输出未经验证的回答。

运行分块检查：

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.chunking.inspect --config .\config.yaml
```
