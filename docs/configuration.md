# PaperBase 配置说明

PaperBase 将可公开的配置模板与机器相关的运行配置分开：

- `config.example.yaml`：提交到 Git 的非敏感模板；
- `config.yaml`：本机实际配置，已由 `.gitignore` 排除；
- `.env.example`：环境变量模板；
- `.env`：LLM 密钥和服务地址，已由 `.gitignore` 排除。

## 初始化

```powershell
Copy-Item .env.example .env
Copy-Item config.example.yaml config.yaml
```

随后在本地 `config.yaml` 中填写 Docling、Embedding 和 Reranker 的模型目录，在 `.env` 中填写 LLM 服务参数。相对路径统一相对于 `config.yaml` 所在目录解析。

## 配置组

| 配置组 | 作用 |
|---|---|
| `project` | 项目标识。 |
| `storage` | PDF、解析检查产物和 staging 工件目录。 |
| `parsing` | Docling 后端、OCR、公式增强、页面清理和章节层级推断。 |
| `chunking` | tokenizer、token 上限、metadata 预算和相邻结构块合并。 |
| `database` | 正式 SQLite 路径和锁等待时间。 |
| `conversation` | 独立会话数据库与指代消解上下文窗口。 |
| `embedding` | Qwen Embedding 模型、设备、batch 和 staging 输出。 |
| `indexing` | FAISS 索引与 manifest 路径。 |
| `reranking` | BGE Cross-Encoder、候选深度和 Final Top-K。 |
| `context_expansion` | Final anchors 的同节相邻块扩展和 token 预算。 |
| `answer_generation` | Evidence 数量、相关性阈值和回答输出预算。 |
| `paper_overview` | 临时论文 Overview 的字段候选与上下文预算。 |
| `explain_section` | Explain Section 的分支覆盖和上下文预算。 |
| `retrieval` | 三路正文召回、RRF 权重、Query Planner 和 Bibliography 配额。 |

## 当前 Retrieval 参数语义

正文候选最多来自三条路径：

1. `dense_resolved`：对经过指代消解的 Query 做 Dense Retrieval；
2. `dense_semantic`：对英文 semantic rewrite 做 Dense Retrieval；
3. `bm25_keywords`：对确定性与 LLM 共同生成的英文关键词做 FTS5/BM25。

相关配置为：

- `dense_top_k_per_query`
- `bm25_keywords_top_k`
- `fused_top_k`
- `rrf_k`
- `dense_resolved_weight`
- `dense_semantic_weight`
- `bm25_keywords_weight`

当前没有 Original BM25 路径。Bibliography 使用独立的 `bibliography_fts`，仅在明确参考文献意图下执行，不进入正文 RRF 或 Cross-Encoder。

## 敏感信息边界

以下内容不得写入 YAML、文档或测试：

- API Key、Token 和密码；
- 带凭证的私有服务 URL；
- 本机用户名和私人目录；
- 未脱敏的论文原文或会话数据库。

`.env.example` 只能保留空值和公开的变量名称。

## 校验

配置由 Pydantic 严格校验。未知字段、类型错误和缺失必填项会在启动时失败，不会静默回退。

```powershell
.\.venv\Scripts\python.exe -c "from paperbase.config import load_settings; print(load_settings())"
```
