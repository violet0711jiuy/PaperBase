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
| `paper_overview` | Workspace 与正式论文 Overview 共用的字段候选与上下文预算。 |
| `explain_section` | Explain Section 的分支覆盖和上下文预算。 |
| `agentic_research` | 单 Agent 循环的预算、逐论文 Query、Coverage 策略与最终 Evidence 上限。 |
| `retrieval` | 三路正文召回、RRF 权重、Query Planner 和 Bibliography 配额。 |

`agentic_research.enabled: true` 只表示功能允许启用；后端需要通过
`paperbase.agentic.create_agentic_research_service(...)` 装配 LangGraph、Research Tools
和真实 LLM。该入口从配置文件同目录的 `.env` 读取 `LLM_*`，并让 Understanding、
Planner、Candidate Selection、Decision、Evidence Analysis 和 Answer 复用同一连接池，
但分别使用节点级输出预算。结构化输出仍会在本地执行严格 Schema 和领域校验，不能把
Provider 的 `json_schema` 声明视为可信输入。

Phase 3 新增的 Coverage 与 Selector 配置包括：

- `coverage_candidate_min_queries_per_paper`：候选论文完成验证前的最少论文级查询数；
- `coverage_exhaustive_min_queries_per_paper`：穷举任务每篇论文的最少查询数，默认覆盖直接查询和替代表达；
- `final_evidence_max_units`：最终 Evidence 总上限；
- `final_evidence_max_per_paper`：单篇论文进入最终 Evidence 的上限。
- `evidence_admission_enabled`：是否启用 Agentic 专用的自动 Evidence 准入；
- `evidence_min_rerank_score`：仅在 reranker 成功时生效的最低准入分数；
- `evidence_source_overlap_threshold`：来源 chunk 集合的重叠去重阈值。结果型表格在
  没有其他候选时会保守保留，并交给后续 Claim 校验。

P2 新增的节点输出预算包括：

- `query_understanding_max_output_tokens`；
- `planner_max_output_tokens`；
- `candidate_selection_max_output_tokens`；
- `decision_max_output_tokens`；
- `evidence_analysis_max_output_tokens`。
- `evidence_analysis_cache_path`：Evidence Analysis 独立 SQLite 缓存路径。缓存键同时包含
  Evidence 指纹、问题、请求目标/信息缺口、Schema 版本和提示词版本；失败分析不会写入。

这些值只控制对应节点的可见输出上限；API Key、Prompt、LLM 原始输出和隐藏推理正文不会
写入诊断记录。运行记录只持久化 token 数、字符数、`finish_reason`、错误阶段和不含输入值
的 ValidationError 类型/字段路径。

正式 Overview 独立保存在 `storage/paper_overviews/<paper_id>/overview.json` 与
`context.json`。生成必须显式调用 `FormalOverviewRepository.generate(...)`；Agent 的
`get_paper_overview` Tool 只读取已有产物，缺失时返回 `unavailable`。

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
