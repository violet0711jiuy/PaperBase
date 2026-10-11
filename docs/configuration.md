# 配置与实际调用参数

核对日期：2026-10-10。字段以 [config.py](../paperbase/config.py) 和 [config.example.yaml](../config.example.yaml) 为准；实际节点装配见 [factory.py](../paperbase/agentic/factory.py)。

## 初始化

```powershell
Copy-Item .env.example .env
Copy-Item config.example.yaml config.yaml
```

YAML 保存路径、模型目录、检索参数和研究循环设置；`.env` 保存 provider、API key 与共享 client 参数。相对路径按配置文件目录解析。不要提交密钥。

## Agentic 模型节点

| 节点 | thinking | 输出契约 | 应用层输出 cap |
|---|---|---|---|
| Decision | 默认 false | `AgentDecisionDraft` strict JSON Schema | 不发送 |
| Planner | false | `ResearchPlanDraft` strict JSON Schema | 不发送 |
| Grounded Answerer | false | `GroundedAnswerDraft` strict JSON Schema | 不发送 |
| Query Understanding / Candidate Selection | false | JSON object + 本地校验 | 不发送 |
| Classic Answer | 基础 client 配置 | Classic draft | `answer_generation.max_output_tokens` |

Decision 关闭 thinking 是当前正式默认值。此前两例实测中它消除了 6 次约 60 秒的超时，但该观察仍需在新架构 live suite 中复验。`json_schema/strict=true` 不能代替本地 Pydantic、Guard 与 source-span 校验。

`llm_sampling_profile=qwen38_recommended` 会覆盖 Agentic 节点采样参数。生产判断应看实际 request diagnostics，而不能只看 `.env` 基础温度。

## Agentic 运行设置

| 配置 | 含义 |
|---|---|
| `enabled` | 是否启用研究服务 |
| `checkpoint_path` / `checkpoint_retention_days` | 当前语义版本的 SQLite checkpoint 与保留期 |
| `max_concurrent_runs` | 并发研究任务上限 |
| `max_agent_iterations` | Decision 研究轮数最终上限 |
| `max_queries_per_paper` | 每篇论文的 canonical query 上限 |
| `node_max_attempts` | 结构化节点与累计非法动作的恢复上限 |
| `decision_temperature` | configured profile 下的 Decision 温度 |
| `decision_thinking_enabled` | Decision thinking 开关，当前 false |
| `decision_timeout_seconds` | 单次 Decision 请求 timeout，不是 case 总时限 |
| `decision_transport_max_retries` | SDK transport retry；当前 0，避免与 node retry 相乘 |
| `planner_temperature` / `answer_temperature` | configured profile 下对应节点温度 |
| `decision_view_*` | Decision 原子视图的目录/历史条数及 gap 字符预算；不含 Evidence 正文预览 |
| `schema_critical_output_mode` | strict schema 节点的 provider 输出模式 |
| `top_k_per_paper` | 工具内部单篇检索返回量 |
| `evidence_admission_enabled` | Registry 前的确定性 Evidence 准入 |
| `evidence_min_rerank_score` | 只用于 normalized higher-is-better reranker score |
| `evidence_source_overlap_threshold` | source chunk 重叠去重阈值 |

Agentic Grounded Answerer始终读取 Registry 中全部去重 Evidence，因此没有 `final_evidence_max_units` 或同类答案截断配置。Decision 只读取 Evidence 元数据目录，Answerer 保留完整语义输入。

## 已废弃配置

以下旧字段不属于运行时 Settings；loader 仅在 YAML 边界丢弃它们，避免旧部署直接报错：

- `max_output_tokens`
- `max_evidence_per_analysis`
- `final_evidence_max_per_paper`
- `final_evidence_max_units`
- `max_no_new_evidence_rounds`
- `evidence_analysis_cache_path`
- `evidence_analysis_timeout_seconds`
- `evidence_analysis_max_output_tokens`
- `evidence_projector_batch_*`
- `projector_temperature`
- `decision_view_mode`

不要在新配置、代码或文档中重新使用这些字段。当前 checkpoint authority 是 `paperbase-agentic-grounded-answer-v1`；旧 checkpoint 不恢复执行。

## Classic 与检索配置

Classic 仍使用 `answer_generation.max_evidence_units/max_output_tokens`。Dense/FTS 候选量、RRF weights、reranker 和 section expansion 使用各自配置，与 Agentic Evidence Pool 的完整传递是两套边界。

实际 provider 健康检查会产生一次外部调用：

```powershell
.\.venv\Scripts\python.exe -m paperbase.llm.healthcheck --config .\config.yaml
```

最新评测口径见[当前验证状态](evaluation/current-baseline.md)。
