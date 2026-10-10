# 配置与实际调用参数

核对日期：2026-10-06。字段与校验以 [config.py](../paperbase/config.py)、[config.example.yaml](../config.example.yaml) 和 [.env.example](../.env.example) 为准；节点实际装配见 [factory.py](../paperbase/agentic/factory.py) 与 [Decision](../paperbase/agentic/decision.py)。

## 初始化

```powershell
Copy-Item .env.example .env
Copy-Item config.example.yaml config.yaml
```

YAML 保存路径、模型目录、检索参数和研究循环设置；.env 保存 provider、API key 和共享 client 参数。相对路径以配置文件目录解析。不要提交本地密钥或把完整运行配置打印到共享日志。

当前 `.env` / `.env.example` 统一使用 ModelScope / Qwen/Qwen3.8-Flash-Next；`config.yaml` / `config.example.yaml` 的 Agentic 采样统一为 qwen38_recommended。模板填自己的 API key，并按本机调整模型目录。根目录只保留这两份 YAML；单次实验配置已移除，历史副本位于本地 output/config_cleanup_20261006/historical_configs/，不参与日常启动。

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_API_KEY=填写自己的密钥
LLM_MODEL=Qwen/Qwen3.8-Flash-Next
LLM_STRUCTURED_OUTPUT_MODE=json_object
LLM_THINKING_ENABLED=false
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=30
LLM_MAX_TOKENS=1600
```

上面的 temperature/mode 是共享基础值，**不是每个节点最终值**。模型名和服务可用性需在实际端点确认。

## 各节点的有效模型配置

下表是当前 qwen38_recommended 配方；共享基础 client 参数和 configured 兼容配方分别定义。

| 节点 | temperature | 输出 mode | 自定义输出 token 上限 |
|---|---|---|---|
| Agentic Decision | 1，thinking=true | 明确覆盖为 json_schema，请求 strict=true | 移除上限参数 |
| Agentic Planner | 0.7，thinking=false | json_schema | 移除上限参数 |
| Agentic Projector | 0.7，thinking=false | json_schema | 移除上限参数 |
| Agentic Answer | 0.7，thinking=false | 最终 generate() 明确覆盖为 json_schema | 移除上限参数 |
| Agentic Query Understanding / Candidate Selection | 0.7，thinking=false | json_object | 移除上限参数 |
| Classic Answer | 共享基础配置 | 共享基础配置 | answer_generation.max_output_tokens |

上表的 native json_schema 描述 openai_compatible 路径。当前模板明确选择 schema_critical_output_mode=json_schema。Decision 自身覆盖为 json_schema；Agentic Answer 虽在 factory 接收该配置，但 generate() 在有 research_requirements 时再次覆盖为 json_schema，因此最终 mode 也不能仅凭 factory 推断。schema-critical client view 不发送项目自定义 max_tokens，但 provider 自身仍可能有长度上限。

Decision 使用 decision_timeout_seconds（当前 60 秒）和 decision_transport_max_retries（当前 0）。SDK facade 共享 HTTP 资源但重试设置独立，节点仍按原 node_max_attempts 恢复，避免 SDK 重试和节点尝试相乘；其他节点 SDK 重试不变。请求 timeout 不是整个 case 的墙钟 deadline。Projector 使用 evidence_analysis_timeout_seconds；Answer 使用 answer_generation.timeout_seconds。所有节点共享 provider/client 基础实现，没有第二套 structured-output client。json_object 仍走本地模型校验；json_schema/strict=true 也不能替代本地验证与 Guard，实际 provider 可能不执行部分 keyword。

审计应查看实际请求的 temperature、response mode、Schema 与 attempts，不能仅凭 YAML/.env 推断。

## Agentic 循环与证据设置

| 配置 | 实际含义 |
|---|---|
| enabled | 是否允许创建研究服务 |
| checkpoint_path / checkpoint_retention_days | 独立 SQLite checkpoint 与保留期 |
| max_concurrent_runs | 服务并发上限，默认 1 |
| max_agent_iterations | research iteration 上限，默认 8；即时替代候选不增加 iteration |
| max_queries_per_paper | 原有逐篇查询限制，默认 4 |
| node_max_attempts | 结构化节点尝试上限；也用于累计非法 iteration 的停止保护，默认 2，合法动作不 reset |
| decision_temperature | Decision 独立采样温度，范围 0–2；不改变其他节点温度 |
| decision_thinking_enabled | Decision 独立思考开关，当前配置与模板均为 false；评测中关闭后消除了 Decision 超时并降低总耗时。实际 OpenAI-compatible 请求为 extra_body.enable_thinking；不改变输出 Schema 或 Guard |
| decision_timeout_seconds | Decision 独立请求超时，默认/当前 60 秒；原节点 attempt pool 保留 |
| decision_transport_max_retries | Decision SDK 内部重试次数，默认/当前 0；请求恢复由 node_max_attempts 管理，其他节点不变 |
| llm_sampling_profile | 当前与模板均为 qwen38_recommended；configured 作为兼容模式沿用独立温度。profile 覆盖 Agentic 节点采样，Classic/base client 不变 |
| decision_view_mode | compact 为 factored View；shadow 对照完整输入序列化，两者共享当前 System Prompt |
| decision_view_* | 支持/soft history/section/文本窗口大小；不裁剪 Runtime hard constraints |
| top_k_per_paper | 工具内部单篇返回上限，默认 5，不是模型 Tool 参数 |
| evidence_admission_enabled | Registry 前确定性证据准入 |
| evidence_min_rerank_score | 仅对 normalized、higher-is-better、ScoreKind.RERANKER 的 score 使用，默认 0.05 |
| evidence_source_overlap_threshold | source chunk Jaccard 重叠处理阈值，默认 0.8 |
| evidence_projector_batch_size | Projector batch 执行大小，默认 4；assessment/cache/Integrity 仍按 Evidence 与 scope |
| evidence_analysis_cache_path | 独立语义 cache SQLite |
| final_evidence_max_units | 最终答案 Evidence 上限，默认 8 |

恢复预检复用原 Guard，在现有 attempt pool 内至多一次即时替代候选；不增加新预算、不自动生成 query。NoProgressSignal 只是策略提示，未配置连续 N 次强制 Finish。

当前配置使用 qwen38_recommended：思考节点（当前仅 Decision）temperature=1、top_p=0.95、presence_penalty=0；非思考节点 temperature=0.7、top_p=0.8、presence_penalty=1.5。共同扩展参数 top_k=20、min_p=0、repetition_penalty=1，通过 extra_body 发送；top_p/presence_penalty 是标准 SDK 参数。profile 覆盖 Decision/Planner/Projector/Answer 的温度，避免内部 client view 再变回旧值；node 构造、超时与修复 view 均保留配方。Schema 与 Guard 不变，provider 若拒绝参数，记录请求错误，不自动丢弃参数或降低输出 mode。

来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage)。推理强度 reasoning_effort 与历史思考保留 preserve_thinking 未加入本次采样配方；现有 Graph/消息历史方式不变。

## Legacy compatibility

agentic_research.max_output_tokens、evidence_analysis_max_output_tokens、evidence_projector_batch_base_tokens、evidence_projector_batch_per_evidence_tokens、evidence_projector_batch_max_output_tokens 仍接受旧配置，但 Agentic 节点已不使用这些输出 cap；已从推荐 config.example.yaml 移除。保留字段是为了 Settings DTO、旧 deployment 和回归解析兼容，不是可生效参数。所有 Agentic 模型请求均省略应用层 max_tokens；provider 默认输出额度和上下文窗口仍然适用。Decision View 的历史数量与字符窗口、迭代和查询预算独立保留。

旧 final_evidence_max_per_paper、max_evidence_per_analysis、max_no_new_evidence_rounds 在兼容加载中移除，不应当成可生效控制参数。其他未知字段由模型校验拒绝；不要添加看似有效但无人读取的配置。

project.name 当前仅为必填元数据，没有运行时读取；retrieval.backend 当前也没有路由消费者，生产工厂固定创建 HybridRetriever。两者保留原配置 DTO，不应把修改其值理解为日志标识或切换 Retriever。其他 backend（parsing/chunking/embedding/indexing/reranking）有真实工厂校验，不能混为一谈。

共享 LLM_THINKING_ENABLED 会被 loader 解析，但基础 OpenAI-compatible client 构造目前固定 enable_thinking=false；ZhipuAI 路径和显式节点 view 另行处理。当前 Agentic thinking 由 decision_thinking_enabled / profile view 显式设置。基础开关不一致是单独 correctness issue，本轮未改。

配置可加载不等于 checkpoint 可恢复。当前 semantic authority 是 projector-requirement-directed-multi-projection-v4；历史 v2/v3 仅用于 parse/inspect/offline replay，resume/resume_stream 拒绝。

## 检索、存储与生成

Classic 的 Query Resolution/Rewrite 开关及 lexical 数量由 retrieval.query_rewrite 控制；Agentic 通过 Decision Action 和 canonical compiler 生成有效请求，Query Understanding hints 不自动接入 Retriever。Dense/FTS 候选数、RRF weights 与 reranker top-k 使用各自配置，不由 Evidence raw_score 反推。

正式正文与 bibliography 索引隔离；正式 Overview 只读已生成产物；临时 workspace 不和正式 KB 混合。详见[存储](pipeline/storage-and-indexing.md)和[两条检索路径](pipeline/retrieval-and-generation.md)。

## 审计与验证

普通 LLM diagnostics 记录节点、attempt、状态、耗时、tokens 与安全错误。现有 Evidence/Action audit 可包含事实值、source anchors、检索请求；不能宣称所有 public audit 都无输入内容。显式 debug/实验 trace 可以保存实际请求，分享前应按其产物范围检查。密钥与 hidden reasoning 不进入交付日志。

预算逻辑计数不等于实际 SDK requests：Answer repair/retries、Decision 多候选和 SDK retries 要分别统计。最近测试和 live 状态见[当前验证状态](evaluation/current-baseline.md)。

```powershell
.\.venv\Scripts\python.exe -m paperbase.llm.healthcheck --config .\config.yaml
```

healthcheck 会实际调用 provider。离线文档检查和单元测试无需执行它。
