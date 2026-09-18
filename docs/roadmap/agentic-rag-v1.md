# PaperBase Agentic RAG v1 冻结架构

> 状态：`FROZEN FOR IMPLEMENTATION`  
> 文档性质：目标架构与实现契约，不代表功能已经全部完成。  
> 当前实施状态：Phase 1 至 Phase 4 已完成首个可运行闭环；Agentic Answer、SQLite 恢复、业务事件流和 Knowledge Base UI 已接通，PostgreSQL 与更高并发优化保留为部署演进项。
> 冻结原则：核心边界、数据契约和状态语义冻结；配额、阈值与排序权重保留为配置项。

## 1. 结论

PaperBase Agentic RAG v1 可以进入实现阶段。

本方案不推翻现有成熟 RAG，而是在现有检索与引用回答链路外增加一层研究编排能力。Agent 负责判断“下一步需要什么证据”，程序负责保证检索范围、数据合法性、预算、覆盖状态和引用可追溯性。

本次冻结的重点不是把所有未来能力一次设计完，而是固定以下不会轻易变化的边界：

- 现有 RAG 是底座，不重写。
- Agent 只做语义决策，不直接控制底层检索细节。
- 多论文检索按论文独立执行，可以批量调度，但不能共享召回配额。
- Tool 只返回事实性检索结果，不自行判断任务是否完成。
- Evidence Registry 统一分配运行期证据编号并维护来源。
- 覆盖率、停止条件和预算由程序确定性计算。
- 最终答案继续复用现有 Grounded Answer Generator。
- 正式知识库 Overview 与 Workspace Overview 分开存储、分开生成。

## 2. 目标与非目标

### 2.1 目标

v1 需要支持以下任务：

- 单论文事实、方法、结果和整体说明。
- 指定多论文比较。
- 针对不同论文生成不同检索问题。
- 多轮补充证据与缺口修复。
- 全库发现与逐论文验证的后续扩展。
- 对正文、参考文献、Overview 和章节上下文使用统一证据模型。
- 在预算耗尽、部分论文检索失败或证据不足时给出诚实的部分回答。

### 2.2 非目标

以下内容不进入 v1 核心实现：

- 多 Agent 协作。
- 独立验证 Agent。
- Web 搜索。
- 长期记忆与跨任务反思。
- Agent 自由修改 Top-K、Rerank 候选数等底层参数。
- 自动生成正式 Overview 作为读取 Tool 的隐式副作用。
- 复杂语义级重叠证据合并。
- 对“未找到证据”作数学意义上的否定证明。

## 3. 总体架构

```text
User Query
    ↓
Scope Resolution
    ↓
Query Understanding
    ↓
Research Planner
    ↓
Agent Decision
    ↓
Research Tools
    ↓
Scoped / Per-paper Retrieval
    ↓
RetrievedEvidence
    ↓
Evidence Registry
    ↓
Evidence Qualification + Coverage + Guards
    ├── 证据不足：回到 Agent Decision
    └── 可以结束：Final Evidence Selector
                         ↓
              Grounded Answer Generator
```

各层职责必须保持清晰：

| 层 | 职责 | 不负责 |
| --- | --- | --- |
| Scope Resolution | 把用户指代解析为可信 `paper_id` 范围 | 生成检索证据 |
| Query Understanding | 生成初始语义查询、关键词、意图和目标 | 多轮研究决策 |
| Research Planner | 将问题拆成有成功条件的子任务 | 调用底层 Retriever |
| Agent | 选择 Tool、论文和下一条查询 | 绕过范围与预算约束 |
| Research Tools | 执行只读检索并返回类型化结果 | 判断覆盖完成或直接写答案 |
| Evidence Registry | 去重、编号、维护证据来源与查询历史 | 判断答案是否正确 |
| Qualification / Coverage | 判断证据关系、归属及任务覆盖状态 | 改写原始检索结果 |
| Final Evidence Selector | 在上下文预算内选择平衡证据 | 生成新的事实 |
| Answer Generator | 基于入选证据生成带引用回答 | 再次自主检索 |

## 4. 与现有 RAG 的关系

现有能力继续作为稳定底座：

- Embedding 与 FAISS Dense Retrieval。
- SQLite FTS5 / BM25 Retrieval。
- RRF 融合。
- Reranker。
- Section-aware Evidence Expansion。
- Bibliography Retrieval。
- Grounded Answer Generator 与引用输出。

正式知识库 Retriever 与 Workspace Retriever 保持为两个独立 Adapter，由新的 `RetrievalService` Facade 统一调用。二者可以复用同一请求和返回契约，但不得为了表面统一而混合数据来源或生命周期。

经典 RAG 与 Agentic RAG 并存：

- 经典模式默认行为保持不变。
- Agentic 模式显式进入新的规划和循环。
- 在 Agentic 链路成熟前，不用自动 Router 替用户切换模式。

## 5. QueryContext 与范围解析

### 5.1 QueryContext

```text
QueryContext
├── original_query
├── resolved_query
├── semantic_query_en?
├── lexical_keywords_en
├── intent
├── scope
├── target_paper_ids
├── scope_source
├── retrieval_goal
├── default_evidence_targets
├── requires_complete_coverage
└── search_bibliography
```

冻结枚举：

```text
Intent = retrieve | explain | compare | synthesize

QueryScope = single_paper | selected_papers | knowledge_base

ScopeSource = trusted_workspace
            | ui_selection
            | resolved_from_query
            | knowledge_base_all

RetrievalGoal = find_evidence
              | find_examples
              | exhaustive_enumeration
              | comparison_support
              | synthesis_support

EvidenceTarget = paper_own_fact
               | method
               | result
               | claim
               | prior_work
               | field_overview
               | reference
               | general
```

### 5.2 不变量

- `single_paper` 必须恰好有一个 `target_paper_id`。
- `selected_papers` 必须至少有两个 `target_paper_ids`。
- `knowledge_base` 的 `target_paper_ids` 为空，由语料快照表达范围。
- 所有目标 ID 必须经过数据库校验，Agent 或 LLM 不得凭标题编造 ID。
- `requires_complete_coverage` 由程序根据 `retrieval_goal` 推导。
- `search_bibliography` 在最终执行前由程序规则确定，不能只依赖 LLM 自报。

### 5.3 Scope Resolver

论文范围解析按可信度顺序执行：

1. Workspace 或 UI 提供的可信上下文。
2. 已知精确 `paper_id`。
3. 精确标题匹配。
4. 受控别名或模糊匹配。
5. 仍有歧义时要求用户澄清。

LLM 可以从自然语言中提取论文名称，但最终只能使用 Resolver 返回并经数据库验证的 `paper_id`。

### 5.4 全库快照

```text
CorpusScopeSnapshot
├── snapshot_id
├── paper_ids
└── created_at
```

全库任务开始时固定成员集合，运行中新增论文不进入本次任务。`snapshot_id` 仅表示排序后 `paper_ids` 的成员快照，不表示 Chunk、Embedding 或索引版本。

## 6. Planner 与多论文动态查询

### 6.1 Research Planner

Planner 是独立 LLM 调用，不绑定 Tool。它输出研究目标和可验证子任务：

```text
ResearchPlan
├── goal
└── subtasks[]
    ├── id
    ├── goal
    ├── evidence_targets
    ├── success_criteria
    └── status
```

Planner 负责把“比较 A 和 B 的方法”拆成至少两个取证目标，再由 Agent 决定具体检索动作。

### 6.2 动态查询原则

多论文问题不能只用一条全局 Query 抢同一个 Top-K。正确模式是：

```text
A 的方法证据 ← 面向 A 的查询
B 的方法证据 ← 面向 B 的查询
        ↓
     联合比较
```

第一轮可以把多篇论文的检索请求批量提交，以降低延迟；但 Batch 只是执行优化，批内每篇论文可以拥有不同 Query、关键词、证据目标和缺失信息。

后续轮次根据 Observation 只补检仍缺证据的论文。例如 A 已有充分方法证据、B 只有背景描述时，只针对 B 生成更精确的 Method 查询或读取章节上下文。

### 6.3 PaperSearchSpec

```text
PaperSearchSpec
├── paper_id
├── query
├── lexical_terms = []
├── evidence_targets = set()
├── missing_information = []
└── target_subtask_id?
```

外层请求：

```text
SearchPapersRequest
├── query_origin = initial_understanding | agent_supplied
├── searches: list[PaperSearchSpec]
└── top_k_per_paper  # 内部配置，不暴露给 Agent
```

`query_origin` 是硬分支：

- `initial_understanding` 可以使用现有 Query Rewriter。
- `agent_supplied` 表示 Agent 已经给出定向 Query，必须绕过 Query Rewriter。
- Agent 提供的 Query 直接进入 Dense Retrieval。
- `lexical_terms` 进入 BM25；为空时使用确定性的 `extract_lexical_terms`，不得再调用 LLM。

## 7. Research Tools

v1 的最小 Tool 集合：

| Tool | 作用 |
| --- | --- |
| `search_knowledge_base` | 在知识库范围发现候选证据或论文 |
| `search_papers` | 对一篇或多篇已解析论文执行异构定向检索 |
| `search_paper` | `search_papers` 的单论文便利封装，不建立第二套实现 |
| `search_references` | 在指定论文的 Bibliography 中检索 |
| `get_paper_overview` | 只读正式或 Workspace 对应来源的 Overview |
| `get_section_context` | 读取指定论文与章节的上下文 |

共同约束：

- Tool 只做检索或读取，不生成最终回答。
- Tool 不计算 `coverage_complete`。
- Tool 不分配 `E#` 或 `R#`。
- Tool 接收的论文范围必须是已验证 `paper_id`。
- Tool 输出必须保留足以追溯到原 Chunk 的来源字段。

## 8. Scoped Retrieval 与公平性

### 8.1 正式知识库范围检索

正式知识库必须支持真正的 `paper_id` 约束：

- BM25 / FTS5 查询使用 `paper_id IN (...)` 过滤。
- 当前语料规模较小时，Dense 可以检索 `index.ntotal` 后按允许的 Vector ID 过滤，从而保证论文范围内的 Top-K。
- 将来语料扩大后可以换成分片索引、带元数据向量库或其他范围检索实现，但不能改变上层契约。

“先取全库较小 Top-K，再过滤目标论文”不满足范围保证，禁止作为正式实现。

### 8.2 每篇论文独立配额

多论文检索必须按论文独立执行以下阶段：

```text
Paper A: Dense + BM25 → RRF → Rerank → Top-K → Expansion
Paper B: Dense + BM25 → RRF → Rerank → Top-K → Expansion
```

不得把 A、B 的候选先混合，再共享一个 RRF、Rerank 候选上限、最终 Top-K 或 Expansion Budget。否则强势论文会挤掉另一篇论文，破坏比较任务的证据公平性。

同一论文内部如果使用多条 Dense Query，必须先在该通道内归一化或聚合，再参与跨通道 RRF，不能因为命中更多改写 Query 而获得成倍通道权重。具体多 Query 归一化算法留待发现能力实现时确定。

## 9. 证据数据契约

### 9.1 RetrievedEvidence

Retriever / Tool 返回候选证据，尚无运行期编号：

```text
RetrievedEvidence
├── source_type
├── paper_id
├── paper_title?
├── seed_chunk_ids: list[str]
├── source_chunk_ids: list[str]
├── section_id?
├── section?
├── page_start?
├── page_end?
├── content
├── score?
└── retrieval_metadata: dict
```

`seed_chunk_ids` 使用复数，因为现有 Expansion 可能合并来自多个 Seed Window 的内容。`source_chunk_ids` 必须包含最终文本实际使用的全部 Chunk。

### 9.2 EvidenceRecord

证据进入 Registry 后形成：

```text
EvidenceRecord
├── evidence_id
├── stable_key
├── source_type
├── paper_id
├── paper_title?
├── source_chunk_ids
├── section_id?
├── section?
├── page_start?
├── page_end?
├── content
├── first_seen_iteration
├── last_seen_iteration
├── supporting_queries: list[str]
└── qualification?
```

Registry 在一次运行内分配：

- 正文、Overview、章节上下文等使用 `E#`。
- 参考文献证据使用 `R#`。

v1 精确去重键冻结为：

```text
stable_key = source_type + paper_id + sorted(source_chunk_ids)
```

部分重叠或语义近似证据的复杂合并延后。现有 Expansion 的父级复用可以保留为尽力优化，但不能成为正确性依赖。

## 10. ToolResult 与批处理状态

### 10.1 类型化信封

```text
ToolResult
├── status: ToolStatus
├── tool_name: ToolName
├── payload: ToolPayload
├── metadata: dict
└── error_code?
```

`ToolPayload` 使用带 `kind` 的判别联合，例如：

- `SearchPapersPayload`
- `KnowledgeBaseSearchPayload`
- `ReferenceSearchPayload`
- `OverviewPayload`
- `SectionContextPayload`

Schema Validator 必须校验 `tool_name` 与 `payload.kind` 匹配。

### 10.2 单论文结果

```text
PaperSearchOutcome
├── paper_id
├── status: ok | empty | error | unavailable
├── query
├── lexical_terms
├── evidence_candidates: list[RetrievedEvidence]
└── error_code?
```

### 10.3 聚合状态语义

| 批内情况 | 顶层状态 |
| --- | --- |
| A `empty`，B `ok` | `ok` |
| A `error`，B `ok` | `partial` |
| 全部 `empty` | `empty` |
| 全部 `error` | `error` |
| 全部 `unavailable` | `unavailable` |

`empty` 表示执行成功但没有结果；`partial` 表示至少有一个执行发生错误或来源不可用，同时仍有可用结果。单篇 `empty` 不得抹掉其他论文的成功结果。

ToolResult 是事实输出，不携带 Coverage 更新。Observation 由运行时根据 ToolResult、Qualification 和策略派生。

## 11. Evidence Qualification

进入 Registry 的证据可由 Agent 在 v1 中标注两个维度：

```text
relation = supports | contradicts | background_only | uncertain

ownership = this_paper | prior_work | general_background | unknown
```

这一步用于避免把 Related Work 中对他人方法的描述误判为“本文采用的方法”。v1 不增加独立 Verifier；必要时可通过提示模板、结构规则和回归集约束 Agent 的标注质量。

## 12. Coverage、停止条件与预算

### 12.1 PaperAssessment

```text
PaperAssessment
├── paper_id
├── status: unvisited | candidate | positive | no_support_found | uncertain
├── queries_tried
├── evidence_ids
└── rationale
```

### 12.2 CoverageState

```text
CoverageState
├── scope_paper_ids
├── assessments
├── strategy: exhaustive | candidate_verification
├── checked_paper_ids      # 派生
├── positive_paper_ids     # 派生
├── remaining_paper_ids    # 派生
└── coverage_complete      # 派生
```

`coverage_complete` 只表示范围内所有论文都执行过规定的验证策略，不表示系统证明了某个否定命题。`no_support_found` 的准确含义是“按当前策略未找到支持证据”。

### 12.3 Exhaustive 验证策略

完整覆盖任务对每篇论文依次执行：

1. 面向“本文方法/事实”的定向查询。
2. 使用替代术语再次查询。
3. 必要时读取 Method / Architecture 等章节上下文。
4. 标记为 `positive`、`no_support_found` 或 `uncertain`。

该策略属于 Phase 2 及以后，不阻塞 Phase 1 检索基础设施。

### 12.4 Guard 与预算

论文级 Guard 与运行级 Guard 必须分开：

- 某篇论文连续无新证据，只停止该论文的继续检索。
- 重复动作可以被拒绝，或要求更换 Query。
- 单篇停止不能自动结束整个多论文任务。
- 达到全局硬限制时，以 Partial 状态结束并说明未完成范围。

预算至少跟踪：

- `agent_iterations`
- `tool_calls`
- `retrieval_queries`
- `queries_per_paper`
- `llm_calls`

具体上限是配置项，不属于冻结架构。

## 13. Overview Repository

### 13.1 两类 Overview 独立存在

```text
Workspace Overview
storage/staging/<workspace_id>/overview/...

Formal Overview Repository
storage/paper_overviews/<paper_id>/overview.json
storage/paper_overviews/<paper_id>/context.json
```

Workspace Overview 服务临时工作区生命周期；Formal Overview 以正式 `paper_id` 为键，可在没有 Workspace 的情况下读取。二者不能互相假定存在。

### 13.2 生成与读取

正式 Overview 从正式数据库中的论文 Chunk 生成，可以复用现有上下文选择、LLM 生成和来源校验逻辑，但触发方式必须显式：

- 用户或管理员操作。
- 入库后的后台任务。
- 独立维护命令。

Agent 的 `get_paper_overview(paper_id)` 是只读 Tool：

- 已存在时返回 `ok` 和 Overview 证据。
- 尚未生成时返回 `unavailable`。
- 不在读取过程中静默调用 LLM 生成，以免隐藏延迟、费用和副作用。

正式 Overview 建议在 Evidence Registry 基础契约稳定后实现，不阻塞 Phase 1。

## 14. Reference Retrieval

Bibliography 继续作为独立检索源：

- 采用现有 BM25 / FTS5 能力。
- Agent 必须给出明确论文范围。
- 结果进入同一 Registry，但使用 `R#` 编号。
- `prior_work` 与 `this_paper` 必须在 Qualification 中区分。

全局参考文献发现不属于 v1 首批实现范围。

## 15. Final Evidence Selector

现有回答生成器存在上下文证据数量上限，因此完成研究后必须经过最终证据选择，不能简单取 Registry 的前若干条。

选择约束：

- 每篇 `positive` 论文至少保留一条核心证据。
- 比较任务的每个关键维度尽可能保留双边证据。
- 方法、结果和直接 Claim 优先于一般背景。
- 必要的 `R#` 可以进入最终上下文。
- 默认过滤仅为 `background_only` 且与回答无直接关系的证据。
- 优先满足 Research Subtask 的 `success_criteria`。
- 保留原有 `E#` / `R#`，便于 Answer Generator 直接引用。
- 总量不得超过生成器的上下文上限。

具体选择权重和上限是配置项，不属于冻结架构。

## 16. 分阶段实施

### Phase 1：Retrieval Foundation

第一阶段只建设可独立验收的检索底座：

1. 新增 `RetrievalService` Facade。
2. 接入 `FormalKnowledgeBaseAdapter`。
3. 接入 `WorkspacePaperAdapter`。
4. 支持正式知识库 `paper_id` 范围检索。
5. 支持 `agent_supplied` Query 绕过 Query Rewriter。
6. 增加确定性 Lexical Fallback。
7. 实现逐论文独立 RRF、Rerank、Top-K 和 Expansion Budget。
8. 实现异构 `search_papers` Batch。
9. 落地 `RetrievedEvidence`、`PaperSearchOutcome` 与 `ToolResult`。
10. 补齐 `section_id` 与完整来源链。
11. 建立契约和回归测试。

Phase 1 明确不实现：Planner、Agent Loop、Coverage、Qualification、Final Evidence Selector、LangGraph 和新的 Answer 流程。

### Phase 2：Planning 与 Agent Loop

- `[COMPLETED]` QueryContext 与 Scope Resolver。
- `[COMPLETED]` Research Planner 与 Subtask；非法 LLM 计划确定性降级。
- `[COMPLETED]` Research Tools 封装；Top-K 等底层上限只由运行时注入。
- `[COMPLETED]` Evidence Registry；正文使用 `E#`，参考文献使用 `R#`，精确来源去重。
- `[COMPLETED]` Agent Decision Loop、Scope Guard、重复动作保护与硬预算。
- `[COMPLETED]` 基础 Observation、逐论文无新增证据抑制与可审计 Action Trace。
- `[COMPLETED]` 动态多论文流程：允许先检索 A，再依据 Observation 为 B 生成不同 Query。

Phase 2 当前保持以下边界：

- 不做 Evidence Qualification、Coverage Manager 或 Final Evidence Selector。
- 不生成最终回答，也不改变现有经典 RAG 的回答链路。
- 正式 Overview Repository 仍属于 Phase 3；读取 Overview 不会隐式触发生成。
- 不引入 LangGraph、多 Agent、Web 搜索或 UI 自动 Router。

### Phase 2.5：最小 LangGraph 执行骨架

> 实施状态：`COMPLETED`。默认 Graph 实现在 `paperbase/agentic/graph.py`，checkpoint DTO 转换在 `paperbase/agentic/checkpoint_serde.py`；生产装配入口为 `paperbase.agentic.create_agentic_research_service`，手写 `ResearchRuntime` 只保留为迁移期行为基准。

LLM 生产边界已经收口：Query Understanding、Research Planner 与 Decision 使用同一个运行时 `OpenAICompatibleChatClient`；三类提示词分别位于 `paperbase/prompts/agentic_query_understanding.py`、`agentic_planner.py` 与 `agentic_decision.py`。外部响应先受 Provider JSON Schema 约束，再由 `paperbase/agentic/schemas.py` 的严格 Pydantic DTO 本地复验，最后执行可信论文范围、计划覆盖及已知 Section 引用等领域校验。Understanding/Planner 校验失败时确定性降级，Decision 校验失败时进入有界修复重试。

Phase 2.5 在 Phase 3 之前引入 LangGraph，但只替换编排机制，不增加新的研究语义。目标是尽早验证状态模型、循环路由和持久化边界，避免 Phase 3 继续依赖手写循环后再进行大规模迁移。

本阶段实现：

1. 增加 LangGraph 运行依赖，并在安装时使用明确的兼容版本范围。
2. 定义可序列化的 `AgentState`，至少包含：
   - Scope Resolution 与可选 Corpus Snapshot；
   - `QueryContext` 与 `ResearchPlan`；
   - 下一步 `AgentAction` 与最近一次 `ToolResult`；
   - Evidence Registry 快照、Observation 与 Action Trace；
   - Budget Usage、逐论文 Query 计数和无新增证据抑制状态；
   - 最终运行状态、停止原因、澄清信息与错误摘要。
3. 将现有业务组件映射为节点：
   - `resolve_scope`；
   - `understand_query`；
   - `build_plan`；
   - `decide_action`；
   - `validate_action`；
   - `execute_tool`；
   - `register_and_observe`；
   - `finish_run`。
4. 使用 conditional edges 表达以下跳转：
   - Scope 未解析时进入 `needs_clarification`；
   - `FinishAction` 直接结束；
   - 非法动作返回 Decision，超过上限后结束；
   - Tool 执行后经 Registry 与 Observation 返回 Decision；
   - 预算耗尽、所有目标均被抑制或出现致命错误时结束。
5. 使用 LangGraph Runtime Context 注入数据库、LLM Client、RetrievalService、Research Tools 和配置；这些运行依赖不得写入可持久化 State。
6. 开发与测试阶段只使用内存 Checkpointer，验证 State 可以 checkpoint / resume；正式持久化后端留到 Phase 4。
7. 保留当前 `ResearchRuntime` 作为迁移期行为基准。完成等价性测试后，它只作为测试参考，不再作为默认生产入口。
8. `AgenticResearchService` 切换为调用编译后的 Graph；现有 Scope、Planner、Guard、Budget、Registry 和 Tool 仍保持框架无关。

实现约束：

- 当前 `EvidenceRegistry` 与 `BudgetAndGuardTracker` 是进程内可变对象，接入前必须拆出可序列化的 `RegistryState`、`BudgetState` 与 `GuardState`。Graph 节点只返回状态增量，不能把这些服务实例直接写入 State。
- Evidence 编号计数、已执行 Action Fingerprints、逐论文 Query 计数、无新增证据计数和被抑制论文集合都必须进入 checkpoint；否则恢复后可能重复查询或重新分配 `E#` / `R#`。
- 持久化 State 使用稳定 DTO 或 JSON-compatible projection；数据库 Row、模型响应对象、数据库连接、Retriever、LLM Client 和锁不得进入 State。
- Tool 节点必须携带稳定的 action fingerprint / invocation key。当前检索 Tool 均为只读操作，允许在崩溃边界安全重试，但 Registry 登记必须保持幂等。
- Node 不直接决定业务阈值；预算与停止判定继续调用 PaperBase Guard，conditional edge 只消费 Guard 的类型化结果。
- 不依赖 LangGraph 默认消息历史表示研究状态；论文 Scope、Evidence、Coverage 与 Budget 必须使用独立字段。

Phase 2.5 明确不实现：

- Evidence Qualification、Coverage Manager 与 Candidate Verification；
- Formal Overview Repository 与 Final Evidence Selector；
- Grounded Answer 节点；
- 正式数据库 Checkpointer、跨进程恢复与部署服务；
- UI Streaming、人工中断审批和多 Agent。

Phase 2.5 的退出条件：

- 手写循环与 LangGraph 对同一脚本化 Decision 序列产生相同的 Evidence IDs、停止状态和预算计数。
- “先查 A，再根据 Observation 使用不同 Query 查 B”的行为保持不变。
- 非法 Scope、重复动作和预算耗尽都在 Tool 调用前被阻止。
- A 的无新增证据抑制不阻塞 B；异构 Batch 仍逐论文统计。
- Graph State 不保存数据库连接、模型实例或其他不可序列化运行依赖。
- Checkpoint 后恢复不会重复执行已经登记成功的 Tool 动作或重新分配 Evidence ID。
- LangGraph 相关测试和当前全部回归测试同时通过。

### Phase 3：Overview 与证据治理

- [x] Formal Overview Repository：按 `storage/paper_overviews/<paper_id>/` 保存，显式生成、只读 Tool 不隐式生成。
- [x] Evidence Qualification：Decision 同轮输出受控的 relation / ownership 标注，只允许引用已登记 Evidence。
- [x] Coverage Manager：从 Scope、Observation 与 Qualification 重算 `PaperAssessment` 和 `CoverageState`。
- [x] Exhaustive / Candidate Verification 策略：穷举默认要求直接查询与替代表达；全库发现查询不计作候选论文验证。
- [x] Final Evidence Selector：过滤 background-only，优先本文直接证据，并保证多篇 positive 论文至少各有一条核心证据。
- [x] 接入 Phase 2.5 Graph：Coverage 进入 Decision State 与 checkpoint，`finish` 受 Coverage Gate 约束，终态输出 selected evidence。

Phase 3 的各项能力继续实现为框架无关的领域组件，再作为明确的治理或收尾步骤接入 Phase 2.5 建立的 Graph；不得把 Qualification、Coverage 或 Selector 逻辑直接写死在路由表达式中。

### Phase 4：LangGraph 产品化与完整链路集成

- [x] 在 Phase 3 已闭环的研究 Graph 后接入 Agentic Answer 节点，并与现有 Grounded Answer Generator 集成。
- [x] 接入正式 SQLite Checkpointer，定义 thread 身份、恢复、重复启动拒绝、删除和按活动时间保留策略；PostgreSQL 留给多进程部署。当前 DeltaChannel 不做不安全的中间 checkpoint 压缩。
- [x] 增加 interrupt/resume、业务化 Streaming、Trace、错误码与回答降级诊断。
- [x] 完成 UI 的快速问答 / Agentic Research 模式选择、全库 / 指定论文范围和任务进度展示；范围不明确时以稳定回答契约进入会话。
- [x] 增加多论文引用有效性、正例论文覆盖与 Evidence 平衡度 Evaluation，并提供本地并发硬上限。
- [x] 删除生产 Service 中的手写循环入口；`ResearchRuntime` 只保留给框架无关的领域回归测试。

LangGraph 在 Phase 2 领域契约稳定后、Phase 3 证据治理前引入。框架负责节点调度、状态持久化、恢复和流式执行；PaperBase 负责 Scope、Plan、Guard、Budget、Evidence 与 Coverage 等业务语义。两层必须保持可独立测试。

## 17. 必须通过的测试

### 17.1 Phase 1

- Agent 提供 Query 时不调用 Query Rewriter。
- 指定论文范围时不返回范围外 Chunk。
- 每篇论文拥有独立最终 Top-K。
- Rerank Candidate Limit 不在论文间共享。
- Expansion Budget 不在论文间共享。
- Batch 中部分失败时保留其他论文的成功结果。
- Batch 中不同论文实际使用各自 Query。
- A 返回 `empty` 时不影响 B 的 `ok` 结果。
- `lexical_terms` 为空时走确定性提取器且不调用 LLM。
- `section_id` 来自真实来源，不由展示文本反推。
- 每条 `RetrievedEvidence` 包含 `paper_id`、Seed / Source Chunk IDs 和 `section_id`。
- Tool 层不提前分配 `E#` / `R#`。
- 现有经典 RAG 回归测试全部继续通过。

### 17.2 Phase 2

- Workspace Scope 不依赖论文已存在于正式知识库。
- UI 选择与显式 `paper_id` 必须通过数据库验证；同名论文返回歧义状态。
- 知识库级任务在运行开始时冻结 Corpus Snapshot。
- Query Understanding 或 Planner 的结构化输出无效时必须确定性降级，且不能改变可信 Scope。
- 比较任务的默认计划必须逐论文覆盖 A、B，并允许两次检索使用不同 Query。
- Agent 不能传入 Top-K；未知论文、未知 Subtask、重复动作在 Tool 执行前拒绝。
- Evidence Registry 对相同来源精确去重，并保留所有 supporting queries。
- Batch 的无新增证据计数必须逐论文计算；A 被抑制不能阻塞 B。
- Tool 后端失败必须返回类型化错误或逐论文错误，不能让循环无界等待。
- 迭代、LLM、Tool、检索总量和逐论文 Query 均受硬预算限制。
- Phase 1 与经典 RAG 的全部回归测试继续通过。

### 17.3 Phase 2.5

- Graph 与手写循环对同一动作序列产生相同 Evidence IDs、Trace、停止状态和预算计数。
- Checkpoint State 只能包含 JSON-compatible DTO，不依赖未注册 Python 类型反序列化。
- 使用严格 MsgPack 模式运行 checkpoint/resume 测试，不能出现类型白名单警告。
- 在 `execute_tool` 后中断并用新 Graph 实例恢复时，不重复执行已完成 Tool。
- Evidence 编号、Action Fingerprints、逐论文预算和抑制状态在恢复后保持不变。
- 动态 A→B Query、非法 Scope、重复动作、预算耗尽与异构 Batch 行为保持不变。
- 已存在的 thread ID 只能恢复，不能静默启动第二个任务；未知 thread ID 不能恢复。
- Graph State 不包含数据库、LLM、Retriever、Tool Executor 或其他运行依赖。
- LangGraph 专项测试和此前全部回归测试同时通过。

### 17.4 Phase 3

- Qualification 只能引用 Registry 中已有的 Evidence ID，未知或重复引用必须拒绝。
- Related Work、一般背景和本文工作使用独立 relation / ownership 语义，不能混为直接证据。
- Candidate Verification 必须执行论文级验证；全库 Discovery Query 不计作候选论文验证。
- Exhaustive 策略默认要求每篇论文至少完成直接 Query 与替代表达 Query。
- `no_support_found` 的理由必须限定为当前策略下未找到，不能表达为事实不存在。
- Agent 在 Coverage 未完成时请求 `finish` 必须被拒绝，且不能绕过预算与非法动作计数。
- Final Evidence Selector 必须过滤 background-only，并在容量允许时覆盖每篇 positive 论文。
- 正式 Overview 缺失时 Tool 返回 `unavailable`，不能在读取路径隐式调用 LLM 生成。
- Qualification、Coverage、Registry 和稳定 Evidence ID 在 checkpoint/resume 后保持一致。
- Phase 3 专项测试和此前全部回归测试同时通过。

### 17.5 Phase 4

- 所有终态先经过 Agentic Answer 节点；澄清、无证据、生成失败和正常回答均返回非空稳定契约。
- 最终回答只能引用 Final Evidence Selector 选出的 E#/R#，回答正文保留 Markdown 段落格式。
- SQLite checkpoint 关闭并重新打开后仍能恢复；`execute_tool` 后恢复不能重复已完成 Tool。
- Streaming 只暴露稳定业务事件，不向 UI 泄漏 Graph State、Prompt、数据库连接或模型对象。
- 相同 thread ID 不能静默开启新任务，未知 thread 不能恢复，明确删除后不能再读取。
- 回答生成异常时保留已选 Evidence，并在 `error_codes` 中提供可诊断错误类型。
- Knowledge Base 的快速问答与 Agentic Research 共用会话历史，Agentic turn 保存 thread ID、回答状态和 Evidence 快照。
- 多论文 Evaluation 必须检测伪造引用、正例论文漏覆盖和 Evidence 严重失衡。
- 超过本地并发上限时立即返回明确忙碌错误，不创建半完成任务。
- Phase 4 专项测试、真实 LLM Answer smoke test 和此前全部回归测试同时通过。

## 18. 仍可调但不阻塞实施的内容

以下事项不再作为架构冻结前置条件：

- 每篇论文的具体 Top-K、Rerank Candidate Limit 和 Expansion Budget。
- 各类运行预算的默认数值。
- Final Evidence Selector 的打分权重。
- Candidate Discovery 的聚合分数与阈值。
- 多 Query 通道内归一化的具体公式。
- Batch 串行或并行执行。
- 部分重叠 Evidence 的合并优化。
- Agent Trace 的 UI 表现形式。

这些内容应通过配置、评测和实现反馈迭代，不应反向修改已经冻结的职责边界和数据语义。

## 19. 冻结决策清单

以下决策从本版起视为冻结：

- Agent 负责语义选择，程序负责确定性约束。
- 正式与 Workspace Retrieval 使用独立 Adapter。
- 多论文采用动态、逐论文 Query；Batch 仅是调度方式。
- 每篇论文独立完成融合、重排、截断和扩展。
- Agent Query 不再经过 Query Rewriter。
- ToolResult 与 Observation、Coverage 分离。
- Tool 产生 `RetrievedEvidence`，Registry 产生 `EvidenceRecord` 和编号。
- v1 使用基于来源 Chunk 集合的精确去重。
- `empty` 与 `error` 语义严格区分。
- 全库任务固定 `CorpusScopeSnapshot`。
- Formal Overview Repository 独立存在，读取不隐式生成。
- `no_support_found` 不是“证明不存在”。
- 最终回答前必须执行跨论文平衡的 Evidence Selection。
- 第一阶段只实现 Retrieval Foundation。

如实现中发现这些契约无法满足，应先形成 Architecture Decision Record，说明冲突、替代方案和迁移影响，再讨论解冻；不得在单个提交中静默改变语义。

## 20. 与旧 Roadmap 的关系

[`multi-paper-rag.md`](multi-paper-rag.md) 保留为多论文能力演进背景。本文件吸收了后续关于动态 Query、工具边界、正式 Overview、Evidence Registry、Coverage 和批处理语义的讨论，是 Agentic RAG v1 的主要实施依据；两者冲突时以本文件为准。
