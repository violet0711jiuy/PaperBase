# PaperBase Agentic RAG v1 冻结架构

> 状态：`FROZEN FOR IMPLEMENTATION`  
> 文档性质：目标架构与实现契约，不代表功能已经全部完成。  
> 当前实施入口：Phase 1 — Retrieval Foundation。  
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

- QueryContext 与 Scope Resolver。
- Research Planner 与 Subtask。
- Research Tools 封装。
- Evidence Registry。
- Agent Decision Loop、Guard 与预算。
- 基础 Observation。

### Phase 3：Overview 与证据治理

- Formal Overview Repository。
- Evidence Qualification。
- Coverage Manager。
- Exhaustive / Candidate Verification 策略。
- Final Evidence Selector。

### Phase 4：工作流集成与产品化

- 将状态机接入 LangGraph 或等价编排层。
- Agentic Answer 流程与现有生成器集成。
- Trace、可观测性与失败诊断。
- UI 模式选择与任务进度展示。
- 多论文 Evaluation 与性能优化。

LangGraph 放在核心状态与契约稳定之后引入，避免把业务语义绑定到框架细节。

## 17. Phase 1 必须通过的测试

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
