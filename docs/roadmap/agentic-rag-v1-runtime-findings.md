# Agentic RAG v1 真实运行发现与修订计划

> 状态：`P0-P1_IMPLEMENTED / P2-P4_PLANNED`
> 记录日期：2026-09-19
> 文档性质：Phase 4 首个真实闭环后的问题记录、架构修订建议与实施优先级。
> 基线关系：本文不直接覆盖《Agentic RAG v1 冻结架构》；涉及冻结语义的修改，应在实现前形成明确契约和迁移测试。

## 1. 记录目的

Phase 4 已接通 Agentic Answer、SQLite checkpoint、业务事件流和 Knowledge Base UI。一次全库方法归纳问题的真实运行表明：最终回答链路可以工作，但 Candidate Discovery、Planner、Decision、Evidence Qualification 和 Coverage 之间仍存在职责耦合与上下文膨胀。

本文记录：

- 真实运行中观察到的行为；
- 已复现的失败模式与证据；
- 当前实现为什么没有生成 Qualification；
- Candidate Selection、Evidence Digest 和 Claim Qualification 的目标设计；
- 后续修改的优先级、推荐实施顺序和验收场景。

## 2. 已验证正常的能力

- 正式知识库 Scope 可以冻结为 `CorpusScopeSnapshot`。
- Query Understanding 能识别综合归纳和穷举式覆盖意图。
- Planner、Decision、Retrieval、Registry、Answer 和 checkpoint 已形成完整调用链。
- `search_papers` 可以对多篇论文独立执行查询并登记稳定 Evidence ID。
- Answer Generator 的输出预算提高到 4000 后，相同问题可以生成有效结构化回答，不再出现最终 Answer JSON 被截断的问题。
- Answer Generator 能在原始 Evidence 中主动识别部分明显噪声，并输出部分覆盖声明。

以上结论仅说明首个闭环可运行，不代表中间证据治理已经达到可信生产标准。

## 3. 真实运行过程摘要

测试问题属于“全库范围内归纳当前有哪些某类方法”的场景。运行过程如下：

1. Query Understanding 将问题识别为 `synthesize + exhaustive_enumeration`。
2. Planner 生成 7 个细分类子任务，提前枚举了多种预期方法类别。
3. 第一轮 Decision 对 Scope 中每篇论文提交相同的宽泛方法检索 Query。
4. Tool 调用成功并登记 16 条 Evidence；其中只有一部分直接相关，其余包括相邻主题、通用时间序列和其他预测任务的内容。
5. 后续 Decision 应当同时选择下一步 Action 并生成 Evidence Qualification，但连续三轮均未产生可接受结果。
6. 运行以 `partial / invalid_action_limit` 结束研究循环。
7. Final Evidence Selector 选择了 8 条 Evidence，其中仍包含若干无关内容。
8. Answer Generator 最终只引用了 4 条直接相关 Evidence，成功生成带覆盖声明的部分回答。

关键运行统计：

| 指标 | 结果 |
|---|---:|
| Agent 迭代 | 4 |
| Tool 调用 | 1 |
| 论文级检索 Query | 4 |
| 登记 Evidence | 16 |
| 无效 Decision | 3 |
| 送入 Answer 的 Evidence | 8 |
| 最终实际引用 | 4 |
| 研究终态 | `partial` |
| 停止原因 | `invalid_action_limit` |
| Answer 状态 | `success` |

## 4. 已确认的问题

### 4.1 Planner 过度展开

Planner 在尚未检索前就按物理方法、统计方法、机器学习、深度学习、混合方法、比较评估和新兴场景拆出 7 个子任务。

影响：

- 把尚未得到证据的方法分类提前固化为计划；
- 容易让后续检索围绕预设答案寻找材料；
- 每轮 Decision 都携带过多 Subtask 状态；
- 增加 Prompt 长度和模型推理难度。

Planner 应规划“需要收集什么证据”，不应在检索前替证据完成领域分类。

### 4.2 全库范围被误解为逐篇深度检索

当前 `knowledge_base_all + exhaustive` 会推动 Agent 对 Scope 中每篇论文执行论文级查询，即使论文标题和主题明显无关。

正确语义应为：

```text
全库范围 = 允许搜索的可信边界
全库覆盖 = 所有论文经过元数据筛选
深度覆盖 = 所有候选论文经过正文验证
```

全库范围不应等价于对每篇论文执行多轮正文检索。

### 4.3 缺少正式 Candidate Selection

当前 Scope Resolution 只回答“哪些论文允许被搜索”，不回答“哪些论文值得为当前问题进入正文检索”。因此明显无关论文仍会消耗 Query、Rerank、Expansion、Decision 上下文和 Answer 上下文。

### 4.4 第一轮 Query 没有体现论文差异

系统契约已允许不同论文使用不同 Query，但本次第一轮对所有论文使用了同一个宽泛 Query。作为初始筛查可以接受，但后续本应根据每篇论文的证据与缺口动态生成不同 Query；由于 Decision 失败，该过程没有发生。

### 4.5 Decision 职责过重

当前一个 Decision 响应同时承担：

- 选择下一步 Action；
- 阅读近期 Evidence；
- 判断 Evidence 与问题的关系；
- 判断内容归属；
- 输出最多 12 条 Qualification；
- 满足复杂 Union JSON Schema。

Action 或 Qualification 任一部分失败，整个响应都不会进入 Graph State。

### 4.6 Evidence 可见性采用“最后 12 条”

当前实现使用 `state.evidence[-12:]`。本次运行中，最重要的前几条直接相关 Evidence 被排除在 Decision Prompt 外，后登记的相邻主题、表格片段和低相关内容反而占据了主要上下文。

Evidence 的可见性应按相关性、当前子任务、论文多样性、Qualification 状态和信息缺口选择，不能按登记时间简单截断。

### 4.7 Decision 输出预算耗尽

当前普通结构化调用使用 1600 completion token。使用检索后的同一份 Decision State 重放两次，得到：

| 尝试 | Prompt token | Completion token | `finish_reason` | 最终 content | 隐藏推理字符数 |
|---:|---:|---:|---|---:|---:|
| 1 | 7880 | 1600 | `length` | 空 | 7220 |
| 2 | 7894 | 1600 | `length` | 空 | 7295 |

这次重放没有得到“格式错误的 JSON”，而是模型把全部 completion 预算消耗在隐藏推理中，最终没有生成可见 JSON。第二次修复请求继续携带接近相同的长上下文和相同预算，因此再次失败。

说明：历史 checkpoint 没有保存当时的原始 LLM content，所以上述数字是相同状态、模型和配置的诊断重放，不应表述为三次历史调用的原始日志；但它复现了与历史结果一致的具体失败路径。

### 4.8 Qualification 没有生成

第一轮 Decision 发生在检索前，当时没有 Evidence，因此 `qualifications=[]` 是正确结果。

检索后，后续 Decision 本应同时输出 Action 和 Qualification。但模型没有生成最终 JSON，`DecisionResult` 无法构造，Graph 不会提交其中任何部分。当前 `qualify_and_assess` 节点只负责应用 Decision 已经生成的 Qualification，不会独立生成 Qualification。

因此本次运行没有任何 Evidence Qualification。

### 4.9 Qualification 不能可靠约束最终回答

当前实现存在三个缺口：

1. `qualification is None` 的 Evidence 默认仍然可以进入 Final Evidence Selector；
2. Qualification 主要影响 Coverage 和排序，没有传递给 Answer Generator；
3. Evidence 级别只有一个 `relation + ownership`，无法表达同一段落中同时存在 prior work 和 this paper claim。

本次最终回答对噪声的过滤主要来自 Answer LLM 自己重新阅读原文，而不是 Qualification 的确定性保护。

### 4.10 诊断信息不足

当前 Trace 将空响应、JSON 语法错误、Schema 错误、业务引用错误和 Guard 拒绝压缩为相近的 `invalid_decision` 信息，checkpoint 没有保存足够的安全诊断元数据，导致历史失败无法准确还原。

## 5. 目标修订方案

### 5.1 Candidate Selection：条件必经节点

Candidate Selection 可以由模型进行语义判断，但不应完全由 Agent 决定“是否需要筛选”。推荐边界：

- Workflow 根据 Scope 决定是否必须进入 Candidate Selection；
- Candidate Selector 根据问题和论文元数据决定候选、待确认和排除集合；
- 程序验证所有 paper_id 均来自冻结快照，并验证集合完整、互斥；
- Agent 后续可以请求扩大或重新检查候选集合，但不能绕过可信 Scope。

建议组件：

- `PaperMetadataRepository`：读取标题、摘要、关键词和正式 Overview；
- `CandidateSelector`：严格结构化语义筛选服务；
- `candidate_selection`：LangGraph 节点；
- 可复用的元数据查询 Tool：供后续扩展使用，但不是全库流程的可选捷径。

推荐输出：

```json
{
  "candidates": [
    {"paper_id": "paper_1", "relevance": "high", "reason": "直接讨论目标主题"}
  ],
  "uncertain": [
    {"paper_id": "paper_2", "reason": "元数据不足，需要浅检索确认"}
  ],
  "excluded": [
    {"paper_id": "paper_3", "reason": "研究对象属于其他预测任务"}
  ]
}
```

仅看标题可能产生假阴性，因此筛选应分层使用标题、摘要、关键词、Overview；`uncertain` 论文可以执行一次低成本浅检索。

### 5.2 Compact Planner

Planner 应只生成规模小、可审计的证据收集计划。普通综合问题建议最多 3 个子任务，Schema 上限建议收紧到 4。

本类问题的合理计划是：

1. 筛选与问题相关的候选论文；
2. 逐篇提取论文提出或综述的目标方法；
3. 合并、去重、分类并评估覆盖边界。

除非用户明确指定，Planner 不应提前枚举尚未由 Evidence 发现的具体方法类别。

### 5.3 Action-only Decision

Decision 只输出下一步 Action，不再同时输出 Qualification。它读取：

- Compact Plan；
- Candidate/Coverage 状态；
- 每篇论文的短 Digest；
- 已尝试 Query；
- `missing_information`；
- 预算和 Guard 状态。

Decision 不应读取无界原始论文段落。

### 5.4 Evidence Analysis：Digest 与 Claim Qualification 合并生成

Tool 返回后自动进入 Evidence Analysis，不把基础压缩做成由 Agent 自主决定是否调用的可选 Tool。

Evidence Analysis 在一次受控处理内完成：

- 相关性判断；
- Evidence Digest；
- 原子 Claim 提取；
- Claim Relation；
- Claim Ownership；
- 来源片段或位置映射；
- `missing_information` 更新。

推荐数据形态：

```json
{
  "evidence_id": "E1",
  "status": "qualified",
  "relevance": "high",
  "digest": "该段回顾已有方法，并介绍本文方法。",
  "claims": [
    {
      "claim_id": "E1:C1",
      "statement": "某方法属于论文回顾的已有工作。",
      "relation": "supports",
      "ownership": "prior_work",
      "source_excerpt": "..."
    },
    {
      "claim_id": "E1:C2",
      "statement": "本文提出新的方法框架。",
      "relation": "supports",
      "ownership": "this_paper",
      "source_excerpt": "..."
    }
  ]
}
```

Digest、Claim、Relation 和 Ownership 可以由同一节点、同一次模型调用生成，但必须保留独立字段，不能压成不可校验的自由文本摘要。

原始 `RetrievedEvidence` 始终保留，Digest 只是给 Agent 阅读的有界表示，不能替代引用原文。

### 5.5 Evidence 状态改为 fail-closed

建议显式区分：

- `pending`：尚未分析；
- `qualified`：Claim 已提取并通过校验；
- `irrelevant`：与问题无关；
- `failed`：Evidence Analysis 失败。

只有 `qualified` 的 Claim/Evidence 可以自动参与 Coverage 和最终事实结论。`pending` 与 `failed` 可以保留给人工核验，但不能默认成为可用证据。

Relation 建议增加 `irrelevant`，避免把完全无关内容勉强标成 `background_only` 或 `uncertain`。

### 5.6 Coverage 语义调整

Coverage 应分为：

- `scope_paper_ids`：冻结知识库边界；
- `screened_paper_ids`：已完成元数据筛选；
- `candidate_paper_ids`：需要正文检索；
- `uncertain_paper_ids`：需要浅检索确认；
- `excluded_paper_ids`：已排除并保存理由；
- `verified_paper_ids`：已完成正文验证。

穷举型问题的完成条件应为：全 Scope 已筛选、全部 Candidate 已验证、全部 Uncertain 已消解，而不是每篇论文都执行固定次数的正文查询。

### 5.7 Final Selector 与 Answer 使用 Claim 元数据

Final Selector 应只选择合格 Claim，并根据问题语义控制 Ownership：

- “本文提出什么”只允许核心结论来自 `this_paper`；
- “目前有哪些方法”可以使用 `prior_work + this_paper`；
- `background_only / irrelevant / pending / failed` 不得形成事实性结论。

Answer Generator 应接收 Claim、Relation、Ownership 和原始 Evidence 映射。内部可以使用 `E1:C1` 级别控制归因，用户引用继续显示 `[E1]`，保持现有 UI 和 provenance 契约稳定。

### 5.8 节点级输出预算

不同节点应使用独立配置，而不是共用一个全局短输出预算。初始建议区间：

| 节点 | 建议输出预算 |
|---|---:|
| Query Understanding | 800～1200 |
| Compact Planner | 1200～1600 |
| Action-only Decision | 1600～3000 |
| Evidence Analysis | 3000～4000 |
| Answer | 4000 |

具体值应通过真实调用统计调整。提高预算只能作为保险，不能替代 Candidate Selection、Prompt 瘦身和职责拆分。

### 5.9 安全可观测性

每次结构化调用至少记录：

- `node`、`attempt`；
- `finish_reason`；
- `prompt_tokens`、`completion_tokens`；
- `content_chars`、`reasoning_chars`；
- `error_stage`；
- 安全、限长的 ValidationError 摘要。

`error_stage` 建议包含：

- `request_error`；
- `empty_content`；
- `invalid_json`；
- `schema_validation`；
- `domain_validation`；
- `guard_rejection`。

不得持久化 API Key 或隐藏推理正文。原始 content 仅可在显式 debug 模式下限长、脱敏保存。

## 6. 优先级

### P0：可信性与运行闭环

1. Decision 改为 Action-only。
2. 增加独立 Evidence Analysis。
3. 引入 Claim 级 Digest、Relation 和 Ownership。
4. 未 Qualification Evidence 改为 fail-closed。
5. Final Selector 与 Answer 接收 Claim Ownership，确定性约束错误归因。

### P1：检索范围与规划质量

> 实施状态：`COMPLETED`（2026-09-19）。

1. 增加 Candidate Selection 条件必经节点。
2. Coverage 改为“全库元数据覆盖 + 候选正文覆盖”。
3. Planner 压缩为 2～3 个证据收集任务。
4. Evidence 可见性从“最后 N 条”改为相关性与论文多样性选择。

### P2：稳定性与诊断

1. 增加节点级 token 配置。
2. 持久化安全调用统计和错误阶段。
3. 根据错误类型选择缩短上下文、提高预算或结构修复，避免无差别重复同一请求。

### P3：检索质量与效率

1. 根据每篇论文的 Digest 和缺口生成动态 Query。
2. 过滤低相关度结果、表格碎片和重复内容。
3. 保持每篇论文 Evidence 多样性。
4. 按 Evidence 指纹、问题和分析 Schema 版本缓存 Evidence Analysis。

### P4：产品审计体验

1. UI 展示 Candidate、Uncertain、Excluded 及理由。
2. 展示每篇论文的 Query、Digest、Claim Ownership 和 Coverage。
3. 展示部分回答和中断的明确原因。

## 7. 推荐实施顺序

```text
Commit 1  LLM 调用诊断信息与错误分类
Commit 2  Compact Planner 与节点级输出预算
Commit 3  Decision 改为 Action-only
Commit 4  Candidate Selection Contract、Service 与 Graph Node
Commit 5  Coverage 迁移为筛选覆盖与候选验证覆盖
Commit 6  Evidence Analysis：Digest、Claim 与 Qualification
Commit 7  Registry 保存 Claim 分析结果和处理状态
Commit 8  Final Selector 改为 fail-closed
Commit 9  Answer Generator 接入 Claim Ownership
Commit 10 动态逐论文 Query 与 Evidence 可见性优化
Commit 11 表格碎片、低相关度和多样性优化
Commit 12 UI 审计展示、契约测试与全量回归
```

每个提交继续要求：原有测试通过、新增契约测试通过，并为真实失败场景增加回归测试。

## 8. 核心验收场景

### 8.0 P0 实施记录（2026-09-19）

本轮已完成 P0 的五项修复：

- Decision 结构化输出已收缩为纯 Action，不再同时生成 Qualification；
- Retrieval 后新增独立 `analyze_evidence` 节点，并将结果写入 checkpoint Registry；
- Evidence Analysis 在同一次受控调用中生成 Digest、Claim、Relation、Ownership 与原文锚点；
- `pending`、`irrelevant`、`failed` Evidence 均为 fail-closed，不能进入最终 Evidence Selector；
- Coverage、Final Selector 和 Answer Prompt 已消费 Claim Relation/Ownership，原始 Evidence 与 E# provenance 保持不变；
- Decision 只读取有界 Digest 与 Claim，不再读取 Evidence 原文；Evidence Analysis 使用独立输出预算。

兼容说明：旧 checkpoint 中仅有 Evidence 级 Qualification 的记录会在加载时迁移为单 Claim；
新生产链路不再生成旧 Qualification。旧 `ResearchRuntime` 可显式注入同一 Evidence Analyzer，
正式产品路径仍以 LangGraph Workflow 为准。

新增回归覆盖：

- 同一 Evidence 中 `prior_work` 与 `this_paper` Claim 的分离；
- `source_excerpt` 不属于原文时重试并最终标记 `failed`；
- 未分析 Evidence 不可被最终选择；
- Decision Prompt 包含 Digest/Claim，但不包含原始 Evidence；
- Evidence Analysis 在 Coverage 前完成并可跨 checkpoint 恢复。

全量验证结果：`275 passed, 15 subtests passed`。

### 8.0.1 P1 实施记录（2026-09-19）

本轮已完成 P1 的四项修复：

- `knowledge_base` Scope 在 Query Understanding 后条件必经 `select_candidates` 节点；显式选择
  的单篇/多篇论文保持用户可信选择，不被自动筛掉；
- Candidate Selector 使用标题、摘要、关键词和已有正式 Overview，输出完整、互斥的
  `candidate / uncertain / excluded` 分区；无效输出保守回退为 `uncertain`；
- Candidate Selection 按 12 篇一批有界执行，避免知识库扩大后产生无界 Prompt；
- Guard 在筛选后锁定 Candidate Scope，禁止重新执行全库检索绕过，也禁止对 `excluded`
  论文执行正文、章节、参考文献或 Overview 动作；
- Coverage 显式区分 `scope / screened / candidate / uncertain / excluded / verified`；
  穷举问题的完成条件改为全 Scope 已筛选且 candidate/uncertain 已验证；
- `uncertain` 论文只要求一次浅层正文验证，避免和高置信 Candidate 一样反复深查；
- Planner Prompt 要求 2～3 个证据收集任务，本地拒绝超过 3 个的模型计划，Schema 上限收紧为 4；
- fallback Planner 不再按论文数量或预设方法类别膨胀，而是稳定生成“收集证据 + 检查缺口/综合”两步计划；
- Decision Evidence 可见性改为问题词项相关性排序，并先为每篇论文保留一条，再按全局相关性
  和单篇上限补齐，避免高分单篇吞没多论文上下文。

新增回归覆盖：

- Candidate 分区完整性、重复/越界 ID 拒绝和保守回退；
- 大 Scope 有界批处理；
- Candidate Selection checkpoint 往返；
- `excluded` 论文无法到达 Tool Executor；
- 全库筛选覆盖、Candidate 正文验证与 Uncertain 浅验证；
- Planner 过度展开时自动回退到 Compact Plan；
- Evidence 可见性的相关性优先和跨论文多样性。

P1 完成后的全量验证结果：`284 passed, 15 subtests passed`。

### 8.1 全库方法归纳

- Planner 最多 3 个任务；
- 明显无关论文不进入正文深查；
- 不确定论文只执行浅检索；
- Answer 不引用其他预测任务内容；
- Coverage 能解释候选、排除和未覆盖范围。

### 8.2 Claim 归属

当同一 Evidence 同时包含“已有工作使用某方法”和“本文提出新方法”时：

- 提取两个独立 Claim；
- 前者为 `prior_work`；
- 后者为 `this_paper`；
- 回答必须使用“论文回顾”和“本文提出”等不同措辞。

### 8.3 Evidence Analysis 失败

- Evidence 状态为 `failed`；
- 不参与自动事实结论；
- 原始 Evidence 保留，可供人工核验；
- 不静默回退为默认合格证据。

### 8.4 Decision 输出耗尽

Checkpoint 能明确记录：

```text
node=decision
finish_reason=length
completion_tokens=<实际值>
content_chars=0
error_stage=empty_content
```

修复路径必须缩短上下文、切换紧凑输入或调整该节点预算，不能只重复同一请求。

## 9. 保持不变的边界

- 现有 Retrieval Service 和 per-paper 检索隔离继续作为底座。
- Tool 仍只返回事实性结果，不自行宣告研究任务完成。
- 原始 Evidence、稳定 E#/R# 和来源 Chunk 映射必须保留。
- Candidate Selection 不能扩大可信 Scope。
- Agent Query 不重新经过 Query Rewriter。
- Answer 继续执行本地 Schema 与引用 ID 校验。
- 快速问答与 Agentic Research 继续并存。

## 10. 最终决策摘要

后续修订的核心不是简单增加 Decision token，而是重新划分职责：

> 全库问题先做可审计的候选论文筛选；Planner 只规划少量证据任务；Decision 只选择 Action；检索后由独立 Evidence Analysis 生成 Claim 级 Digest 与 Qualification；Coverage、Selector 和 Answer 只消费经过校验的 Claim，并始终保留到原始 Evidence 的可追溯关系。
