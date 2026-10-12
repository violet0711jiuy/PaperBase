# 字段事实源与语义边界

核对日期：2026-10-12。派生视图只用于传输或展示，不得成为第二份权威状态。

| 信息 | 权威字段 / 模块 | 含义与边界 |
|---|---|---|
| 原问题与范围 | 用户输入 + `ScopeResolver` | 模型不可扩大可信 corpus |
| 问题理解 | `QueryContext` | 规范化问题、意图、可信范围和证据类型；不保存检索 query/关键词 |
| 回答义务 | 原问题 + `ResearchPlan.requirements` | 原问题规定范围；`requirement_id/description/required/evidence_targets` 组织宽粒度义务，不预设事实关系或检索顺序 |
| Decision wire 动作 | `ActionDraft` Schema | 一轮一个工具；检索/扩展严格绑定单个 `requirement_id`，搜索文字使用 `semantic_query + lexical_terms` |
| 内部 Decision 动作 | 校验后的 `AgentAction` | wire contract 编译后的执行对象；可用复数字段记录批次/历史，但当前检索动作仍须恰好一个 Requirement |
| 实际检索请求 | `CanonicalSearchRequest` | effective query、lexical tokens、identity 的唯一事实源 |
| 执行合法性与预算 | `GuardState` + receipts | 区分 attempted/executed/successful/failed |
| 原始检索结果 | `RetrievedEvidence` | 保存正文、chunk/window、section/page 和 score 语义 |
| Evidence 身份与 provenance | `EvidenceRegistry` / `EvidenceRecord` | E#/REF#、检索 query、请求绑定和首次出现；不判断支持 |
| Answerer 完整输入 | `ResearchRunResult.answer_input_evidence_ids` | 某次答案阶段看到的整个 Registry 快照 |
| Requirement 答案 | `GroundedAnswerAttempt.requirements` | 有引用的完整/部分 answer 与尚未回答的具体 remaining_gap；citation 合法不代表内容完整 |
| 引用锚点 | `GroundedCitation` | Evidence ID + source span IDs；excerpt 由程序恢复 |
| 实际引用集 | `GroundedAnswerAttempt.evidence_ids` | 只含真正被答案引用的 Evidence |
| accepted answers | Graph state | 同一 Evidence 快照内用于 scoped repair/聚焦重读的合法完成项；新 Evidence 到达后不跨快照冻结 |
| Requirement 状态 | `RequirementState` | satisfied 只表示已有已校验答案；否则按实际 gap/尝试状态派生 |
| 对外答案 | `ResearchAnswer` | renderer 从已校验 RequirementAnswer 确定性生成 |
| 当前 checkpoint | `paperbase-agentic-grounded-answer-v1` | 旧语义版本不迁移执行 |
| 完成回合审计 | `AgenticAuditSnapshot` | `grounded-answer-v1`，包含 retrieval、answer input、citation 与成本信息 |

关键区别：`requested_requirement_ids` 不是支持；`answer_input_evidence_ids` 不是引用；`selected_evidence` 当前等于实际引用 Evidence；span 校验通过只证明引用可追溯，不证明自然语言推断必然正确。

实现入口：[retrieval contracts](../paperbase/retrieval/contracts.py)、[canonical requests](../paperbase/retrieval/requests.py)、[Registry](../paperbase/agentic/evidence_registry.py)、[Grounded Answerer](../paperbase/agentic/grounded_answer.py)、[Graph](../paperbase/agentic/graph.py)、[Audit](../paperbase/agentic/audit.py)。
