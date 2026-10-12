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
| 整体答案 | `GroundedAnswerAttempt.answer` | 围绕原问题的单份正文，不是 Requirement 段落拼接 |
| Requirement 覆盖 | `GroundedAnswerAttempt.requirements` | `RequirementCoverage` 的 requirement_id/evidence_ids/remaining_gap；只描述整体正文实际覆盖，complete/status 派生 |
| 实际引用集 | `GroundedAnswerAttempt.citations` | 正文 [E#]/[REF#] 标记对应的完整 Evidence ID；不声称句子级 span 精度 |
| 覆盖修复 | Answerer 调用内 | 固定合法整体正文，只修复失败覆盖项；下一次语义重读不继承旧 complete |
| Requirement 状态 | `RequirementState` | satisfied 只表示已有已校验答案；否则按实际 gap/尝试状态派生 |
| 对外答案 | `ResearchAnswer` | 保留整体正文，requirement_coverage 是同一覆盖权威的展示投影，仅附 gap 提示 |
| Decision 覆盖视图 | `requirements[].answer_coverage` | 覆盖状态、Evidence IDs、gap，不重复正文，不是独立 Requirement 答案 |
| 当前 checkpoint | `paperbase-agentic-grounded-answer-v2` | 旧语义版本不迁移执行 |
| 完成回合审计 | `AgenticAuditSnapshot` | `grounded-answer-v2`，包含 retrieval、answer input、Evidence 引用与成本信息 |
| 供应商请求拒绝 | `LLMRequestError` → `LLMCallRecord.validation_diagnostics` | `invalid_request` 区分参数/权限等不可原样重试的拒绝；只保存有限安全代码、参数路径与 HTTP/关键词摘要，不保存原始错误体 |

关键区别：requested_requirement_ids 不是支持；answer_input_evidence_ids 不是引用；selected_evidence 等于整体正文实际引用 Evidence；Evidence 引用合法不代表整窗每个 chunk 已被回答。GroundedCitation、RequirementAnswer、accepted_answers、grounded_requirements 已废除，不以同名不同义字段兼容旧 checkpoint。

实现入口：[retrieval contracts](../paperbase/retrieval/contracts.py)、[canonical requests](../paperbase/retrieval/requests.py)、[Registry](../paperbase/agentic/evidence_registry.py)、[Grounded Answerer](../paperbase/agentic/grounded_answer.py)、[Graph](../paperbase/agentic/graph.py)、[Audit](../paperbase/agentic/audit.py)。
