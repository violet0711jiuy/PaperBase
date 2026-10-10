# 字段事实源与语义边界

核对日期：2026-10-08。完整流程见[当前架构](architecture/agentic-research.md)。派生视图不能成为另一份权威状态。

| 信息 | 权威 | 派生或使用边界 |
|---|---|---|
| 论文原文、section/page、chunk 定位 | SQLite 或隔离 workspace | embedding_text 用于检索，不取代 raw_text |
| 用户原问题与范围 | 用户输入 + ScopeResolver | 全库使用本轮冻结快照；模型不得扩大范围 |
| 问题解释与冷启动 hints | 校验后的 QueryContext | 提供建议，不自动注入 Retriever |
| 全库候选筛选 | CandidateSelection.assessments | ID 分区派生；screening_complete 只表示筛选 |
| 回答义务 | ResearchPlan.requirements | ID/description/required/evidence_targets/projection_type 的正式定义 |
| 工具输入 | 通过 Schema/local/Guard 的 AgentAction | 模型原始 JSON 不可执行；requirement_ids 不等于支持 |
| 有效检索请求与重复身份 | CanonicalSearchRequest 编译器 | Guard 与执行器同一 identity；trace raw query 不一定参与有效请求 |
| 预算、成功请求与禁止动作 | GuardState + execution receipts | capabilities/View 是投影；failed 不冒充 successful |
| 证据分数 | raw_score/score_kind/score_direction/retrieval_rank | 来源内解释；无统一 calibration；legacy score deprecated |
| Evidence ID、请求目标、来源与出现历史 | EvidenceRegistry | E#/REF# 是引用空间，R# 仅 Requirement |
| 当前 assessment | EvidenceRecord.assessment.projections[] | legacy singleton alias 不代表多个投影组 |
| 已评估 scope | 显式 evaluated_requirement_ids | 请求 IDs 与 Plan IDs 的交集；历史缺失不推断 |
| 有效局部支持 | Integrity 核验后的 verified_support_ids | suggested supported IDs 和 requested IDs 均不是已核验支持 |
| Projector cache 命中 | semantic request fingerprint + Evidence identity | 绑定 scoped Requirement/type/问题/goal/schema/prompt；旧无指纹数据不猜测命中 |
| Requirement 局部支持状态 | RequirementState | SATISFIED 仅表示有 valid support |
| Critical anchor coverage | RequirementCoverage | complete 是 anchor 覆盖，不是语义完整性 |
| Decision 输入与 no-progress | Graph State 派生的 factored View | builder 不调 LLM、不改 State；信号不进入 Guard |
| 当前 rejection 与历史 | last_action_rejection / action_rejections | 当前指针按执行结果清除；历史不重新激活 |
| 最终 Evidence 与允许支持 | Selector + CompletionView | 使用选中且经核验的支持，不推断 Answer complete |
| Answer 语义 complete/partial/unsupported | 校验后的 AnswerSemanticDraft.requirement_status | Answer LLM 可判断局部支持不足以回答核心；程序不猜测语义完整性，不修改上游 assessment |
| unsupported、coverage evidence IDs 与 overall | Coverage Compiler + 引用 parser | 无 selected support 时补 unsupported；有映射时保留 Answer 的语义否决，unsupported 引用集为空，全部 unsupported 则 insufficient |
| 对外答案 | ResearchAnswer | 公开 DTO 不等于 provider draft；Classic/Agentic 独立 draft |
| 未完成任务持久化状态 | LangGraph checkpoint，当前 v4 | v2/v3 只读审计，禁止当前 resume；不迁移旧 support |
| 已完成 KB 回合 | ConversationStore 的 answer/evidence/audit/metadata | pending task 仅指向 checkpoint；历史 sidecar 只读兼容 |

兼容入口保留部分旧 Claims、score、singleton projection 或审计字段以解析历史数据。这些字段不产生新生产权威，也不允许旧 semantic state 无声进入当前执行链。新写入和历史读取必须分别说明。

实现入口：[retrieval contracts](../paperbase/retrieval/contracts.py)、[canonical requests](../paperbase/retrieval/requests.py)、[Registry](../paperbase/agentic/evidence_registry.py)、[Projector](../paperbase/agentic/evidence_projection.py)、[Analyzer Protocol](../paperbase/agentic/evidence_analysis.py)、[Integrity](../paperbase/agentic/grounding.py)、[RequirementState](../paperbase/agentic/requirement_state.py)、[Selector](../paperbase/agentic/selector.py)、[Answer compiler](../paperbase/generation/answer_coverage.py)、[Graph](../paperbase/agentic/graph.py)。
