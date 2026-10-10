# Contracts 与 Authorities

Snapshot `paperbase-architecture-snapshot-v1`；生成时间 `2026-10-07T15:33:50.053750+08:00`（Asia/Shanghai）。Commit `e365bf9e27eaab29f554a3c3f8ec441b2c21a9d6`；dirty=`True`。

当前生产树 SHA256：`54766b03327438e05f56f3d7a6777907c5c089541c9eb1aba8463905c02f5e7d`（158 个 paperbase/app Python 文件，包含未提交文件）。配置 bundle SHA256：`59af7d9d7c859eace4993289182544c47fecbed28954334261d64a225b9ac4e5`。历史 commit 不代表当前 dirty tree，精确路径和哈希见 [source_hashes.json](source_hashes.json)。

## Authority Matrix

| Concept | Producer | Authority | Consumers | Not authority for |
|---|---|---|---|---|
| QueryContext | ScopeResolver + QU | 可信 scope/意图与 advisory hints | Planner/Decision/Tools | canonical execution input |
| ResearchPlan / ResearchRequirement | Planner | 回答义务/required/投影 shape | Decision/Projector/Selector/Answer | 支持成立/完整答案 |
| AgentAction | Decision + local validation | 候选工具命令 | Graph/Guard/Executor | 运行合法性 |
| CanonicalSearchRequest | retrieval.requests compiler | 生效请求与逐 request identity | Guard/Executor/Retriever | 语义近重复 |
| RuntimeActionCapabilities | Graph State derived builder | 结构可行域和 runtime hard data | Decision View/Schema | 替代 formal Guard |
| ActionRejectionContext | 共享 preflight / formal runtime | 拒绝事实和当前恢复状态 | Decision/trace/checkpoint | 生成新的 query |
| RetrievedEvidence | Retriever/Tool | 原始来源、score kind/direction/rank | Admission/Registry | 跨来源统一 relevance |
| EvidenceRecord | Registry | identity/provenance/binding/assessment | Integrity/State/Selector/Answer | 独立语义完整性 |
| EvidenceProjection | Projector | 指定局部 shape 的语义抽取 | Integrity/Registry | 自动正式 support |
| verified_support_ids | grounding.py | 正式局部 source-grounded support | State/Selector/CompletionView | 独立 entailment/completeness |
| RequirementState | derive/finalize helpers | 研究尝试与局部 support 状态 | Graph Finish/Decision/audit | Answer complete |
| CandidateSelection | Candidate Selector | KB paper screening disposition | Guard/Coverage/Decision | Requirement answer coverage |
| AnswerSemanticDraft | Answer LLM | 文本和支持项 complete/partial 声明 | Compiler/local validator | unsupported/API citation 数组 |
| AnswerRequirementCoverage | coverage compiler | required rows/actual cited mapped support/unsupported | ResearchAnswer/overall/E1 | 独立 complete/partial 语义裁决 |
| ResearchAnswer | Answer bridge + compiler / deterministic branch | 最终 public 输出 | Service/UI/E1 | 独立答案质量证明 |

Guard 是 research Action 的 runtime legality authority。Finish 的研究 closure 在 Graph `_assess_action`，通用 Guard 对 Finish 直接接受；两者负责不同条件，不应把 closure 的逻辑归属写错。Selector/support/Answer wrappers 复用正式 helper，不另建支持判断。低层 request identity 与 batch Action aggregate identity 也分别有明确域。

## Contract Catalog

以下字段由当前 dataclass/Pydantic/union introspection 提取；这里列 contract 字段定义，不保存实例 Evidence、真实 queries 或 provider 原文。checkpoint 标记包括嵌套于 Observation/Registry/Graph 的持久化；RuntimeActionCapabilities 是从 Graph 数据重建的 view，provider 草稿保持临时。

### QueryContext

用途：Scope / resolved query / advisory hints。Producer：ScopeResolver + QueryUnderstandingService；Consumer：Planner, Decision, Tools。

| Variant | Source | Fields |
|---|---|---|
| QueryContext | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L393) | `original_query` (str); `resolved_query` (str); `intent` (Intent); `scope` (QueryScope); `target_paper_ids` (tuple[str, ...]); `scope_source` (ScopeSource); `retrieval_goal` (ResearchGoal); `default_evidence_targets` (frozenset[EvidenceTarget]); `semantic_query_en` (str \| None); `lexical_keywords_en` (tuple[str, ...]) |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### ResearchPlan

用途：Ordered answering obligations。Producer：ResearchPlanner；Consumer：Graph, Decision, Projector, Selector, Answer。

| Variant | Source | Fields |
|---|---|---|
| ResearchPlan | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L485) | `goal` (str); `requirements` (tuple[ResearchRequirement, ...]); `planning_mode` (PlanningMode) |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### ResearchRequirement

用途：Requirement ID / semantics / required / target / projection shape。Producer：ResearchPlanner；Consumer：Search bindings, Projector, Integrity, Answer。

| Variant | Source | Fields |
|---|---|---|
| ResearchRequirement | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L437) | `requirement_id` (str); `description` (str); `evidence_targets` (frozenset[EvidenceTarget]); `required` (bool); `projection_type` (EvidenceProjectionType \| None) |

Visibility：public domain (nested Plan)。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### AgentAction

用途：Typed chosen tool command。Producer：LLMDecisionAgent after local validation；Consumer：Graph, Guard, Tool executor。

| Variant | Source | Fields |
|---|---|---|
| SearchPapersAction | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L503) | `searches` (tuple[PaperSearchSpec, ...]); `action_name` (str) |
| SearchKnowledgeBaseAction | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L514) | `query` (str); `lexical_terms` (tuple[str, ...]); `requirement_ids` (tuple[str, ...]); `action_name` (str) |
| SearchReferencesAction | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L527) | `paper_id` (str); `query` (str); `lexical_terms` (tuple[str, ...]); `requirement_ids` (tuple[str, ...]); `action_name` (str) |
| GetSectionContextAction | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L542) | `paper_id` (str); `section_id` (str); `requirement_ids` (tuple[str, ...]); `action_name` (str) |
| GetPaperOverviewAction | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L555) | `paper_id` (str); `requirement_ids` (tuple[str, ...]); `action_name` (str) |
| FinishAction | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L566) | `reason` (str); `action_name` (str) |

Visibility：public domain union。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### CanonicalSearchRequest

用途：Compiler-normalized effective retrieval input and identity。Producer：retrieval.requests compilers / compile_action_requests；Consumer：Guard, Executor, Retriever, receipts。

| Variant | Source | Fields |
|---|---|---|
| CanonicalPaperSearchRequest | [paperbase/retrieval/requests.py](../../../paperbase/retrieval/requests.py#L70) | `paper_ids` (tuple[str, ...]); `semantic_query` (str); `lexical_terms` (tuple[str, ...]); `lexical_query` (str); `top_k` (int); `source` (str); `workspace_id` (str \| None); `retrieval_parameters` (dict[str, object]); `kind` (Literal['search_papers']) |
| CanonicalReferenceSearchRequest | [paperbase/retrieval/requests.py](../../../paperbase/retrieval/requests.py#L83) | `paper_ids` (tuple[str, ...]); `lexical_terms` (tuple[str, ...]); `lexical_query` (str); `top_k` (int); `kind` (Literal['search_references']) |
| CanonicalKBSearchRequest | [paperbase/retrieval/requests.py](../../../paperbase/retrieval/requests.py#L92) | `paper_ids` (tuple[str, ...]); `semantic_query` (str); `lexical_terms` (tuple[str, ...]); `lexical_query` (str); `top_k` (int); `retrieval_parameters` (dict[str, object]); `scope` (Literal['paper_ids', 'all']); `kind` (Literal['search_knowledge_base']) |

Visibility：internal domain union。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### RuntimeActionCapabilities

用途：Structural feasible domain derived from current Graph State。Producer：decision._runtime_action_capabilities；Consumer：Decision View / dynamic Schema / local validation。

| Variant | Source | Fields |
|---|---|---|
| RuntimeActionCapabilities | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L254) | `valid_requirement_ids` (tuple[str, ...]); `valid_paper_ids` (tuple[str, ...]); `known_section_ids_by_paper` (tuple[tuple[str, tuple[str, ...]], ...]); `available_section_ids_by_paper` (tuple[tuple[str, tuple[str, ...]], ...]); `tool_budget_state` (dict[str, object]); `unavailable_capabilities` (tuple[str, ...]); `recent_unavailable_actions` (tuple[RuntimeUnavailableAction, ...]); `unavailable_paper_queries` (tuple[RuntimeUnavailablePaperQuery, ...]); `available_tools` (tuple[str, ...]); `search_paper_ids` (tuple[str, ...]); `reference_paper_ids` (tuple[str, ...]); `overview_paper_ids` (tuple[str, ...]); `capability_reasons` (tuple[str, ...]) |

Visibility：internal derived view + compatibility DTO。checkpoint：否（派生/临时）。序列化：derived JSON/view or local provider validation; not stored as provider draft。
### ActionRejectionContext

用途：Current/historical rejection feedback, not replacement Action。Producer：Decision preflight / Graph runtime validation；Consumer：Decision View, audit, checkpoint。

| Variant | Source | Fields |
|---|---|---|
| ActionRejectionContext | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L166) | `action_name` (str); `action_arguments` (dict[str, object]); `requirement_ids` (tuple[str, ...]); `rejection_code` (str); `rejection_reason` (str); `guard_name` (str); `iteration` (int); `recovery_action` (str); `rejection_stage` (str); `blockers` (tuple[object, ...]); `validator_name` (str); `action_canonicalization` ('ActionCanonicalization \| None') |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### RetrievedEvidence

用途：Retrieval content locator / provenance / explicit score contract。Producer：Tools and retrieval adapters；Consumer：Admission, Registry。

| Variant | Source | Fields |
|---|---|---|
| RetrievedEvidence | [paperbase/retrieval/contracts.py](../../../paperbase/retrieval/contracts.py#L153) | `source_type` (EvidenceSourceType); `paper_id` (str); `seed_chunk_ids` (tuple[str, ...]); `source_chunk_ids` (tuple[str, ...]); `content` (str); `paper_title` (str \| None); `section_id` (str \| None); `section` (str \| None); `page_start` (int \| None); `page_end` (int \| None); `score` (float \| None); `retrieval_metadata` (dict[str, object]); `raw_score` (float \| None); `score_kind` (ScoreKind); `score_direction` (ScoreDirection); `retrieval_rank` (int \| None) |

Visibility：public tool/domain DTO。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### EvidenceRecord

用途：Run-local Evidence identity, binding, successful assessment。Producer：EvidenceRegistry；Consumer：Projector, Integrity, State, Selector, Answer, audit。

| Variant | Source | Fields |
|---|---|---|
| EvidenceRecord | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L857) | `evidence_id` (str); `evidence` (RetrievedEvidence); `supporting_queries` (tuple[str, ...]); `first_seen_iteration` (int); `last_seen_iteration` (int); `originating_tools` (tuple[str, ...]); `fingerprint` (str \| None); `requested_evidence_targets` (frozenset[EvidenceTarget]); `requested_requirement_ids` (tuple[str, ...]); `assessment` (EvidenceAnalysis \| None) |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### EvidenceProjection

用途：Typed local structured extraction, not final completeness。Producer：LLMEvidenceProjector；Consumer：Integrity, Registry, Selector, Answer。

| Variant | Source | Fields |
|---|---|---|
| EnumerationProjection | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L657) | `evidence_id` (str); `entities` (tuple[ProjectedEntity, ...]); `supported_requirement_ids` (tuple[str, ...]); `support_anchors` (tuple[RequirementSupportAnchor, ...]); `projection_type` (EvidenceProjectionType) |
| ComparisonProjection | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L674) | `evidence_id` (str); `comparison_dimensions` (tuple[str, ...]); `supported_requirement_ids` (tuple[str, ...]); `proposed_requirement_ids` (tuple[str, ...]); `support_anchors` (tuple[RequirementSupportAnchor, ...]); `projection_type` (EvidenceProjectionType) |
| FactProjection | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L710) | `evidence_id` (str); `facts` (tuple[FactItem, ...]); `projection_type` (EvidenceProjectionType) |
| MechanismProjection | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L743) | `evidence_id` (str); `mechanism` (str); `steps` (tuple[str, ...]); `inputs` (tuple[str, ...]); `outputs` (tuple[str, ...]); `supported_requirement_ids` (tuple[str, ...]); `proposed_requirement_ids` (tuple[str, ...]); `support_anchors` (tuple[RequirementSupportAnchor, ...]); `projection_type` (EvidenceProjectionType) |

Visibility：public domain union nested in assessment。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### EvidenceAnalysis

用途：Per-Evidence assessment aggregate and explicit evaluated scope。Producer：Projector + merge_analysis_groups；Consumer：Registry / checkpoint / diagnostics。

| Variant | Source | Fields |
|---|---|---|
| EvidenceAnalysis | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L773) | `evidence_id` (str); `status` (EvidenceAnalysisStatus); `digest` (str \| None); `claims` (tuple[EvidenceClaim, ...]); `projection` (EvidenceProjection \| None); `projection_rejections` (tuple[str, ...]); `error_code` (str \| None); `evaluated_requirement_ids` (tuple[str, ...]); `projections` (tuple[EvidenceProjection, ...]) |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### ProjectionSemanticRequest

用途：Question + scoped Requirement meaning used in semantic cache identity。Producer：build_projection_semantic_request；Consumer：Projector prompt scope / cache fingerprint。

| Variant | Source | Fields |
|---|---|---|
| ProjectionSemanticRequest | [paperbase/agentic/projection_semantics.py](../../../paperbase/agentic/projection_semantics.py#L25) | `question` (str); `plan_goal` (str); `projection_type` (EvidenceProjectionType); `requirements` (tuple[ProjectionRequirementSemantic, ...]) |

Visibility：internal ephemeral。checkpoint：否（派生/临时）。序列化：derived JSON/view or local provider validation; not stored as provider draft。
### RequirementState

用途：Research attempt / local verified support status。Producer：track/derive/finalize_requirement_states；Consumer：Decision, Graph Finish closure, audit。

| Variant | Source | Fields |
|---|---|---|
| RequirementState | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L471) | `requirement_id` (str); `status` (RequirementStatus); `evidence_ids` (tuple[str, ...]) |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### CandidateSelection

用途：Frozen corpus candidate / uncertain / excluded disposition。Producer：LLMCandidateSelector (metadata-only) or explicit-scope candidate derivation；Consumer：Coverage, Guard, Decision。

| Variant | Source | Fields |
|---|---|---|
| CandidateSelection | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L351) | `scope_paper_ids` (tuple[str, ...]); `assessments` (tuple[CandidateAssessment, ...]) |

Visibility：public domain。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### AnswerSemanticDraft

用途：Untrusted provider text + supported-required semantic status。Producer：Answer LLM；Consumer：Local JSON/schema/domain parser and deterministic compiler。

| Variant | Source | Fields |
|---|---|---|
| AnswerSemanticDraft | [paperbase/generation/answer_generator.py](../../../paperbase/generation/answer_generator.py#L47) | `direct_answer` (str); `evidence_explanation` (str); `reading_interpretation` (str \| None); `requirement_status` (dict[str, Literal['complete', 'partial']]); `coverage_note` (str \| None) |

Visibility：internal provider draft。checkpoint：否（派生/临时）。序列化：derived JSON/view or local provider validation; not stored as provider draft。
### CompletionView

用途：Selected verified support map for required IDs。Producer：build_completion_view；Consumer：Answer Schema / allowed coverage。

| Variant | Source | Fields |
|---|---|---|
| CompletionView | [paperbase/agentic/answering.py](../../../paperbase/agentic/answering.py#L16) | `selected_verified_support_by_requirement` (tuple[tuple[str, tuple[str, ...]], ...]) |

Visibility：internal ephemeral。checkpoint：否（派生/临时）。序列化：derived JSON/view or local provider validation; not stored as provider draft。
### AnswerRequirementCoverage

用途：Required Requirement status and actually cited allowed support。Producer：compile_answer_requirement_coverage；Consumer：derive_overall_answer_status, ResearchAnswer, E1。

| Variant | Source | Fields |
|---|---|---|
| AnswerRequirementCoverage | [paperbase/generation/answer_coverage.py](../../../paperbase/generation/answer_coverage.py#L24) | `requirement_id` (str); `status` (Literal['complete', 'partial', 'unsupported']); `evidence_ids` (tuple[str, ...]) |

Visibility：public final DTO nested in ResearchAnswer。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。
### ResearchAnswer

用途：Final user-facing typed result。Producer：generate_agentic_answer / Graph deterministic short circuits；Consumer：Service / UI / audit / E1。

| Variant | Source | Fields |
|---|---|---|
| ResearchAnswer | [paperbase/agentic/contracts.py](../../../paperbase/agentic/contracts.py#L1094) | `status` (str); `answer` (str); `citations` (tuple[str, ...]); `insufficient_evidence` (bool); `partial_answer` (bool); `coverage_note` (str \| None); `requirement_coverage` (tuple[AnswerRequirementCoverage, ...]); `overall_status` (str \| None); `llm_diagnostics` (tuple[LLMCallRecord, ...]) |

Visibility：public final DTO。checkpoint：是（直接或嵌套）。序列化：Pydantic JSON validation then domain DTO; nested through checkpoint_serde。

## 序列化与兼容边界

Domain DTO 通过 checkpoint_serde TypeAdapter 递归 JSON 序列化，最终由 SqliteSaver 保存。不是将 provider Pydantic 草稿原样放入 checkpoint。ResearchAnswer 和 AnswerRequirementCoverage 是 public domain DTO；AnswerSemanticDraft、CompletionView、ProjectionSemanticRequest 是 internal 临时对象。

旧 Evidence.score 明确为 deprecated/legacy_unknown，不能依数值符号补猜 score kind/direction。旧 singleton projection accessor 仍可读取单投影兼容数据，但 aggregate 正式结构为 projections[]。历史 evaluated_requirement_ids 为空时保持为空；historical verified support 保留用于审计，但不能迁移到新 scope 执行。

RetrievedEvidence 的 raw_score 可为空，retrieval_rank 为同次 retrieval 的有序名次。ScoreKind 实际枚举为 dense / reranker / rrf / bm25 / bm25_fts5 / non_ranked / other / legacy_unknown；ScoreDirection 为 higher_is_better / lower_is_better / not_applicable。正式 FTS5 BM25 使用 lower_is_better，workspace BM25 使用 higher_is_better；reranker/RRF 为 higher_is_better，non_ranked 为 not_applicable。rank 只能作为既有选择信号，不是无定义的跨 channel calibration。Admission 的 min_rerank_score=0.05 只作用于明确标记 normalized 的 reranker channel。

| Current | Legacy / compatibility boundary |
|---|---|
| canonical effective search identity | raw paper/query fingerprints retained only as compatibility notes |
| advisory QU hints | implicit QU retrieval input authority retired; Classic rewrite is still active |
| heterogeneous scoped projections[] | singleton read accessor / historical schema, not aggregate authority |
| AnswerSemanticDraft + compiler | provider coverage array archived/test-only; Classic draft active independently |
| v4 execution gate | v2/v3 parse/inspect offline only |
| strict Decision json_schema | old json_object Decision retired |

Classic 的 rewrite 和 Answer provider draft 仍在自身生产路径活跃，不能标为整个系统已删除；Agentic 已退出的仅是其旧的隐式 QU 输入/coverage 数组 authority。

## Authority conflict audit

当前 `authority_conflicts = []`。自动核对范围是 search identity、Requirement support、Answer coverage、citation、overall status、duplicate 六个指定 helper 的 AST 定义归属，加上五个 delegation/DTO 检查；人工复核各层不同责任。不存在同名竞争 owner，不等于程序已证明任意 helper 永远不会存在语义冲突。完整边界和 source locations 在 JSON `authority_audit`。

| Delegation check | Result |
|---|---|
| Selector support wrapper | True |
| Answer bridge uses shared overall helper | True |
| preflight and formal Guard use same admission helper | True |
| search fingerprint delegates canonical identity | True |
| typed Action adapter excludes qualifications | True |
