# Effective Config、版本、Checkpoint 与 Evaluation

Snapshot `paperbase-architecture-snapshot-v1`；生成时间 `2026-10-07T15:33:50.053750+08:00`（Asia/Shanghai）。Commit `e365bf9e27eaab29f554a3c3f8ec441b2c21a9d6`；dirty=`True`。

当前生产树 SHA256：`54766b03327438e05f56f3d7a6777907c5c089541c9eb1aba8463905c02f5e7d`（158 个 paperbase/app Python 文件，包含未提交文件）。配置 bundle SHA256：`59af7d9d7c859eace4993289182544c47fecbed28954334261d64a225b9ac4e5`。历史 commit 不代表当前 dirty tree，精确路径和哈希见 [source_hashes.json](source_hashes.json)。

## 1. 实际客户端组装

Provider `openai_compatible`，model `Qwen/Qwen3.8-Flash-Next`，host `api-inference.modelscope.cn`。只记录 endpoint host 与 credential 是否配置，不记录 API key、完整带 credential endpoint、.env 内容、Prompt 或 hidden reasoning。生产 factory 注入 SDK sentinel 后对各节点真实 client view 调用 complete_json_diagnostic，截获最终 create 参数；7 次 mock SDK create、0 次 provider。合成 schema ID 仅用于 metadata 检查，不使用用户 State，不验证 provider 真正强制 schema 的能力。

Base client（来自当前 .env 的实际 loader，隔离宿主 process override）：

```json
{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "temperature": 0.2,
  "output_mode": "json_object",
  "strict": false,
  "schema_name": null,
  "schema_sha256": null,
  "top_p": null,
  "presence_penalty": null,
  "extra_body": {
    "enable_thinking": false
  },
  "timeout": 30.0,
  "max_tokens": 1600,
  "max_tokens_key_present": true
}
```

Agentic `qwen38_recommended` 通过 immutable view 覆盖 base 参数：

| Node | T | Thinking | top_p | presence | top_k / min_p / repetition | Mode / strict | max_tokens sent | Timeout s |
|---|---|---|---|---|---|---|---|---|
| query_understanding | 0.7 | False | 0.8 | 1.5 | 20 / 0.0 / 1.0 | json_object / False | None | 30.0 |
| candidate_selection | 0.7 | False | 0.8 | 1.5 | 20 / 0.0 / 1.0 | json_object / False | None | 30.0 |
| planner | 0.7 | False | 0.8 | 1.5 | 20 / 0.0 / 1.0 | json_schema / True | None | 30.0 |
| decision | 1.0 | True | 0.95 | 0.0 | 20 / 0.0 / 1.0 | json_schema / True | None | 30.0 |
| projector | 0.7 | False | 0.8 | 1.5 | 20 / 0.0 / 1.0 | json_schema / True | None | 60.0 |
| answer | 0.7 | False | 0.8 | 1.5 | 20 / 0.0 / 1.0 | json_schema / True | None | 60.0 |
| retrieval | None | None | None | None | None / None / None | None / None | None | None |
| selector | None | None | None | None | None / None / None | None / None | None | None |

六个 Agentic 节点 max_tokens key 都不发送（不是发送 null），不保证 provider 没有自身输出限制。decision thinking=true 由 node view 实际设定；当前 base thinking=false。应用仍有时间、iteration、输入 view、检索/上下文限额。LLM sampling top_k=20 与 Retriever top-k 是不同参数。

Classic 未使用 Agentic profile：保持 base temperature=0.2/json_object/thinking=false（base max_tokens=1600），Classic Answer 用其自身 max_output_tokens=4000；Overview 3000、Explain 2500；query rewrite 仍按 base/client 路径。不要把 .env 温度理解为当前 Decision 温度，也不要把 Agentic 不发送 cap 理解为 Classic cap 被删除。

## 2. Budget / attempt / context

```json
{
  "max_agent_iterations": 8,
  "max_queries_per_paper": 4,
  "node_max_attempts": 2,
  "actual_node_attempts": {
    "planner_max_attempts": 2,
    "decision_max_attempts": 2,
    "projector_max_attempts": 2,
    "candidate_max_attempts": 2,
    "candidate_batch_size": 12
  },
  "decision_recovery_replacements_per_rejected_candidate": 1,
  "invalid_counter_reset_on_valid_action": false,
  "agentic_application_output_cap": null,
  "base_client_max_tokens": 1600,
  "sdk_transport_default_max_retries": 2,
  "answer_request_error_retries": 1,
  "answer_structural_citation_repair": 1,
  "timeouts": {
    "base": 30.0,
    "projector": 60.0,
    "answer": 60.0
  },
  "tool_top_k_per_paper": 5,
  "projector_batch_size": 4,
  "final_evidence_max_units": 8,
  "selector_cap_is_absolute": false,
  "decision_view": {
    "decision_view_mode": "compact",
    "decision_view_support_limit": 6,
    "decision_view_recent_observations": 2,
    "decision_view_recent_queries": 3,
    "decision_view_recent_actions": 3,
    "decision_view_sections": 10,
    "decision_view_scope_chars": 600,
    "decision_view_unresolved_limit": 6,
    "decision_view_unresolved_chars": 240,
    "decision_view_entity_limit": 12
  },
  "compact_recent_event_window": 3,
  "classic": {
    "answer_max_output_tokens": 4000,
    "expansion_max_total_tokens": 6000,
    "bibliography_top_k": 5
  },
  "retrieval": {
    "dense_top_k_per_query": 20,
    "bm25_keywords_top_k": 20,
    "fused_top_k": 50,
    "rrf_k": 60,
    "reranker_candidate_top_k": 40,
    "reranker_final_top_k": 5,
    "retrieval_scope": "Dense/FTS/RRF/reranker counts, independent of sampling top_k=20."
  },
  "ignored_legacy_agentic_caps": [
    "max_output_tokens",
    "evidence_analysis_max_output_tokens",
    "evidence_projector_batch_base_tokens",
    "evidence_projector_batch_per_evidence_tokens",
    "evidence_projector_batch_max_output_tokens"
  ]
}
```

实际 research max iterations=8、每 paper queries=4，node_max_attempts=2；一次拒绝最多 immediate replacement 1 次，并受剩余 candidate pool 限制。invalid iteration 累计，成功合法执行不归零。实际 Planner 内部 max_attempts=2 是 constructor default，factory 当前未显式把 node_max_attempts 传进去；两者当前相等，不意味着未来改配置自动同步。Projector group/batch repair 后可逐个 fallback，transport/node attempts/研究 iteration 不应混为同一计数。

SDK `2.53.0` 的 transport 默认重试 2，没有 factory 另行覆盖；它是传输层，不等同 Answer request-error retry 或 Decision 自身候选重试。Projector/Answer timeout=60 秒，其余四个当前为 30 秒。Selector max_units=8 是选择目标，为 required-support 覆盖 phase 1 可超限；Agentic Answer 不再次截断。batch4 是 Projector transport envelope，不是一个统一语义 scope。

Decision 当前 recent_research_events 窗口=3；support scope/unresolved/section/entity 等按配置限制显示，底层 Graph State 仍保留。section metadata 与 runtime 合法 ID domain 分开。retrieval dense20/FTS20/fused50/RRF k60/reranker candidate40/final5/tool top-k5，context expansion max6000 tokens/neighbor1；此为上下文与检索边界，不是采样参数。

旧 Agentic output-cap 配置字段仍可 parse 以维持 DTO 兼容，但当前 uncapped views 不使用这些值。project.name 仍是必需 metadata，retrieval.backend 当前不驱动 route 切换，factory 固定 Hybrid；快照仅记录实际行为。

## 3. Prompt / Schema / semantic versions

| Node | Prompt version | Schema name | Schema version |
|---|---|---|---|
| query_understanding | UNVERSIONED | paperbase_query_understanding | UNVERSIONED |
| planner | UNVERSIONED | paperbase_research_plan | UNVERSIONED |
| decision | UNVERSIONED | paperbase_agent_action | UNVERSIONED |
| projector | requirement-directed-multi-projection-v13 | paperbase_evidence_transport_batch, paperbase_evidence_projection | evidence-projection-schema-v20-requirement-multi-group |
| answer | answer-semantic-status-v3 | paperbase_grounded_answer_semantic_v2 | answer-semantic-status-object-v2 |
| candidate_selection | UNVERSIONED | paperbase_candidate_selection | UNVERSIONED |

UNVERSIONED 表示没有显式 version constant，schema 的 name/native DTO title 不能充当版本。Prompt source SHA 和实际文本 SHA（只有哈希，没有原文）在 JSON `prompt_versions`；source_hashes 可精确区分无版本组件。Projector schema/cache version 管投影 group 结构，projection semantic request `projection-semantic-v1` 管语义 fingerprint 输入，两者不是 checkpoint version。Classic Answer schema name 为 paperbase_grounded_answer。

## 4. Checkpoint acceptance 与 DTO domain

Semantic authority `projector-requirement-directed-multi-projection-v4`；checkpoint namespace `paperbase-agentic-ref-evidence-v2`；serializer 组件没有独立 version，记 `UNVERSIONED`。namespace 不等同语义版本。

| Stored semantic authority | DTO parse/inspect/roundtrip | resume / resume_stream |
|---|---|---|
| 当前 v4 | 允许 | 允许，继续正常 legality checks |
| 旧 v2 / v3 | 离线兼容允许 | 明确拒绝，在执行 graph 前 |
| 缺失 / 不匹配 | 可兼容读取的数据按 DTO 能力解析 | 明确拒绝 |

不裁剪/推断/回填旧 evaluated scope，不迁移历史 verified support。当前数据结构能 parse 不代表其旧语义可执行。

Checkpoint serializer：checkpoint_serde TypeAdapter JSON DTOs + SQLite SqliteSaver JsonPlusSerializer; pickle_fallback=False / allowed modules empty。Graph 持久字段如下（名称来自 TypedDict）：

`semantic_authority_version, thread_id, original_query, trusted_workspace, selected_paper_ids, knowledge_base_all, scope_resolution, corpus_snapshot, query_context, candidate_selection, plan, requirement_states, action, canonical_search_requests, action_canonicalization, tool_result, evidence_filter_report, pending_analysis_ids, registry_state, guard_state, last_action_rejection, action_rejections, observations, traces, terminal_status, stop_reason, clarification, error_codes, selected_evidence, answer, next_node, result, llm_diagnostics`

checkpoint_serde adapter domain：

```json
[
  "tuple[paperbase.retrieval.requests.CanonicalPaperSearchRequest | paperbase.retrieval.requests.CanonicalReferenceSearchRequest | paperbase.retrieval.requests.CanonicalKBSearchRequest, ...]",
  "<class 'paperbase.agentic.scope_resolver.TrustedWorkspaceScope'>",
  "<class 'ScopeResolution'>",
  "<class 'CandidateSelection'>",
  "<class 'CorpusScopeSnapshot'>",
  "<class 'CoverageState'>",
  "tuple[EvidenceQualification, ...]",
  "tuple[EvidenceRecord, ...]",
  "<class 'EvidenceFilterReport'>",
  "<class 'QueryContext'>",
  "<class 'ResearchPlan'>",
  "<class 'RequirementState'>",
  "tuple[RequirementState, ...]",
  "SearchPapersAction | SearchKnowledgeBaseAction | SearchReferencesAction | GetSectionContextAction | GetPaperOverviewAction | FinishAction",
  "<class 'ActionCanonicalization'>",
  "<class 'paperbase.retrieval.contracts.ToolResult'>",
  "<class 'RegistryState'>",
  "<class 'GuardState'>",
  "<class 'ActionRejectionContext'>",
  "tuple[ActionRejectionContext, ...]",
  "tuple[ResearchObservation, ...]",
  "tuple[ActionTrace, ...]",
  "tuple[paperbase.llm.client.LLMCallRecord, ...]",
  "<class 'ResearchRunResult'>",
  "<class 'ResearchAnswer'>"
]
```

Action、canonical execution requests/receipts、QueryContext、Plan、Registry/assessment、RequirementState、Observations/current与历史rejection、Budget、Answer 等 domain 数据直接或嵌套保存。Runtime服务/连接、guard_preflight closure、Compact View、CompletionView、ProjectionSemanticRequest 不保存；provider drafts 不 checkpoint。preflight callback 每轮重新组装，不是历史可恢复 callable。

## 5. Validated deterministic baseline

Golden：[eval/datasets/golden_dataset_v1_2.jsonl](../../../eval/datasets/golden_dataset_v1_2.jsonl)，SHA256 `05eb18b529073ac71930ef42cee8a7527fcba57c82f48b80e070bc04602a092b`；冻结 40 cases，已从当前库以只读方式验证 metadata/ID/domain：

| Primary taxonomy | Cases |
|---|---|
| bibliography | 4 |
| experiment | 6 |
| fact | 8 |
| method | 8 |
| result | 6 |
| synthesis | 4 |
| unanswerable | 4 |

E1 `agentic-eval-v2` 负责确定性 contract、provenance、support/citation mapping、控制诊断；不是 semantic correctness 分数。E2 `SemanticAnswerJudge` 目前 **NOT IMPLEMENTED**（仅 Protocol/DTO 扩展点，未被调用）。本次生成/测试零 provider，不生成新的 live Evaluation，也不把 deterministic 全测当成 model quality baseline。

## 6. Experimental observations

当前 sampling 是 **candidate sampling baseline, not yet full 40-case validated**。最新 manifest 只跑一次 method_002；3 research iterations、3 Decision calls、2 Tool calls、0 duplicates、0 invalid、agent_finished，生产 Answer 声明 complete。这次没有触发 recovery，因此没有证明 rejection recovery 稳定。Decision prompt tokens=16311，completion=2327，total=18638。不是 8-case 或 40-case 基线，也不是 E2 confirmed complete。

下列仅保留允许的现有运行 metadata，排除了 raw trace、问题/正文/Answer 内容：

```json
{
  "status": "candidate sampling baseline; not full 40-case validated",
  "latest_manifest": {
    "run_id": "run",
    "evaluation_schema_version": "agentic-eval-v2",
    "dataset_sha256": "05eb18b529073ac71930ef42cee8a7527fcba57c82f48b80e070bc04602a092b",
    "case_count": 1,
    "timestamp": "2026-10-06T13:45:01.841346+00:00",
    "mode": "live",
    "production_semantic_authority_version": "projector-requirement-directed-multi-projection-v4"
  },
  "latest_cases": [
    {
      "case_id": "method_002",
      "runs": 1,
      "run_status": "completed",
      "stop_reason": "agent_finished",
      "decision_model_calls": 3,
      "decision_iterations": 3,
      "decision_prompt_tokens": 16311,
      "decision_completion_tokens": 2327,
      "duplicate_rejections": 0,
      "formal_guard_rejections": 0,
      "invalid_iterations": 0,
      "tool_calls": 2,
      "section_context_calls": 0,
      "all_decision_outputs_schema_legal": true,
      "answer_status": "success",
      "answer_overall_status": "complete",
      "answer_model_calls": 1,
      "sdk_create_calls": 8
    }
  ],
  "sampling_profile": "qwen38_recommended",
  "recovery_triggered": false,
  "current_profile_fixed_8_case_stability_validated": false,
  "current_profile_full_40_case_validated": false,
  "independent_semantic_judge": false,
  "prior_profile_control_observations": [
    {
      "case_id": "method_002",
      "runs": 1,
      "run_status": "partial",
      "stop_reason": "invalid_action_limit",
      "decision_model_calls": 8,
      "decision_iterations": 5,
      "decision_prompt_tokens": 46518,
      "decision_completion_tokens": 1072,
      "duplicate_rejections": 5,
      "formal_guard_rejections": 0,
      "invalid_iterations": 2,
      "tool_calls": 3,
      "section_context_calls": 0,
      "all_decision_outputs_schema_legal": true,
      "answer_status": "insufficient_evidence",
      "answer_overall_status": "insufficient",
      "answer_model_calls": 0,
      "sdk_create_calls": 13
    },
    {
      "case_id": "fact_001",
      "runs": 1,
      "run_status": "completed",
      "stop_reason": "agent_finished",
      "decision_model_calls": 2,
      "decision_iterations": 2,
      "decision_prompt_tokens": 9801,
      "decision_completion_tokens": 194,
      "duplicate_rejections": 0,
      "formal_guard_rejections": 0,
      "invalid_iterations": 0,
      "tool_calls": 1,
      "section_context_calls": 0,
      "all_decision_outputs_schema_legal": true,
      "answer_status": "success",
      "answer_overall_status": "complete",
      "answer_model_calls": 1,
      "sdk_create_calls": 7
    }
  ],
  "evidence_sources": [
    {
      "path": "output/qwen_recommended_method002_20261006/run/run_manifest.json",
      "sha256": "0e4e41322d355dea208ef3f069978a3219b426e923e707c135cd03b055774ced"
    },
    {
      "path": "output/qwen_recommended_method002_20261006/results.json",
      "sha256": "559824c9efaf87ef88c991549b6489e68afeb4ec3c5a62f32bf8b5df776857c6"
    }
  ],
  "caveat": "Different runs/plans/projections/sampling; no causal parameter attribution. complete is production model status, not independent semantic truth."
}
```

旧 T=0 strict run 的 method_002 出现重复拒绝并 invalid_action_limit；fact_001 正常。旧 Projector trace 显示上游有效空 FACT 支持与另一类可修复 MECHANISM domain error；没有证据证明 silent converter support loss。历史 Prompt/配置/Plan/检索结果不同，不能把两次差异直接归因温度或 thinking，也不能当作当前遗漏频率。

## 7. 重生成与漂移比较

先在 .venv 中进行零 provider 校验；测试使用只读生产数据与临时 SQLite 隔离（guard 脚本在 output 中，禁止 network）。

```powershell
$env:PYTHONPATH = (Get-Location).Path + ';' + (Join-Path (Get-Location) 'output/architecture_snapshot_20261007')
.venv/Scripts/python.exe -X utf8 -m pytest tests -q -p snapshot_pytest_safety
.venv/Scripts/python.exe -X utf8 scripts/generate_architecture_snapshot.py --test-log output/architecture_snapshot_20261007/pytest_full.log
.venv/Scripts/python.exe -X utf8 scripts/generate_architecture_snapshot.py --compare OLD/architecture_snapshot.json NEW/architecture_snapshot.json
```

执行时需把 pytest stdout/stderr 保存为指定 UTF-8 log，并检查 exit code（本次由 subprocess runner 完成）。不传 --test-log 时 full_pytest=NOT_RUN，validation 不会冒称通过。可用 --output 指定 docs/architecture 下另一目录作 diff。配置/版本/哈希/Schema/Graph/DTO 会重新读取；语义解释是人工复核文本，若 production tree 改变则 requires_manual_architecture_review=true、validation fail，需人工核查，不能自动批准新树。CLI __main__ wrapper 会执行入口，所以只 compile；对应 library implementation 全部 import。该界限在 validation 显式记录。
