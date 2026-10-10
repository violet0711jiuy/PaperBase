# 当前已知限制与文档漂移

Snapshot `paperbase-architecture-snapshot-v1`；生成时间 `2026-10-07T15:33:50.053750+08:00`（Asia/Shanghai）。Commit `e365bf9e27eaab29f554a3c3f8ec441b2c21a9d6`；dirty=`True`。

当前生产树 SHA256：`54766b03327438e05f56f3d7a6777907c5c089541c9eb1aba8463905c02f5e7d`（158 个 paperbase/app Python 文件，包含未提交文件）。配置 bundle SHA256：`59af7d9d7c859eace4993289182544c47fecbed28954334261d64a225b9ac4e5`。历史 commit 不代表当前 dirty tree，精确路径和哈希见 [source_hashes.json](source_hashes.json)。

本文件只记录当前真实边界与证据不足；没有添加生产功能/Prompt规则/采样试验。已修复的 score contract、canonical request、local-scope、多投影 cache、Answer compiler、rejection serializer 不继续列为未解决缺陷。

## 未解决限制（7 项）

### L01 — Decision can repeat a visible rejected effective request; current T=1 profile recovery stability is untested.

- Affected layer：Decision / recovery
- Severity / kind：HIGH / UNRESOLVED
- Current mitigation：Shared Guard rejection + one bounded same-iteration replacement + cumulative invalid iteration stop.
- Observed evidence：Prior T=0 strict-json-schema method_002: 5 duplicate rejections / invalid_action_limit despite full canonical feedback. Latest T=1 run has no rejection, so does not validate recovery.
- Planned next validation：Separate authorized fixed 8-case stability / saved failure-State comparison; not run by snapshot.
### L02 — Projector can return structurally valid extraction with empty proposed support; source Integrity cannot invent omitted semantic support.

- Affected layer：Projector / support establishment
- Severity / kind：MEDIUM / UNRESOLVED
- Current mitigation：Typed/domain/source validation, scoped group repair/fallback, PRE/POST diagnostics, support monotonicity.
- Observed evidence：Historical structural trace confirms upstream empty FACT assertions plus a distinct domain rejection repaired for MECHANISM; no silent converter support loss found. That older T=0 run is not a frequency measurement for current profile.
- Planned next validation：Evaluate support establishment with current profile separately, using existing PRE/POST metadata.
### L03 — complete/partial and relevance/entailment are model semantic judgments without independent current E2 validation.

- Affected layer：Answer / Projector / evaluation
- Severity / kind：HIGH / UNRESOLVED
- Current mitigation：Deterministic IDs, provenance, actual citations, selected-support mapping; no program claim of semantic completeness.
- Observed evidence：AnswerSemanticDraft uses complete/partial; compiler only checks allowed support and citations. SemanticAnswerJudge is Protocol/DTO only.
- Planned next validation：Calibrated independent E2/human semantic validation in a separate task.
### L04 — qwen38_recommended has only a successful single method_002 observation, not a full stability/40-case production baseline.

- Affected layer：Evaluation / model configuration
- Severity / kind：HIGH / UNRESOLVED
- Current mitigation：Run manifests, frozen Golden, exact request audits; distinguish candidate config from validated quality.
- Observed evidence：Latest manifest case_count=1; current-profile fixed 8-case and 40-case have not been run.
- Planned next validation：Separate authorized stability suite before formal 40-case.
### L05 — Semantic near-duplicates remain legal when canonical retrieval-relevant inputs differ.

- Affected layer：Guard / Decision strategy
- Severity / kind：INFO / DESIGN_BOUNDARY
- Current mitigation：Exact canonical identity dedup + no-progress/targeted-support hints.
- Observed evidence：Guard compares identities; regression test explicitly permits semantic-near-duplicate distinct requests. No semantic detector implemented.
- Planned next validation：Measure redundant strategies separately; no forced semantic detector in current production.
### L06 — Sending strict=true does not prove the provider enforces every Schema keyword or prevents semantic omission.

- Affected layer：Provider structured output
- Severity / kind：MEDIUM / UNRESOLVED
- Current mitigation：Local Pydantic/domain validation, Runtime Guard, bounded repairs/fallback.
- Observed evidence：Older strict requests still produced empty query/local Schema rejection; current one-case success is not enforcement proof. Native request metadata now independently verified offline.
- Planned next validation：Provider/schema reliability study separately, without removing local validation.
### L07 — Base OpenAI-compatible constructor ignores a true thinking_enabled flag and initializes enable_thinking=false.

- Affected layer：Base client configuration
- Severity / kind：LOW / UNRESOLVED
- Current mitigation：Current .env is false; current Agentic views explicitly set thinking, so current configured calls are unaffected.
- Observed evidence：load_llm_runtime_settings parses flag; OpenAICompatibleChatClient.__init__ hard-codes false.
- Planned next validation：Separate base-client contract fix/regression; snapshot must not modify it.

L01/L02 的证据来自旧采样 live 或历史保存 trace，故记录为能力风险/未验证稳定性，不能把旧失败率宣称为当前 profile 失败率。L03 的本地 anchor 校验保证原文来源与 domain，不证明语义蕴含。L05 是精确去重的设计边界，不是已承诺而缺失的 semantic Judge。L07 是 base constructor 的潜在配置问题，当前 false base 与显式 Agentic views 不受影响。

## DOCUMENTATION_DRIFT（0 处）



## Authority audit 与检查范围

本次六项核心 owner/helper + delegation 审计报告 0 个冲突，JSON `authority_conflicts` 为 `[]`。这是限定检查结果，不是全系统语义正确性的证明。Finish closure 在 Graph、candidate screening 与 Answer completeness、request/batch identity 和 Classic/Agentic 合约均明确区分。

观察来源只读取 [output/qwen_recommended_method002_20261006/run/run_manifest.json](../../../output/qwen_recommended_method002_20261006/run/run_manifest.json)、[output/qwen_recommended_method002_20261006/results.json](../../../output/qwen_recommended_method002_20261006/results.json)、[output/prompt_contract_fix_live_2case_20261006/results.json](../../../output/prompt_contract_fix_live_2case_20261006/results.json)、[output/projector_structural_trace_20261006/projector_structural_trace_report.md](../../../output/projector_structural_trace_20261006/projector_structural_trace_report.md) 的既有记录与当前源码，不导出其中 Evidence/Answer 内容。未来 next validation 只是建议，未被本轮自动授权或执行。

## 本轮校验

Validation passed=`True`，provider calls=0；15 项检查详情见 [snapshot_validation.json](snapshot_validation.json)。本轮原始 production/config/tests/Golden/docs/checkpoint 的哈希前后核对，测试对生产 SQLite 的写入隔离到临时目录。安全隔离只属于本次 tooling，未改变 Graph/checkpoint 生产实现。
