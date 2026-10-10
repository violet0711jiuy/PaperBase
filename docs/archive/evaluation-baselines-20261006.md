# Historical evaluation snapshots — 2026-10-06

这是清理前 current-baseline 的归档。下文“当前”“最新”“仅离线”等措辞只描述各次实验当时的快照，不是当前版本结论。模型、Prompt、采样与 State 不同的独立 run 不能用于单变量因果比较。当前参数和最后验证结果见 [Current baseline](../evaluation/current-baseline.md)。

---

# 当前验证状态与历史 Retrieval 基线

核对日期：2026-10-06。下面分开记录当前 Agentic smoke 与历史 Classic benchmark，不能互相替代。

## 最新采样配方实验：method_002 单次 complete

启用 qwen38_recommended：仅 Decision 思考，温度 1/top_p 0.95/presence_penalty 0；其余 Agentic 节点非思考，温度 0.7/top_p 0.8/presence_penalty 1.5，共同 top_k 20/min_p 0/repetition_penalty 1。真实请求均携带这些参数，8 次外层 SDK create 均成功，没有参数拒绝或静默回退。

method_002 仅运行一次：3 轮、3 Decision calls、2 Tools、0 duplicate/invalid/Schema failure，agent_finished。R1/R2 均建立 verified support，Answer 原文保存、校验通过，overall complete，引用 E1。Decision 总 tokens 18638，上次仅开思考为 35798。

本轮同时改变其他 Agentic 节点采样，Plan 从上次三项变为两项，Projection 状态不同，不能由单次 live 证明某个参数的因果效果或稳定性。没有追加 case 或独立 Judge。

本地详见 [官方采样配方单次报告](../../output/qwen_recommended_method002_20261006/report.md)。Targeted 39 passed；full 1036 passed，21 subtests passed。

## 前一次思考实验：仅 Decision 开启思考，method_002 为 partial

Decision temperature=1、enable_thinking=true、strict json_schema，其余节点仍关闭思考。method_002 仅真实运行一次，5 轮、5 Decision calls、4 Tools、0 duplicate、0 invalid/Schema failure、agent_finished。provider 每次返回非空 reasoning_content（仅保存字符数），证明本次确实产生思考输出。

本次 Planner 拆为 R1/R2/R3（额外包含两项之间的权衡关系），与前一次两项不同。Answer 原文保存且校验通过，R1 complete、R2 unsupported、R3 partial，overall partial。Decision 总 tokens 35798，前一次不开思考为 35140；本次 completion tokens 4612，前一次 799。单次独立 run 不证明思考的净效果或稳定性，未追加 case。

本地详见 [思考模式单次报告](../../output/decision_thinking_method002_20261006/report.md)。Targeted 46 passed；full 1031 passed，21 subtests passed。

## 前一次重试：method_002 原始 Answer 已保存，结果 partial

保持 Decision temperature=1、thinking=false、strict json_schema、无应用层输出 token cap，再运行 method_002 一次。Decision 6 calls、4 Tools、0 duplicate、0 invalid iteration，agent_finished；第 4 轮第一次候选 empty query 违反 Schema，在第二次候选修复。

Answer 调用一次，原始 message.content 已保存：1474 字符、641 completion tokens，JSON/Schema/domain 校验通过，status=success、overall=partial（R1 unsupported，R2 complete，引用 E1）。本次没有 fallback，没有复现前一次 Answer JSON invalid；单次不同状态的 live 不能证明稳定性。两套 case runner 已默认保存每次 Answer 最终输出到 cases/<case_id>.answer_llm_outputs.jsonl，包括失败与修复响应；原文独立于公共 audit/Graph/评分，不保存隐藏推理。

本地详见 [原始输出重试报告](../../output/answer_raw_capture_method002_20261006/report.md)。专项 14 passed；full 1027 passed，21 subtests passed；没有追加其他 case。

## 前一次温度实验：method_002 单次，无重复但 Answer fallback

2026-10-06 单独将 Decision temperature 配置为 1，保持 thinking 关闭与 strict json_schema，method_002 仅真实运行一次。Decision 3 calls、0 duplicate、0 invalid、2 Tools，agent_finished；E1 建立 R1/R2 局部 verified support。最终 Answer 调用一次，在 AnswerSemanticDraft JSON 解析阶段失败（ANSWER_JSON_INVALID），进入 fallback，端到端未成功。

当前 Agentic 请求已移除应用层输出 token 上限；前一个 temperature=0 run 仍发送 Decision max_tokens=8000。独立 live 的 Planner/Projector 状态也不同，不能将变化全部归因于温度。此次没有 rejection/recovery，不能证明 recovery 或长期策略稳定，也没有追加其他 case。Full pytest tests：1013 passed，21 subtests passed。

本地详见 [温度 1 单次报告](../../output/decision_temperature1_method002_20261006/report.md)。

## 前一轮修复版本：两例真实复验仍未通过稳定性门槛

2026-10-06 Prompt/契约修复后，method_002 与 fact_001 各运行一次。两者 Decision 实际均为 temperature=0/json_schema/strict=true，所有 Action 输出 Schema 合法。method_002 为 8 model calls、5 duplicate、2 invalid iterations、3 Tools，停止于 invalid_action_limit，Answer insufficient；fact_001 为 2 calls、0 duplicate/invalid、1 Tool，agent_finished，Answer complete。

method_002 的 3.2/5.2 在完整可见的 raw blocked Action 与 canonical request/identity 反馈下仍原样重复，当前 serializer 对这条路径已由实际输入验证。References 字段修复本次没有触发对应 live 分支，仍只有离线回归。两例结论 DECISION_RECOVERY_NOT_STABLE，没有追加 8-case 或 40-case。

本地详见 [两例报告](../../output/prompt_contract_fix_live_2case_20261006/report.md)。此为独立 run，不把与旧结果的差异当作 Prompt 的因果净效果；fact_001 未触发 recovery，不能证明恢复成功率。

## 修复前最近完整 8-case：历史快照

该历史实验使用 ModelScope / Qwen/Qwen3.8-Flash-Next，Decision 实际请求 temperature=0、json_schema、strict=true。先 method_002/result_001 各一次，通过阶段控制检查后固定 8-case 各一次，共 10 research runs；没有追加 replay/provider probe 或正式 40-case。

| 固定 8-case | Decision 模型 calls | preflight duplicate | invalid iterations | Tools | stop | Answer overall |
|---|---:|---:|---:|---:|---|---|
| fact_001 | 8 | 5 | 2 | 3 | invalid_action_limit | insufficient |
| method_001 | 5 | 1 | 0 | 3 | agent_finished | complete |
| method_002 | 2 | 0 | 0 | 1 | agent_finished | complete |
| experiment_001 | 3 | 0 | 0 | 2 | agent_finished | complete |
| result_001 | 2 | 0 | 0 | 1 | agent_finished | complete |
| synthesis_001 | 7 | 0 | 1 | 4 | agent_finished | partial |
| bibliography_001 | 3 | 0 | 0 | 2 | agent_finished | complete |
| unanswerable_001 | 4 | 0 | 0 | 3 | agent_finished | insufficient |

前置 2-case：method_002 为 8 model calls、3 duplicates、1 invalid iteration、agent_finished/partial；result_001 为 2 calls、0 duplicates、0 invalid、agent_finished/complete。独立 run 的上游支持变化不能据此归因于 recovery 修复。

最终 gate：**DECISION_RECOVERY_NOT_STABLE；未进入正式 40-case。** fact_001 在明确可见的 raw blocked Action 下仍重复，Guard 正确拒绝，没有无限 retry；synthesis_001 两次 empty query 违反已发送的 Schema minLength，被本地校验捕获。发送 strict=true 不证明 provider 实际强制执行全部 keyword。

实际 wire RCA 另外发现 canonical identity/有效请求摘要在原 repair serializer 中被丢弃。运行结束后已修补 allowlisted rejected_effective_request 转发。**该最终补丁只有离线验证，未再做 live**；上表不能冒充修补后代码的真实运行结果。

已有最后一次代码回归记录：targeted 109 passed；full 1001 passed，21 subtests passed。这是前一轮代码验收结果，本次文档维护没有重跑 pytest、没有调用 LLM。确定性测试证明契约/恢复边界，不证明模型策略稳定或 Answer 语义正确。

后续 Prompt/契约审核发现 References 拒绝摘要字段错配及 lexical 反馈上限残留，现已另行修复并统一 Decision 模式和 Planner 义务来源；修复记录位于本地 output/prompt_contract_fix_20261006/。上述 live 表格仍对应修复前快照，不能当作当前版本已通过稳定性验收。没有追加真实模型运行。

本地详细工件：

- [Decision Recovery report](../../output/decision_rejection_recovery_20261006/decision_rejection_recovery_report.md)
- [root cause matrix](../../output/decision_rejection_recovery_20261006/root_cause_matrix.json)
- [live 8-case results](../../output/decision_rejection_recovery_20261006/live_8case_results.json)
- [live / delivery scope check](../../output/decision_rejection_recovery_20261006/scope_check.json)

这些 output 文件是本地实验产物，不随仓库发布；文档正文保留可公开理解的结论。E1 evaluator agentic-eval-v2 已冻结，E2 semantic Judge 未实现。[评测设计](../evaluation/design.md)区分来源契约诊断与语义质量验收。

## 历史 Classic Context-free Retrieval

以下是原先保存的 Retrieval benchmark，不是当前 Agentic 40-case 验证，也未用当前代码重新运行。

Dataset：eval/datasets/golden_dataset_v1_2.jsonl，内容 cases 32、bibliography 4、unanswerable 4。逐 case trace 位于本地 eval/results/；Hit/Recall 内容分母与 bibliography 分母分别计算。

| Metric | Result |
|---|---:|
| Hit@5 | 0.937500 |
| Recall@5 | 0.859375 |
| MRR@5 | 0.715625 |
| Bibliography Intent F1 | 1.000000 |
| Bibliography Hit@5 | 1.000000 |

### Evidence Expansion

对 32 条 answerable content cases 的 Top-5 anchors 做 section-aware expansion：

| Metric | Before | Expanded | Delta |
|---|---:|---:|---:|
| Hit | 0.937500 | 0.968750 | +0.031250 |
| Recall | 0.859375 | 0.932292 | +0.072917 |

历史记录：recovery cases 1；平均 expansion latency 3.792 ms；平均 expanded chunks 8.4375，最大 13；errors 0。method_010 的 chunk_0016 anchor 补回 chunk_0017；synthesis_001 当时仍未命中五个跨位置 gold chunks。

| Slice | Raw Recall@5 | Expanded Recall | Delta |
|---|---:|---:|---:|
| method | 0.687500 | 0.916667 | +0.229167 |
| synthesis | 0.500000 | 0.625000 | +0.125000 |
| single_hop | 0.950000 | 1.000000 | +0.050000 |
| multi_hop | 0.708333 | 0.819444 | +0.111111 |

新的复现将验证当前代码，结果应另存快照，不能覆盖历史数值：

```powershell
.\.venv\Scripts\python.exe eval\scripts\validate_golden_dataset.py
.\.venv\Scripts\python.exe eval\scripts\run_retrieval_eval.py
.\.venv\Scripts\python.exe eval\scripts\build_retrieval_diagnostic.py
```

Retrieval eval 需要论文、索引、本地模型与 provider；离线 diagnostics 读取既有 trace。完整 ablation、Answer correctness/faithfulness/completeness Judge 均不包含在上述历史数字中。
