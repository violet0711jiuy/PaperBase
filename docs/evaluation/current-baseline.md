# 当前验证基线

核对日期：2026-10-10。

## 当前代码基线

| 项目 | 当前值 |
|---|---|
| Agentic architecture | Raw Evidence Pool → Grounded Answerer → deterministic source-span validation |
| Decision thinking | 默认 false |
| Checkpoint authority | `paperbase-agentic-grounded-answer-v1` |
| Public audit | `grounded-answer-v1` |
| Evaluator | `agentic-grounded-answer-eval-v3` |
| Agentic output cap | 不发送应用层 cap；Grounded Answerer 接收全部 Registry Evidence |
| Golden | `golden_dataset_v1_2.jsonl`，40 cases，SHA256 `05eb18b529073ac71930ef42cee8a7527fcba57c82f48b80e070bc04602a092b` |

## Live 验证状态

新 Grounded Answer 架构已于 2026-10-10 单独重跑 `method_002`，结果 **FAIL**：gold recall 100%，Decision thinking=false、timeout=0，但两次 Grounded Answer 都把同一 Evidence 中多个分离区间放入一条 citation，触发旧连续性规则，最终 `fatal_error/fallback`。实际 Answer 文本已覆盖 MPIWcapt 与 C_i，因此该失败定位为 citation representation/validation failure，而不是 retrieval miss 或 answer semantic omission。

该 run 耗时 75.9 秒、39,212 tokens，Decision view chars 为 `2102 → 8047 → 11464 → 13900`，另有一次重复 C_i 搜索被 Guard 拦截。报告保存在本地 `output/decision_atomic_method002_rerun1_live_20261010/`，不随仓库发布。

随后代码已修复但**尚未再次 live 验证**：有序稀疏 spans 现在拆成多个精确连续 anchors；乱序/未知/重复仍拒绝。用该 run 的真实 E1 与 span IDs 离线重放已全部通过。Decision view 去除重复 Evidence、答案正文和重复 rejection 信息，同一保存状态重建后的前三轮为 `2106 → 4188 → 5792`。这些是离线 replay，不得写成新的 live 通过结果。

此前关闭 Decision thinking 的两例实验中，timeout 从 6 次降为 0、总耗时从 9 分 14 秒降至 2 分 18 秒，但同时采样 profile 有变化，故它只支持“thinking 路径是主要嫌疑”，不构成严格因果隔离。新架构保持 thinking=false，并将在新 live gate 中继续记录。

## 下一轮验证

按以下顺序执行，失败即停：

1. 再次运行 `method_002`，验证 span 拆分、答案完成状态、审计中的 Answer 调用和实际 Decision view
2. `synthesis_001`
3. `unanswerable_001`
4. `method_001`、`fact_001`
5. 固定 8-case
6. 对关键 case 重复运行，再考虑全 40-case

`method_002` 的最低验收是：gold 正常进入 Registry，全部 Registry Evidence 出现在 answer input，MPIWcapt 与 C_i 都形成带合法 span 的 RequirementAnswer，且不因同一 Evidence 的原始检索绑定而限制跨 Requirement 使用。

`synthesis_001` 重点检查 gold recall、answer input coverage、citation coverage、Decision timeout、实际 token 和 wall time。`unanswerable_001` 必须正确拒答，不能为了 complete 强行引用不相关 Evidence。

历史实测和旧口径数据见 [evaluation archive](../archive/evaluation-baselines-20261006.md)。本地 `output/` 报告不随仓库发布。
