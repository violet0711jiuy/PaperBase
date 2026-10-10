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

新 Grounded Answer 架构尚未执行 provider live case，因此目前没有可宣称的 case 通过率、稳定性或成本基线。旧架构最近一次 gate 是 `method_002`：gold recall 100%，但中间语义投影漏掉 MPIWcapt 支持，最终 partial。这个失败直接促成了本次架构替换；其 Projector/span/repair 数值与 v3 不同口径，不能当成新系统结果。

此前关闭 Decision thinking 的两例实验中，timeout 从 6 次降为 0、总耗时从 9 分 14 秒降至 2 分 18 秒，但同时采样 profile 有变化，故它只支持“thinking 路径是主要嫌疑”，不构成严格因果隔离。新架构保持 thinking=false，并将在新 live gate 中继续记录。

## 下一轮验证

按以下顺序执行，失败即停：

1. `method_002`
2. `synthesis_001`
3. `unanswerable_001`
4. `method_001`、`fact_001`
5. 固定 8-case
6. 对关键 case 重复运行，再考虑全 40-case

`method_002` 的最低验收是：gold 正常进入 Registry，全部 Registry Evidence 出现在 answer input，MPIWcapt 与 C_i 都形成带合法 span 的 RequirementAnswer，且不因同一 Evidence 的原始检索绑定而限制跨 Requirement 使用。

`synthesis_001` 重点检查 gold recall、answer input coverage、citation coverage、Decision timeout、实际 token 和 wall time。`unanswerable_001` 必须正确拒答，不能为了 complete 强行引用不相关 Evidence。

历史实测和旧口径数据见 [evaluation archive](../archive/evaluation-baselines-20261006.md)。本地 `output/` 报告不随仓库发布。
