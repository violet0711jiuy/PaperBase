# 当前验证基线

核对日期：2026-10-11。

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

新 Grounded Answer 架构已于 2026-10-10 再次单独重跑 `method_002`。核心正确性门禁已通过：R1/R2 均 complete，span anchors 3/3，Answer 正式 audit 2/2，Decision view 与离线重建逐字一致，且 C_i 只检索一次。Gold recall 100%，Decision thinking=false、timeout=0，耗时 72.6 秒，已知用量 14,220 tokens。

该 run 唯一未通过的是旧“不相邻翻倍”门禁：Decision view 为 `2115 → 4512 → 4512`，首次召回四条 Evidence 后增长 2.13 倍，随后不再增长。定位表明增长中包含 Evidence 正文预览；这与“Decision 只决定动作、Grounded Answerer 才读正文”的边界不符，因此 Decision Evidence catalog 已进一步改为纯元数据，移除 `source_preview` 及其配置。用本轮保存状态离线重建为 `2115 → 3785 → 3785`，最大相邻倍率约 1.79；这是离线 replay，改动仍需下一次 live run 验证。

本轮另有 5 次无 usage 的连接错误（4 次 Decision、1 次 Answer），均为 `APIConnectionError → ConnectError → ConnectionResetError`；Answer 重试后成功。它们属于传输稳定性问题，不应记作 Decision timeout、模型输出错误或架构正确性失败，但会造成 wall time 与总 token 不完整。

此前关闭 Decision thinking 的两例实验中，timeout 从 6 次降为 0、总耗时从 9 分 14 秒降至 2 分 18 秒，但同时采样 profile 有变化，故它只支持“thinking 路径是主要嫌疑”，不构成严格因果隔离。新架构保持 thinking=false，并将在新 live gate 中继续记录。

## 下一轮验证

按以下顺序执行，失败即停：

1. 再次运行 `method_002`，验证 metadata-only Decision view 的相邻增长门禁，并继续记录连接错误
2. `synthesis_001`
3. `unanswerable_001`
4. `method_001`、`fact_001`
5. 固定 8-case
6. 对关键 case 重复运行，再考虑全 40-case

`method_002` 的最低验收是：gold 正常进入 Registry，全部 Registry Evidence 出现在 answer input，MPIWcapt 与 C_i 都形成带合法 span 的 RequirementAnswer，且不因同一 Evidence 的原始检索绑定而限制跨 Requirement 使用。

`synthesis_001` 重点检查 gold recall、answer input coverage、citation coverage、Decision timeout、实际 token 和 wall time。`unanswerable_001` 必须正确拒答，不能为了 complete 强行引用不相关 Evidence。

历史实测和旧口径数据见 [evaluation archive](../archive/evaluation-baselines-20261006.md)。本地 `output/` 报告不随仓库发布。
