# 评测设计与边界

核对日期：2026-10-11。当前 Agentic 确定性评测契约是 `agentic-grounded-answer-eval-v4`。

## 三层评测

| 层级 | 作用 |
|---|---|
| Context-free Retrieval | Hit@5、Recall@5、MRR、bibliography routing 与 expansion |
| Agentic deterministic evaluation | 定位 retrieval → registry → answer input → cited 的证据流失、控制失败、span 合法性与成本 |
| Semantic evaluation | correctness、faithfulness、Requirement completeness、citation entailment；尚未形成独立自动 Judge |

确定性校验只能证明 ID 和原文范围合法，不能证明引用在语义上支持答案。生产 Grounded Answerer 的 complete/partial 也不是外部真值。

## Golden

正式数据为 [golden_dataset_v1_2.jsonl](../../eval/datasets/golden_dataset_v1_2.jsonl)，40 cases；SHA256 `05eb18b529073ac71930ef42cee8a7527fcba57c82f48b80e070bc04602a092b`。`reference_answer` 和 `required_facts` 只用于评测，不能注入 Agent。

候选 [golden_dataset_v1_3.jsonl](../../eval/datasets/golden_dataset_v1_3.jsonl)
只为 `synthesis_001` 增加人工审核的 `evidence_alternatives`。字段的 key 是
canonical Gold chunk；每个内层列表是一条必须全部出现的 all-of 路径，多条路径之间为 OR。
严格 chunk recall 仍只按 `relevant_evidence` 计算；`evidence_path_coverage` 单独报告
canonical 或任一替代路径的覆盖率，不能把两者混成一个指标，也不能由运行结果自动反推替代路径。

固定 smoke：`fact_001`、`method_001`、`method_002`、`experiment_001`、`result_001`、`synthesis_001`、`bibliography_001`、`unanswerable_001`。

## 新架构门禁顺序

1. `method_002`：验证全部 gold 已召回时，完整 Evidence Pool 能回答 MPIWcapt 与 C_i。
2. `synthesis_001`：验证多证据综合、answer-input/citation funnel、Decision timeout 与成本。
3. `unanswerable_001`：验证没有充分证据时正确拒答。
4. `method_001` 与 `fact_001`：覆盖较普通的事实/方法路径。
5. 通过后再跑固定 8-case；最后才进行重复稳定性和 40-case。

每个 gate 必须记录 dataset hash、代码 commit、provider/model、真实 sampling、预算、token/latency、timeout 和 case-level audit。live provider 运行需要显式授权，不由单元测试自动触发。

详见[确定性诊断契约](e1_diagnostic.md)和[当前基线](current-baseline.md)。
