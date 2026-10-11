# Agentic Grounded Answer 确定性诊断

版本：`agentic-grounded-answer-eval-v4`。该 evaluator 不调用语义 Judge、不修改生产 Agent 或 Golden。

## Evidence funnel

```text
gold chunks
  → retrieved chunks
  → Registry Evidence
  → answer_input_evidence_ids
  → cited_evidence_ids / source_span_ids
```

| 阶段 | 关键字段 | 典型失败 |
|---|---|---|
| retrieval | `agent_retrieved_source_chunk_ids` | `RETRIEVAL_MISS` |
| registry | Evidence `source_chunk_ids` | 检索结果未进入 Registry |
| answer input | `answer_input_evidence_ids` | `GOLD_EVIDENCE_NOT_SHOWN_TO_ANSWERER` |
| grounding | `cited_evidence_ids`、Requirement citations | `GOLD_ANSWER_INPUT_NOT_CITED`、`REQUIRED_REQUIREMENT_GAP` |
| control | stop reason、rejections、budget | `BUDGET_EXHAUSTED`、`INVALID_ACTION_STOP` |
| answer validation | LLM diagnostics | schema/domain/citation failure |

`selected_evidence_ids` 在当前架构中是实际引用 Evidence 的兼容投影，不是一个独立选择阶段。完整输入必须看 `answer_input_evidence_ids`。

每个 funnel 阶段同时保留两套不可混报的量：

- `gold_*_chunk_ids` / strict recall：只认 `relevant_evidence` 中的 canonical chunk ID；
- `path_covered_gold_*_chunk_ids` / evidence path coverage：canonical chunk 本身命中，或人工审核的某条 all-of `evidence_alternatives` 路径全部命中，才算覆盖该 canonical 单元。

替代路径只来自冻结 Golden，不由单次运行结果或模型自动生成。它用于避免把已获得等价证据误判为检索/grounding 失败，不会重写或抹去严格 recall。

## Requirement 记录

每个 Requirement 审计包含：

- `requirement_id/description/required/status`
- 已校验 `answer`
- `citations[{evidence_id, source_span_ids}]`
- 未回答时的 `remaining_gap`

required Requirement 只有同时具备 answer 与 citation 才算 grounded complete。状态字段不能替代实际答案和引用。

## Span 校验

Answerer 只能引用为每条 Evidence 预切分的 span ID。本地校验要求 ID 存在且顺序正确，并从源文本恢复 excerpt。有序但不连续的选择按 gap 确定性拆成多个连续 anchor，不扩张引用范围；乱序、重复、未知 Evidence、未知 span 或过度碎片化仍是硬失败。程序不模糊匹配模型复制文本。

## 解释边界

- strict Gold recall 100% 只证明 Retriever 找到了 canonical 标注 chunk；path coverage 100% 表示 canonical 或人工审核的等价证据路径已完整覆盖。
- Evidence 进入 Registry 不证明 Answerer 看到了它；由 answer input 阶段确认。
- Answerer 看到 Evidence 但未引用，是 grounding/semantic read 问题，不应归因于 Retriever。
- span 100% 合法只证明可追溯，不证明 entailment。
- token 少可能来自超时或提前停止，必须结合完成状态与未知 usage calls 解读。

公开审计版本为 `grounded-answer-v1`，保存 canonical effective requests、request receipts、Registry provenance、答案输入、引用和安全诊断，不保存 hidden reasoning。
