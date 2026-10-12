# Agentic Grounded Answer 确定性诊断

版本：`agentic-grounded-answer-eval-v5`，trace 为 `case-trace-grounded-answer-v3`。该 evaluator 不调用语义 Judge、不修改生产 Agent 或 Golden。当前答案与引用粒度是整体答案 / 完整 Evidence；历史 span observer 不直接兼容新 contract。

## Evidence funnel

```text
gold chunks
  → retrieved chunks
  → Registry Evidence
  → answer_input_evidence_ids
  → cited_evidence_ids（完整 Evidence）
```

| 阶段 | 关键字段 | 典型失败 |
|---|---|---|
| retrieval | `agent_retrieved_source_chunk_ids` | `RETRIEVAL_MISS` |
| registry | Evidence `source_chunk_ids` | 检索结果未进入 Registry |
| answer input | `answer_input_evidence_ids` | `GOLD_EVIDENCE_NOT_SHOWN_TO_ANSWERER` |
| grounding | `cited_evidence_ids`、Requirement evidence_ids/gap | `GOLD_ANSWER_INPUT_NOT_CITED`、`REQUIRED_REQUIREMENT_GAP` |
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
- `evidence_ids`：整体正文中覆盖该项所引用的 Evidence
- `remaining_gap`：未回答 / 部分回答的具体缺口

每项不再另存答案段落或 span citation。required 项需 evidence_ids 非空且 gap=null，整体正文有引用，才可派生 complete。状态字段不能替代内容复核；RequirementTrace.answer_present 只表示整体正文对该项存在自报覆盖部分，并非一段独立 Requirement answer。

## Evidence 引用校验

本地校验要求引用属于实际输入 Evidence、ID 去重、正文 [E#]/[REF#] 标记与 citations 一致、覆盖引用是正文引用子集。新输入必须逐份 content 与 Registry 原文完全一致且没有重复 source_spans 文本。正文引用失败重试整体；覆盖字段失败只修复失败项，不重写合法正文，不把元数据修复当作新增答案内容。

`invalid_request` 表示供应商拒绝请求，不是模型 JSON/领域校验失败；没有返回模型输出时引用与必要内容为未验证。Observer 的 SDK/audit 对齐应把该阶段计入 transport failure，而不是返回了答案但业务拒绝；零响应对应零可捕获输出，不得将空捕获文件误报为漏捕获。安全 provider 诊断使用现有 validation_diagnostics 字段，不把它计作模型修复成功。

Span validation 为 N/A，不计为通过，也不与旧 span 成功率比较。被引 Evidence 包含一个 gold chunk，不代表那个 chunk 的必要事实已经写入答案；仍需独立 content_review。不得只因 whole-Evidence 引用扩宽了 gold_cited 集就宣称质量提升。

## 解释边界

- strict Gold recall 100% 只证明 Retriever 找到了 canonical 标注 chunk；path coverage 100% 表示 canonical 或人工审核的等价证据路径已完整覆盖。
- Evidence 进入 Registry 不证明 Answerer 看到了它；由 answer input 阶段确认。
- Answerer 看到 Evidence 但未引用，是 grounding/semantic read 问题，不应归因于 Retriever。
- Evidence ID 合法只证明引用可追溯，不证明 entailment 或必要内容已解释。
- token 少可能来自超时或提前停止，必须结合完成状态与未知 usage calls 解读。

公开审计版本为 `grounded-answer-v2`，保存 canonical effective requests、request receipts、Registry provenance、答案输入、Evidence 引用和安全诊断，不保存 hidden reasoning。旧结果仍按其旧版本读入，不覆盖历史产物；旧 checkpoint 不恢复运行。Golden 与内容覆盖口径不因架构修改而放宽。
