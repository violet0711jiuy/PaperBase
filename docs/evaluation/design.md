# 评测设计与已实现边界

核对日期：2026-10-06。当前要区分三种工作，不能共用“评测已完成”的结论。

| 层级 | 状态 | 验证内容 |
|---|---|---|
| Context-free Retrieval benchmark | 已有脚本与历史基线 | 召回、排名、bibliography routing、扩展效果 |
| Agentic E1 / agentic-eval-v2 | deterministic evaluator 已实现并冻结；已运行固定 smoke | provenance funnel、接口状态、控制失败与成本诊断 |
| Agentic E2 semantic evaluation | 未实现；只有扩展 Protocol/DTO | Answer correctness/faithfulness/completeness/citation support，需后续校准 |

当前架构见[Agentic Research](../architecture/agentic-research.md)，具体定义见[E1 契约](e1_diagnostic.md)，最新结果见[验证状态](current-baseline.md)。

## Frozen Golden

正式数据：[golden_dataset_v1_2.jsonl](../../eval/datasets/golden_dataset_v1_2.jsonl)，40 cases：fact 8、method 8、experiment 6、result 6、synthesis 4、bibliography 4、unanswerable 4。

SHA256：05eb18b529073ac71930ef42cee8a7527fcba57c82f48b80e070bc04602a092b。dataset 本轮未改。字段包含 question、primary_type/tags、paper_id、answerable、reference_answer、required_facts、relevant_evidence 和 expected_bibliography_intent；后者是 boolean，没有 expected_bibliography 列表。

reference_answer/required_facts 用于人工和未来语义评测，不能直接注入 Agent。relevant_evidence 保存 chunk 与 section/page provenance。论文有答案但系统没找到属于检索/支持失败，不能改标 unanswerable。

Context-free benchmark 的 question 必须自包含；gold paper_id 用于评分，不偷偷作为 Retriever 上下文。Active-paper scoped Agentic QA 则明确使用受信任 scope，并记录在 run manifest；不同场景的数值不能混报。

## Retrieval benchmark

现有脚本见[eval README](../../eval/README.md)。Hit@K 检查至少一次命中，Recall@K 衡量 gold chunks 覆盖，MRR 衡量首个命中位置。按问题类型、single/multi-hop 与语言等 tags 做 slice。分别报告 RRF/reranker 前后与 expansion 后结果，不把 expanded recall 称为原始 Recall@5。

Classic bibliography intent F1 与 Bibliography Hit@5 单独报告。Agentic 的 search_references 是自主 Tool，不套用经典 intent routing 的同一解释。

Dense-only、Dense+BM25、rewrite、reranker 等 ablation 是后续实验安排，不宣称已有完整消融结果。旧设计中的 E1/E2/E3/E4 检索消融编号不再使用，避免与 Agentic E1/E2 混淆。

## Agentic E1

E1 从真实 result/audit 或 typed trace 派生 retrieved → registered → verified → selected → cited funnel，使用来源 chunk/window coverage，**不评正文语义正确性**。chunk 命中不等于 support established，verified support 也不等于 Requirement 完整回答。

可报告 duplicate、schema/local/Guard rejection、no-support streak、stop reason、Answer 引用域和调用成本。诊断阈值只属于 evaluator，不改变生产策略或 budget。历史 trace 缺少某字段则 unknown/null，不回填支持 scope、不推断 requests。

Decision iteration、Decision model attempts、正式 Tool 执行、逻辑 budget counters 分开报告。最新实验的自定义 Decision calls 表示模型调用 attempts，不等同 E1 标准 decision_calls（distinct iteration）；详见[E1 契约](e1_diagnostic.md)。

固定 smoke 为 fact_001、method_001、method_002、experiment_001、result_001、synthesis_001、bibliography_001、unanswerable_001。当前 qwen38_recommended 仅有 method_002 单次成功，还没有该 profile 的固定 8-case stability suite 或正式 Agentic 40-case；旧 profile 的 gate 失败保留在归档，不能把单次成功、旧 8-case 或历史 replay 宣称为当前版本全量通过。

现有 v1 run_agentic_eval/run_agentic_only_eval 入口仍保留用于兼容。当前诊断采用 run_agentic_e1，不以 v1 useful-action/progress 启发式作为新的语义完整性标准。

## 待实现的 E2

未来需要对 correctness、faithfulness、Requirement completeness、citation support 做人工或经校准的语义评测。引用 ID 合法可以确定性验证；引用是否真正支持句子不能仅凭 ID/来源 chunk 判定。

Judge 上线前需人工校准，冻结 rubric/模型、区分开发与保留集，并记录误判。不把生产 Answer 的 complete/partial 自报状态作为外部语义真值，不因测试绿色而省略验证。当前未调用 E2 Judge，也不输出 overall semantic correctness 分数。

## 可复现记录

每轮保留 dataset SHA、scope、provider/model、实际节点 temperature/mode、budget、代码 commit 与 dirty-content hash、source artifact hashes、UTC 时间、token/latency 和 case-level trace。live 之前与之后的修补使用不同快照；历史模型或输入未保存则标 unknown。

所有 live 有明确运行范围；离线 replay 不创建模型服务、不调用 provider。密钥/hidden reasoning 不进报告；显式 debug 实际输入与现有 public audit 分别管理。
