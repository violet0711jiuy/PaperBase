# PaperBase Evaluation

本目录保存可公开复用的 Golden Dataset、人工审核记录和 Classic Retrieval Evaluation 工具。Agentic E1 实现在 paperbase/evaluation，详见 [E1 契约](../docs/evaluation/e1_diagnostic.md)。两条路径的结果不能混报。
评测脚本只观察正式 PaperBase 检索链路并计算确定性指标，不会修改生产
RAG / Retrieval / Generation 实现。

Agentic live case 的 Answer 原始最终输出自动保存到 `cases/<case_id>.answer_llm_outputs.jsonl`，包括失败和修复尝试；无 Answer 响应时文件为空。原文仅用于本地排错，不包含 hidden reasoning，也不进入公共 audit 或确定性评分。详见 [E1 契约](../docs/evaluation/e1_diagnostic.md)。

## 目录内容

```text
eval/
├── candidates/          # Candidate Goldens 与人工审核表
├── datasets/            # Golden v1、v1.1、v1.2 与候选 v1.3
├── scripts/             # 生成、冻结、校验、评测与诊断脚本
└── results/             # 每次运行重新生成，已由 Git 忽略
```

当前 Retrieval Benchmark 默认读取：

```text
eval/datasets/golden_dataset_v1_2.jsonl
```

v1.2 含 40 条 context-free Query。`paper_id` 只用于 Paper-level / Evidence-level
Ground Truth 校验与评分，不会作为检索范围传给正式 Retriever。

## Benchmark 论文清单

Golden v1.2 基于以下 4 篇论文。论文 PDF 受原出版来源约束，不在本仓库中重新分发；
要复现历史 Retrieval 基线，必须准备相同版本的 PDF，并通过当前 Parser / Chunker clean rebuild，
使 `paper_id` 与 Ground Truth chunk IDs 一致。

| paper_id | 论文标题 |
|---|---|
| `paper_5b6a1007fa7514bf` | Enhanced wind speed forecasting for sustainable power systems: A deep learning framework unifying deterministic predictions and uncertainty quantification |
| `paper_b12197625a863197` | Forecast of Fine Particles in Chengdu under Autumn-Winter Synoptic Conditions |
| `paper_b7a064b63171eaee` | ESDTW: Extrema-based shape dynamic time warping |
| `paper_c162376bc253ae7d` | Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting |

如果使用不同 PDF 版本或改变解析、章节层级、分块配置，chunk boundary 与 chunk ID
可能变化。此时 Validator 会报告 missing chunk 或 metadata mismatch，不能把旧 Golden
直接用于评分，也不能根据相似文本自动猜测新的 Ground Truth。

## 运行顺序

在项目根目录执行：

```powershell
# 1. 校验 Golden Schema、Evidence chunk 和当前 SQLite metadata
.\.venv\Scripts\python.exe eval\scripts\validate_golden_dataset.py

# 2. 使用正式 HybridRetriever 跑完整 Retrieval Baseline
.\.venv\Scripts\python.exe eval\scripts\run_retrieval_eval.py

# 3. 根据 case_results.jsonl 生成分阶段与失败诊断
.\.venv\Scripts\python.exe eval\scripts\build_retrieval_diagnostic.py

# 4. 根据同一份 trace 审计 Query Planner
.\.venv\Scripts\python.exe eval\scripts\build_query_planner_audit.py
```

默认输出位于：

```text
eval/results/validation/
eval/results/retrieval_full/
```

其中 Full Retrieval 会调用 LLM Query Planner、Embedding、SQLite/FTS5、FAISS 和
Reranker，因此需要完整本地配置、模型、正式知识库与可用 LLM API。Diagnostic 与
Audit 只读取已经保存的 trace，不会再次调用模型。

## 数据版本

| 版本 | 作用 |
|---|---|
| `candidate_goldens.jsonl` | 第一阶段候选集合，不直接作为正式评分集。 |
| `golden_dataset_v1.jsonl` | 人工审核后首次冻结版本。 |
| `golden_dataset_v1_1.jsonl` | 将 context-free Query 改写为尽可能 self-contained。 |
| `golden_dataset_v1_2.jsonl` | 修正过严或过宽 Evidence 后的当前 Retrieval Benchmark。 |
| `golden_dataset_v1_3.jsonl` | 候选版本；仅为 `synthesis_001` 增加人工审核的 all-of Evidence 替代路径，不改变 canonical chunk 集合或严格 recall。 |

历史 Retrieval 数值及当前 Agentic smoke 状态见 [`docs/evaluation/current-baseline.md`](../docs/evaluation/current-baseline.md)，
Schema、类型、Tags 和指标定义见
[`docs/evaluation/design.md`](../docs/evaluation/design.md)。

## 安全边界

- 不提交 `eval/results/`，避免把大体积、可重复生成的 trace 固化到仓库；
- 不在 Evaluation 中修改正式 Retriever 参数或注入 Golden `paper_id`；
- 不自动覆盖冻结数据集；
- chunk 不存在或边界改变时必须人工 remap，不能做文本相似度猜测；
- 完整 Retrieval Ablation 与 E2 semantic Judge 未实现；Ragas/DeepEval 未接入。Agentic E1 已实现并有真实 smoke，不能再表述为所有 Generation/Agentic 验证均未开始；E1 不等于语义质量评测。
