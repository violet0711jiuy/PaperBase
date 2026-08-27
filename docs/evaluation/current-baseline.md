# Current Retrieval Baseline

> Dataset：`eval/datasets/golden_dataset_v1_2.jsonl`
> 场景：Context-free Retrieval Benchmark
> 内容 Case：32；Bibliography：4；Unanswerable：4

该页面只记录当前冻结基线。完整逐 Case trace 和可再生成报告位于本地 `eval/results/`，不提交到 Git。

## Overall Retrieval

| Metric | Result |
|---|---:|
| Hit@5 | 0.937500 |
| Recall@5 | 0.859375 |
| MRR@5 | 0.715625 |
| Bibliography Intent F1 | 1.000000 |
| Bibliography Hit@5 | 1.000000 |

## Evidence Expansion

生产 Section-aware Expansion 只处理 32 条 answerable content cases 的 Final Top-5 anchors。

| Metric | Before | Expanded | Delta |
|---|---:|---:|---:|
| Hit | 0.937500 | 0.968750 | +0.031250 |
| Recall | 0.859375 | 0.932292 | +0.072917 |

| Diagnostic | Result |
|---|---:|
| Recovery cases | 1 |
| Avg expansion latency | 3.792 ms |
| Avg expanded chunks | 8.4375 |
| Max expanded chunks | 13 |
| Expansion errors | 0 |

唯一 recovery case 为 `method_010`：Final anchor `chunk_0016` 通过生产 Expansion 补回 GT `chunk_0017`。`synthesis_001` 在 Expansion 后仍未命中其五个跨位置 GT chunks。

## Slice Recall Before → After

| Slice | Raw Recall@5 | Expanded Recall | Delta |
|---|---:|---:|---:|
| method | 0.687500 | 0.916667 | +0.229167 |
| synthesis | 0.500000 | 0.625000 | +0.125000 |
| single_hop | 0.950000 | 1.000000 | +0.050000 |
| multi_hop | 0.708333 | 0.819444 | +0.111111 |

## 复现

```powershell
.\.venv\Scripts\python.exe eval\scripts\validate_golden_dataset.py
.\.venv\Scripts\python.exe eval\scripts\run_retrieval_eval.py
.\.venv\Scripts\python.exe eval\scripts\build_retrieval_diagnostic.py
```

当前 Baseline 不包含 Retrieval Ablation、Generation Evaluation、Ragas、DeepEval 或 LLM Judge。
