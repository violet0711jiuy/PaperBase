# PaperBase Evaluation Design

> 版本：v1.0
> 当前范围：单论文 RAG Evaluation
> 目标：建立一套可复现、可自动运行、能够定位问题来源的 PaperBase 评测流程。

## 当前实施状态

| 阶段 | 状态 |
|---|---|
| Candidate Generation / Human Review / Golden Freeze | `COMPLETED` |
| Golden Dataset Validation | `COMPLETED` |
| Query Planner Audit | `COMPLETED` |
| Deterministic Retrieval Evaluation | `COMPLETED` |
| Expansion-aware Evaluation | `COMPLETED` |
| Retrieval Ablation | `PLANNED` |
| Generation / Citation Evaluation | `PLANNED` |
| LLM Judge、DeepEval、Ragas | `PLANNED` |

当前冻结数据集和可复现指标见 [Current Retrieval Baseline](current-baseline.md)。本文中标为后续 Phase 的内容不代表已经实现。

---

## 1. 评测目标与范围

PaperBase 当前完整问答链路为：

```text
User Query
→ Query Rewrite / Bibliography Routing
→ Dense + BM25 Retrieval
→ Weighted RRF
→ Reranker
→ Context Expansion
→ LLM Generation
→ Answer + Citation
```

Evaluation 不只看最终答案是否“看起来正确”，而是分别验证：

- Retrieval 是否召回正确证据；
- Reranker 是否把正确证据排到前面；
- Bibliography Routing 是否正确；
- LLM 是否基于 Evidence 正确回答；
- 答案是否忠于当前 Evidence；
- Citation 是否真实且能够支持结论；
- 论文没有答案时是否能够拒答；
- Query Rewrite、BM25、Reranker 等模块是否真正带来收益；
- 不同方案在质量、Latency 和 Cost 之间如何取舍。

当前 v1 **只评单论文能力**，暂不包含：

- 跨论文比较；
- 多论文综合；
- Agent / Tool Calling；
- Web Search；
- 多模态图片理解。

---

## 2. Golden Dataset 设计

Golden Dataset 是一组经过人工确认的测试样本。
LLM 可以先根据论文生成 Candidate Goldens，但必须人工审核后才能进入正式 Golden Dataset。

构建流程：

```text
论文 / Chunks
    ↓
LLM 按固定类型生成 Candidate Goldens
    ↓
人工检查问题、答案、Evidence、标签
    ↓
修改 / 删除不合格 Case
    ↓
冻结 Golden Dataset
```

### 2.1 问题类型与数量

Evaluation v1 计划约 40 条：

| 类型 | 数量 | 主要测试内容 |
|---|---:|---|
| `fact` | 8 | 明确事实、基础召回 |
| `method` | 8 | 方法、机制、语义理解 |
| `experiment` | 6 | Baseline、参数、数据划分、实验设置 |
| `result` | 6 | 指标、表格、消融和实验结果 |
| `synthesis` | 4 | 同一论文内多处 Evidence 综合 |
| `bibliography` | 4 | References 路由与检索 |
| `unanswerable` | 4 | 拒答与证据不足判断 |
| **总计** | **40** | |

其中：

- `fact`：通常 1～2 个局部 Chunk 就能回答；
- `method`：重点测试中文 Query → 英文论文、Semantic Retrieval 和 Query Rewrite；
- `experiment`：适合测试 BM25 对模型名、数据集名、参数和数字的帮助；
- `result`：测试实验结果、表格、指标和消融结果召回；
- `synthesis`：需要同一篇论文多个位置共同回答，不涉及跨论文；
- `bibliography`：只有明确询问引用 / References 时才属于该类；
- `unanswerable`：问题合理，但论文本身确实没有答案。

### 2.2 Tags

除 `primary_type` 外，每个 Case 可增加 Tags，用于后续 Slice Evaluation：

```text
zh_query_en_doc
english_query

exact_term
semantic_paraphrase

single_hop
multi_hop

bibliography_intent

easy
medium
hard
```

例如：

```text
“模型怎么动态判断不同节点之间的关系？”
```

可以标记为：

```json
[
  "zh_query_en_doc",
  "semantic_paraphrase",
  "single_hop",
  "medium"
]
```

---

### 2.3 Golden Case 结构

#### Context-free Query 的自包含约束

`golden_dataset_v1_2.jsonl` 用于 **Context-free Retrieval Benchmark**。每条 `question`
必须在没有会话历史、当前论文页面或隐藏 `paper_id` 提示时，仍能明确识别目标论文、模型、
方法、数据集或研究任务。

- `paper_id` 只用于 Ground Truth 标注和评分，不能作为 Retriever 的额外查询上下文；
- 不能只用“该模型”“该方法”“本文”“这篇论文”“the study”等无先行词指代；
- 如果先行词已在同一条 Query 内明确出现，可以继续使用后续指代；
- Bibliography Query 必须说明要查询哪篇源论文的 References；
- Active-paper Scoped QA 属于另一种评测场景，应单独运行和报告。

正式数据建议保存为：

```text
eval/datasets/golden_dataset_v1.jsonl
```

每行一个 JSON：

```json
{
  "id": "method_001",
  "question": "D²STGNN如何学习节点之间的动态空间依赖？",
  "primary_type": "method",
  "tags": [
    "zh_query_en_doc",
    "semantic_paraphrase",
    "single_hop",
    "medium"
  ],
  "paper_id": "paper_001",
  "answerable": true,
  "reference_answer": "模型根据输入状态动态学习节点关系，并将得到的动态空间关系用于后续信息传播。",
  "required_facts": [
    "节点关系由数据驱动学习",
    "空间关系会随输入状态变化",
    "动态关系用于空间信息传播"
  ],
  "relevant_evidence": [
    {
      "paper_id": "paper_001",
      "section": "Dynamic Graph Learning",
      "page_start": 5,
      "page_end": 6,
      "chunk_ids": [
        "chunk_0031",
        "chunk_0032"
      ]
    }
  ],
  "expected_bibliography_intent": false
}
```

字段说明：

- `id`：Case 唯一标识；
- `question`：真正输入 PaperBase 的用户问题；
- `primary_type`：问题主类型；
- `tags`：语言、表达方式、推理深度和难度；
- `paper_id`：目标论文；
- `answerable`：论文中是否有足够 Evidence 回答；
- `reference_answer`：人工确认的参考答案，不要求模型逐字一致；可以包含这些必需事实之外的有用细节，但不能要求系统必须复述所有这些细节
- `required_facts`：完整答案必须覆盖的核心事实，用于 Completeness；一个答案要被判定为“实质上完整回答了这个问题”时，最低限度必须包含的语义事实
- `relevant_evidence`：Retrieval 的 Ground Truth，用于 Recall@K、MRR 等指标；
- `expected_bibliography_intent`：该问题是否应该开启参考文献辅助检索。

`relevant_evidence` 同时保存 `chunk_ids` 和 `section/page`。
Chunk ID 便于代码自动计算指标；Section/Page 用于人工核验，并在未来重新 Chunk 后提供稳定定位。

对于不可回答问题：

```json
{
  "id": "unanswerable_001",
  "question": "作者下一步计划在哪个新的数据集上验证模型？",
  "primary_type": "unanswerable",
  "tags": [
    "zh_query_en_doc",
    "hard"
  ],
  "paper_id": "paper_002",
  "answerable": false,
  "reference_answer": null,
  "required_facts": [],
  "relevant_evidence": [],
  "expected_bibliography_intent": false
}
```

### 2.4 人工审核重点

LLM 生成 Candidate 后，人工主要检查四件事：

1. **Question**：是否自然、是否像真实用户问题、是否重复；
2. **Reference Answer**：是否真的正确，是否超出论文原文；
3. **Evidence**：是否真正支持答案，Chunk/Page/Section 是否正确；
4. **Type / Tags**：类型、难度、Bibliography Intent 是否合理。

特别注意：

```text
论文有答案，但系统没找到
→ Retrieval Failure

论文本身没有答案
→ Unanswerable
```

不能因为当前系统检索失败，就把问题标成不可回答。

---

## 3. Retrieval 与 Reranker Evaluation

Retrieval 的目标不是回答问题，而是：

> 把正确 Evidence 找回来，并尽量排在前面。

### 3.1 核心指标

#### Hit@K

Top-K 中是否至少存在一个正确 Evidence。

适合判断：

> “至少有没有找到一个关键证据？”

建议记录：

```text
Hit@5
Hit@10
```

#### Recall@K

Ground Truth Evidence 中，有多少被 Top-K 找回来：

```text
Recall@K
=
Top-K 中召回的 Relevant Evidence 数量
/
Ground Truth Relevant Evidence 总数
```

建议：

```text
Recall@5
Recall@10
Recall@20
Recall@40
```

如果一个问题需要 3 个 Evidence，而 Top-5 只找到其中 2 个：

```text
Recall@5 = 2 / 3
```

所以 Recall@K 对 `multi_hop` / `synthesis` 问题尤其重要。

#### MRR

MRR（Mean Reciprocal Rank）关注第一条正确 Evidence 排得多靠前：

```text
第一条正确 Evidence rank = 1 → 1.0
rank = 2 → 0.5
rank = 5 → 0.2
```

所有 Case 取平均得到 MRR。

#### Precision@K

Top-K 中有多少比例是真正相关 Evidence：

```text
Precision@K
=
Relevant 数量 / K
```

它用于观察 Retrieval 是否过于嘈杂。

PaperBase 初始 Candidate 阶段以 Recall 为主，因此 Precision@40 不需要特别高，后续由 Reranker 提纯。

### 3.2 NDCG@K

NDCG 只有在 Evidence 存在不同相关等级时更有价值，例如：

```text
3 = 直接回答
2 = 强支持
1 = 弱相关
0 = 无关
```

Evaluation v1 暂不强制使用。
如果当前 Ground Truth 只标 Relevant / Irrelevant，`Recall@K + MRR` 已足够。

### 3.3 Reranker 怎么评

比较：

```text
Rerank 前
vs
Rerank 后
```

重点看：

```text
Hit@5
Recall@5
MRR
First Relevant Rank
```

理想状态：

```text
Retrieval Top-40
→ 保证高 Recall

Reranker
→ 把 Relevant Evidence 提升到 Top-5
```

因此 Reranker 不要求提高 Top-40 Recall，而是要让正确 Evidence 在最终小 Context 中留下并排得更靠前。

---

## 4. Bibliography、Generation、Citation 与拒答

### 4.1 Bibliography Routing

Golden Dataset 中：

```text
expected_bibliography_intent
```

系统输出：

```text
search_bibliography
```

直接作为二分类问题，用 Python 自动计算：

```text
Accuracy
Precision
Recall
F1
```

正例：

```text
“这篇论文有没有引用 Graph WaveNet？”
→ true
```

负例：

```text
“Graph WaveNet 和本文模型有什么区别？”
→ false
```

还应记录：

```text
Bibliography Hit@5
```

用于判断真正的引用问题能否找到正确 Reference Entry。

对于普通问题可以额外记录：

```text
Bibliography Noise Rate
=
普通 Query Top-K 中 bibliography chunks 数
/
普通 Query Top-K 总 chunks 数
```

理想值接近 0。

---

### 4.2 Generation Metrics

最终生成阶段重点评：

#### Answer Correctness

答案相对于论文事实是否正确。

建议使用：

```text
LLM-as-a-Judge + 人工校准
```

推荐 0～4 分：

```text
4 = 完全正确且关键事实完整
3 = 核心正确，仅有轻微遗漏
2 = 部分正确，有明显遗漏或错误
1 = 大部分错误
0 = 完全错误或与论文矛盾
```

#### Faithfulness

答案中的事实是否都受到**本轮实际提供给 LLM 的 Evidence**支持。

注意：

即使某个事实在论文其他位置是真的，但没有进入当前 Context，模型自行补充出来，仍然属于 Faithfulness 问题。

#### Answer Relevance

答案是否真正回答用户问题。

正确但答非所问，也应该降低 Relevance。

#### Completeness

根据：

```text
required_facts
```

判断完整答案应该包含的关键信息是否都覆盖。

例如：

```text
required_facts = [A, B, C]
```

模型只说 A，则可能没有明显事实错误，但 Completeness 较低。

---

### 4.3 LLM-as-a-Judge

LLM Judge 适合判断：

```text
Correctness
Faithfulness
Answer Relevance
Completeness
Citation Support
```

但这些可以稳定用代码判断的指标不要交给 LLM：

```text
Recall@K
MRR
Bibliography Intent
Citation ID 是否存在
Answerability Label
Latency
```

正式批量使用 Judge 前，先人工评分约 10～20 条，并让 Judge 对相同 Case 评分。

如果人工与 Judge 判断大致一致，再批量使用。

这个过程称为：

```text
Judge Calibration
```

---

### 4.4 Citation Evaluation

分两层：

#### Citation Validity

检查模型输出的：

```text
S1
S3
```

是否真的存在于本轮 Evidence 中。

完全由 Python 自动计算，目标应尽量达到 100%。

#### Citation Support

检查引用 Evidence 是否真的支持对应 Claim。

例如：

```text
“模型使用 Dataset A [S2]”
```

如果 S2 实际只讲模型结构：

```text
Citation Validity = Pass
Citation Support = Fail
```

Citation Support 使用 LLM Judge 或人工判断。

---

### 4.5 Answerability / Refusal

Golden：

```text
answerable
```

系统：

```text
predicted_answerable
```

作为二分类任务，自动计算：

```text
Accuracy
Precision
Recall
F1
```

特别关注两类错误：

```text
False Refusal
= 论文有答案，但系统拒答

Failed Refusal
= 论文无答案，但系统仍然回答
```

后者风险更高，因为容易形成 Hallucination。

---

## 5. Ablation Experiments

Ablation 的目的是：

> 去掉某个模块重新跑同一套 Golden Dataset，看这个模块到底有没有带来可测量收益。

### E1 — Dense Only

```text
Original Query
→ Embedding
→ FAISS
```

作为基础 Baseline。

### E2 — Dense + BM25

```text
Original Dense
+
Original BM25
→ RRF
```

验证 Sparse Retrieval 的贡献。

### E3 — Hybrid + Query Rewrite

```text
Original Dense
Original BM25
Rewritten Dense
Rewritten BM25
→ Weighted RRF
```

验证 Query Rewrite 是否提升跨语言 / 语义模糊问题的 Recall。

### E4 — Full Retrieval + Reranker

```text
Hybrid Retrieval
→ Candidate Top-K
→ Reranker
→ Final Top-K
```

验证 Reranker 是否让 Relevant Evidence 更集中地进入最终 Context。

推荐最终比较：

| Configuration | Hit@5 | Recall@5 | Recall@10 | Recall@20 | MRR | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense Only | | | | | | |
| Dense + BM25 | | | | | | |
| + Query Rewrite | | | | | | |
| + Reranker | | | | | | |

注意：

最终不是单纯选择 Recall 最高的方案，而是综合考虑：

```text
Quality
Latency
Cost
Complexity
```

例如：

```text
Query Rewrite OFF:
Recall@10 = 84%
Latency = 2.2s

Query Rewrite ON:
Recall@10 = 92%
Latency = 3.0s
```

表示：

```text
+8 个百分点 Recall@10
代价：
+0.8s Latency
+1 次 LLM 调用
```

这就是 Quality–Latency / Cost Trade-off。

---

## 6. 实验记录与结果输出

### 6.1 自动记录的系统结果

每个 Case 建议记录：

```text
case_id
question

retrieved_chunk_ids
reranked_chunk_ids
final_context_ids

predicted_bibliography_intent
predicted_answerable

actual_answer
citation_ids

retrieval_latency
rerank_latency
generation_latency
total_latency

metric_scores
failure_type
```

Case-level 结果建议保存：

```text
eval/results/.../case_results.jsonl
```

汇总结果：

```text
summary.json
summary.csv
```

最终人工可读报告：

```text
docs/evaluation_report.md
```

---

### 6.2 Latency

至少记录：

```text
Retrieval Latency
Rerank Latency
Generation Latency
Total Latency
```

如果方便，可以进一步拆：

```text
Query Rewrite
Embedding
Dense Retrieval
BM25
RRF
Context Expansion
```

建议最终报告：

```text
Average Total Latency
```

数据量足够时再加：

```text
Median
P95
```

如果能够获得 API Token Usage，可选记录：

```text
rewrite tokens
generation tokens
estimated cost
```

---

### 6.3 实验可复现性

每次正式实验至少记录：

```text
experiment_id
timestamp
dataset_version
git_commit
embedding_model
reranker_model
generation_model
query_rewrite_enabled
retrieval_top_k
rerank_top_k
rrf_weights
```

避免以后看到一个指标，却不知道当时对应哪套代码和配置。

---

### 6.4 Slice Evaluation

除了 Overall Metric，还应该按类型分析：

```text
fact
method
experiment
result
synthesis
bibliography
unanswerable
```

以及：

```text
exact_term
semantic_paraphrase

single_hop
multi_hop

easy
medium
hard
```

例如：

```text
Overall Recall@10 = 91%
Fact Recall@10 = 98%
Method Recall@10 = 90%
Synthesis Recall@10 = 68%
```

这样才能发现系统真正薄弱的场景。

---

### 6.5 Failure Analysis

失败 Case 建议标记：

```text
retrieval_miss
ranking_failure
reranker_drop
context_incomplete
generation_error
hallucination
citation_error
routing_error
false_refusal
failed_refusal
```

核心目的不是让标签特别复杂，而是能够回答：

> 这次失败究竟发生在哪一层？

---

## 7. Evaluation 实施流程

推荐按以下顺序完成，不一次性把所有东西堆在一起。

### Phase 1 — Candidate Golden Generation

基于当前真实论文和 Chunk：

```text
LLM
→ candidate_goldens.jsonl
```

按本文定义的类型和数量生成。

---

### Phase 2 — Human Review

人工检查：

```text
Question
Reference Answer
Required Facts
Evidence
Type / Tags
Answerability
Bibliography Intent
```

审核完成后冻结：

```text
eval/datasets/golden_dataset_v1.jsonl
```

如果数据量允许，进一步拆：

```text
Dev Set
+
Test Set
```

建议约：

```text
70% Dev
30% Test
```

Dev 用于调参数，Test 只用于最终结果。

---

### Phase 3 — Deterministic Evaluation

先实现 Python 自动指标：

```text
Hit@K
Recall@K
Precision@K
MRR

Bibliography Intent
Bibliography Hit@5
Bibliography Noise Rate

Answerability
Citation Validity

Latency
```

---

### Phase 4 — Retrieval Ablation

固定 Golden Dataset，依次跑：

```text
Dense Only
Dense + BM25
+ Query Rewrite
+ Reranker
```

得到 Retrieval Ablation 表。

---

### Phase 5 — Generation Evaluation

完整跑：

```text
Question
→ Retrieval
→ Reranker
→ Context
→ LLM
```

保存：

```text
Actual Answer
Context
Citation
Answerability
```

---

### Phase 6 — LLM Judge

增加：

```text
Correctness
Faithfulness
Answer Relevance
Completeness
Citation Support
```

先人工校准 Judge，再批量运行。

PaperBase 推荐：

```text
Custom Python
+
DeepEval（可选）
```

其中 Python 负责确定性指标；DeepEval 或类似框架负责 LLM-based semantic metrics。

Golden Dataset 本身始终由 PaperBase 自己维护为 JSONL，不绑定某个第三方框架。

---

### Phase 7 — Final Report

生成：

```text
docs/evaluation_report.md
```

建议包含：

1. Dataset Composition
2. Evaluation Setup
3. Retrieval Ablation
4. Bibliography Evaluation
5. Generation Evaluation
6. Latency / Cost
7. Slice Evaluation
8. Failure Analysis
9. Final Configuration
10. Limitations

---

## 8. 推荐目录

```text
docs/
└── evaluation/
    ├── design.md
    └── current-baseline.md

eval/
├── candidates/
│   └── candidate_goldens.jsonl
│
├── datasets/
│   ├── golden_dataset_v1.jsonl
│   ├── golden_dataset_v1_1.jsonl
│   └── golden_dataset_v1_2.jsonl
│
├── scripts/
│   ├── generate_candidates.py
│   ├── validate_golden_dataset.py
│   ├── run_retrieval_eval.py
│   ├── build_retrieval_diagnostic.py
│   └── build_query_planner_audit.py
│
└── results/                 # 本地生成，由 .gitignore 排除
```

目录可以根据当前 PaperBase 实际代码结构小幅调整。

---

## 9. Evaluation v1 完成标准

当前不提前硬规定：

```text
Recall@10 必须达到 95%
```

之类的目标。

Evaluation v1 真正完成的标准是：

- 有经过人工审核的 Golden Dataset；
- Retrieval Metrics 可以自动重复计算；
- Dense / BM25 / Query Rewrite / Reranker 有 Ablation 结果；
- Bibliography Routing 有独立测试；
- Unanswerable 能够量化；
- Generation Correctness 和 Faithfulness 被评估；
- Citation Validity 能自动检查；
- Latency 被记录；
- Failure 可以定位到具体 Pipeline Stage；
- Final Configuration 是根据实验结果选择，而不是凭主观感觉确定。

核心原则：

> 不问“系统看起来好不好”，而问：
>
> **哪个模块提升了哪项能力、提升了多少、付出了什么代价，以及剩余 Failure Mode 是什么。**
