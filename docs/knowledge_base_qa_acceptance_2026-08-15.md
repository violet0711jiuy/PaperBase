# Knowledge Base QA 验收记录（2026-08-15）

## 范围与方法

本轮针对已重新解析、重新入库并重建 FAISS 的三篇论文进行端到端验收。所有问题均用中文提问，论文标题、模型名等专有名词保留英文。每题均通过以下链路实际运行：

```text
中文问题 → Query Rewrite → Dense / BM25 / bibliography FTS → RRF → BGE Reranker
→ 同节邻居扩展 → LLM 回答（JSON Schema + Pydantic 校验）
```

验收时先根据 SQLite `chunks` 真相表确认标准答案，再检查实际的回答事实、命中文章、`[E#]/[R#]` 行内引用、`citations` 字段和降级行为。

## 汇总

| 编号 | 类型 | 结果 | 核心结论 |
| --- | --- | --- | --- |
| QA-01 | 单论文数据细节 | 通过 | 正确回答 NREL、两处地点、站点 ID、年份与 100 m。 |
| QA-02 | 单论文方法细节 | 通过 | 正确说明 DSTF 的扩散/固有信号分解及目的。 |
| QA-03 | 深层方法细节 | 通过 | 精确命中 `5.3 Dynamic Graph Learning`，并解释动态矩阵构建。 |
| QA-04 | 单论文整体概括 | 通过 | 研究目标、COST733 + CMAQ、污染结论和预测时效均正确。 |
| QA-05 | 跨论文比较 | 不通过 | 仍只召回 D2STGNN，缺少风速论文证据；安全拒答正确，但覆盖不足。 |
| QA-06 | 参考文献事实 | 通过 | `search_bibliography=true`，正确返回 Graph WaveNet 作者与年份。 |
| QA-07 | 正文 + References 联合问题 | 有条件通过 | 正确利用正文解释基线/局限，但“为什么引用”属于基于相关工作和实验定位的解释，不是作者直接陈述。 |
| QA-08 | 无答案问题 | 有条件通过 | 未编造 GPU 小时数，但本次回答生成降级为 `fallback`，没有输出明确的“证据不足”文本。 |

结果为 **5/8 通过、2/8 有条件通过、1/8 不通过**。其中 QA-08 的“有条件通过”是安全性通过、用户体验未完全通过；主要后续问题仍是跨论文召回覆盖，而非答案编造。

---

## QA-01：风速数据集的地点与时间（单论文数据细节）

**问题**

```text
在论文 Enhanced wind speed forecasting for sustainable power systems 中，主风速数据来自哪些地点、站点 ID、年份和测量高度？
```

**标准答案**

数据来自 NREL：Lake Huron 离岸风场（site ID 110197）和 Pennsylvania 陆上风场（site ID 71764）；使用 2011–2012 年数据，轮毂高度为 100 m。依据 `4.1. Datasets`，`paper_5b6a1007fa7514bf_chunk_0039`，PDF 第 10 页。

**实际结果**

回答完整给出两处地点、两个站点 ID、2011–2012 年和 100 m，并使用 `[E1]`；`E1` 精确指向 `4.1. Datasets` / `chunk_0039`。

**判定：通过。**

## QA-02：DSTF 分解的隐藏信号（单论文方法细节）

**问题**

```text
在论文 Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting 中，DSTF 将交通时间序列分解成哪两类隐藏信号？这样做的目的是什么？
```

**标准答案**

DSTF 以数据驱动方式将时间序列分成扩散信号（diffusion signals）和固有信号（inherent signals）。目的在于针对两类不同空间时间特性分别建模，以提升交通预测准确性。核心依据为 `4 THEDECOUPLEDFRAMEWORK`、`4.1 Residual Decomposition Mechanism` 和 `4.2 Estimation Gate`。

**实际结果**

回答正确给出两类信号，说明扩散模块使用时空局部卷积、固有模块使用 RNN 与自注意力，并以 `[E1]`–`[E6]` 标注对应证据。首要证据命中 `chunk_0015`（PDF 第 3 页），且同节扩展覆盖 `4.1` 与 `4.2`。

**判定：通过。**

## QA-03：动态图库构建（深层方法细节）

**问题**

```text
在论文 Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting 中，dynamic graph learning 如何使用动态特征和静态前向/后向转移矩阵构建动态图？
```

**标准答案**

模型用历史观测、日/周时间嵌入和源/目的节点嵌入构造前向、后向动态特征；通过自注意力得到成对掩码，再分别与静态前向/后向转移矩阵逐元素结合，形成随时间变化的动态图。依据 `5.3 Dynamic Graph Learning`，`chunk_0033`–`0035`，PDF 第 6–7 页。

**实际结果**

回答给出了动态特征来源、前/后向特征矩阵、自注意力掩码和与静态矩阵逐元素相乘的关系，并引用 `[E1]`。`E1` 正确由 `chunk_0033`–`0035` 组成，命中目标章节。

**判定：通过。** 这是对“细粒度问题 + 同节邻居扩展”的有效验证。

## QA-04：成都细颗粒物论文的整体总结（整体问题）

**问题**

```text
请概括论文 Forecast of Fine Particles in Chengdu under Autumn-Winter Synoptic Conditions 的研究目标、使用的主要方法，以及关于细颗粒物污染的总体结论。
```

**标准答案**

论文评估气象因子预报对成都秋冬细颗粒物预测的影响；使用 COST733 客观天气分类与 CMAQ 空气质量模型。高压后部、均压和低压条件下更容易发生污染，高压底部条件下较少发生；24 h 预测通常优于 48 h 和 72 h，2 m 相对湿度与 10 m 风速是关键因子。

**实际结果**

回答完整覆盖研究目标、COST733、CMAQ、四类天气条件中的污染差异、预测时效和关键气象因子，使用 `[E1][E3]`。`E1` 指向 `4. Conclusions`，`E3` 指向 Abstract。

**判定：通过。**

## QA-05：两篇论文预测对象与输出形式（跨论文比较）

**问题**

```text
比较论文 Enhanced wind speed forecasting for sustainable power systems 与 Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting 的预测对象和输出形式。
```

**标准答案**

风速论文预测风速，并同时输出确定性点预测与预测区间；D2STGNN 面向路网中的未来交通状态/交通序列预测。答案必须同时有两篇论文的正文证据。

**实际结果**

检索结果 10 条均来自 D2STGNN；回答明确表示缺少风速论文内容，因而不能比较，`insufficient_evidence=true`，没有编造答案。

**判定：不通过（但安全降级正确）。** 该问题再次证明：Query 中同时出现多个明确论文标题时，现有统一 Top-K 仍可能被其中一篇论文占满。后续应加入“每个显式论文实体至少保留若干正文候选”的跨论文召回配额或多锚点检索。

## QA-06：Graph WaveNet 的参考文献（参考文献事实）

**问题**

```text
论文 Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting 是否引用了 Graph WaveNet？如果引用，请给出该参考文献的作者和年份。
```

**标准答案**

是。该条目为 Zonghan Wu、Shirui Pan、Guodong Long、Jing Jiang、Chengqi Zhang，2019 年。

**实际结果**

`rewrite_plan.search_bibliography=true`，英文关键词包含 `Graph WaveNet`；回答正确给出作者和 2019 年，并引用 `[R1]`。`R1` 是 `REFERENCES` / `chunk_0090` / PDF 第 13 页。

**判定：通过。**

## QA-07：为什么引用 Graph WaveNet（正文 + References 联合）

**问题**

```text
论文 Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting 的作者为什么引用 Graph WaveNet？请根据正文说明，不要只给出书目信息。
```

**标准答案**

应以正文说明 Graph WaveNet 是交通预测比较中的时空基线，结合 GNN 与 Gated TCN 建模时空依赖；D2STGNN 再以解耦框架处理其未充分建模的非扩散/固有信号问题。参考文献仅用于确认书目信息，不能单独支撑“为什么”。

**实际结果**

`search_bibliography=true`，同时得到正文 E# 和参考文献 R# 候选。最终回答引用 `[E2][E4][E5]`，将 Graph WaveNet 说明为比较基线，并解释 D2STGNN 的解耦动机；`E5` 来自 `2 RELATEDWORK`，`E2` 来自 `6.2 The Performance of D 2 STGNN`。

**判定：有条件通过。** 回答与正文比较和相关工作一致，但“作者为什么引用”是从论文的比较位置与局限性归纳出的解释，不是一个作者明确写出的单句因果声明；产品回答应保留这种表述边界。

## QA-08：三篇论文的 GPU 训练总时长（无答案）

**问题**

```text
当前知识库中的三篇论文合计使用了多少 GPU 训练小时？请给出精确数值。
```

**标准答案**

三篇论文的现有证据并未给出可相加的精确 GPU 训练小时数。正确行为是明确说明证据不足，不能编造数值。

**实际结果**

检索命中了 D2STGNN 的实验与效率章节，但回答生成返回 `answer_status=fallback`、`answer=null`，没有输出具体 GPU 小时数，也没有虚构引用。

**判定：有条件通过。** 安全性正确（未编造），但用户体验不完整：理想行为应返回“当前论文证据不足，无法给出精确总时长”，并标记 `insufficient_evidence=true`。这属于 LLM 结构化输出降级路径的稳定性问题，后续可单独设计确定性的无答案文案。

## 本轮结论

重新解析和分块后，细节题的目标章节命中明显改善：D2STGNN 的动态图题不再停留在 Introduction，而是命中 `5.3` 并携带连续邻居证据。单论文事实、细节、整体概括和 bibliography-aware 路径已经具备可演示性。

下一阶段最有价值的改进仍是跨论文问题的“论文实体覆盖”策略；其次是将 Generation 的 `fallback` 显式映射为用户可读的证据不足回答，而不是只返回空 `answer`。
