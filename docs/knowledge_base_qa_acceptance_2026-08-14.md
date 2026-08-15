# Knowledge Base QA 验收记录（2026-08-14）

## 范围与方法

本次验收针对当前已入库的三篇论文：

1. Saeed et al. (2025), *Enhanced wind speed forecasting for sustainable power systems*；
2. Shao et al. (2022), *Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting*；
3. Yang et al. (2023), *Forecast of Fine Particles in Chengdu under Autumn-Winter Synoptic Conditions*。

每道题的标准答案先由 SQLite `chunks` 真相表的原文确认，再运行：

```powershell
.\.venv\Scripts\python.exe -m paperbase.answer "<question>" --top-k 5
```

验收同时检查：事实正确性、是否命中正确论文/章节、`citations` 是否能对应 `evidence`，以及回答文本是否真的带有 `[E#]/[R#]` 行内标注。

## 汇总

| 编号 | 类型 | 结果 | 说明 |
| --- | --- | --- | --- |
| QA-01 | 单论文事实 | 有条件通过 | 事实与证据正确；`citations=["E1"]`，但答案正文遗漏 `[E1]`。 |
| QA-02 | 单论文方法 | 有条件通过 | 正确说明自注意力动态调节空间依赖；未完整说明动态特征的构成。 |
| QA-03 | 单论文事实 | 通过 | 时间范围、4 个秋冬期和章节证据均正确。 |
| QA-04 | 跨论文比较 | 不通过 | 仅召回交通论文，缺少风速论文证据，系统安全返回证据不足。 |
| QA-05 | 无答案 | 通过 | 未编造训练时间，正确返回证据不足。 |

结果为 **2/5 完全通过、2/5 有条件通过、1/5 不通过**。不通过项是跨论文召回覆盖问题，而不是 LLM 编造问题。

---

## QA-01：风速论文的数据来源（单论文事实）

**问题**

```text
In the paper Enhanced wind speed forecasting for sustainable power systems, what data sources, locations, and time period were used for the main wind-speed dataset?
```

**标准答案与原文依据**

主数据来自 NREL，包含 Lake Huron 的海上风场（site ID 110197）和 Pennsylvania 的陆上风场（site ID 71764）；使用 2011–2012 年数据，轮毂高度为 100 m。标准证据为风速论文 `4.1. Datasets`，`paper_5b6a1007fa7514bf_chunk_0039`，PDF 第 10 页。

**运行结果**

```text
The main wind-speed dataset used in the paper was sourced from two locations:
an offshore wind farm in Lake Huron (site ID 110197) and an onshore wind farm
in Pennsylvania (site ID 71764), both provided by NREL. The data spanned 2011–2012...
```

- `answer_status`: `success`
- `citations`: `["E1"]`
- `E1`: 风速论文 / `4.1. Datasets` / `chunk_0039`

**判定：有条件通过。** NREL、两地、站点 ID、年份和 100 m 均正确；但 `answer` 正文没有按 Prompt 要求写出 `[E1]`，仅 JSON 的 `citations` 字段有该引用。

## QA-02：D2STGNN 的动态图构建（单论文方法）

**问题**

```text
How does the Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting construct its dynamic graphs?
```

**标准答案与原文依据**

D2STGNN 以历史观测作为动态特征，并结合静态信息、日/周时间信息和源/目的节点嵌入；通过自注意力计算成对掩码，再与静态前/后向转移矩阵逐元素结合，形成随时间变化的动态图。依据：`5.3 Dynamic Graph Learning`，`paper_c162376bc253ae7d_chunk_0033`–`0035`，PDF 第 6–7 页。

**运行结果**

```text
The dynamic graph in D2STGNN is constructed using a dynamic graph learning module
based on a self-attention mechanism, which learns latent correlations between time
series to adjust the road network-based spatial dependency dynamically ... [E1].
```

- `answer_status`: `success`
- `citations`: `["E1"]`
- `E1`: D2STGNN / `1 INTRODUCTION` / `chunk_0007`–`0009`

**判定：有条件通过。** 核心机制（自注意力、动态空间依赖、基于路网调整）正确且有行内证据；但检索停留在 Introduction，未召回 `5.3` 的具体特征构成和矩阵计算，答案细节不完整。

## QA-03：成都细颗粒物论文的秋冬时间范围（单论文事实）

**问题**

```text
In Forecast of Fine Particles in Chengdu under Autumn-Winter Synoptic Conditions,
what months define the autumn-winter period and how many autumn-winter periods were analyzed?
```

**标准答案与原文依据**

秋冬定义为每年 11 月至次年 2 月；分析了 2018–2019、2019–2020、2020–2021、2021–2022 共 4 个秋冬期。依据：`2.3. Forecast Evaluation Method and Dataset`，`paper_b12197625a863197_chunk_0011`，PDF 第 4 页。

**运行结果**

```text
The autumn-winter period is defined as November through February. Four
autumn-winter periods were analyzed: 2018-2019, 2019-2020, 2020-2021,
and 2021-2022 [E1].
```

- `answer_status`: `success`
- `citations`: `["E1"]`
- `E1`: 成都细颗粒物论文 / `2.3. Forecast Evaluation Method and Dataset` / `chunk_0011`–`0012`

**判定：通过。** 事实、论文归属、章节和行内引用均正确。

## QA-04：风速论文与 D2STGNN 的预测对象对比（跨论文）

**问题**

```text
Compare the forecast targets of Enhanced wind speed forecasting for sustainable
power systems and Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting.
```

**标准答案与原文依据**

风速论文预测风速，并同时生成确定性点预测与预测区间；D2STGNN 预测未来交通信号/交通状况（论文实验包含交通速度和交通流量数据）。依据分别为：

- 风速论文 Abstract，`paper_5b6a1007fa7514bf_chunk_0003`，PDF 第 1 页；
- D2STGNN Abstract/Preliminaries，`paper_c162376bc253ae7d_chunk_0001`、`0014`，PDF 第 1、3 页。

**运行结果**

```text
The provided evidence only discusses D2STGNN, which targets traffic flow prediction.
There is no evidence ... about Enhanced wind speed forecasting ..., so a direct
comparison ... cannot be made.
```

- `answer_status`: `success`
- `insufficient_evidence`: `true`
- 实际命中均为 D2STGNN；没有风速论文的 `E#`。

**判定：不通过。** 系统没有猜测，安全行为正确；但跨论文问题应同时召回两个明确论文对象，目前检索覆盖不足。

## QA-05：三篇论文 GPU 总训练时间（无答案）

**问题**

```text
What is the exact total GPU training time in hours across all three papers in this knowledge base?
```

**标准答案**

当前三篇论文的已解析内容没有提供可相加的“总 GPU 训练小时数”，因此正确行为是不回答具体数字，并标记证据不足。

**运行结果**

```text
The exact total GPU training time in hours across all three papers ... is not
provided in the evidence ... does not include specific GPU training time measurements.
```

- `answer_status`: `success`
- `insufficient_evidence`: `true`
- `citations`: `[]`

**判定：通过。** 没有把效率比较或模型参数臆造成训练时长。

## 结论与下一步

当前单论文事实问答和无答案拒答已经可用；方法题能答核心机制但需要进一步评估细节召回。最明显的下一项改进是跨论文检索：当 Query 明确出现多个论文标题时，应为每个论文名建立一个目标锚点或候选配额，再融合其正文证据，避免一个强相关论文挤掉另一个对比对象。

另外，QA-01 表明 JSON Schema 中的 `citations` 字段校验正常，但 Prompt 对 `answer` 行内 `[E#]` 的遵循不是完全稳定；之后可在 Python 校验中要求非 `insufficient_evidence` 的回答至少出现一个与 `citations` 一致的行内引用。
