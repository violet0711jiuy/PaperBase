# 检索与生成：Classic 与 Agentic

核对日期：2026-10-06。两条路径共享正式知识库、Retriever/client 和来源契约，但研究编排与 Answer draft 不同。

## Classic RAG

```text
用户问题与会话上下文
→ Query Resolution / Retrieval Rewrite
→ resolved_query Dense + semantic_query_en Dense + lexical FTS5
→ Weighted RRF → Reranker → Top-K anchors
→ Section-aware Evidence Expansion
→ Grounded Answer
```

经典 Query Planner 负责指代解析、英文 semantic rewrite 与 lexical hints。是否启用 rewrite、候选量和最终 top-k 以配置为准。不要把这条自动多路 rewrite 链描述为 Agentic 每次工具调用的必经流程。

正文只检索 content 索引；经典 bibliography intent 可触发独立 References FTS。References 不进入正文 FAISS。扩展使用 source section 与邻居，原文定位继续保存。

Classic Answer 使用自己的生成 draft 和确定性引用/拒答校验。生产 citation 命名空间为 E#/REF#，R# 不用作参考文献 Evidence ID。

## Agentic Research

完整编排见[当前架构](../architecture/agentic-research.md)。Query Understanding 仅提供冷启动 hints，实际工具输入由经过 Schema/local/Guard 的 Decision Action 确定。

| 工具 | 有效输入与输出 |
|---|---|
| search_papers | 逐篇 canonical semantic + lexical 请求，内部 top_k_per_paper |
| search_knowledge_base | 受信任 corpus scope 内 canonical semantic + lexical 请求 |
| search_references | 独立 lexical-only FTS；显式 lexical 存在时 raw query 仅 trace |
| get_section_context | capabilities 中合法 paper/section 的来源文本，non-ranked |
| get_paper_overview | 正式库已有 Overview，只读，non-ranked |

canonical 编译器是有效请求/重复 identity 的唯一权威，Guard 与 executor 使用同一编译结果。不通过改变 lexical 顺序、空格或 requirement binding 绕过重复保护。Action 的 requirement_ids 表示请求意图；支持只能由 Evidence-local Projector 与 Integrity 建立。

```text
ToolResult → Admission → Registry / Observation
→ scoped multi-projection → Integrity verified support
→ Decision 继续研究或原 Guard 接受 Finish
→ Selector → CompletionView → AnswerSemanticDraft
→ Coverage Compiler → ResearchAnswer
```

Projector scope 为 requested_requirement_ids 与 Plan IDs 交集；多个 Requirement 可生成不同类型组。Selector 不把 RequirementState.SATISFIED 当作完整回答。Answer LLM 对有选中局部支持的项判断 complete/partial/unsupported，程序补无映射项 unsupported、构造引用交集与 overall；全部 unsupported 时拒答，不修改上游局部支持，不新增 semantic Judge。

## Retrieval Score

来源内 raw_score 不能跨 Retriever 直接比较。RetrievedEvidence 明确标注 score_kind、score_direction、retrieval_rank：FTS5 BM25 lower-is-better；Dense/reranker/RRF 和 workspace BM25 higher-is-better；non-ranked raw_score 为 None。没有统一 calibration，旧 score 只用于显式兼容。

Admission 阈值只适用于明确 normalized 的 higher-is-better reranker score。Bibliography 的负数 FTS5 score 不会因此被低分拒收。后续选择依据支持、质量、来源与 rank；检索 rank 也不等于语义支持或答案完整性。

## 配置与检查入口

临时 workspace 的章节解释由 LLM 判断当前正文是否足以形成有用解释，chunk 存在不自动等于充分。证据不足结果必须具有空要点和空来源；有依据的解释仍需真实 context 引用。程序保留段落双换行，并把配置中的 max_key_points 传入实际 Prompt。

新生成 Overview 的空内容字段使用“当前速览证据未覆盖”，与缺失标题“未提供论文标题”分开；这不推断全论文没有说明。生成边界兼容旧“论文未明确说明”占位符，历史保存文件不自动改写。

[配置说明](../configuration.md)、[Retriever](../../paperbase/retrieval/hybrid_retriever.py)、[canonical compiler](../../paperbase/retrieval/requests.py)、[Tool executor](../../paperbase/agentic/tools.py)、[Answer](../../paperbase/generation/answer_generator.py)。

Classic 检索检查：

```powershell
.\.venv\Scripts\python.exe -m paperbase.retrieval "D²STGNN 如何学习动态空间依赖？" --config .\config.yaml
```

实际 LLM/provider 执行与离线 replay 分开；CLI 参数以对应模块 --help 为准。当前 Agentic 稳定性和历史 Retrieval 数值分别见[验证状态](../evaluation/current-baseline.md)。
