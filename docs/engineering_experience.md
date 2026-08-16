# PaperBase 工程经验与实验记录

> 用途：记录 PaperBase 在真实业务论文上发现的问题、提出的假设、实际对照实验、
> 最终方案和明确的限制。本文档是项目经验沉淀，不是产品需求文档；所有结论都应能
> 回到代码、配置、测试输出或原始论文进行核验。

## 维护约定

每完成一个业务模块的一个 Step，在本文件追加一条记录，至少包含：

1. **业务场景与目标**：用户真正需要解决什么问题。
2. **现象与证据**：问题在哪些真实输入中出现，如何复现。
3. **候选方案与实验**：试过什么，而不只记录最后成功的方案。
4. **最终决策与原因**：为什么采用当前方案，以及为什么没有采用其他方案。
5. **验证结果**：运行命令、样本和客观输出。
6. **限制与后续计划**：哪些情况尚不可靠，后续如何验证或替换。

不要在此记录 API Key、Token、个人信息或模型缓存中的敏感内容。配置参数记录在
[`config.yaml`](../config.yaml)，敏感项未来放在未提交的 `.env`。

---

# Step 1：科研论文解析（Parser）

**阶段状态：已完成当前 Docling 基线验证；复杂公式和复杂表格保留为可替换解析器的后续评估项。**

## 1. 业务目标与范围

PaperBase 面向英文科研论文，用户以中文提问。解析阶段需要把 PDF 转为供后续 RAG 使用的
结构化文本，至少保留：

- 论文标题、章节标题与阅读顺序；
- 页码来源、表格、图注；
- 双栏论文的正文顺序；
- 可供后续分块和嵌入使用的 Markdown。

本阶段**不做** Embedding、FAISS、Reranker、LLM、SQLite、Streamlit，也不做图片多模态理解。
原始 PDF 始终保留在 `storage/papers/`，模型缓存保持在 D 盘。

## 2. 验证样本与运行环境

### 真实论文样本

| 样本 | 特征 | 用途 |
| --- | --- | --- |
| `Saeed 等 - 2025 - Enhanced wind speed forecasting for sustainable po.pdf` | Elsevier 首页双栏、展示公式、行内公式、复杂统计表、图表 | 主压力样本 |
| `Shao 等 - 2022 - Decoupled Dynamic Spatial-Temporal Graph Neural Ne.pdf` | 双栏科研论文、章节与表格 | 回归样本 |

### 环境基线

- Windows + Python 3.10.4；
- NVIDIA RTX 5060，8 GB VRAM；
- Docling 2.119.0，CUDA 可用；
- Docling 本地模型目录：`D:/AI_Workspace/AI_Models/docling_models`；
- 当前公式模型预设：`granite_docling`。

## 3. 初始解析架构与一个 Windows 兼容性修复

### 问题

论文文件名包含中文时，直接把 Windows 路径传给 Docling 的 PDF 后端不稳定；此外，Windows
环境下模型隐式启用 `torch.compile` 会触发 GBK 编码相关错误。

### 决策

在 [`docling_parser.py`](../paperbase/parsing/docling_parser.py) 中：

- 以 `BytesIO` + `DocumentStream(name="paper.pdf")` 传递 PDF 内容，避免把中文路径交给
  不稳定后端；原始 PDF 不复制、不改名、不修改。
- 显式使用 CUDA，并关闭布局模型与公式模型的编译开关。
- 当前数字型论文关闭 OCR、图片分类、图片描述与图表数据抽取；这些能力超出本阶段范围。

### 结果

两篇含中文文件名的真实 PDF 均可稳定解析。这个修复与论文内容无关，是 Windows 本地部署的
工程兼容性处理。

## 4. 标题识别：不修改原始标签，增加可审计回退规则

### 现象

Docling 对部分论文首页没有标记 `title`，而是标记成 `section_header`。若直接依赖标签，后续
SQLite 元数据可能缺标题或误把期刊名、摘要标题当成论文标题。

### 方案

1. 优先使用 Docling 的显式 `title` 标签。
2. 没有显式标题时，只从首页 `section_header` 取候选。
3. 过滤 `ARTICLE INFO`、`ABSTRACT`、期刊主页等噪声，要求候选长度足够。
4. 将最终业务标题、标题来源、所有候选分别写入统一解析结果；不修改 Docling 原始 `label`。

### 结果与价值

两篇样本均提取到正确论文标题。保留 `title_source` 和 `title_candidates` 的做法，使标题选择可
复核，也避免后续排查时无法区分“模型输出”与“业务修复”。

## 5. 公式模型对照：CodeFormulaV2 与 Granite-Docling

### 业务问题

PDF 文本层会将上下标、分式与希腊字母压扁。例如 `z_{1-α/2}` 若变成普通字符序列，会影响数学
表达、精确检索和最终回答可靠性。

### 实验

| 方案 | 观察结果 | 结论 |
| --- | --- | --- |
| 不启用公式富化 | 展示公式常被压扁为普通文本 | 不可接受 |
| `codeformulav2` | 可生成部分 LaTeX，但出现超长生成、公式边界错误、符号结构错误 | 不作为默认 |
| `granite_docling` | 对已被 Docling 标记为展示公式的对象整体更稳定；可恢复部分分式、上下标与希腊字母 | 作为当前默认 |

### 当前实现

- 启用 Docling `do_formula_enrichment`，使用 `granite_docling` 预设；
- 仅移除可确定的模型尾部垃圾：`</formula` 标记，以及带 `\quad (编号)` 的公式编号后续正文；
- `orig` 保留原始文本层，富化后的 LaTeX 写入 `text`，便于追溯；
- 不对 LaTeX 做全局“去空格”。多数 TeX 空格不改变渲染，而粗暴删除可能破坏 `\text{...}` 等合法结构。

### 已知限制

公式富化只处理版面模型先识别为 `FORMULA` 的对象。行内公式如果仍被识别为普通正文，不会进入
Granite 模型；因此不能保证所有 `σ_pooled`、`χ²`、大 O 复杂度等行内表达式恢复为正确 LaTeX。

## 6. 连续编号公式被误识别为表格

### 现象

在 Saeed 论文 2.2 节中，LSTM 的六行公式左对齐、编号右对齐。布局模型将其识别为两列表格，
Markdown 变为 `| 公式文本 | (编号) |`，破坏了公式语义。

### 证据与规则

真实表格和该对象的关键差异为：

- 恰好两列、至少两行；
- 每一行左单元格包含等号；
- 右单元格必须严格匹配连续的纯数字公式编号，如 `(2)` 到 `(7)`；
- 不含表头、图注；
- 编号必须逐行连续。

### 最终方案

仅在同时满足以上全部证据时，将该表的每行转为 `FormulaItem`，并保留原始单元格文本与页码
provenance。不能满足时一律保留为表格，避免误伤真实实验结果表。

### 验证结果

Saeed 论文中该伪表格被修复为 6 个公式；原第 3、11 页的真实表格仍然保留。检查报告中的
`docling.corrected_formula_table_count` 为 `1`。

### 限制

该修复解决的是“公式块被错误分类成表格”，不等于从扁平 PDF 文本恢复完整 LaTeX。若原始文本层
已丢失上下标和根号，不能安全地凭正则猜测数学式。

## 7. 首页双栏阅读顺序与字母间距标题

### 现象

Saeed 首页采用：左栏 `ARTICLE INFO`、右栏 `ABSTRACT`，下方开始两栏 `Introduction`。Docling
按左栏优先输出，造成 Introduction 的前半段插入摘要之前，后半段又在摘要之后。

同页栏目标题还被原样抽取为：

```text
A R T I C L E  I N F O
A B S T R A C T
```

### 方案

- 只对 `TITLE` 和 `SECTION_HEADER` 执行逐字大写的空格规范化：
  `A B S T R A C T → ABSTRACT`；不处理正文和公式。
- 在首页同时找到摘要与 Introduction，且摘要实际位于 Introduction 之后时，检查摘要标题与后续
  正文是否位于同一栏。证据充分时才将摘要块移到 Introduction 前。
- 阅读顺序修复只调整文档树中已有元素的引用顺序，不重写文本、页码、父节点或原始内容。

### 验证结果

Saeed 输出顺序已稳定为：`ARTICLE INFO → ABSTRACT → 1. Introduction`；Introduction 左右栏内容
连续。Shao 未满足该特定异常条件，因此没有被重排。

### 泛化边界

该规则覆盖常见首页双栏、摘要与 Introduction 的交错问题，并兼容 `1. Introduction`、
`I. Introduction`、无编号 `Introduction`。不同出版社的三栏、特殊摘要名、跨页首页布局仍需要
通过检查报告抽样验证，不能承诺完全自动正确。

## 8. PDF 断词产生的软连字符

### 现象

PDF 自动换行会把 `alternatives` 抽取成 `al­ ternatives`，其中包含不可见的 U+00AD soft hyphen。
它会影响精确关键词匹配，并增加后续文本处理噪声。

### 方案与安全边界

只移除 U+00AD 及紧随空白；保留普通 ASCII 连字符。因此：

```text
al­ ternatives     → alternatives
state-of-the-art   → state-of-the-art
weather-dependent  → weather-dependent
```

公式对象不经过此规则，以免改动 LaTeX。

### 验证结果

重新解析后，Saeed 修复 86 个文本项，Markdown 中 U+00AD 数量从 150 降为 0；Shao 本来就是 0。

### 普通 ASCII 连字符造成的跨行断词（后续补充）

新增单栏论文暴露出另一种 PDF 文本层问题：`low-pres-\nsure`、`humid-\nity` 使用的是普通
ASCII 连字符，而非 U+00AD。原先只清理软连字符，因此错误会同时进入解析 Markdown、Step 2
的 `raw_text`、FTS5 与 embedding；这不是 HybridChunker 的 token 硬切问题。

修复放在 Step 1 的 Docling 原生文档清洗阶段，覆盖同一文本项内的换行，也覆盖
`humid-` / `ity` 这类被拆成两个相邻 Docling TextItem 的情形。对高置信度英语构词片段移除行末
连字符并拼词；对 `model-\nbased`、`state-of-the-\nart` 等歧义的真实复合词，仅去掉排版换行、
保留连字符。公式对象不参与该规则。这样所有后续模块消费同一份修复结果，不需要在 FTS、embedding
或回答阶段各补一次。

## 9. 复杂表格与行内公式的失败对照实验

### 业务问题

Saeed 第 13 页的 pooled standard deviation 行内公式包含根号和上下标；第 19 页包含多处大 O
复杂度表达式。第 13 页 Table 9 还出现 `PICP`/`PINRW` 合并、统计值和解释错列。

### 对照实验

| 候选方案 | 结果 | 决策 |
| --- | --- | --- |
| TableFormer V1 accurate（当前） | Table 9 错列，复杂宽表仍有缺陷，但整体是当前两篇样本中较稳定基线 | 保留 |
| TableFormer V1，关闭 cell matching | Table 9 仍合并部分行，还出现重复行 | 不采用 |
| TableFormer V2 | Table 9 减少部分错位，却漏掉 `PINRW`，且多个宽表变差 | 不作为默认 |
| Granite-Docling 整页 VLM pipeline | 行内 pooled 公式仍未恢复为正确 LaTeX，Table 9 被压缩为单行 | 不替换主流程 |

### 结论

这类错误已经发生在原生结构阶段，非 Markdown 渲染问题。对数学和统计结果做正则猜测会制造“看似
正确、实际错误”的内容，风险高于保留原始文本。因此当前策略是：保留 Docling 基线、如实暴露限制、
不对高风险行内公式和统计表作不可审计的自动修复。

## 10. 可替换解析器接口与配置化

### 业务需求

当前 Docling 在普通正文、标题、双栏顺序和部分展示公式上可用，但复杂表格、行内公式的准确性有限。
项目需要以后能对比 MinerU 等解析器，而不能让 Chunk、检索、SQLite 与 UI 绑定 Docling。

### 当前架构

统一接口定义在 [`paperbase/parsing/base.py`](../paperbase/parsing/base.py)：

- `PaperParser`：所有解析器必须实现的协议；
- `ParsedPaper`：统一输出，包含 `parser_id`、`markdown`、标题元数据、`diagnostics`；
- `native_document`：仅供解析器专用检查工具使用，业务层禁止依赖其类型。

Docling 适配器将 `document.export_to_markdown()` 写入 `ParsedPaper.markdown`。未来解析器只需产生
同一契约的 Markdown 与元数据，后续模块无需变化。

配置入口在 [`config.yaml`](../config.yaml)：

```yaml
parsing:
  backend: docling
  docling:
    device: cuda
    formula_preset: granite_docling
```

[`paperbase/config.py`](../paperbase/config.py) 负责严格校验和路径解析，
[`paperbase/parsing/factory.py`](../paperbase/parsing/factory.py) 根据 `backend` 创建解析器。
当前只注册 `docling`；将来接入 MinerU 时，新增适配器、配置小节和工厂分支，再改为
`backend: mineru` 即可。

## 11. 当前运行与验证方式

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.parsing.inspect --config .\config.yaml
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

默认命令会读取 `config.yaml` 中的论文目录、D 盘模型目录、CUDA、公式模型与输出目录。最近一次
回归结果：

| 样本 | 页数 | 结构项数 | 关键修复 |
| --- | ---: | ---: | --- |
| Saeed 2025 | 20 | 339 | 1 个伪公式表、86 个软连字符、2 个标题、1 次首页顺序重排 |
| Shao 2022 | 14 | 272 | 5 个标题间距规范化；未触发其他修复 |

产物位于 `storage/parsed/granite_docling/`，包括统一 Markdown、Docling 原生 JSON 和检查报告。报告
会记录 `parser_id`、公式模型预设以及各项修复计数，保证可回溯。

## 12. 面试表达要点

1. **以真实论文驱动解析评估**：没有只看“能否跑通”，而是用双栏、公式、复杂表格等困难样本做
   结构化验证。
2. **保守修复而非文本猜测**：对编号连续的伪公式表使用多条件证据转换；对数学含义不确定的行内公式
   不做正则伪修复。
3. **结果可审计**：保留原始文本、标题来源、解析器标识、修复计数与检查 JSON。
4. **工程可演进**：用解析器无关的 `ParsedPaper` 契约和工厂模式隔离 Docling，支持未来 A/B 测试
   MinerU 等解析方案。
5. **配置与敏感信息分离**：模型路径、GPU、解析器选择集中在 YAML；未来密钥进入 `.env`。

## 13. 后续解析工作（尚未执行）

当整体 RAG 链路完成并具备明确的检索评测集后，再进行解析器替换实验：

1. 在隔离环境安装候选解析器；
2. 对同一 PDF 集合输出统一 `ParsedPaper`；
3. 对标题、阅读顺序、行内/展示公式、表格、图注、速度、显存和检索 Recall@K 做对比；
4. 只有在真实样本和下游检索指标都更优时，才把 `parsing.backend` 切换到新实现。

在此之前，不应因为单一页面效果不佳就替换已经稳定工作的主解析链路。

---

# Step 1 补充验证：MDPI 单栏论文与重叠文本层

**验证日期：2026-08-12；本次只验证，不修改解析规则。**

## 1. 新增样本与目的

新增论文：`Yang 等 - 2023 - Forecast of Fine Particles in Chengdu under Autumn.pdf`。

这是一篇 11 页的 MDPI `Toxics` 单栏论文，首页有左侧出版信息栏，正文位于右侧主栏，正文页包含
跨行合并单元格的统计表。验证目的是确认此前针对双栏 Elsevier 论文的修复不会伤害单栏论文，并观察
另一家出版社版式的解析边界。

## 2. 运行结果

仍使用配置化默认命令和 `docling` 解析器：

```powershell
.\.venv\Scripts\python.exe -m paperbase.parsing.inspect --config .\config.yaml
```

本次为避免重复耗时，只对新增论文执行同一解析器和检查产物写入流程。结果：

| 指标 | 结果 |
| --- | --- |
| 页面数 | 11 |
| 结构项数 | 145 |
| 标题 | 正确：`Forecast of Fine Particles in Chengdu under Autumn-Winter Synoptic Conditions` |
| 章节标题 | 识别 13 个 |
| 图注 | 6 个 |
| 表格 | 3 个 |
| 展示公式 | 1 个，相关系数公式 LaTeX 输出正常 |
| 既有修复触发情况 | 伪公式表 0、软连字符 0、字母间距标题 0、首页摘要重排 0 |

## 3. 表现良好的部分

- 论文标题、作者、摘要、关键词及 `1. Introduction` 的主顺序正确；
- 单栏正文在普通页面中保持由上到下的阅读顺序；
- 第 4 页展示公式 `R` 的分子、分母、根号和编号被恢复为 LaTeX；
- 第 6 页 Table 1 的 5 列、4 行统计数值与原 PDF 对照正确；
- 此前为 Elsevier 首页做的 `ABSTRACT → Introduction` 重排没有误触发，说明规则没有对该
  单栏 MDPI 首页造成副作用。

## 4. 新发现：不可见/重叠的审稿文本层导致正文重复

### 证据

渲染后的原 PDF 第 3 页只显示正式论文内容与 Figure 1。但 Docling 的结构 JSON 同页还包含：

```text
, x FOR PEER REVIEW
were obliquely rotated and classified into several groups ...
```

随后才是一次完整的 `2.2. Model Parameters`。Markdown 中还出现：

- `FOR PEER REVIEW` 共 3 次；
- Figure 1 图注连续重复 2 次；
- 2.2 节正文中混入前一页的旧文本；
- `3.3 ... 3.3.1 ...` 两个视觉上独立的标题被合并为一个 section header。

这说明该 PDF 的可提取文本层中存在额外或重叠的审稿版文本。视觉渲染不显示这些内容，但普通 PDF
文本抽取会读取它们。问题不是此前的双栏顺序规则造成的。

### 当前决策

**不在本轮自动删除。**依据单个关键词删除 `FOR PEER REVIEW` 可行，但无法可靠判断紧随的隐藏段落
范围、重复图注与其他出版社的合法文本。贸然按字符串去重可能删掉正文中的重复论述或图注。后续应先
设计“隐藏/重叠文本层检测”实验，再建立带页码、bbox、文本相似度和视觉位置证据的保守过滤规则。

## 5. 跨行合并表格的边界

### Table 1（第 6 页）

结构与数据正确，证明普通横向表格当前可用。

### Table 2（第 7 页）

原表中每个天气模式跨 3 行，对应 24 h、48 h、72 h。当前结果的大部分数值正确，但 `Low pressure`
行被错误合并为 `24 h 48 h`，并混入页码文本 `8 of`。

### Table 3（第 8 页）

同样存在跨 3 行单元格：部分 Forecast Duration、温度/湿度/风速数值被合并到同一单元格，
`High-pressure bottom` 也出现重复行。

### 当前决策

这些错误发生在表格结构识别阶段，Markdown 只是呈现了错误结构。当前不手写拆行规则：不同论文的
合并单元格语义不同，按固定行数重排会给统计问答引入静默错误。此类表格将作为后续解析器对比、
表格专用检测和下游检索评测的重要样本。

## 6. 对当前 Parser 的结论

该样本说明“单栏”不等于“低难度 PDF”。当前 Docling 基线对普通单栏正文、标题、展示公式和简单
表格已可用，但对带审稿残留文本层、侧栏出版元数据、跨行表格的 PDF 仍需要人工检查。现阶段最正确
的工程动作是保留检查报告和原始 JSON，使风险在进入 RAG 前可见，而不是用不可审计的文本清洗掩盖。

---

# Step 1 补充修复：页眉页脚、审稿文本层与重复图注

**验证日期：2026-08-12；本次变更已在 Saeed、Shao、Yang 三篇真实论文上完整回归。**

## 1. 业务场景与可视化核验

Yang（MDPI `Toxics`）的第 3 页视觉渲染只显示正式页眉：

```text
Toxics 2023, 11, 777                         3 of 11
```

但解析输出却混入了按行拆开的：

```text
Toxics
2023
,
11
, x FOR PEER REVIEW
3 of 11
```

原 PDF 视觉上没有这段审稿文字。对 Docling 原生 JSON 的页码和 bbox 检查表明：PDF 的可提取
文本层同时保存了正式版本和不可见的审稿层；两层位置重叠，普通文本抽取无法自动区分。该问题不是
Markdown 渲染造成，也不是此前的首页双栏重排规则造成。

## 2. 页眉、页脚的独立字段

### 决策

在统一结果 [`ParsedPaper`](../paperbase/parsing/base.py) 新增：

```text
page_furniture: tuple[PageFurniture, ...]
```

其中每项包含 `page_no`、`location`（header/footer）和 `text`。纯页码例如 `3 of 11`、
`Page 3` 会直接丢弃；期刊名、卷期、文章号、DOI、arXiv 编号等保留为来源元数据，不再进入
Markdown、后续 Chunk 或检索语料。

Docling 有一个实现细节：部分页眉/页脚仅存在于 `document.texts` 的原生对象列表，并没有挂在正文
树上，因此本就不会导出 Markdown。实现同时读取完整对象列表以提取元数据；若某页眉/页脚实际挂在
正文树，才从树中删除。这样不会为了“删除”而破坏 Docling 的对象注册表。

### 结果

检查报告新增 `page_furniture` 字段。三篇样本分别提取到：

| 样本 | 有信息价值的页眉/页脚项 | 示例 |
| --- | ---: | --- |
| Saeed 2025 | 23 | `Energy 335 (2025) 137979`、DOI |
| Shao 2022 | 1 | `arXiv:2206.09112v4 [cs.LG] 5 Sep 2022` |
| Yang 2023 | 12 | `Toxics 2023, 11, 777`、MDPI DOI |

配置开关位于 `config.yaml` 的 `parsing.docling.remove_page_furniture`，默认 `true`。

## 3. 不可见审稿文本层：只清理有双重证据的碎片

### 候选方案与风险

| 方案 | 风险 | 决策 |
| --- | --- | --- |
| 全文按相同句子去重 | 论文中合法的重复表述、图注或引用可能被删除 | 不采用 |
| 只删除 `FOR PEER REVIEW` 字样 | 剩余的卷期碎片、旧正文仍污染 Markdown | 不采用 |
| 标记 + 页码 + 同一 bbox 顶边 | 只处理可证明来自同一不可见层的文本 | 采用 |

### 最终规则

只有同时满足下面两项时才删除：

1. 页面上存在 `FOR PEER REVIEW`（不区分大小写）；
2. 候选文本同页、同为普通文本对象，且 bbox 顶边与该标记相差不超过 1 pt。

另外，在已命中这种审稿页的正文行首，仅删除可确定的 `N of M` 分页前缀；不会重写其后的论文正文。
该规则受 `parsing.docling.remove_peer_review_artifacts` 控制，默认 `true`。

### 验证结果

Yang 中删除了 16 个叠加文本项，`FOR PEER REVIEW` 和 `3 of 11` 在最终 Markdown 中均为 0。Saeed、
Shao 都没有匹配该标记，因此此规则在两篇旧样本上的触发数为 0。

## 4. 同一图注对象的重复与合并标题

### 重复图注

Yang 的 Figure 1、Figure 2、Figure 3 各被拼接为一次完整图注加一次高度相似的完整图注。规则仅处理
`CAPTION` 对象，要求第二段再次以相同的 `Figure/Table + 编号` 开头，且两段在空白规范化后的相似度
至少为 0.90。它不是全文去重器。

结果：Yang 的三个重复图注各保留一份；Saeed、Shao 的图注修复数均为 0。

### 合并标题

Yang 的 `3.3` 与 `3.3.1` 被识别在同一个 `section_header` 中。仅当一个标题对象中恰好出现一个新的
“数字编号 + 大写标题”边界时才拆分；正文、表格单元格和公式完全不参与。结果为两个独立标题，且
三篇回归中仅 Yang 触发 1 次。

## 5. 跨页表格中的不完整分页残片

Yang Table 2 的 `Low pressure` 行右侧原为 `- 8 of`，其中 `8 of` 是下一页页眉被压入表格的残片。
本次仅删除表格单元格**末尾**的悬空 `N of`：带完整数量语义的 `8 of 11 samples` 不匹配，也不重排
任何行列或数值。该规则在 Yang 触发 1 次，在 Saeed、Shao 均为 0。

这使分页碎片不再进入 Markdown，但 Table 2/3 的跨行单元格错位仍是 Docling 表格结构识别的限制，
仍按第 5 节的结论保留为后续解析器替换实验的对比样本，**不通过猜测数值自动修复**。

## 6. 最终回归结论

运行命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -m paperbase.parsing.inspect --config .\config.yaml
```

单元测试为 6/6 通过。三篇真实论文的结构项数与既有关键修复如下：

| 样本 | 结构项数 | 原有修复回归 | 新规则触发 |
| --- | ---: | --- | --- |
| Saeed 2025 | 339 | 伪公式表 1、软连字符 86、标题 2、首页顺序 1，均保持原值 | 全部 0 |
| Shao 2022 | 272 | 标题间距 5，均保持原值 | 全部 0 |
| Yang 2023 | 130 | 不触发旧规则 | 合并标题 1、审稿叠加层 16、分页前缀 1、表格残片 1、图注去重 3 |

这次的核心经验是：**先用 PDF 视觉渲染证明“可见内容”和“文本层内容”不同，再用标签、页码、bbox
和对象类型构成删除证据链**。对无法形成证据链的复杂表格和正文重复，宁可保留并交由后续解析器
对比，也不制造静默数据错误。

---

# Step 2：结构感知分块（HybridChunker）

**验证日期：2026-08-12；本阶段只产生检查 JSONL，不写 SQLite、不生成 embedding、不建立 FAISS。**

## 1. 业务目标与输入输出

Parser 已验证标题、章节、页码、阅读顺序、图注与表格基础结构后，Step 2 的任务是将论文拆为适合
检索的内容块，同时保留“它来自哪篇论文、哪个章节、哪些页、前后相邻什么内容”。不采用固定字符数
或“每页 300 tokens + overlap”的切法，因为论文一页可同时出现方法、图注与实验，页边界不是语义边界。

流程为：

```text
ParsedPaper / DoclingDocument
→ Docling HybridChunker
→ PaperChunk 列表
→ JSONL + 人工检查报告
```

统一 `PaperChunk` 契约定义在 [`paperbase/chunking/base.py`](../paperbase/chunking/base.py)。当前保存：

```text
chunk_id, vector_id(None), paper_id, paper_title, chunk_index,
raw_text, embedding_text, section, page_start, page_end,
raw_token_count, embedding_token_count, prev_chunk_id, next_chunk_id
```

`raw_text` 是后续交给 LLM 的结构化原文；`embedding_text` 额外添加论文标题和章节路径，服务中文问题到
英文论文段落的跨语言检索。两者刻意分离，避免检索元数据污染最终的回答证据。

## 2. Tokenizer 对齐实验与模型下载

### 问题

Docling `HybridChunker` 未显式配置时默认使用 `sentence-transformers/all-MiniLM-L6-v2` tokenizer。
这会产生两个问题：

1. 默认初始化会尝试向 C 盘 Hugging Face 缓存下载 MiniLM；
2. Step 4 将使用 `Qwen/Qwen3-Embedding-0.6B` 生成向量，MiniLM 的 token 边界与 Qwen 不一致，
   `max_tokens=512` 无法准确代表最终 embedding 输入长度。

### 决策

将 `Qwen/Qwen3-Embedding-0.6B` 完整下载到：

```text
D:/AI_Workspace/AI_Models/hf_models/Qwen3-Embedding-0.6B
```

Step 2 使用 `local_files_only=True` 从此目录加载 Qwen tokenizer；实际验证了中英混合文本的 token 计数，
没有访问网络或 C 盘缓存。模型权重不加载到 GPU，因此本阶段不占用额外推理显存。

分块预算配置为：

```yaml
max_tokens: 512
embedding_metadata_reserve_tokens: 64
```

即 HybridChunker 的正文 token 预算为 448，最终加上 `Paper title`、`Section`、`Content` 字段后重新用
Qwen tokenizer 计数。检查报告会记录 `over_limit_chunk_count`，而不是假设预留总能充分。

## 3. 结构与元数据映射

Docling HybridChunker 原生提供：正文块、标题层级和每个底层对象的 provenance。适配器采用如下映射：

| PaperChunk 字段 | 来源/规则 |
| --- | --- |
| `paper_id` | 原始 PDF SHA-256 前 16 位，避免中文文件名和当前路径影响稳定性 |
| `chunk_id` | `paper_<hash>_chunk_<四位序号>` |
| `section` | HybridChunker 的标题路径，以 ` > ` 串接 |
| `page_start/page_end` | chunk 内所有 provenance 页码的最小/最大值 |
| `prev/next_chunk_id` | 同论文内相邻索引，供 Step 8 再按章节做安全邻居扩展 |
| `vector_id` | 当前为 `None`；Step 5 建立统一 FAISS 时才分配 |

跨页内容不拆回页级，而是保留如 `page_start=1, page_end=2`，最终 Citation 可覆盖完整来源范围。

## 4. 发现与保守处理：孤立出版类型标签

首次在 Yang 上运行得到一个只有 `Article`、1 token、无章节路径的 chunk。这不是论文知识，若进入向量库
会成为无意义候选。

没有采用“删除所有短 chunk”的做法，因为短公式、图注与关键词也可能有检索价值。最终仅过滤同时满足：

1. 无章节路径；
2. token 数不超过 4；
3. 规范化文本精确等于 `Article`、`Research article` 或 `Review article`。

Yang 因此从 39 个变为 38 个有效 chunk；短图注、短公式和有章节归属的短文本均保留。该行为由单元测试
覆盖，并作为 `chunking.skipped_layout_noise_chunk_count` 写入报告。

## 5. 三篇真实论文回归

运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -m paperbase.chunking.inspect --config .\config.yaml
```

结果：10/10 单元测试通过；三篇论文的最终检查如下。

| 样本 | 有效 chunk | 最大 embedding tokens | 超过 512 的 chunk | 跨页 chunk | 新噪声过滤 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saeed 2025 | 133 | 492 | 0 | 18 | 0 |
| Shao 2022 | 94 | 477 | 0 | 11 | 0 |
| Yang 2023 | 38 | 488 | 0 | 7 | 1 |

另外验证：三篇所有 chunk 的前后邻居 ID 均完整连通；Yang 的 `FOR PEER REVIEW`、`3 of 11` 仍为 0，
说明 Step 1 清理结果被原生结构分块直接继承；Figure 1 与 Table 1 各保留一次。Saeed 首页 chunk 顺序为
`ARTICLE INFO → ABSTRACT → 1. Introduction`，与 Parser 的双栏顺序修复一致；LSTM 公式仍作为同一
结构 chunk 的正文保留。

## 6. 当前限制与下一步

HybridChunker 保留了 Step 1 的所有真实结构优点，也保留其上游限制：复杂表格的合并单元格错位、部分
行内/展示公式识别不完整，不能由分块层擅自“修复”。此外，参考文献会产生大量 chunk；当前先保留，
等 Step 6 有真实查询与检索指标后再决定是否将 `References` 单独降权或排除，不能在无评测前凭直觉删除。

下一步是 **Step 3：SQLite Metadata**：将已稳定的 `documents`、`chunks` 元数据正式落库，但仍不生成
embedding 或建立 FAISS。

---

# Step 1 补充修复：正文编号列表被误判为章节

**验证日期：2026-08-12；修复位置：Step 1 Parser 文档结构层。**

## 现象与根因

Saeed 论文的 `1. Introduction` 末尾包含连续的四条研究贡献。第一条跨第 3、4 页：

```text
1) Development of a unified deep learning framework ... prediction intervals ...
2) Introduction of an adaptive quantile violation mechanism ...
```

原 PDF 中它们是正文编号列表，不是新的一级章节。Docling 却将第 1 条的前半句标为
`section_header`；后续第 2 至 4 条则位于一个 `ListGroup` 中、已被标为 `ListItem`。如果让 Step 2
修正 `section`，只能掩盖一个字段错误，Markdown、页码 provenance 和后续任何替代分块器仍会继承错误
文档树。

## 通用修复规则

Parser 在导出 Markdown、计算标题和交给 Chunker 之前，保守地把错误结构还原。它不依赖作者、论文名或
固定页码，而是同时要求以下证据：

1. 当前根节点确实被 Docling 标为 `section_header`，文本精确符合 `N) 内容`；
2. 文本长度不少于 `parsing.docling.list_style_heading_min_chars`（默认 `80`），避免改写短标题；
3. 在下一个真实 `section_header` 之前、最多八个相邻根对象内，存在连续的 `(N+1)` 枚举列表项；该项
   可以是普通文本，也可以位于 Docling 的 `ListGroup` 子树中；
4. 原节点必须带 provenance，才能保留可靠页码。

满足条件时，Parser 用保留原 marker、正文、内容层和 provenance 的 `ListItem` 替换原
`section_header`。因此 Markdown 会出现完整 `1) ...` 正文，Chunker 自然继承 `1. Introduction`，
而 Citation 仍能覆盖第 3 至 4 页。短的 `1) Study area`、没有连续下一项的长标题、以及下一个真实章节
之后才出现的数字都不会触发转换。

同时，Parser 对未标注的 running header/footer 使用独立的通用规则：相同单行文本必须至少跨两页、位于
页面顶部或底部 12%、且四个 bbox 坐标都在 2 pt 容差内，才从正文移到 `page_furniture`。这解决了作者名
页眉混入跨页列表的问题，但不会按具体作者名删除正文。

## 验证结果

目标论文的原生 DoclingDocument 已出现 `list_item`（marker 为 `1)`），解析诊断为
`docling.converted_list_style_heading_count=1`。Step 2 不再含任何伪章节回退代码；目标 chunk
`paper_5b6a1007fa7514bf_chunk_0015` 自然得到：

```text
section: 1. Introduction
page_start: 3
page_end: 4
raw_text: - 1) Development ... prediction intervals ...
embedding_text 的 Section: 1. Introduction
```

三篇 PDF 的 Parser 回归均成功；该规则的转换计数依次为 Saeed `1`、Shao `0`、Yang `0`。分块回归结果：

| 样本 | Chunk 数 | 最大 embedding tokens | 超限 |
| --- | ---: | ---: | ---: |
| Saeed 2025 | 132 | 492 | 0 |
| Shao 2022 | 94 | 477 | 0 |
| Yang 2023 | 38 | 488 | 0 |

单元测试覆盖了括号编号格式、连续下一项证据、下一个真实章节的停止条件、以及 running header 的固定坐标
要求；该次新增后测试为 `15/15` 通过。这个案例的经验是：**结构错误应在最早产生结构的 Parser 层修复；下游
模块只消费统一结构，不针对单篇论文重写业务语义。**

---

# Step 1 补充修复：论文前置元数据语义块

**验证日期：2026-08-12；修复位置：Step 1 Parser 与 `ParsedPaper` 统一契约。**

## 问题与目标

旧结构将标题后的作者、单位、邮箱等内容继承为“论文标题” section；是否能正确得到
`Abstract`、`Keywords` 又取决于出版社是否把它们标为标题。例如：Yang 的 `Abstract:`、
`Keywords:` 是普通文本；Shao 的 `PVLDBReference Format:` 和 `PVLDBArtifact Availability:`
被当作正文标题；后者还把代码链接、通讯作者、许可证和期刊脚注混在一个 chunk 候选中。

这不是 token 上限问题。若先在 Chunker 中按文本做切割，未来替换分块器、SQLite 或检索策略时仍会
重新得到不一致的前置信息。因此本次只修复 Step 1 的语义结构，暂不改 Step 2 和 doc2query。

## 方案

新增解析器无关的 `ParsedPaper.front_matter`，每个 `FrontMatterBlock` 保存：

```text
block_type / canonical_section / text / page_start / page_end /
source_item_count / detection_method / confidence
```

所有文本都来自 PDF/Docling 原始内容，系统不通过 LLM 补全作者、摘要或关键词。Parser 使用以下组合证据：

1. **受控标题别名**：忽略大小写、空格、标点，统一 `ABSTRACT`、`PVLDBReference Format:`、
   `PVLDBArtifact Availability:` 等为标准类型；未知标题完全保留。
2. **行内强标签**：仅把行首 `Abstract:`、`Keywords:`、`Citation:`、`Received:`、`Copyright:`
   等显式标签提升为对应语义；没有标签的长段落绝不猜测为摘要。
3. **标题锚定的作者单位区间**：在论文标题之后、下一个标题或强标签之前，且内容含机构、邮箱或通讯
   信号时，创建 `authors_affiliations`。作者名、单位编号和邮箱保持在同一原始块中，不擅自拆实体。
4. **Availability 容器细分**：已有“可用性”标题下的代码链接、通讯作者、许可证、期刊脚注按明确内容
   信号分别输出为 `availability`、`correspondence`、`rights`、`publication_info`。
5. **侧栏不重排**：首页侧栏可能在 Docling 阅读顺序中插入 Introduction 中间。它们只作为独立
   front_matter 块记录，不移动原生正文节点，避免破坏正文续写和 provenance。

对于可确定的首页结构，Markdown 同步标准化为 `Authors and affiliations`、`Abstract`、`Keywords`、
`Publication information`、`Availability`。这让人类检查结果也与统一语义一致。

## 验证

| 样本 | 识别到的前置元数据块 | 关键验证点 |
| --- | --- | --- |
| Saeed 2025 | 作者单位、关键词、摘要 | `ARTICLE INFO` 仅包含 Keywords 时被安全收敛为 `Keywords`。 |
| Shao 2022 | 作者单位、摘要、出版信息、代码可用性、通讯作者、许可证、期刊脚注 | `Artifact Availability` 容器被细分，代码链接不再和版权文本混在同一语义块。 |
| Yang 2023 | 作者单位、行内摘要、行内关键词、出版信息、版权 | `Abstract:` 与 `Keywords:` 被提升为标题；错排侧栏独立记录而未吞入摘要。 |

项目单元测试为 `19/19` 通过，覆盖别名标准化、行内标签、普通正文的反例，以及 Availability 容器细分。

## 后续边界

本次没有生成 doc2query，也没有把这些块重新分块或建立向量。下一次 Step 2 实验应以
`front_matter.block_type` 为唯一输入：作者单位作为关联 metadata chunk，摘要与关键词各自独立，
并通过受控检索提示、词法/结构化匹配和 short2big 验证“作者是谁”“代码在哪里”等查询。这样可以把
解析准确性与召回收益分开评估，避免无法判断效果来自哪个环节。

---

# Step 3：SQLite Metadata 正式入库

**验证日期：2026-08-12；范围：只建立文档、前置元数据与原始 chunks 的正式存储。**

## 为什么不能只保留 JSONL

Step 2 的 `*.chunks.jsonl` 适合人工检查：一行包含一个完整 chunk，便于直接打开、diff 和复现。
但它不适合作为正式业务数据层：按论文、前置元数据、页码、section 和未来 vector ID 查询时，需要扫描
文件、手工关联，无法提供外键、唯一性和原子更新保证。因此 JSONL 继续保留为检查产物，SQLite 成为
正式 metadata store。

## 设计

使用项目内的单文件 `storage/paperbase.sqlite3`，无需下载或启动额外服务。核心关系为：

```text
documents (paper_id)
├── front_matter (paper_id foreign key)
└── chunks (paper_id foreign key)
```

`documents` 保存 PDF 完整 SHA-256、来源路径、标题、解析/分块器标识和 diagnostics；
`front_matter` 保存 Step 1 的标准语义块；`chunks` 保存 raw/embedding 文本、section、页码、token
计数和前后邻居。PDF 二进制不进入 SQLite，`vector_id` 暂为 `NULL`，不提前建立 embedding 或 FAISS。

导入器始终使用：

```text
PDF → Parser → HybridChunker → SQLite
```

而不是从 Markdown / JSONL 反推，因为后两者不能完整恢复 Docling 结构、provenance 和前置元数据。

## 幂等性与一致性

同一 PDF 的 `paper_id` 来自文件完整哈希，重导入时在一个事务内替换该论文记录及其级联子记录：

- 成功：新解析/分块结果完整替换旧版本，不会追加重复 chunks；
- 失败：事务回滚，避免半篇论文；
- 导入前验证哈希身份、chunk ID 唯一、连续的 `chunk_index` 和有效邻居引用；
- 已有非空 `vector_id` 的论文拒绝走替换入口，提前防止未来 FAISS 与 SQLite 不一致。

## 验证结果

新增 4 个 SQLite 单元测试，覆盖三表关联、同一 PDF 的替换导入、非法邻居引用时不写入半成品、
以及已分配 `vector_id` 时拒绝覆盖；全项目测试由 `19/19` 增至 `23/23` 通过。

真实导入三篇论文后：

| 样本 | 前置元数据块 | chunks |
| --- | ---: | ---: |
| Saeed 2025 | 3 | 132 |
| Shao 2022 | 7 | 94 |
| Yang 2023 | 5 | 39 |
| 合计 | 15 | 265 |

实际再次导入 Yang 显示 `replaced`，总行数保持 `documents=3`、`front_matter=15`、`chunks=265`；
`integrity_check=ok`、外键违规为 0、chunk 顺序与邻居引用异常均为 0，且 `vector_id` 非空数量为 0。

## 经验

在 RAG 中，JSONL 适合可读的中间检查，SQLite 适合稳定身份、关联约束与增量生命周期。两者并存而不
相互替代。重要的是：在尚未建立向量索引前就明确阻止“覆盖已分配 vector_id 的文档”，能把日后最难
排查的 metadata/index 脱节问题变成可见错误。

---

# Step 3 补充修正：前置元数据不再重复落库

**修正日期：2026-08-13；范围：SQLite schema V1 → V2。**

## 发现的问题

原 Step 3 将 `ParsedPaper.front_matter` 的全文写入独立 `front_matter` 表，同时 Step 1 已经把同样内容
作为标准 section 留在文档树，因此 Step 2 的 `chunks` 也会保存它。两张表产生了重复文本。更关键的是，
后续只有 `chunks` 会进入 embedding/FAISS 并持有 `vector_id`：若前置元数据仅从独立表查询，向量命中后
无法按统一 ID 回查，也会导致作者、摘要、关键词等信息缺少语义召回路径。

## 修正方案

保留 Step 1 的 `ParsedPaper.front_matter` 作为**解析阶段的语义证据**，但 SQLite 只保留两张正式表：

```text
documents (paper_id)
└── chunks (paper_id foreign key；正文和前置元数据的唯一文本源)
```

每个 chunk 新增：

- `content_kind`：`body` 或 `front_matter`；
- `front_matter_type`：仅当前者为 `front_matter` 时有值，例如 `authors_affiliations`、`abstract`、`keywords`。

Chunker 不重新切分任何文本，而是用 Step 1 已识别块的“标准 section 末级标题 + 页码范围相交”标记现有
chunk。这样是解析器无关的通用规则，兼容 `Abstract` 与 `A B S T R A C T` 的字距化标题，不针对某篇
论文的特定正文写例外。

SQLite 首次打开 V1 文件时，在一个事务中回填上述类型字段、删除重复 `front_matter` 表并升级
`schema_version` 为 `2`。迁移失败会回滚；迁移完成后建议完整重导入，让最新 Parser/Chunker 重新生成
所有标记。

## 验证结果

新增回归测试覆盖：新库不创建 `front_matter` 表、前置类型 chunk 可查询、重复导入仍原子替换、已分配
`vector_id` 时拒绝覆盖，以及 V1 历史库迁移后保留原始 chunk 文本并正确回填类型。项目完整测试为
`25/25` 通过。

已先备份旧库，再将实际 `storage/paperbase.sqlite3` 从 V1 迁移到 V2，并按最新 Parser/Chunker 重新导入
三篇样本。最终只有 `documents`、`chunks`、`schema_info` 三张表，共 `documents=3`、`chunks=265`：

| 样本 | chunks | `front_matter` 类型 chunks |
| --- | ---: | ---: |
| Saeed 2025 | 132 | 3（作者单位、摘要、关键词） |
| Shao 2022 | 94 | 4（作者单位、摘要、出版信息、可用性） |
| Yang 2023 | 39 | 3（作者单位、摘要、关键词） |

`PRAGMA integrity_check=ok`、`PRAGMA foreign_key_check=0`，且当前 `vector_id` 非空数量为 0，符合尚未进入
Embedding/FAISS 阶段的边界。

---

# Step 4：本地 Embedding 向量工件

## 目标与边界

Step 4 从正式 SQLite 的 `chunks.embedding_text` 读取当前知识库的 265 条 chunk，使用本地
`Qwen/Qwen3-Embedding-0.6B` 生成文档向量。它不重新执行 PDF Parse/Chunk，不写入 SQLite 向量列，
不建立 FAISS，也不分配 `vector_id`；这些 ID 必须等 Step 5 的 `FAISS add_with_ids` 成功后再回写。

这保证新增论文的“已解析 chunk 集”与 embedding 输入严格一致，也使向量重建不必再次加载 Docling 模型。

## 方案

模型用 `sentence-transformers` 从 D 盘本地目录离线加载。文档侧不附加 query instruction；未来中文查询
再在 query 编码侧使用 retrieval instruction。输出统一转为 `float32` 并再次 L2 归一化，使后续
`IndexFlatIP` 的内积等于余弦相似度。

向量写入 staging 而非 SQLite：`vectors.npy` 保存 `(N, D)` 矩阵，`records.jsonl` 保存
`row_index -> chunk_id` 映射与输入文本哈希，`manifest.json` 保存模型、维度、数量、输入指纹和文件校验和。
因此 Step 5 能验证行映射与文件完整性后再创建正式索引，不会把文件篡改或半写入的混合批次静默加入 FAISS。

## 验证结果

最小 CUDA 实测先对两条英文文档成功生成 `2 × 1024` 的 `float32` 单位向量。随后对三篇样本文献完成正式
运行，得到 `265 × 1024` 向量；行级 `chunk_id` 映射与 SQLite 的 `paper_id, chunk_index` 稳定排序完全一致，
范数范围为 `0.99999988–1.00000012`，`vector_id` 非空数量仍为 `0`。

新增工件写入/读取完整性测试、文件篡改拒绝测试、SQLite 输入顺序与“已有 vector_id 时拒绝重建”测试；
全项目回归测试为 `27/27` 通过。

---

# Step 5：全局 FAISS 索引与 SQLite 映射

## 目标与边界

将 Step 4 的 265 条 staging 向量建立为**唯一**的知识库索引，而不是按论文拆分多个 `.faiss` 文件。
索引选择 `IndexIDMap2(IndexFlatIP)`：文档向量已归一化，因此内积等价于余弦相似度；Flat 索引无需训练，
将来可作为增量 `add_with_ids` 的基础。当前仅实现首次全库建库，不提前实现上传论文的增量业务服务。

## 映射方案

Step 4 的 `records.jsonl` 在建库时将临时的“向量第 i 行”指向 `chunk_id`。系统先验证这些 chunk 和
`embedding_text` 哈希仍与 SQLite 完全一致，再给每条记录分配全局 ID `1..265`：同一个 ID 一方面通过
`FAISS.add_with_ids` 写入索引，另一方面在一个 SQLite 事务中写入 `chunks.vector_id`。

最终线上链路不依赖 `.npy` 或 JSONL：

```text
Query vector → FAISS vector_id → SQLite chunk → raw_text / page / section / neighbors
```

## 一致性处理

因为 FAISS 文件和 SQLite 文件无法组成同一事务，首次建库实现了 pending journal。候选索引与完整映射先
落入 journal，SQLite 成功提交映射后才原子发布索引与 manifest；若中断，下次运行会根据 journal 判断
是安全清理未提交候选，还是补完已提交映射对应的索引发布。manifest 绑定索引校验和、维度、数量、
embedding 输入指纹和 ID 分配指纹。

后续一次“文本修复后全库重建”验证了另一个恢复边界：旧索引和新候选索引可能拥有完全相同的
`vector_id` 集合，却对应不同的向量内容。因此发布恢复不能只比对维度和 ID；现在还必须比对候选
journal 记录的索引 SHA-256。若正式索引内容旧、候选索引校验正确，就原子替换正式索引；两者都不匹配
才停止发布。该规则避免 SQLite 新映射误连到旧 embedding。

测试还发现 FAISS 1.15 的 Windows 路径 I/O 在包含中文用户名的临时目录失败。因此改用 FAISS 的内存
序列化与 Python 文件写入，输出仍是标准 FAISS 二进制格式，同时避开路径编码问题。

## 验证结果

三篇样本文献成功建立 `265 × 1024` 的全局索引，SQLite 中 265 条 chunks 的 `vector_id` 范围为 `1..265`。
CLI 的 FAISS—SQLite—embedding 工件交叉验证通过；前 10 条向量的 self-search 均返回自身 ID，最低分为
`0.99999994`，没有遗留 pending journal。新增首次建库、陈旧 embedding 拒绝、SQLite 已提交后的恢复发布
三项索引测试；全项目回归测试为 `30/30` 通过。

---

# Step 6：混合检索、受约束 Query 改写与可降级在线链路

## 目标与边界

在不改变 Step 1–5 的 PDF、解析、chunk、SQLite 正文和 FAISS 向量的前提下，把用户问题召回成带来源证据的 chunks。此阶段不生成最终答案、不引入 reranker，也不将 LLM 生成文本写入数据库。这样可以把“召回是否正确”和“回答是否正确”分开评估。

## 初版方案：五类逻辑通道 + 加权 RRF

原始问题始终走两条基础路径：Qwen 稠密检索 Top-20 和 SQLite FTS5/BM25 Top-10。LLM 只作为补充查询规划器，返回严格 JSON：最多一条完整语义改写、最多三条中文短关键词、最多三条英文短关键词。语义改写走第二条稠密路径；中英文关键词分别走两组 BM25 路径。

不同模型的原始分数不可混加，因此结果按 `weight / (rrf_k + rank)` 做 RRF 融合，默认 `rrf_k=60`。同类中的多条关键词会均分该类总权重，避免模型因为多输出关键词而篡改排序。结果保存每一次命中的 route、query、rank、原始分数和实际权重，方便后续离线评估或接入 cross-encoder reranker。

## 工程契约

1. 正式线上检索只依赖 `paperbase.faiss`、FAISS manifest 和 SQLite；不读取 `vectors.npy`/`records.jsonl`。启动时根据 SQLite 的 `vector_id` 和 `embedding_text` 重算映射指纹，校验三者一致。
2. SQLite schema V3 新增 FTS5 派生表 `chunks_fts`，索引论文标题、章节和 raw_text。其内容由 `chunks` 触发器维护，正文事实源仍只有 `documents`、`chunks`。
3. Prompt 从业务代码剥离到 `paperbase/prompts/query_rewrite.py`，将同一用途的 system prompt、user 模板、JSON 修复 prompt 放在同一个 Python 文件，便于版本管理和 A/B 实验。
4. `.env` 只存 API Key、模型地址和运行时参数；`config.yaml` 只存非敏感检索阈值、权重和开关。`.env.example` 提供可提交模板。
5. LLM 网络失败、空响应或 JSON 不合约时，默认 `fallback_to_original=true`，无声但可观测地退化为原始稠密 + 原始 BM25，不阻断查询。

## 验证与一次真实问题的发现

新增 Query 改写 JSON 契约与失败降级测试、RRF 去重/证据保留测试、FTS5 检索测试、正式索引不依赖 staging 的验证；全量回归由 `30/30` 增至 `33/33` 通过。实际数据库由 schema V2 原子迁移至 V3，迁移前备份，结果仍为 `documents=3`、`chunks=265`，BM25 能正常返回候选。

以“LSTM 风速预测论文的作者是谁？”做端到端检查时，FAISS—SQLite 路径成功执行；但 ModelScope 的最小 Chat Completions 请求返回空 `choices`（无响应 ID），改写器按契约降级。因此本次结果仅来自原始稠密路径，并偏向包含 LSTM 的实验表，而未能验证“作者”前置 metadata 的 LLM 关键词召回。这是一次重要的工程结论：必须把外部 LLM 不可用视为正常分支，并在 Prompt/HTTP 可用性与检索质量两层分别记录。

下一步应先在 ModelScope 控制台确认当前模型的推理服务权限和可调用 model ID；服务正常后，以作者、摘要、数据集、方法缩写、中英混合问题各做一组召回评测，再决定是否进入 rerank 与最终回答生成阶段。

---

## Step 6 方案收敛：四路召回与英文关键词组

在确认知识库当前以英文论文为主后，删除中文关键词 BM25 路径，避免低命中率通道增加复杂度。LLM 的 JSON 契约收敛为一个 `semantic_query` 和一个 `lexical_keywords_en` 列表；前者只做一次 Rewritten Dense Top-20，后者共同编译为一条 OR 型 Rewritten BM25 Top-20，例如 `LSTM OR "wind speed prediction" OR author`，不再把三个关键词拆成三次 BM25 调用。

最终候选来自四条固定路径：Original Dense Top-20、Original BM25 Top-10、Rewritten Dense Top-20、Rewritten BM25 Top-20。它们按 `chunk_id` 聚合去重，并采用加权 RRF 得到 Top-40；下一阶段才将这 40 条送入本地 BGE Reranker，输出 Top-5。这样先稳定并评测召回，再独立评测重排，不把两个变量混在同一次改动中。

---

## Step 6 补充：Bibliography-aware Retrieval 与规则优先路由

**问题**：参考文献中的论文名、作者名和年份具有很强的词法匹配能力，容易在 Dense/BM25 中压过正文；但它们通常不能单独回答“方法如何构建、为何有效、用了哪些 baseline”等正文问题。反过来，“有没有引用 Graph WaveNet？”或“[15] 是哪篇论文？”又必须查看参考文献条目。

**解决方案**：Chunk 在解析结构中标记为 `content` 或 `bibliography`。正文进入主 FAISS 与正文 FTS5；参考文献仅保留在 SQLite 和独立 bibliography FTS5。普通检索不触碰参考文献索引；`search_bibliography=true` 时仍先走完整正文 Hybrid Retrieval，再追加受 `bibliography_top_k` 限制的少量参考文献候选。

引用路由采用“高精度规则优先、LLM 兜底”：明确的“是否/有没有引用”“参考文献”“[15] 是哪篇”等直接为 `true`；明确的方法比较、baseline、构建机制问题直接为 `false`；其余问题才使用 Query Rewriter 的结构化判断。规则结果会覆盖 LLM 的误判，因此 LLM 失败或禁用时，明确引用问题仍可查询 bibliography FTS5。`作者为什么引用 Graph WaveNet？` 属于 `true`，但不是 bibliography-only：正文用于回答“为什么”，参考文献用于确认被引条目。

**验证**：单元测试覆盖 `Related Work → content`、`References → bibliography`、普通问题不查询 bibliography、明确引用/编号问题开启该索引、比较问题强制关闭该索引，以及 LLM 失败时的引用检索降级路径。

**进一步收敛**：对于“风速预测那篇论文的参考文献中哪些也是风速预测”这类“某篇论文 + 宽主题”问题，仅用 `wind speed prediction` 全库检索会过宽。现在先由正文四路召回按 RRF 证据聚合出目标 `paper_id`，再将 bibliography FTS5 的关键词 OR 查询限制在该论文内；无法定位目标论文时不退化为全库 bibliography 搜索。来源记录的 `bm25_bibliography.query` 同时改为真实的英文关键词串，便于检查实际检索词。

**输出队列调整**：参考文献 BM25 不再写入正文 RRF 候选池，也不使用低权重参与融合。明确引用问题中，先输出剩余名额的正文 RRF 结果，再直接追加最多 `bibliography_top_k` 条、按 bibliography BM25 排序的参考文献；`--top-k 20` 且命中 5 条时即为正文 Top-15 + 参考文献 Top-5。这样既不扰动正文排序，也不会让参考文献因 RRF 分数过低在截断前消失。

---

## Step 7：本地 Cross-Encoder Reranking

**目标**：Step 6 的 Dense/BM25/RRF 优先保证召回覆盖，但高位候选仍可能只是词语相近。Step 7 使用本地 `BAAI/bge-reranker-v2-m3` 对“完整检索问题 + 论文标题/章节/原文”成对打分，只调整正文候选的相关性顺序。

**实现与边界**：正文 RRF Top-40 进入 Cross-Encoder，默认输出正文 Top-5；每条输出保留 `pre_rerank_rank`、`rerank_score`、原 RRF `fused_score` 和全部 `source_matches`，从而可以区分召回问题与重排问题。参考文献继续是独立 bibliography BM25 队列，不输入 Cross-Encoder、不参与 RRF；明确引用问题仍以正文证据解释语义、以参考文献确认条目。模型仅从 D 盘本地目录离线加载，加载/显存/推理异常时自动回退到 Step 6 RRF 顺序。

**验证**：使用 CUDA 对一条风速预测实验正文与一条无关参考文献进行离线打分，归一化相关性分别为 `0.746212` 与 `0.002436`。单元测试覆盖正文实际重排、重排前名次保留、参考文献不参与重排和故障回退。

---

## Step 8：同节邻居扩展与受证据约束的论文问答

**问题**：Reranker 命中的 chunk 可能只包含一个方法步骤或一段结论的后半部分，直接发送给 LLM 容易丢失限定条件；粗暴拼接前后 chunk 又会跨章节，甚至混入 References。

**方案**：把 Reranker 正文结果视为种子。系统只从 SQLite 的 `chunks` 真相表读取相同 `paper_id + section` 且 `section_type='content'` 的连续 `chunk_index`，以 `neighbor_window=1` 构成相邻窗口、合并重叠窗口，并用完整 chunk 计入 token 预算。参考文献始终独立为 `R#`，不做邻居扩展。

**回答约束**：新增业务型 Prompt，分别约束方法、比较、数据集/实验和引用问题；其中 R# 只证明书目信息，引用原因必须由正文 E# 支持。回答采用 JSON Schema + Pydantic 校验，所有 `[E#]/[R#]` 都必须属于本次证据集合；模型失败、结构不合法或伪造引用时只回退到可审阅证据，绝不生成无依据结论。

**额外保护**：在无论文选择器的 CLI 中，“这篇论文/本文”若同时召回多篇论文，程序直接要求用户明确标题，而不是让 LLM 任意选择候选论文。

**结构化输出取舍（2026-08-15）**：移除了 Query Rewrite 与最终回答中的 JSON Repair Prompt 和第二次 LLM 修复调用。现在由 OpenAI-compatible API 的 `response_format=json_schema` 尽可能约束字段，再由 Pydantic 的严格模型（字段类型、必填字段、`extra="forbid"`）进行本地验证。原因是 Repair 仍是一次新的生成，可能改写原有语义、增加调用成本，并使失败行为难以复现；当结构不合法时，Query Rewrite 直接回退原始 Query，回答生成直接回退可审阅证据。新增测试确认错误 JSON 只触发一次模型调用。

**验证**：离线测试覆盖同节连续扩展、章节/References 隔离、完整组 token 预算和伪造引用回退；全量 55 项测试通过。该阶段是纯读取链路，不需要重新 parse、chunk、ingest 或重建 FAISS。

---

# v0.2 Step 2 补充：Paper Overview 的字段感知 Context Selector

**记录日期：2026-08-16；范围：仅优化临时论文工作区的 Paper Overview Context 构造，不修改 v0.1 正式知识库检索链路、标题识别、PDF 解析、Chunk、Embedding 或 FAISS。**

## 1. 问题与原始流程

Paper Overview 是对一篇临时上传论文的整篇速览，不走 Temporary FAISS / Reranker。初版流程直接读取
temporary workspace 已保存的 `parsed_paper.json` 与 `chunks.jsonl`，按 `Abstract`、`Introduction`、
`Method`、`Experiments`、`Conclusion` 等大类分别取该类最靠前的若干 chunk，合并后一次发送给 LLM：

```text
每类章节的前 N 个 chunk
→ 合并、按总长度截断
→ 一次 LLM 调用
→ Paper Overview JSON
```

该方案能够避免重新 Parse / Embedding，也能排除 `References`，但有明显的 **section prefix bias
（章节前缀偏差）**：章节长度较长时，固定选择开头块会把“章节开场说明”误当作代表证据。

在真实的 ESDTW 论文 temporary workspace 中，具体表现为：

- Introduction 的前段主要是研究背景，后部的明确 contributions 可能未入选；
- 方法章只覆盖前部，`3.1`、`3.3 Algorithm overview`、`3.4 Time complexity analysis` 等后续模块容易丢失；
- Experiments 先选到“开展了哪些实验”的介绍，而 `4.3 Sequence alignment results`、`4.5 Time series classification results`
  中的具体指标和数值未必入选；
- `4.7 Discussion` 的失败案例或限制说明没有与 Conclusion 同等的入选机会。

根因不是 Chunker 切分错误，也不是向量检索失败；而是 Overview 的上下文选择只利用了“所属大章节”和
“文件顺序”，没有利用已有的 section/subsection 结构、chunk 的章节内位置和文本中的弱证据信号。

## 2. 设计目标与约束

这次不将七个字段拆成七次 LLM 调用。仍保持一份共享 Context 与一次结构化生成，但让程序先为各字段选择
可能相关的 evidence：

```text
字段候选 evidence
→ chunk_id 合并与去重
→ 全局 token 预算筛选
→ 一份 Overview Context
→ 一次 LLM 调用
```

选择器的分数只用于“让哪些原文块有机会进入 Context”，不是论文事实，也不直接生成结论。最终结论仍由 LLM
依据所给原文生成，并接受来源 ID 校验。

## 3. 最终举措

在 [`paperbase/overview/service.py`](../paperbase/overview/service.py) 中实现了字段感知、结构感知的确定性
selector，并把全部上限集中到 [`config.yaml`](../config.yaml) 的 `paper_overview` 配置：

| Overview 字段 | 主要候选规则 |
| --- | --- |
| `research_problem` | Abstract；Introduction 前/中部；`problem`、`challenge`、`however`、`limitation` 等问题信号 |
| `contributions` | Abstract；Introduction 后部；`contribution`、`we propose`、`our work`、`in summary` 等信号 |
| `main_method` | 识别为方法章的块优先；先覆盖不同 subsection，再补同 subsection 的高分块 |
| `datasets` | Dataset/Data/Benchmark/UCR 等文本或章节信号 |
| `experimental_setup` | Setup、Baseline、Metric、Implementation、Settings 等实验设置信号 |
| `main_results` | Results/Performance/Comparison、Table/Figure、指标词与数值密度；其 token 优先级最高 |
| `limitations` | Discussion、Limitations、Failure Case、Error Analysis、Conclusion，以及 `sensitive`、`drawback`、`future work` 等明确信号 |

对于“方法章未显式写 Method，而实验为第 N 章”的常见论文结构，程序把第 `N-1` 章中尚未有明确类别的块补为
Method 候选。这是章节编号与已有 section metadata 的保守推断，不改写原始章节名或正文。

每个入选 chunk 在 Context 中保留已有元数据，并新增 selector 辅助标签：

```text
[chunk_id: paper_..._chunk_0044]
[category: experiments]
[section: 4.3. Sequence alignment results]
[overview_roles: main_results]
raw_text...
```

`overview_roles` 仅表示该块可能服务的字段，不能被解释为程序已经判定了论文结论。

## 4. Token 预算与来源边界

旧版是“每一大类固定块数”。新版采用共享的 `max_total_context_tokens`：先合并 `chunk_id` 并去重，再按
`main_method` / `main_results` 高于其他字段的优先级入预算；重复背景与低优先级块在空间不足时被跳过。Abstract
最多保留少量块，避免摘要重复占用方法和结果的容量。

预算直接复用 v0.1 Chunker 已写入 `raw_token_count`（本地 Qwen tokenizer 对 `raw_text` 的计数）。单块超过
`max_tokens_per_chunk` 时才按比例截断正文；历史工件缺失该字段时使用保守的“字符数 / 4”估算。该计数是稳定的
Context 预算近似值，不等于远端 LLM 服务端对系统 Prompt、标签和输出 JSON 的精确 token 计费。

来源字段也遵循“确定性信息不交给 LLM”的原则：LLM 对每个 Overview 字段只返回 `content` 与
`source_chunk_ids`；程序校验 ID 必须属于本次 Context，并由 `chunk_id → section` 元数据回填最终 API/JSON 的
`source_sections`。这样避免模型重复生成可由代码可靠导出的章节名称。

## 5. 真实论文验证与结果

在 `staging_d371213bfbeb45f38e9919b53cccdeaa` 的 ESDTW 真实论文上执行：

```powershell
.\.venv\Scripts\python.exe -m paperbase.overview staging_d371213bfbeb45f38e9919b53cccdeaa
```

selector 的 debug 输出显示：候选合并前有 16 个唯一 chunk，最终在 `4800` token 预算内保留 14 个、共
`4501` token，且无重复 ID。关键覆盖如下：

| 字段 | 最终代表性证据 |
| --- | --- |
| Contributions | Abstract 与 Introduction 后部 `chunk_0009` |
| Main method | 第 3 章、`3.1`、`3.3 Algorithm overview`、`3.4 Time complexity analysis` |
| Datasets / setup | `4.1 Datasets and performance measure` 与实验设置块 |
| Main results | `4.3 Sequence alignment results` 的 `chunk_0042/0044/0045`，以及 `4.5 Time series classification results` 的 `chunk_0048` |
| Limitations | `4.7 Discussion` 的 `chunk_0061` 与第 5 章 Conclusion |

相应 Overview 已能基于结果段写出 `14/18`、`46/84` 等论文明确给出的实验数字，并从 Discussion / Conclusion
提取“对局部极值、噪声和极值数量变化敏感”等明确限制。说明这次改动解决了测试样本中的章节前缀偏差，而不是只让
Context 变长。

## 6. 验证、结论与限制

- 专用测试覆盖：Introduction 后部 contribution、多个 Method subsection、含数值的 Results、Discussion
  limitation、References 排除、最终 ID 去重与全局 token 预算；`4/4` 通过。
- 全项目单元测试：`68/68` 通过。
- 真实运行仍是一次 `complete_json` 调用；Overview 只读取 staging 中的 parsed/chunks 产物，未触发 Parser、
  Embedder、Temporary Index、正式 FAISS 或 FTS5。
- 运行前后比较正式 `paperbase.sqlite3`、`paperbase.faiss`、`paperbase.faiss.manifest.json` 的 SHA-256，均无变化。

**结论：该问题已在当前真实样本和单元测试范围内解决。**新的 selector 能够主动为字段寻找跨 subsection、跨
实验结果和 Discussion 的证据，且仍保持一次 LLM 调用、可审计来源和正式 KB 隔离。

**尚有限制：**关键词和章节名规则目前以中英文常见学术写法为主；非常规标题、其他语言或结果完全只存在图片中的
论文仍可能遗漏候选，需要积累更多样本后增加评测集。`raw_token_count` 也只是 Context 预算，不是远端模型的
精确计费。此前发现的论文标题/作者单位解析异常属于 Parser 质量问题，未纳入本次 selector 修复。
