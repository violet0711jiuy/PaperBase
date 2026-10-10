# PaperBase 文档索引

当前实现说明核对日期：2026-10-06。实现有变更时，先核对代码和真实调用，再更新文档。

| 文档 | 用途 |
|---|---|
| [README](../README.md) | 项目入口、安装与运行 |
| [UI 阅读指南](../app/STREAMLIT_UI_GUIDE.md) | 问答模式、Service 接线与恢复入口 |
| [Agentic Research 当前架构](architecture/agentic-research.md) | LangGraph 执行链、接口语义、恢复和 Answer 边界 |
| [字段事实源](field-source-of-truth.md) | 唯一权威与派生关系 |
| [配置](configuration.md) | 生效的节点参数、配置兼容与预算含义 |
| [解析与分块](pipeline/parsing-and-chunking.md) | PDF 到可溯源文本 |
| [存储与索引](pipeline/storage-and-indexing.md) | SQLite、FAISS 和增量入库 |
| [检索与生成](pipeline/retrieval-and-generation.md) | Classic 与 Agentic 两条生产路径 |
| [评测设计](evaluation/design.md) | Retrieval benchmark、E1 与待实现 E2 的区别 |
| [当前验证状态](evaluation/current-baseline.md) | 当前 model/profile/v4/E1/Golden 与最后一次 smoke |
| [历史基线归档](archive/evaluation-baselines-20261006.md) | 独立模型实验、旧 8-case 与 Classic benchmark 快照 |
| [E1 确定性诊断契约](evaluation/e1_diagnostic.md) | 冻结 evaluator、历史兼容与运行入口 |
| [评测数据与脚本](../eval/README.md) | Golden Dataset 和 Retrieval 工具 |

旧 Agentic 实施路线图与重复的 evaluation_rules 已移除，其中的架构已被生产接口替代。阶段性诊断、旧验收记录及个人复习资料已移到本地 `output/documentation_cleanup_20261006/historical/`，用于历史审计，不作为当前实现依据。清理清单及删除前副本位于同一输出目录；本地工件不随仓库发布。

维护规则：当前文档只描述已有行为；未实现能力明确标为待实现。测试通过、局部 support、筛选完成和 Answer 语义完整性分别表述。历史数值注明所测路径与快照，不把离线 replay 当成当前 live 验证。文档不保存密钥、私有机器绝对路径或隐藏推理。
