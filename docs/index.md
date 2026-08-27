# PaperBase 文档索引

本目录只保存与当前代码、公开使用和后续开发直接相关的文档。历史验收记录、一次性排障笔记和完整工程日志保存在本地 `docs/_local/`，不提交到 Git。

## 使用与配置

- [项目 README](../README.md)：项目能力、架构、安装和快速启动。
- [配置说明](configuration.md)：如何从模板创建本地配置，以及各配置组的职责。

## 当前系统设计

- [解析与分块](pipeline/parsing-and-chunking.md)：PDF 解析、章节层级、前置元数据、Bibliography 分类和 Chunk 契约。
- [存储与索引](pipeline/storage-and-indexing.md)：SQLite schema v5、FTS5、Embedding、FAISS、一致性检查和增量 Promotion。
- [检索与生成](pipeline/retrieval-and-generation.md)：Query Planner、三路正文召回、Bibliography 路由、RRF、Reranker、Evidence Expansion 和回答生成。

## Evaluation

- [Evaluation Design](evaluation/design.md)：Golden Schema、指标定义、切片、失败分析和后续评测规划。
- [Current Baseline](evaluation/current-baseline.md)：当前冻结数据集、Retrieval Baseline 和 Expansion-aware 指标。

## Roadmap

- [Multi-paper RAG Roadmap](roadmap/multi-paper-rag.md)：多论文比较、发现和全库归纳的后续计划。

## 文档维护规则

1. 正式文档只描述当前实现，未来设计必须明确标为 `PLANNED`。
2. 不在正式文档中固化本机绝对路径、API Key、论文原文或本地数据库统计。
3. 当前配置值以 [`config.example.yaml`](../config.example.yaml) 为准；文档只解释语义，不复制完整配置。
4. 可重复生成的 Evaluation 输出保存在 `eval/results/`，不提交到 Git。
5. 已解决问题应由测试和 Git 历史追溯，不长期保留为当前问题文档。
