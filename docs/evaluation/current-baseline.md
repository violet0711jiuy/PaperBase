# 当前验证基线

核对日期：2026-10-10。本页记录当前配置与最后一次真实验证；[历史采样实验、8-case 与 Classic benchmark](../archive/evaluation-baselines-20261006.md) 单独归档。

| 项目 | 当前值 |
|---|---|
| Provider / model | ModelScope，openai_compatible；Qwen/Qwen3.8-Flash-Next |
| Sampling profile | qwen38_recommended |
| Decision | 实际请求 thinking=false；temperature=0.7；top_p=0.8；presence_penalty=1.5（qwen38_recommended 覆盖配置中的 decision_temperature=1） |
| Decision transport | timeout=60 秒；SDK max_retries=0；本轮实际 timeout=0 |
| 其他 Agentic 节点 | thinking=false；temperature=0.7；top_p=0.8；presence_penalty=1.5 |
| 共同采样参数 | top_k=20；min_p=0；repetition_penalty=1 |
| 输出契约 | Decision / Planner / Projector / Answer：json_schema、strict=true；QU / Candidate：json_object；Agentic 不发送应用层输出 token cap |
| Checkpoint authority | projector-requirement-directed-multi-projection-v4 |
| Evaluator | agentic-eval-v2，确定性 E1 diagnostics；E2 尚未实现 |
| Golden | eval/datasets/golden_dataset_v1_2.jsonl，40 cases；SHA256 05eb18b529073ac71930ef42cee8a7527fcba57c82f48b80e070bc04602a092b |

最近真实 smoke 是 **2026-10-10 atomic gate1，method_002 单次，FAIL**。使用 `.venv` 和隔离的 checkpoint / Projector cache；5 iterations、5 Decision calls、4 Tools、12 次 SDK create 均成功，stop=agent_finished，overall=partial。C_i（R2）在第 2 轮建立真实 source-span verified support，之后不再搜索 R2；MPIWcapt（R1）仍 unsupported，未达到 complete 验收。

Golden recall=100%（3/3），Span validation=100%（1/1 support candidate），Projector repair=0/4，Decision timeout / duplicate / Schema repair / 原样 timeout retry 均为 0。实际 Decision 搜索项每项仅绑定一个 Requirement，原始 LLM lexical_terms 为 4–5 个非空关键词，输出没有 evidence_targets，也没有 satisfied/support_found 状态暗示。Decision view chars 为 2238 → 3569 → 5660 → 6335 → 6486，最大相邻增长 1.595 倍，后两轮增长收敛；单个 case 不能证明所有 case 的输入全局有界。

本轮耗时 117.2 秒，总 prompt/completion/total tokens 为 47427 / 2948 / 50375，未知 usage calls=0；Decision 为 11680 / 599 / 12279。首轮 Projector 已生成 MPIWcapt 机制文字，却没有提出 R1 support candidate 或 supported_requirement_id；后续改写检索未补齐支持。原始 Answer 仅声明 R2 complete，最终 R1 unsupported / overall partial，不能因召回全部 Golden 或正文提到 MPIWcapt 而算作通过。

按失败即停门禁，第一阶段剩余 synthesis_001 / bibliography_001 / unanswerable_001、第二阶段四个回归 case 和两轮额外重复样本均未运行。当前配置未通过固定 8-case smoke，尚无三次稳定性结论。本次没有 E2 Judge，也没有运行 Classic Retrieval 40-case 或 Agentic 40-case；本轮未重跑 pytest，历史测试数量不作为当前验证结果。

本地 [本轮报告与失败定位](../../output/decision_atomic_smoke_20261010/report.md)、[逐项门禁](../../output/decision_atomic_gate1_20261010/cases/method_002.gate.json)、[实际 SDK 请求回执](../../output/decision_atomic_gate1_20261010/llm_call_receipts.jsonl) 和 [run manifest](../../output/decision_atomic_gate1_20261010/run_manifest.json) 保存真实结果。之前的单次 complete 记录见 [2026-10-06 smoke](../../output/qwen_recommended_method002_20261006/report.md)，不能替代当前验收。更新前文档另存于 [baseline_before.md](../../output/decision_atomic_smoke_20261010/baseline_before.md)。output 工件不随仓库发布。

采样覆盖、configured fallback、共享 .env 与兼容参数见[配置](../configuration.md)；完整职责链见唯一的[当前架构](../architecture/agentic-research.md)。
