# PaperBase

PaperBase 是面向学术论文的 RAG 知识库与阅读辅助应用，支持 PDF 解析、Overview、Explain Section、Ask This Paper，以及 Classic RAG 和 Agentic Research 两条问答路径。前端使用 Streamlit，正文与 bibliography 分开存储，回答保留证据来源。

## 启动

主开发环境为 Windows / PowerShell / Python 3.10。使用项目虚拟环境；PyTorch 按 CPU/CUDA 环境安装，模型权重与论文数据在本地准备。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt

hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B
hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
docling-tools models download --output-dir models/docling

Copy-Item .env.example .env
Copy-Item config.example.yaml config.yaml
```

在 `.env` 填写自己的 provider、model 与 API key，在 `config.yaml` 设置模型路径和设备；CPU 环境把 parsing.docling.device、embedding.device、reranking.device 改为 cpu。PyTorch CUDA 安装说明见 [PyTorch](https://pytorch.org/get-started/locally/)。仓库不包含论文 PDF、权重、密钥或索引。

当前 `.env.example` 已对齐 ModelScope / Qwen/Qwen3.8-Flash-Next，填写自己的 LLM_API_KEY 即可使用。`config.example.yaml` 使用 qwen38_recommended，与当前本机 Agentic 采样一致；模型目录仍需按本机路径设置。有效参数见[配置说明](docs/configuration.md)。

```powershell
python -c "from paperbase.config import load_settings; load_settings(); print('Configuration valid')"
streamlit run app/app.py
```

Paper Workspace 可上传单篇临时论文、阅读并加入知识库；Knowledge Base 查询正式论文库。已有批量 PDF 放入 storage/papers 后，可用 `python -m paperbase.rebuild --config .\config.yaml` clean rebuild，再用 `python -m paperbase.indexing --config .\config.yaml --verify` 校验；日常单篇入库使用 UI。详见 [UI 指南](app/STREAMLIT_UI_GUIDE.md)。

## 当前架构

```text
PDF → Docling → structure-aware chunks → SQLite / FTS5 + FAISS

Classic: query rewrite → Dense / BM25 → RRF → reranker → expansion → Answer

Agentic: Scope → advisory Query Understanding → Candidates → Requirement Plan
→ Decision → shared Guard preflight / formal Guard → Tools → Admission → Registry
→ Evidence-local multi-projection / Integrity → Decision loop
→ Selector → Answer semantic status → deterministic Coverage Compiler
```

Canonical Search compiler 是有效请求与 identity 的唯一权威；Guard 裁决合法性和重复请求。Projector 提取结构语义，Integrity 建立 verified support。Answer LLM 判断有支持 Requirement 的 complete/partial；程序构造 unsupported、引用、coverage IDs、overall 与 flags。RequirementState.SATISFIED 仅表示已有局部支持。

完整定义以唯一的[当前 Agentic 架构](docs/architecture/agentic-research.md)为准；[字段事实源](docs/field-source-of-truth.md)列出跨层职责。当前 checkpoint authority 为 `projector-requirement-directed-multi-projection-v4`；旧版本可离线解析，但不能 resume。

## 当前配置与验证

本地 profile 是 qwen38_recommended：Decision thinking=true / T=1；其他 Agentic 节点 thinking=false / T=0.7。configured profile 仍支持独立节点温度。Agentic 不发送应用层输出 cap；共享 .env 与 Classic 的输出预算保持独立。旧 cap 不再出现在推荐 YAML，保留解析兼容。

最近 method_002 单次为 complete，3 iterations、2 Tools、0 duplicate / invalid / Schema failures；当前 profile 尚未完成固定 8-case stability suite 或正式 Agentic 40-case。E1 是来源与控制契约的确定性诊断，E2 语义质量 Judge 尚未实现。见[当前基线](docs/evaluation/current-baseline.md)及[历史实验](docs/archive/evaluation-baselines-20261006.md)。

## 测试与评测

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe eval\scripts\validate_golden_dataset.py
```

冻结 Golden v1.2 为 40 cases；Validator 需要对应正式 SQLite 与论文 chunk。E1 离线 replay 读取已有结果，输出放到独立目录，不调用 provider：

```powershell
.\.venv\Scripts\python.exe -m paperbase.evaluation.run_agentic_e1 --source output/qwen_recommended_method002_20261006/run --case-id method_002 --output eval/results/e1_latest_replay
```

Classic Retrieval benchmark 使用 `eval/scripts/run_retrieval_eval.py`，需要论文、索引、本地模型和 provider；Agentic 真实评测使用 E1 CLI 的 `--live`。完整参数、固定 smoke 和运行记录见 [eval README](eval/README.md) 与 [E1 契约](docs/evaluation/e1_diagnostic.md)。Answer 原始最终输出由 case runner 保存到 cases/<case_id>.answer_llm_outputs.jsonl；其用途是本地排错，不参与公共审计评分。

## 目录

```text
app/                    Streamlit UI 与 Service
paperbase/agentic/       Graph、Plan、Decision、Guard、Registry、Projector、Selector
paperbase/generation/   Answer 与 Coverage Compiler
paperbase/retrieval/    Canonical requests、Hybrid Retrieval、Tool result contracts
paperbase/evaluation/   E1 evaluator 与 live / offline runners
paperbase/              parsing / chunking / embedding / indexing / llm / reading services
tests/                  当前契约、兼容与历史 RCA 回归
eval/                   冻结 Golden、审核记录、Classic benchmark 脚本
docs/                   当前架构、配置、评测；archive/ 保存历史基线
storage/                本地论文、索引、checkpoint、evaluation
output/                 本地诊断与实验快照
config.example.yaml     当前 YAML 模板
.env.example            共享 provider / client 模板
```

[文档索引](docs/index.md)连接安装、配置、存储、检索和评测说明。历史工件保留用于回归与审计，不作为当前实现权威。
