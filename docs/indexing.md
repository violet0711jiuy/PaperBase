# Step 5：全局 FAISS 向量索引

Step 5 将 Step 4 已验证的 embedding 工件写入唯一的正式全局 FAISS 索引，并把同一批
`vector_id` 回写到 SQLite。MVP 使用：

```text
IndexIDMap2(IndexFlatIP(1024))
```

`IndexFlatIP` 不需要训练，可支持以后直接追加向量；当前阶段只实现空知识库的首次全量建库，
增量上传与“Add to Knowledge Base”将在对应业务服务阶段复用相同映射契约实现。

## 正式知识库由两部分组成

```text
storage/paperbase.faiss
  └── 265 个归一化向量 + 对应整数 vector_id

storage/paperbase.sqlite3 / chunks
  └── vector_id → chunk_id → 论文、section、页码、raw_text、邻居关系
```

FAISS 不保存论文标题、页码或原文；它只负责相似度搜索与返回 ID。查询时，FAISS 返回的例如
`[17, 92, ...]` 会用于 SQLite 查询 `WHERE vector_id IN (...)`，再由程序取回真实证据。

## 建库流程

```text
vectors.npy 第 i 行
→ records.jsonl 第 i 行取得 chunk_id 与文本哈希
→ 校验 SQLite chunk 集和 embedding_text 未变化
→ 为每个 chunk 分配全局唯一 vector_id（首次为 1..N）
→ FAISS add_with_ids(vectors, vector_ids)
→ SQLite chunks.vector_id 写入相同 ID
```

`vector_id` 不等于 `chunk_index`，也不等于长期使用的 `.npy` 行号。`records.jsonl` 只在建库时把
临时行号精确映射到 `chunk_id`；一旦建库完成，正式关系就是 FAISS ID 与 SQLite `vector_id`。

## 一致性与中断恢复

FAISS 二进制文件与 SQLite 无法放进同一个文件系统事务，因此建库使用 pending journal：

1. 先构建候选 FAISS 文件并写入 journal，journal 记录完整的 `vector_id → chunk_id` 分配和索引校验和；
2. 用单个 SQLite 事务回写全部 `vector_id`；
3. 原子发布 FAISS 文件与 manifest；
4. 删除 journal。

若进程在第 2、3 步间中断，下次执行建库命令会读取 journal：SQLite 尚未提交则清理候选文件；SQLite
已完整提交则验证候选索引并完成发布。索引 manifest 还会保存索引文件哈希、维度、向量数量、embedding
输入指纹和 `vector_id` 分配指纹。任何一项不一致都会拒绝继续，而不会静默查询错配数据。

Windows 上 FAISS 原生文件路径 API 对含中文用户名的目录不稳定，当前实现使用 FAISS 标准二进制序列化
加 Python `Path` 写入，因此临时目录与正式目录均可安全包含中文字符。

## 配置与命令

```yaml
indexing:
  backend: faiss_flat_ip
  index_path: storage/paperbase.faiss
  manifest_path: storage/paperbase.faiss.manifest.json
```

首次建库：

```powershell
cd D:\AI_Workspace\projects\PaperBase
.\.venv\Scripts\python.exe -m paperbase.indexing --config .\config.yaml
```

只读交叉验证：

```powershell
.\.venv\Scripts\python.exe -m paperbase.indexing --config .\config.yaml --verify
```

验证会同时检查：FAISS 维度、向量数、真实 ID 集合，SQLite 的完整 `vector_id → chunk_id` 映射，
以及 Step 4 工件的 records/text-hash/manifest。当前三篇样本文献已建立 `265 × 1024` 索引，
`vector_id=1..265`。
