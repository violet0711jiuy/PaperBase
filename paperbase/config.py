"""PaperBase 的集中式非敏感配置加载与校验。

本模块只读取项目内的 YAML 配置文件。API Key、访问 Token 等敏感信息不属于这里，
后续生成模型接入时应从未提交的 ``.env`` 文件读取。配置模型采用严格校验：拼写错误
或尚未实现的参数会在启动时立即报错，避免程序静默使用默认值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationInfo, field_validator


class ConfigurationError(ValueError):
    """配置文件缺失、格式错误或不符合 PaperBase 配置契约时抛出的异常。"""


class ProjectSettings(BaseModel):
    """项目级标识配置。后续可在此扩展版本、运行环境等非敏感参数。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class StorageSettings(BaseModel):
    """PaperBase 自己管理的本地数据目录。"""

    model_config = ConfigDict(extra="forbid")

    papers_dir: Path
    parsed_dir: Path
    staging_dir: Path

    @field_validator("papers_dir", "parsed_dir", "staging_dir", mode="after")
    @classmethod
    def resolve_storage_path(cls, value: Path, info: ValidationInfo) -> Path:
        """将相对路径统一解析为相对 ``config.yaml`` 所在目录的绝对路径。"""
        return _resolve_path(value, info)


class DoclingSettings(BaseModel):
    """Docling 适配器当前支持的全部非敏感参数。"""

    model_config = ConfigDict(extra="forbid")

    artifacts_path: Path
    device: str = "cuda"
    enable_ocr: bool = False
    enable_formula_enrichment: bool = True
    formula_preset: str = "granite_docling"
    compile_layout_model: bool = False
    compile_formula_model: bool = False
    remove_page_furniture: bool = True
    remove_peer_review_artifacts: bool = True
    list_style_heading_min_chars: int = Field(ge=32, le=512)

    @field_validator("artifacts_path", mode="after")
    @classmethod
    def resolve_artifacts_path(cls, value: Path, info: ValidationInfo) -> Path:
        """允许模型目录使用相对或绝对路径，但最终始终交给解析器绝对路径。"""
        return _resolve_path(value, info)


class FrontMatterSettings(BaseModel):
    """所有解析器共享的论文前置元数据识别参数。"""

    model_config = ConfigDict(extra="forbid")

    # 此开关属于统一 ParsedPaper 契约，而不是某个 PDF 工具的私有选项。未来 MinerU
    # 适配器接入后，也必须依据相同语义输出 front_matter。
    enabled: bool = True
    max_pages: int = Field(default=2, ge=1, le=4)


class ParsingSettings(BaseModel):
    """所有解析器共享的选择项，以及当前 Docling 的专属配置。"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(min_length=1)
    inspection_output_dir: Path
    front_matter: FrontMatterSettings = Field(default_factory=FrontMatterSettings)
    docling: DoclingSettings

    @field_validator("inspection_output_dir", mode="after")
    @classmethod
    def resolve_inspection_output_dir(
        cls, value: Path, info: ValidationInfo
    ) -> Path:
        """使检查产物目录始终相对项目配置文件解析。"""
        return _resolve_path(value, info)


class ChunkingSettings(BaseModel):
    """所有分块器共享的运行参数，以及当前 Docling Hybrid 实现的专属参数。"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(min_length=1)
    max_tokens: int = Field(ge=64, le=8192)
    embedding_metadata_reserve_tokens: int = Field(ge=16, le=512)
    tokenizer_path: Path
    inspection_output_dir: Path
    repeat_table_header: bool = True
    merge_peers: bool = True

    @field_validator("tokenizer_path", "inspection_output_dir", mode="after")
    @classmethod
    def resolve_chunking_path(cls, value: Path, info: ValidationInfo) -> Path:
        """将 tokenizer 与检查产物目录统一解析为绝对路径。"""
        return _resolve_path(value, info)


class DatabaseSettings(BaseModel):
    """SQLite 元数据层的非敏感连接参数。"""

    model_config = ConfigDict(extra="forbid")

    path: Path
    busy_timeout_ms: int = Field(default=5_000, ge=0, le=60_000)

    @field_validator("path", mode="after")
    @classmethod
    def resolve_database_path(cls, value: Path, info: ValidationInfo) -> Path:
        """数据库文件可使用相对路径，但运行时始终使用项目内绝对路径。"""
        return _resolve_path(value, info)


class EmbeddingSettings(BaseModel):
    """Step 4 生成文档向量的非敏感、可替换运行参数。"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_path: Path
    device: str = Field(min_length=1)
    batch_size: int = Field(ge=1, le=512)
    normalize_embeddings: bool = True
    output_dir: Path

    @field_validator("model_path", "output_dir", mode="after")
    @classmethod
    def resolve_embedding_path(cls, value: Path, info: ValidationInfo) -> Path:
        """模型路径与 staging 输出目录均支持相对/绝对配置，但交给运行层的必须是绝对路径。"""
        return _resolve_path(value, info)


class IndexingSettings(BaseModel):
    """Step 5 正式向量索引的非敏感、可替换配置。"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(min_length=1)
    index_path: Path
    manifest_path: Path

    @field_validator("index_path", "manifest_path", mode="after")
    @classmethod
    def resolve_index_path(cls, value: Path, info: ValidationInfo) -> Path:
        """索引与清单文件均可相对 config.yaml 指定，运行时统一为绝对路径。"""
        return _resolve_path(value, info)


class QueryRewriteSettings(BaseModel):
    """LLM Query 改写的边界参数；密钥与模型地址仍只从 ``.env`` 读取。

    一次 LLM 调用只产生一条语义改写问题和一组英文关键词；关键词组会作为一条
    Rewritten BM25 路径执行，而非拆成多条独立 BM25 路径。
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_lexical_keywords_en: int = Field(default=3, ge=1, le=5)
    # 只保留近期检索对话，避免较早历史覆盖当前用户问题；0 表示禁用历史上下文。
    max_context_turns: int = Field(default=4, ge=0, le=16)
    # 引用意图命中后，bibliography FTS5 最多提供的辅助候选数；不参与主 FAISS。
    bibliography_top_k: int = Field(default=5, ge=1, le=20)
    fallback_to_original: bool = True


class RetrievalSettings(BaseModel):
    """Step 6 混合召回与 RRF 融合的全部非敏感、可调参数。"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(min_length=1)
    dense_top_k_per_query: int = Field(default=20, ge=1, le=200)
    bm25_original_top_k: int = Field(default=10, ge=1, le=200)
    bm25_rewrite_top_k: int = Field(default=20, ge=1, le=200)
    fused_top_k: int = Field(default=40, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=1_000)
    dense_original_weight: float = Field(default=1.0, gt=0, le=10)
    dense_rewrite_weight: float = Field(default=1.0, gt=0, le=10)
    bm25_original_weight: float = Field(default=0.7, gt=0, le=10)
    bm25_rewrite_weight: float = Field(default=1.0, gt=0, le=10)
    query_instruction: str = Field(min_length=20, max_length=1_000)
    query_rewrite: QueryRewriteSettings = Field(default_factory=QueryRewriteSettings)


class RerankingSettings(BaseModel):
    """Step 7 本地 Cross-Encoder 重排序的非敏感、可替换运行参数。"""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(min_length=1)
    enabled: bool = True
    model_id: str = Field(min_length=1)
    model_path: Path
    device: str = Field(min_length=1)
    batch_size: int = Field(ge=1, le=128)
    candidate_top_k: int = Field(ge=1, le=200)
    final_top_k: int = Field(ge=1, le=50)
    max_length: int = Field(ge=64, le=8192)
    normalize_scores: bool = True

    @field_validator("model_path", mode="after")
    @classmethod
    def resolve_reranker_model_path(cls, value: Path, info: ValidationInfo) -> Path:
        """将本地 reranker 模型目录统一解析为绝对路径，避免运行时隐式联网下载。"""
        return _resolve_path(value, info)


class ContextExpansionSettings(BaseModel):
    """Step 8 按论文与章节边界扩展相邻证据块的可调参数。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    neighbor_window: int = Field(default=1, ge=0, le=8)
    max_total_tokens: int = Field(default=6_000, ge=256, le=32_000)


class AnswerGenerationSettings(BaseModel):
    """Step 8 基于已检索证据生成答案的非敏感运行参数。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_evidence_units: int = Field(default=8, ge=1, le=30)


class PaperOverviewSettings(BaseModel):
    """v0.2 单篇论文速览的章节级上下文预算。"""

    model_config = ConfigDict(extra="forbid")

    # 每类核心章节最多选取的 chunk 数，防止 Introduction 或 Experiments 独占 Prompt。
    max_chunks_per_section: int = Field(default=3, ge=1, le=10)
    # 一个 chunk 写入 Prompt 时允许保留的最大字符数，超出时只截取前段并显式标注。
    max_chars_per_chunk: int = Field(default=2_000, ge=300, le=12_000)
    # 所有章节内容的总字符预算；标题与 Prompt 固定说明不计入此预算。
    max_total_context_chars: int = Field(default=18_000, ge=2_000, le=80_000)
    # 无法匹配核心章节名称时，最多保留多少个前序正文 chunk 作为受控兜底。
    max_fallback_chunks: int = Field(default=2, ge=0, le=10)
    # Overview 有八个带来源字段，所需 JSON 通常比 Query Rewrite 的全局默认输出更长。
    max_output_tokens: int = Field(default=1_600, ge=400, le=8_000)


class AppSettings(BaseModel):
    """PaperBase 当前阶段的完整、解析器无关配置根对象。"""

    model_config = ConfigDict(extra="forbid")

    project: ProjectSettings
    storage: StorageSettings
    parsing: ParsingSettings
    chunking: ChunkingSettings
    database: DatabaseSettings
    embedding: EmbeddingSettings
    indexing: IndexingSettings
    reranking: RerankingSettings
    retrieval: RetrievalSettings
    context_expansion: ContextExpansionSettings = Field(
        default_factory=ContextExpansionSettings
    )
    answer_generation: AnswerGenerationSettings = Field(
        default_factory=AnswerGenerationSettings
    )
    paper_overview: PaperOverviewSettings = Field(
        default_factory=PaperOverviewSettings
    )

    # 配置文件路径不是用户在 YAML 中填写的业务参数，因此作为 Pydantic 私有属性保存。
    _config_path: Path = PrivateAttr()

    @property
    def config_path(self) -> Path:
        """返回实际加载的配置文件绝对路径，供日志、诊断报告和错误提示使用。"""
        return self._config_path


def default_config_path() -> Path:
    """返回仓库默认 ``config.yaml`` 的绝对路径，不在 import 时读取文件。"""
    return Path(__file__).resolve().parents[1] / "config.yaml"


def load_settings(config_path: Path | str | None = None) -> AppSettings:
    """读取 YAML 并返回经过路径解析与类型校验的应用配置。

    不创建目录、不下载模型，也不读取环境变量；因此可以安全地在 CLI、测试与未来的
    Streamlit 启动代码中复用。路径值中的 ``~`` 会展开，而相对路径以配置文件目录为准。
    """
    resolved_config_path = Path(config_path or default_config_path()).expanduser()
    resolved_config_path = resolved_config_path.resolve()
    if not resolved_config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {resolved_config_path}")

    try:
        raw_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML configuration: {resolved_config_path}"
        ) from error

    if not isinstance(raw_config, dict):
        raise ConfigurationError(
            "Configuration root must be a YAML mapping, for example 'parsing: ...'."
        )

    try:
        settings = AppSettings.model_validate(
            raw_config,
            context={"config_dir": resolved_config_path.parent},
        )
    except ValueError as error:
        raise ConfigurationError(
            f"Invalid PaperBase configuration: {resolved_config_path}\n{error}"
        ) from error

    settings._config_path = resolved_config_path
    return settings


def _resolve_path(value: Path, info: ValidationInfo) -> Path:
    """按加载器提供的配置目录解析一个 YAML 路径字段。"""
    candidate = value.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    config_dir = (info.context or {}).get("config_dir")
    if not isinstance(config_dir, Path):
        # 该分支主要保护未来有人直接构造嵌套配置模型的情况；常规 load_settings 调用
        # 始终提供 config_dir，因此不会触发。
        raise ConfigurationError("Relative paths require loading through load_settings().")
    return (config_dir / candidate).resolve()
