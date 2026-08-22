"""所有 Streamlit 页面使用的 UI → backend 适配器。

这里把后端对象转换成适合页面展示的视图模型，避免页面直接接触 SQLite 行、
FAISS 对象或 staging JSON。Knowledge Base 页面通过这里创建/切换会话、调用正式
问答链路并恢复 Evidence；Paper Workspace 的 Overview、Explain Section 和 Ask
This Paper 都通过这里适配现有后端流程。
"""

# 延迟解析类型标注。
from __future__ import annotations

# ``dataclass`` 用来定义页面需要的不可变视图模型。
from dataclasses import dataclass
# ``hashlib`` 用于定位后端 Explain artifact 的稳定文件名。
import hashlib
# ``json`` 用来读取受控的 workspace manifest 和解析标题文件。
import json
# ``Path`` 用来安全地拼接、校验和读取项目内路径。
from pathlib import Path
# 这里的 SQLite 连接只用于正式库论文元数据展示，不加载 FAISS 或检索器。
import sqlite3
# ``TemporaryDirectory`` 用于保存网页上传的短生命周期副本；真正的 staging
# workspace 仍然只由后端写入配置指定的 storage/staging。
from tempfile import TemporaryDirectory

# AppSettings 是后端统一配置模型，load_settings 读取项目 config.yaml。
from paperbase.config import AppSettings, load_settings
# 直接复用 staging 论文问答服务，不在前端复制检索或答案生成逻辑。
from paperbase.ask_paper import AskThisPaperResult, create_ask_this_paper_service
# ConversationStore 负责持久化 conversation，ConversationRecord 是轻量元数据对象。
from paperbase.conversations import (
    FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
    ConversationRecord,
    ConversationStore,
    ConversationTurn,
)
from paperbase.generation import AnswerServiceResult, create_answer_service
# MetadataDatabase 复用后端已有的数据库初始化和单论文读取能力。
from paperbase.database import MetadataDatabase
from paperbase.overview.service import (
    PaperOverview,
    PaperOverviewError,
    run_paper_overview_stage,
)
from paperbase.explain_section.service import (
    ExplainSection,
    ExplainSectionError,
    run_explain_section_stage,
)
# Promotion 负责把 staging 中已经完成的论文工件安全增量加入正式知识库。
from paperbase.promotion.service import PromotionError, PromotionResult, promote_workspace
# 复用后端现有 Section Tree / chunk 只读仓库，不自行解析标题编号。
from paperbase.staging.sections import WorkspaceSectionRepository
# 复用完整 staging pipeline，不在前端重复 Parser、Chunker 或临时索引流程。
from paperbase.staging.service import (
    TemporaryWorkspaceError,
    delete_temporary_workspace,
    run_temporary_workspace_stage,
)


@dataclass(frozen=True)
class KnowledgeBasePaper:
    """正式知识库论文在列表页面中的最小展示模型。"""

    # 后端生成的稳定论文 ID。
    paper_id: str
    # 可靠的论文标题；没有时为 None，页面会回退到文件名。
    title: str | None
    # 入库时保存的源 PDF 文件名。
    source_filename: str
    # 论文的 chunk 总数量，用于页面统计展示。
    total_chunk_count: int

    @property
    def display_title(self) -> str:
        """返回页面应显示的标题，标题缺失时回退到文件名。"""
        # ``or`` 可以同时处理 None 和空字符串两种没有标题的情况。
        return self.title or self.source_filename


@dataclass(frozen=True)
class KnowledgeBaseConversation:
    """Knowledge Base 会话列表中的轻量摘要，不把完整历史放进页面状态。"""

    conversation_id: str
    title: str
    turn_count: int
    updated_at: str


@dataclass(frozen=True)
class PaperConversation:
    """当前 staging workspace 下的独立会话摘要。"""

    conversation_id: str
    title: str
    turn_count: int
    updated_at: str


@dataclass(frozen=True)
class KnowledgeBaseEvidence:
    """只保留面向用户的 Evidence 字段，隐藏 chunk、向量和检索分数。"""

    evidence_id: str
    kind: str
    paper_title: str
    section: str | None
    page_start: int | None
    page_end: int | None
    text: str


@dataclass(frozen=True)
class KnowledgeBaseTurnMetadata:
    """答案状态的用户可读子集，用于显示证据不足或覆盖范围提示。"""

    answer_status: str
    insufficient_evidence: bool
    partial_answer: bool
    coverage_note: str | None


@dataclass(frozen=True)
class WorkspaceSummary:
    """一个 staging workspace 在 Paper Workspace 页面中的摘要模型。"""

    # workspace 目录名，例如 ``staging_xxx``。
    workspace_id: str
    # 该工作区对应的稳定论文 ID。
    paper_id: str
    # 用户上传时的原始 PDF 文件名。
    source_filename: str
    # 解析阶段发现的论文标题；无标题时为 None。
    title: str | None
    # workspace manifest 中记录的全部 chunk 数。
    total_chunk_count: int
    # parsed_paper.json 中稳定解析出的章节数量。
    section_count: int
    # 是否已经通过 promotion 加入正式 Knowledge Base。
    added_to_kb: bool

    @property
    def display_title(self) -> str:
        """返回工作区页面使用的标题，缺失时回退到源文件名。"""
        # 这样页面永远有一个可识别的标题。
        return self.title or self.source_filename


@dataclass(frozen=True)
class PaperOverviewSource:
    """论文概览字段对应的可读来源，不向页面暴露 chunk 内部 ID。"""

    section: str | None
    page_start: int | None
    page_end: int | None
    text: str


class PaperOverviewArtifactError(RuntimeError):
    """概览工件缺失以外的读取错误，供页面显示可读的恢复提示。"""


class PaperUploadError(RuntimeError):
    """网页上传无法创建 staging workspace 时的用户可读错误。"""


class WorkspaceActionError(RuntimeError):
    """加入知识库或删除临时论文失败时的用户可读错误。"""


@dataclass(frozen=True)
class SectionSummary:
    """Section Tree 中供 UI 使用的最小章节模型。"""

    section_id: str
    title: str
    level: int
    parent_section_id: str | None
    page_start: int | None
    page_end: int | None
    has_children: bool
    has_content: bool


@dataclass(frozen=True)
class SectionExplanationView:
    """Explain Section 结构化结果的 UI 视图，不暴露 context/debug 字段。"""

    section_id: str
    section_title: str
    mode: str
    explanation: str
    key_points: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    insufficient_evidence: bool


@dataclass(frozen=True)
class SectionExplanationSource:
    """Explain 来源的用户可读字段。"""

    section: str | None
    page_start: int | None
    page_end: int | None
    text: str


class SectionExplanationArtifactError(RuntimeError):
    """Explain artifact 缺失以外的读取错误。"""


class PaperBaseService:
    """把 Streamlit 页面与 PaperBase 后端实现隔离开的窄适配层。"""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        answer_service: object | None = None,
    ) -> None:
        """初始化配置、正式元数据数据库和独立会话存储。"""
        # 测试可以传入临时 settings；正常运行时从默认 config.yaml 读取。
        self._settings = settings or load_settings()
        # 创建后端 MetadataDatabase，页面不需要知道 SQLite 文件的具体细节。
        self._database = MetadataDatabase(
            # 使用配置中的正式论文元数据数据库路径。
            self._settings.database.path,
            # 沿用后端配置的 SQLite 锁等待时间。
            busy_timeout_ms=self._settings.database.busy_timeout_ms,
        )
        # 会话数据库与正式论文数据库分离，符合 ConversationStore 的设计。
        self._conversations = ConversationStore(
            # 使用配置中的 conversations.sqlite3 路径。
            self._settings.conversation.path,
            # 同样沿用配置中的并发等待时间。
            busy_timeout_ms=self._settings.conversation.busy_timeout_ms,
        )
        # 正式问答链按需装配；页面只在用户真正发送问题时加载模型和索引。
        # 测试可注入一个替身，避免初始化真实 embedding/FAISS/LLM。
        self._answer_service = answer_service
        # KB 回答的证据快照由 Service 持久化，供 rerun 后恢复历史 Evidence。
        # 目录位于 storage 下的独立区域，不与 staging workspace 混用。
        self._answer_artifact_dir = self._settings.storage.papers_dir.parent / "knowledge_base_answers"
        # Ask This Paper 服务按 workspace 延迟装配并缓存；浏览 Overview/Explain 时不加载
        # embedding、BM25、reranker 或模型，真正提交问题时才初始化对应论文链路。
        self._paper_ask_services: dict[str, object] = {}

    def list_knowledge_base_papers(self) -> tuple[KnowledgeBasePaper, ...]:
        """返回正式 Knowledge Base 的论文元数据，不访问向量或检索链路。"""
        # 先确保正式数据库的元数据表存在，并完成后端自己的 schema 校验。
        self._database.initialize()
        # MetadataDatabase 当前没有批量列出 documents 的接口。
        # 这里将只读的展示查询集中放在 Service，页面本身不会直接连接 SQLite。
        with sqlite3.connect(self._database.path) as connection:
            # 一次查询同时拿到论文基本字段和它的 chunk 数量。
            rows = connection.execute(
                # documents 是论文主表，chunks 用于统计每篇论文的分块数。
                """
                SELECT documents.paper_id, documents.paper_title, documents.source_filename,
                       COUNT(chunks.chunk_id) AS total_chunk_count
                FROM documents
                LEFT JOIN chunks ON chunks.paper_id = documents.paper_id
                GROUP BY documents.paper_id
                ORDER BY documents.updated_at DESC, documents.paper_id
                """
            ).fetchall()
        # 把数据库行转换成页面使用的不可变视图模型。
        return tuple(
            # 每个元组元素对应一篇正式入库论文。
            KnowledgeBasePaper(
                # SQLite 返回的值显式转换为普通 Python 字符串。
                paper_id=str(row[0]),
                # 标题通过辅助函数清理空白，并保留 None 语义。
                title=_optional_text(row[1]),
                # 文件名是正式库记录中的源文件名。
                source_filename=str(row[2]),
                # COUNT 结果转换为 int，供 Streamlit metric 使用。
                total_chunk_count=int(row[3]),
            )
            # 遍历 SQL 查询返回的所有行。
            for row in rows
        )

    def list_workspaces(self) -> tuple[WorkspaceSummary, ...]:
        """扫描 staging 根目录的直接子目录，并返回有效 workspace 摘要。"""
        # 读取配置中的 staging 根目录。
        staging_dir = self._settings.storage.staging_dir
        # staging 目录不存在时返回空元组，让页面显示空状态。
        if not staging_dir.is_dir():
            return ()
        # 临时列表用于收集所有通过 manifest 校验的工作区。
        workspaces: list[WorkspaceSummary] = []
        # 只扫描 staging 根目录的直接子目录，不递归进入内部产物目录。
        for root in staging_dir.iterdir():
            # 只处理目录，并要求目录名以 staging_ 开头。
            if not root.is_dir() or not root.name.startswith("staging_"):
                continue
            # 读取并校验 workspace.json；无效工作区由辅助方法返回 None。
            workspace = self._read_workspace(root)
            # 只有有效 manifest 才加入返回结果。
            if workspace is not None:
                workspaces.append(workspace)
        # 按 workspace ID 倒序返回，通常可以让较新的 UUID 工作区排在前面。
        return tuple(sorted(workspaces, key=lambda item: item.workspace_id, reverse=True))

    def create_workspace_from_pdf_upload(
        self, *, filename: str, content: bytes
    ) -> WorkspaceSummary:
        """把上传的单个 PDF 交给既有 staging pipeline，并返回新 workspace 摘要。

        Streamlit 的 UploadedFile 只在本次请求内存在。这里先在系统临时目录保存一
        份受控副本，再调用 ``paperbase.staging`` 已有入口；页面不会自行创建
        staging 目录、解析 PDF 或写入向量/FAISS 产物。
        """
        safe_filename = _safe_pdf_filename(filename)
        if not content:
            raise PaperUploadError("上传的 PDF 文件为空，请重新选择文件。")
        # PDF 规范允许文件头前存在少量字节；在前 1 KiB 内检查签名只是快速
        # 输入校验，最终能否解析仍完全由后端 parser 决定。
        if b"%PDF-" not in content[:1024]:
            raise PaperUploadError("文件看起来不是有效的 PDF，请重新选择文件。")

        try:
            with TemporaryDirectory(prefix="paperbase-upload-") as temporary_root:
                source_pdf = Path(temporary_root) / safe_filename
                source_pdf.write_bytes(content)
                workspace = run_temporary_workspace_stage(
                    settings=self._settings, source_pdf=source_pdf
                )
        except TemporaryWorkspaceError as error:
            raise PaperUploadError(str(error)) from error
        except OSError as error:
            raise PaperUploadError("上传文件暂时无法保存或读取，请稍后重试。") from error
        except Exception as error:  # noqa: BLE001 - 不把 parser/model traceback 直接交给 UI。
            raise PaperUploadError("论文解析或索引创建失败，请检查后端模型配置后重试。") from error

        summary = self.get_workspace(workspace.workspace_id)
        if summary is None:
            # staging stage 成功时按契约一定会发布 workspace.json；这里作为服务层
            # 边界校验，避免页面进入一个无法恢复的半完成 workspace。
            raise PaperUploadError("论文工作区创建后无法读取，请重新上传。")
        return summary

    def get_workspace(self, workspace_id: str) -> WorkspaceSummary | None:
        """读取一个 workspace，并拒绝越出 staging 根目录的路径。"""
        # 先检查 ID 的格式，阻止绝对路径和路径穿越字符串进入拼接流程。
        if not _valid_workspace_id(workspace_id):
            return None
        # 将候选路径解析为绝对路径，便于后面进行边界比较。
        root = (self._settings.storage.staging_dir / workspace_id).resolve()
        # 只有 staging 根目录的直接子目录才是合法 workspace。
        if root.parent != self._settings.storage.staging_dir.resolve():
            return None
        # 复用统一的 manifest 读取逻辑，保证单个读取和列表读取行为一致。
        return self._read_workspace(root)

    def add_workspace_to_knowledge_base(self, workspace_id: str) -> PromotionResult:
        """复用后端 Promotion，将一个 staging workspace 加入正式知识库。

        页面只提供当前 workspace ID；SQLite、正式 PDF、FAISS 和 embedding
        artifact 的原子发布仍完全由 ``paperbase.promotion`` 处理。
        """
        try:
            self._require_workspace(workspace_id)
            return promote_workspace(settings=self._settings, workspace_id=workspace_id)
        except (ValueError, PromotionError) as error:
            raise WorkspaceActionError(str(error)) from error
        except Exception as error:  # noqa: BLE001 - 不把发布链路 traceback 交给 UI。
            raise WorkspaceActionError("加入知识库失败，请检查论文工件和后端配置后重试。") from error

    def delete_workspace(self, workspace_id: str) -> None:
        """删除一个 staging 工作区，不影响已经存在的正式知识库论文。"""
        try:
            self._require_workspace(workspace_id)
            delete_temporary_workspace(
                staging_dir=self._settings.storage.staging_dir,
                workspace_id=workspace_id,
            )
        except (ValueError, TemporaryWorkspaceError) as error:
            raise WorkspaceActionError(str(error)) from error
        except OSError as error:
            raise WorkspaceActionError("临时论文工作区删除失败，请稍后重试。") from error
        # 清理当前 Streamlit 进程中按 workspace 缓存的 Ask 服务；真正的数据删除
        # 仍由 staging 后端入口完成，页面不直接触碰工作区文件。
        self._paper_ask_services.pop(workspace_id, None)

    def get_paper_overview(self, workspace_id: str) -> PaperOverview | None:
        """读取已经落盘的 Paper Overview；没有工件时返回 ``None``。

        这个方法只读 ``overview/overview.json``，不会因为页面刷新自动调用
        LLM。工件存在但格式损坏时抛出专用异常，由页面提供“重新生成”入口。
        """
        root = self._workspace_root(workspace_id)
        if root is None:
            return None
        overview_path = root / "overview" / "overview.json"
        if not overview_path.is_file():
            return None
        try:
            payload = json.loads(overview_path.read_text(encoding="utf-8"))
            return PaperOverview.model_validate(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise PaperOverviewArtifactError(
                "论文概览文件无法读取，请重新生成。"
            ) from error

    def generate_paper_overview(self, workspace_id: str) -> PaperOverview:
        """按需调用后端 Overview stage，并返回新生成的结构化结果。

        解析、分块和 LLM 调用全部由 ``paperbase.overview`` 负责；Service
        只做 workspace 校验和适配，不复制后端的筛选或提示词逻辑。
        """
        root = self._workspace_root(workspace_id)
        if root is None:
            raise PaperOverviewError("论文工作区不存在或已失效。")
        try:
            outcome = run_paper_overview_stage(
                settings=self._settings,
                workspace_id=workspace_id,
            )
        except PaperOverviewError:
            raise
        except Exception as error:  # noqa: BLE001 - 页面统一处理后端失败。
            raise PaperOverviewError("论文概览生成失败，请稍后重试。") from error
        return outcome.overview

    def get_paper_overview_sources(
        self, workspace_id: str, source_chunk_ids: tuple[str, ...] | list[str]
    ) -> tuple[PaperOverviewSource, ...]:
        """根据 Overview 保存的来源 ID恢复章节、页码和原文。

        chunks JSONL 由 Service 读取并转换成最小展示模型，页面不直接打开
        staging 文件，也不会把 chunk_id、vector_id 或检索分数传到 UI。
        """
        root = self._workspace_root(workspace_id)
        if root is None:
            return ()
        requested_ids = tuple(
            item.strip() for item in source_chunk_ids if isinstance(item, str) and item.strip()
        )
        if not requested_ids:
            return ()
        records_by_id: dict[str, dict[str, object]] = {}
        chunks_path = root / "chunks" / "chunks.jsonl"
        try:
            for line in chunks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                chunk_id = record.get("chunk_id")
                if isinstance(chunk_id, str) and chunk_id in requested_ids:
                    records_by_id[chunk_id] = record
        except (OSError, json.JSONDecodeError):
            return ()

        sources: list[PaperOverviewSource] = []
        for chunk_id in requested_ids:
            record = records_by_id.get(chunk_id)
            if record is None:
                continue
            raw_text = _optional_text(record.get("raw_text"))
            if not raw_text:
                continue
            sources.append(
                PaperOverviewSource(
                    section=_optional_text(record.get("section")),
                    page_start=_optional_page(record.get("page_start")),
                    page_end=_optional_page(record.get("page_end")),
                    text=raw_text,
                )
            )
        return tuple(sources)

    def get_section_tree(self, workspace_id: str) -> tuple[SectionSummary, ...]:
        """从 ParsedPaper 的真实 Section Tree 返回可展示章节，不猜层级。"""
        snapshot = self._section_repository().load(workspace_id)
        return tuple(
            SectionSummary(
                section_id=section.section_id,
                title=section.section_title,
                level=section.section_level,
                parent_section_id=section.parent_section_id,
                page_start=section.page_start,
                page_end=section.page_end,
                has_children=bool(snapshot.get_children(section.section_id)),
                has_content=bool(snapshot.get_subtree_chunks(section.section_id)),
            )
            for section in snapshot.get_section_tree()
        )

    def get_section_explanation(
        self, workspace_id: str, section_id: str
    ) -> SectionExplanationView | None:
        """读取已有 Explain artifact；没有生成结果时返回 ``None``。"""
        snapshot = self._section_repository().load(workspace_id)
        # 先校验 section_id，避免失效 ID 被误认为“尚未生成”。
        snapshot.get_section(section_id)
        artifact_path = self._section_explanation_path(snapshot.root_dir, section_id)
        if not artifact_path.is_file():
            return None
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            explanation = ExplainSection.model_validate(payload)
            if explanation.section_id != section_id:
                raise ValueError("Explain artifact section_id does not match requested section.")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SectionExplanationArtifactError(
                "章节解释文件损坏或无法读取，请重新生成。"
            ) from error
        return _section_explanation_view(explanation)

    def generate_section_explanation(
        self, workspace_id: str, section_id: str
    ) -> SectionExplanationView:
        """调用既有 Explain Section workflow，不在前端拼接 Prompt。"""
        snapshot = self._section_repository().load(workspace_id)
        snapshot.get_section(section_id)
        try:
            outcome = run_explain_section_stage(
                settings=self._settings,
                workspace_id=workspace_id,
                section_id=section_id,
            )
        except ExplainSectionError:
            raise
        except Exception as error:  # noqa: BLE001 - 页面统一显示友好错误。
            raise ExplainSectionError("章节解释生成失败，请稍后重试。") from error
        return _section_explanation_view(outcome.explanation)

    def get_section_explanation_sources(
        self, workspace_id: str, source_chunk_ids: tuple[str, ...] | list[str]
    ) -> tuple[SectionExplanationSource, ...]:
        """按 Explain 返回的真实 chunk ID恢复 Section、页码和原文。"""
        snapshot = self._section_repository().load(workspace_id)
        requested_ids = tuple(
            item.strip() for item in source_chunk_ids if isinstance(item, str) and item.strip()
        )
        if not requested_ids:
            return ()
        chunks_by_id = {chunk.chunk_id: chunk for chunk in snapshot.chunks}
        sections_by_id = {section.section_id: section for section in snapshot.sections}
        sources: list[SectionExplanationSource] = []
        for chunk_id in requested_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or not chunk.raw_text.strip():
                continue
            section_title = chunk.section
            if not section_title and chunk.section_id in sections_by_id:
                section_title = sections_by_id[chunk.section_id].section_title
            sources.append(
                SectionExplanationSource(
                    section=section_title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.raw_text,
                )
            )
        return tuple(sources)

    def create_kb_conversation(self) -> ConversationRecord:
        """在稳定的正式 KB scope 下创建一个全新的独立 conversation。"""
        # ConversationStore 会生成唯一 ID，并把记录持久化到 conversations.sqlite3。
        return self._conversations.create_conversation(
            # scope_type 明确表明这是跨论文的 Knowledge Base 会话。
            "knowledge_base", FORMAL_KNOWLEDGE_BASE_SCOPE_ID
        )

    def list_kb_conversations(self) -> tuple[KnowledgeBaseConversation, ...]:
        """列出正式 KB 会话，并用第一条用户问题生成稳定的临时标题。"""
        records = self._conversations.list_conversations(
            "knowledge_base", FORMAL_KNOWLEDGE_BASE_SCOPE_ID
        )
        summaries: list[KnowledgeBaseConversation] = []
        for record in records:
            turns = self._conversations.get_turns(record.conversation_id)
            first_query = turns[0].user_query if turns else None
            summaries.append(
                KnowledgeBaseConversation(
                    conversation_id=record.conversation_id,
                    title=_conversation_title(first_query),
                    turn_count=len(turns),
                    updated_at=record.updated_at,
                )
            )
        return tuple(summaries)

    def get_conversation_turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        """读取正式 KB 会话的完整历史，并在入口处验证 scope。"""
        self._conversations.get_conversation(
            conversation_id,
            expected_scope_type="knowledge_base",
            expected_scope_id=FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
        )
        return self._conversations.get_turns(conversation_id)

    def ask_knowledge_base(self, query: str, conversation_id: str) -> AnswerServiceResult:
        """把问题和会话 ID交给现有正式 KB QA 链路，并保存本轮 Evidence 快照。"""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("问题不能为空。")
        # 先验证 scope，避免调用链内部误接收 workspace 或其他 KB 的 ID。
        self._conversations.get_conversation(
            conversation_id,
            expected_scope_type="knowledge_base",
            expected_scope_id=FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
        )
        result = self._get_answer_service().answer_query(
            normalized_query,
            conversation_id=conversation_id,
        )
        # ConversationStore 已由后端自动追加 turn；这里仅把同一结果中的真实 Evidence
        # 写入 Service 管理的快照文件，供历史恢复使用，不重新检索也不复制 RAG 逻辑。
        self._persist_answer_artifact(result)
        return result

    def get_answer_evidence(self, result: AnswerServiceResult) -> tuple[KnowledgeBaseEvidence, ...]:
        """把本轮后端 Evidence 转成页面视图模型，不暴露内部 chunk 字段。"""
        return tuple(_evidence_view(item) for item in result.expansion.evidence)

    def get_conversation_turn_evidence(
        self, turn: ConversationTurn
    ) -> tuple[KnowledgeBaseEvidence, ...]:
        """从 turn 的 evidence_json 恢复 Evidence，旧版本再回退到 sidecar。

        新版本的正式来源是 ConversationStore 的 nullable 字段；sidecar 只为
        之前已经生成的历史产物保留兼容读取，不会重新检索。
        """
        database_evidence = _evidence_from_json_text(turn.evidence_json)
        if database_evidence is not None:
            return database_evidence
        payload = self._read_answer_artifact(turn)
        if payload is None:
            return ()
        raw_evidence = payload.get("evidence")
        if not isinstance(raw_evidence, list):
            return ()
        return tuple(
            evidence
            for item in raw_evidence
            if isinstance(item, dict)
            for evidence in (_evidence_from_payload(item),)
            if evidence is not None
        )

    def get_conversation_turn_metadata(
        self, turn: ConversationTurn
    ) -> KnowledgeBaseTurnMetadata | None:
        """恢复答案状态字段，支持 UI 清晰提示证据不足和部分覆盖。"""
        payload = self._read_answer_artifact(turn)
        if payload is None:
            return None
        return KnowledgeBaseTurnMetadata(
            answer_status=_optional_text(payload.get("answer_status")) or "unknown",
            insufficient_evidence=payload.get("insufficient_evidence") is True,
            partial_answer=payload.get("partial_answer") is True,
            coverage_note=_optional_text(payload.get("coverage_note")),
        )

    def create_workspace_conversation(self, workspace_id: str) -> ConversationRecord:
        """为一个已存在的 staging workspace 创建独立会话。"""
        # 创建前先确认 workspace 存在，避免产生指向不存在论文的会话。
        if self.get_workspace(workspace_id) is None:
            # 用清晰的业务错误替代底层 SQLite 或文件错误。
            raise ValueError(f"Workspace does not exist: {workspace_id}")
        # workspace 会话的 scope_id 就是精确的 workspace ID。
        return self._conversations.create_conversation("workspace", workspace_id)

    def list_paper_conversations(self, workspace_id: str) -> tuple[PaperConversation, ...]:
        """列出指定 staging workspace 的 Ask This Paper 会话。"""
        self._require_workspace(workspace_id)
        records = self._conversations.list_conversations("workspace", workspace_id)
        summaries: list[PaperConversation] = []
        for record in records:
            turns = self._conversations.get_turns(record.conversation_id)
            first_query = turns[0].user_query if turns else None
            summaries.append(
                PaperConversation(
                    conversation_id=record.conversation_id,
                    title=_conversation_title(first_query),
                    turn_count=len(turns),
                    updated_at=record.updated_at,
                )
            )
        return tuple(summaries)

    def create_paper_conversation(self, workspace_id: str) -> ConversationRecord:
        """仅为当前 staging workspace 创建新会话，不自动切换到正式 KB。"""
        return self.create_workspace_conversation(workspace_id)

    def get_paper_conversation_turns(
        self, conversation_id: str, *, workspace_id: str | None = None
    ) -> tuple[ConversationTurn, ...]:
        """读取 Paper 会话历史，并校验它属于 workspace scope。"""
        record = self._conversations.get_conversation(
            conversation_id,
            expected_scope_type="workspace",
            expected_scope_id=workspace_id,
        )
        if workspace_id is not None and record.scope_id != workspace_id:
            # expected_scope_id 已经会抛出 scope error；这里保留显式分支让契约清晰。
            raise ValueError("Paper conversation does not belong to the current workspace.")
        return self._conversations.get_turns(conversation_id)

    def ask_this_paper(
        self, workspace_id: str, conversation_id: str, query: str
    ) -> AskThisPaperResult:
        """把问题交给现有 AskThisPaperService，并把用户可见 Evidence 快照补入 turn。"""
        self._require_workspace(workspace_id)
        self._conversations.get_conversation(
            conversation_id,
            expected_scope_type="workspace",
            expected_scope_id=workspace_id,
        )
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("问题不能为空。")
        result = self._get_paper_ask_service(workspace_id).ask(
            normalized_query,
            conversation_id=conversation_id,
        )
        # AskThisPaperService 已完成检索、回答和 turn 追加；这里仅把最终 Evidence
        # 的最小 snapshot 写进同一 turn，历史读取不需要重新检索。
        if result.conversation_id:
            turns = self._conversations.get_turns(result.conversation_id)
            if turns:
                self._conversations.update_turn_evidence(
                    turns[-1].turn_id,
                    _evidence_snapshot_json(
                        result.expansion.evidence,
                        result.answer.citations,
                        answer_status=result.answer.status,
                    ),
                )
        return result

    def get_paper_conversation_turn_evidence(
        self, turn: ConversationTurn
    ) -> tuple[KnowledgeBaseEvidence, ...]:
        """恢复 Paper turn 的 Evidence snapshot，旧 turn 回退到 ask_paper 工件。"""
        database_evidence = _evidence_from_json_text(turn.evidence_json)
        if database_evidence is not None:
            return database_evidence
        payload = self._read_paper_answer_artifact(turn)
        if payload is None or not isinstance(payload.get("evidence"), list):
            return ()
        return tuple(
            evidence
            for item in payload["evidence"]
            if isinstance(item, dict)
            for evidence in (_evidence_from_payload(item),)
            if evidence is not None
        )

    def get_paper_conversation_turn_metadata(
        self, turn: ConversationTurn
    ) -> KnowledgeBaseTurnMetadata | None:
        """恢复 Ask This Paper 的降级/覆盖提示，不暴露工件中的检索字段。"""
        payload = self._read_paper_answer_artifact(turn)
        if payload is None:
            return None
        return KnowledgeBaseTurnMetadata(
            answer_status=_optional_text(payload.get("answer_status")) or "unknown",
            insufficient_evidence=payload.get("insufficient_evidence") is True,
            partial_answer=payload.get("partial_answer") is True,
            coverage_note=_optional_text(payload.get("coverage_note")),
        )

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        """读取一个持久化 conversation；调用页面负责验证它属于哪个 scope。"""
        # Store 负责检查 ID 是否存在并返回轻量的会话元数据。
        return self._conversations.get_conversation(conversation_id)

    def _get_answer_service(self) -> object:
        """延迟创建正式问答服务，避免只浏览页面时加载重型模型。"""
        if self._answer_service is None:
            self._answer_service = create_answer_service(
                config_path=self._settings.config_path,
                conversation_scope_id=FORMAL_KNOWLEDGE_BASE_SCOPE_ID,
            )
        return self._answer_service

    def _get_paper_ask_service(self, workspace_id: str) -> object:
        """按 workspace 延迟装配并复用现有 Ask This Paper 后端服务。"""
        service = self._paper_ask_services.get(workspace_id)
        if service is None:
            service = create_ask_this_paper_service(
                workspace_id=workspace_id,
                config_path=self._settings.config_path,
            )
            self._paper_ask_services[workspace_id] = service
        return service

    def _require_workspace(self, workspace_id: str) -> WorkspaceSummary:
        """确认 workspace 存在，避免创建或问答指向失效 staging 目录。"""
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            raise ValueError("当前 staging workspace 不存在或已失效。")
        return workspace

    def _read_paper_answer_artifact(
        self, turn: ConversationTurn
    ) -> dict[str, object] | None:
        """安全读取 Ask This Paper 已写入的结果工件，仅用于恢复用户可见字段。"""
        if not turn.audit_path:
            return None
        try:
            path = Path(turn.audit_path).resolve()
            staging_root = self._settings.storage.staging_dir.resolve()
            if not path.is_relative_to(staging_root) or not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _persist_answer_artifact(self, result: AnswerServiceResult) -> None:
        """保存本轮答案和真实 Evidence，写入失败不影响已经完成的回答。"""
        conversation_id = result.conversation_id
        if not conversation_id:
            return
        try:
            turns = self.get_conversation_turns(conversation_id)
            if not turns:
                return
            turn = turns[-1]
            payload = {
                "conversation_id": conversation_id,
                "turn_index": turn.turn_index,
                "answer_status": result.answer.status,
                "answer": result.answer.answer,
                "citations": list(result.answer.citations),
                "insufficient_evidence": result.answer.insufficient_evidence,
                "partial_answer": result.answer.partial_answer,
                "coverage_note": result.answer.coverage_note,
                "evidence": [
                    _evidence_payload(item) for item in result.expansion.evidence
                ],
            }
            path = self._answer_artifact_path(conversation_id, turn.turn_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            # Evidence 是增强展示能力；即使文件系统暂时不可写，主回答仍应可见。
            return

    def _answer_artifact_path(self, conversation_id: str, turn_index: int) -> Path:
        """返回由后端生成 ID 组成的快照路径，不接受用户提供的任意路径。"""
        if not conversation_id or "/" in conversation_id or "\\" in conversation_id:
            raise ValueError("Invalid conversation ID.")
        return self._answer_artifact_dir / f"{conversation_id}_{turn_index}.json"

    def _read_answer_artifact(self, turn: ConversationTurn) -> dict[str, object] | None:
        """读取并校验指定 turn 的快照，避免页面直接打开 Service 管理的文件。"""
        try:
            path = self._answer_artifact_path(turn.conversation_id, turn.turn_index)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("conversation_id") != turn.conversation_id:
            return None
        if payload.get("turn_index") != turn.turn_index:
            return None
        return payload

    def _section_repository(self) -> WorkspaceSectionRepository:
        """创建后端 Section Tree 只读仓库，统一以 staging 根目录为边界。"""
        return WorkspaceSectionRepository(self._settings.storage.staging_dir)

    @staticmethod
    def _section_explanation_path(root: Path, section_id: str) -> Path:
        """复用 Explain backend 的稳定 section artifact 命名规则。"""
        artifact_stem = hashlib.sha256(section_id.encode("utf-8")).hexdigest()[:16]
        return root / "explain_sections" / f"{artifact_stem}.json"

    def _workspace_root(self, workspace_id: str) -> Path | None:
        """返回通过边界校验的 workspace 路径，页面不需要处理真实路径。"""
        if self.get_workspace(workspace_id) is None:
            return None
        return (self._settings.storage.staging_dir / workspace_id).resolve()

    def _read_workspace(self, root: Path) -> WorkspaceSummary | None:
        """读取单个 workspace manifest，并转换成页面摘要模型。"""
        # workspace.json 是 staging 工作区的入口清单。
        manifest_path = root / "workspace.json"
        # 文件缺失、权限错误或 JSON 格式错误都视为无效工作区。
        try:
            # 以 UTF-8 读取并解析 JSON 对象。
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 列表页面跳过坏目录，避免一个坏 workspace 阻塞全部论文。
            return None
        # 目录名和 manifest 内 workspace_id 必须一致，防止错配。
        if not isinstance(manifest, dict) or manifest.get("workspace_id") != root.name:
            return None
        # 读取并清理 manifest 中的稳定论文 ID。
        paper_id = _required_text(manifest.get("paper_id"))
        # 读取并清理用户上传时的源文件名。
        source_filename = _required_text(manifest.get("source_filename"))
        # chunk 数必须是整数，才能可靠地显示页面统计。
        total_chunk_count = manifest.get("total_chunk_count")
        # 三个关键字段缺失或类型错误时，当前 manifest 不可供页面使用。
        if paper_id is None or source_filename is None or not isinstance(total_chunk_count, int):
            return None
        # 组织成不可变的 workspace 页面模型。
        return WorkspaceSummary(
            # 目录名已经通过上面的等值检查，因此可直接作为 workspace ID。
            workspace_id=root.name,
            # 使用清洗后的 paper ID。
            paper_id=paper_id,
            # 使用清洗后的源文件名。
            source_filename=source_filename,
            # 标题优先从正式库读取，其次从 parsed artifact 读取。
            title=self._workspace_title(root, paper_id),
            # manifest 中的 chunk 数已经验证为 int。
            total_chunk_count=total_chunk_count,
            # 章节数来自同一个 staging workspace 的 parsed artifact。
            section_count=self._workspace_section_count(root),
            # 只有明确为 True 才表示已加入正式 KB，其他值均按未加入处理。
            added_to_kb=manifest.get("added_to_kb") is True,
        )

    def _workspace_title(self, root: Path, paper_id: str) -> str | None:
        """按可靠性顺序读取 workspace 标题。"""
        # promotion 后正式库中的标题是最可靠的来源，因此先查询正式 metadata。
        document = self._database.get_document(paper_id)
        # 如果正式库中有记录，则优先使用其中的非空 paper_title。
        if document is not None:
            # 清理数据库标题中的多余空白。
            title = _optional_text(document["paper_title"])
            # 只有标题非空时才提前返回。
            if title:
                return title
        # 未 promotion 或正式库没有标题时，回退到 parsed_paper.json。
        try:
            # 解析 staging 中保存的结构化解析结果。
            parsed = json.loads((root / "parsed" / "parsed_paper.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 解析文件缺失或损坏时，交给上层使用源文件名。
            return None
        # 只有 JSON 是对象时才读取 paper_title，避免对列表等类型调用 get。
        return _optional_text(parsed.get("paper_title")) if isinstance(parsed, dict) else None

    def _workspace_section_count(self, root: Path) -> int:
        """读取 parsed_paper.json 中的章节列表长度，失败时返回 0。"""
        # 章节树是 staging 解析产物，页面只需要章节数量，不需要暴露 section_id。
        try:
            # 复用 workspace 内已经生成的结构化解析文件。
            parsed = json.loads((root / "parsed" / "parsed_paper.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 解析文件不存在或损坏时，不阻塞页面其他信息显示。
            return 0
        # 只有 sections 是列表时才把长度作为可靠章节数量。
        sections = parsed.get("sections") if isinstance(parsed, dict) else None
        return len(sections) if isinstance(sections, list) else 0


def _valid_workspace_id(workspace_id: str) -> bool:
    """判断 workspace ID 是否符合目录边界要求。"""
    # 必须使用后端生成的 staging_ 前缀。
    return workspace_id.startswith("staging_") and "/" not in workspace_id and "\\" not in workspace_id


def _required_text(value: object) -> str | None:
    """把任意值规范化为非空文本，否则返回 None。"""
    # 当前 UI 只需要文本字段，因此统一交给文本清洗函数处理。
    value = _optional_text(value)
    # 保留一个独立函数名，表达 manifest 字段是必需值的语义。
    return value


def _optional_text(value: object) -> str | None:
    """清理字符串首尾和内部多余空白，非字符串或空白值返回 None。"""
    # manifest/SQLite 可能返回 None、数字或其他类型，先做类型保护。
    if not isinstance(value, str):
        return None
    # ``split`` 再 ``join`` 会把连续空白统一成一个普通空格。
    normalized = " ".join(value.split())
    # 空字符串不作为有效标题或 ID 返回。
    return normalized or None


def _safe_pdf_filename(value: str) -> str:
    """保留上传文件名的显示价值，同时拒绝路径片段和非 PDF 扩展名。"""
    # ``Path.name`` 去除本机路径片段；额外替换反斜杠，兼容网页客户端可能上传
    # 的 Windows 风格文件名，避免它在 POSIX 环境中被当作普通字符保留下来。
    filename = Path(str(value).replace("\\", "/")).name.strip()
    if not filename or filename in {".", ".."} or Path(filename).suffix.casefold() != ".pdf":
        raise PaperUploadError("请上传一个 PDF 文件。")
    return filename


def _conversation_title(first_query: str | None) -> str:
    """用第一条问题生成不调用 LLM 的会话标题。"""
    if not first_query:
        return "新会话"
    normalized = " ".join(first_query.split())
    # 约 20～30 个字符足以在下拉框中识别会话，同时避免标题挤压布局。
    return normalized if len(normalized) <= 28 else f"{normalized[:28]}…"


def _evidence_view(item: object) -> KnowledgeBaseEvidence:
    """将后端 EvidenceUnit 转成仅包含用户关心定位信息的视图。"""
    return KnowledgeBaseEvidence(
        evidence_id=str(getattr(item, "evidence_id", "")),
        kind=str(getattr(item, "kind", "content")),
        paper_title=str(getattr(item, "paper_title", "") or "未命名论文"),
        section=_optional_text(getattr(item, "section", None)),
        page_start=_optional_page(getattr(item, "page_start", None)),
        page_end=_optional_page(getattr(item, "page_end", None)),
        text=str(getattr(item, "text", "") or ""),
    )


def _evidence_payload(item: object) -> dict[str, object]:
    """保存 Evidence 快照时只写 UI 后续需要的公开字段。"""
    view = _evidence_view(item)
    return {
        "evidence_id": view.evidence_id,
        "kind": view.kind,
        "paper_title": view.paper_title,
        "section": view.section,
        "page_start": view.page_start,
        "page_end": view.page_end,
        "text": view.text,
    }


def _evidence_snapshot_json(
    evidence: object,
    citations: object = (),
    *,
    answer_status: str | None = None,
) -> str | None:
    """序列化 Ask This Paper 本轮最终可展示的 Evidence。"""
    if answer_status in {"needs_clarification", "insufficient_evidence"}:
        return None
    citation_set = {
        str(item).strip()
        for item in (citations if isinstance(citations, (tuple, list, set)) else ())
        if str(item).strip()
    }
    items = tuple(evidence) if isinstance(evidence, (tuple, list)) else ()
    selected = (
        tuple(item for item in items if str(getattr(item, "evidence_id", "")) in citation_set)
        if citation_set
        else items
    )
    return json.dumps(
        [_evidence_payload(item) for item in selected], ensure_ascii=False
    )


def _evidence_from_payload(payload: dict[str, object]) -> KnowledgeBaseEvidence | None:
    """校验快照字段后恢复 Evidence，损坏的单条记录不会影响其它证据。"""
    evidence_id = _optional_text(payload.get("evidence_id"))
    text = payload.get("text", payload.get("raw_text"))
    if not evidence_id or not isinstance(text, str) or not text.strip():
        return None
    return KnowledgeBaseEvidence(
        evidence_id=evidence_id,
        kind=_optional_text(payload.get("kind", payload.get("type"))) or "content",
        paper_title=_optional_text(payload.get("paper_title")) or "未命名论文",
        section=_optional_text(payload.get("section")),
        page_start=_optional_page(payload.get("page_start")),
        page_end=_optional_page(payload.get("page_end")),
        text=text,
    )


def _evidence_from_json_text(value: str | None) -> tuple[KnowledgeBaseEvidence, ...] | None:
    """解析 ConversationStore 中的 evidence_json；NULL/损坏值交给 sidecar 回退。"""
    if not value:
        return None
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    evidence = tuple(
        item_evidence
        for item in payload
        if isinstance(item, dict)
        for item_evidence in (_evidence_from_payload(item),)
        if item_evidence is not None
    )
    return evidence


def _section_explanation_view(explanation: ExplainSection) -> SectionExplanationView:
    """将后端 Explain 模型转换成页面需要的稳定字段。"""
    return SectionExplanationView(
        section_id=explanation.section_id,
        section_title=explanation.section_title,
        mode=explanation.mode,
        explanation=explanation.explanation,
        key_points=tuple(explanation.key_points),
        source_chunk_ids=tuple(explanation.source_chunk_ids),
        insufficient_evidence=explanation.insufficient_evidence,
    )


def _optional_page(value: object) -> int | None:
    """只接受整数页码，避免把 None 或错误文本展示给用户。"""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
