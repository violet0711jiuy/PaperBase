"""从 Temporary Paper Workspace 生成可追溯的单篇论文速览。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbase.config import AppSettings, PaperOverviewSettings
from paperbase.llm.client import (
    ChatCompletionClient,
    LLMRequestError,
    LLMRuntimeSettings,
    OpenAICompatibleChatClient,
    load_llm_runtime_settings,
)
from paperbase.prompts.paper_overview import (
    PAPER_OVERVIEW_SYSTEM_PROMPT,
    build_paper_overview_user_prompt,
)


_MISSING = "论文未明确说明"
_CORE_SECTION_ORDER = (
    "abstract",
    "introduction",
    "method",
    "experiments",
    "conclusion",
)


class PaperOverviewError(RuntimeError):
    """临时工作区不完整、Overview 输出不可信或写入失败时抛出。"""


class _OverviewDraftField(BaseModel):
    """LLM 返回的字段正文及其引用的 context chunk ID。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(default_factory=list)


class _PaperOverviewDraft(BaseModel):
    """API JSON Schema 使用的最小草稿结构；section 名由程序可信回填。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    paper_title: str = Field(min_length=1)
    research_problem: _OverviewDraftField
    main_method: _OverviewDraftField
    contributions: _OverviewDraftField
    datasets: _OverviewDraftField
    experimental_setup: _OverviewDraftField
    main_results: _OverviewDraftField
    limitations: _OverviewDraftField


class OverviewField(BaseModel):
    """最终展示字段：正文、来源 chunk 与由程序校验出的章节路径。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_sections: list[str] = Field(default_factory=list)


class PaperOverview(BaseModel):
    """可落盘、可供未来前端直接展示的单篇论文速览。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    paper_title: str = Field(min_length=1)
    research_problem: OverviewField
    main_method: OverviewField
    contributions: OverviewField
    datasets: OverviewField
    experimental_setup: OverviewField
    main_results: OverviewField
    limitations: OverviewField


@dataclass(frozen=True)
class OverviewContextChunk:
    """一个进入 Prompt 的已截断 chunk 及其稳定来源信息。"""

    chunk_id: str
    section: str
    category: str
    text: str


@dataclass(frozen=True)
class OverviewContext:
    """按章节选择后的 Prompt 数据，不含向量、FAISS 或其他论文内容。"""

    paper_title: str | None
    chunks: tuple[OverviewContextChunk, ...]

    @property
    def rendered_text(self) -> str:
        """将每个来源明确标为 chunk ID，供模型引用并供本地校验。"""
        return "\n\n".join(
            "[chunk_id: {chunk_id}]\n[category: {category}]\n[section: {section}]\n{content}".format(
                chunk_id=chunk.chunk_id,
                category=chunk.category,
                section=chunk.section or "未标注章节",
                content=chunk.text,
            )
            for chunk in self.chunks
        )


@dataclass(frozen=True)
class PaperOverviewOutcome:
    """本次 Overview 的可审计结果及其落盘位置。"""

    overview: PaperOverview
    overview_path: Path
    context_path: Path
    selected_chunk_count: int


def run_paper_overview_stage(
    *, settings: AppSettings, workspace_id: str
) -> PaperOverviewOutcome:
    """按配置为一个已存在的 temporary workspace 创建 Overview。"""
    workspace_root = _workspace_root(settings.storage.staging_dir, workspace_id)
    runtime_settings = load_llm_runtime_settings(settings.config_path.parent / ".env")
    # 不修改 .env 的共享默认值；仅本次 Overview 请求提升其自身 JSON 输出上限。
    overview_runtime_settings = _with_overview_output_budget(
        runtime_settings, settings.paper_overview.max_output_tokens
    )
    client = OpenAICompatibleChatClient(overview_runtime_settings)
    return create_paper_overview(
        workspace_root=workspace_root,
        settings=settings.paper_overview,
        client=client,
    )


def create_paper_overview(
    *,
    workspace_root: Path,
    settings: PaperOverviewSettings,
    client: ChatCompletionClient,
) -> PaperOverviewOutcome:
    """只读临时 parsed/chunks 产物并写入 overview，不重跑 Parse、Embedding 或检索。"""
    context = build_overview_context(workspace_root=workspace_root, settings=settings)
    if not context.chunks:
        raise PaperOverviewError("Temporary workspace has no usable non-bibliography chunks.")
    try:
        raw_response = client.complete_json(
            system_prompt=PAPER_OVERVIEW_SYSTEM_PROMPT,
            user_prompt=build_paper_overview_user_prompt(
                paper_title=context.paper_title,
                context=context.rendered_text,
            ),
            json_schema=_PaperOverviewDraft.model_json_schema(),
            schema_name="paperbase_paper_overview",
        )
        overview = _validate_and_finalize_overview(
            draft=_PaperOverviewDraft.model_validate_json(raw_response), context=context
        )
    except (LLMRequestError, ValidationError, ValueError, TypeError) as error:
        raise PaperOverviewError("Paper Overview generation returned an invalid result.") from error
    return _write_overview_artifacts(
        workspace_root=workspace_root, overview=overview, context=context
    )


def build_overview_context(
    *, workspace_root: Path, settings: PaperOverviewSettings
) -> OverviewContext:
    """从 workspace 的结构化产物构造受预算限制的章节级上下文。"""
    root = workspace_root.resolve()
    parsed_record = _load_json_object(root / "parsed" / "parsed_paper.json")
    raw_chunks = _load_chunk_records(root / "chunks" / "chunks.jsonl")
    candidates = [
        record
        for record in raw_chunks
        if record.get("section_type") == "content" and _record_text(record)
    ]
    if not candidates:
        return OverviewContext(paper_title=_optional_string(parsed_record.get("paper_title")), chunks=())

    categories_by_chunk_id = _categorize_records(candidates)
    selected: list[OverviewContextChunk] = []
    remaining_chars = settings.max_total_context_chars
    for category in _CORE_SECTION_ORDER:
        matches = [
            record
            for record in candidates
            if categories_by_chunk_id.get(_optional_string(record.get("chunk_id"))) == category
        ]
        for record in matches[: settings.max_chunks_per_section]:
            chunk = _context_chunk(record, category, settings.max_chars_per_chunk, remaining_chars)
            if chunk is None:
                break
            selected.append(chunk)
            remaining_chars -= len(chunk.text)
            if remaining_chars <= 0:
                break
        if remaining_chars <= 0:
            break

    # 部分论文标题完全非标准时仍提供少量早期正文；这不是全量 PDF 回退，也不会包含 References。
    if not selected and settings.max_fallback_chunks:
        for record in candidates[: settings.max_fallback_chunks]:
            chunk = _context_chunk(record, "fallback", settings.max_chars_per_chunk, remaining_chars)
            if chunk is None:
                break
            selected.append(chunk)
            remaining_chars -= len(chunk.text)

    return OverviewContext(
        paper_title=_optional_string(parsed_record.get("paper_title")),
        chunks=tuple(selected),
    )


def _validate_and_finalize_overview(
    *, draft: _PaperOverviewDraft, context: OverviewContext
) -> PaperOverview:
    """拒绝伪造来源，并从真实 chunk 元数据回填每个字段的章节路径。"""
    source_sections = {chunk.chunk_id: chunk.section for chunk in context.chunks}

    def finalize(field: _OverviewDraftField) -> OverviewField:
        content = field.content.strip()
        chunk_ids = _unique_strings(field.source_chunk_ids)
        unknown_ids = set(chunk_ids).difference(source_sections)
        if unknown_ids:
            raise ValueError(f"Overview references chunks outside the provided context: {unknown_ids}")
        if content == _MISSING:
            if chunk_ids:
                raise ValueError("A missing Overview field must not cite a chunk.")
            return OverviewField(content=content)
        if not chunk_ids:
            raise ValueError("A non-missing Overview field must cite at least one context chunk.")
        return OverviewField(
            content=content,
            source_chunk_ids=chunk_ids,
            source_sections=_unique_strings(source_sections[chunk_id] for chunk_id in chunk_ids),
        )

    expected_title = context.paper_title or _MISSING
    if draft.paper_title.strip() != expected_title:
        raise ValueError("Overview paper_title must exactly match the workspace metadata.")
    return PaperOverview(
        paper_title=expected_title,
        research_problem=finalize(draft.research_problem),
        main_method=finalize(draft.main_method),
        contributions=finalize(draft.contributions),
        datasets=finalize(draft.datasets),
        experimental_setup=finalize(draft.experimental_setup),
        main_results=finalize(draft.main_results),
        limitations=finalize(draft.limitations),
    )


def _write_overview_artifacts(
    *, workspace_root: Path, overview: PaperOverview, context: OverviewContext
) -> PaperOverviewOutcome:
    """在当前临时工作区内保存结果和已送入模型的来源清单。"""
    output_dir = workspace_root.resolve() / "overview"
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "overview.json"
    context_path = output_dir / "context.json"
    overview_path.write_text(
        json.dumps(overview.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    context_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "paper_title": context.paper_title,
                "selected_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "section": chunk.section,
                        "category": chunk.category,
                    }
                    for chunk in context.chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return PaperOverviewOutcome(
        overview=overview,
        overview_path=overview_path,
        context_path=context_path,
        selected_chunk_count=len(context.chunks),
    )


def _section_category(record: dict[str, Any]) -> str | None:
    """按章节语义而非固定完整标题归类常见中英文论文结构。"""
    section = " ".join(str(record.get("section") or "").casefold().split())
    front_matter_type = str(record.get("front_matter_type") or "").casefold()
    if front_matter_type == "abstract" or "abstract" in section or "摘要" in section:
        return "abstract"
    if any(token in section for token in ("introduction", "background", "overview", "引言", "背景")):
        return "introduction"
    if any(token in section for token in ("method", "methodology", "approach", "proposed", "framework", "model", "algorithm", "方法", "模型", "框架")):
        return "method"
    if any(token in section for token in ("experiment", "evaluation", "result", "performance", "empirical", "实验", "结果", "评估", "评价")):
        return "experiments"
    if any(token in section for token in ("conclusion", "discussion", "limitation", "future work", "结论", "讨论", "局限", "展望")):
        return "conclusion"
    return None


def _categorize_records(records: list[dict[str, Any]]) -> dict[str, str]:
    """先用章节语义归类，再为非标准方法标题提供保守的编号结构回退。"""
    categories: dict[str, str] = {}
    for record in records:
        chunk_id = _optional_string(record.get("chunk_id"))
        category = _section_category(record)
        if chunk_id and category:
            categories[chunk_id] = category

    # 很多论文把方法章节直接命名为模型/算法名，例如“3. ESDTW”，并不出现 method。
    # 若实验主章节为第 N 节，则它紧邻的第 N-1 节及子节通常是论文的方法主体；该规则
    # 只在没有任何关键词命中方法章节时启用，避免覆盖更明确的语义标题。
    if "method" not in set(categories.values()):
        experiment_numbers = [
            number
            for record in records
            if _section_category(record) == "experiments"
            for number in [_top_level_section_number(record)]
            if number is not None
        ]
        if experiment_numbers:
            inferred_method_number = min(experiment_numbers) - 1
            if inferred_method_number > 0:
                for record in records:
                    chunk_id = _optional_string(record.get("chunk_id"))
                    if (
                        chunk_id
                        and chunk_id not in categories
                        and _top_level_section_number(record) == inferred_method_number
                    ):
                        categories[chunk_id] = "method"
    return categories


def _top_level_section_number(record: dict[str, Any]) -> int | None:
    """提取 ``3.2`` / ``3`` 中的顶级章节编号；无编号章节返回 None。"""
    section = _optional_string(record.get("section")) or ""
    pieces = section.split(maxsplit=1)
    if not pieces:
        return None
    first_piece = pieces[0].rstrip(".")
    top_level = first_piece.split(".", maxsplit=1)[0]
    return int(top_level) if top_level.isdigit() else None


def _context_chunk(
    record: dict[str, Any], category: str, max_chunk_chars: int, remaining_chars: int
) -> OverviewContextChunk | None:
    """在双重字符预算内保留 chunk 开头的完整可读内容。"""
    if remaining_chars < 1:
        return None
    raw_text = _record_text(record)
    limit = min(max_chunk_chars, remaining_chars)
    text = raw_text if len(raw_text) <= limit else raw_text[:limit].rstrip() + "\n[本 chunk 后续内容已截断]"
    chunk_id = _optional_string(record.get("chunk_id"))
    if not chunk_id:
        raise PaperOverviewError("Workspace chunk is missing chunk_id.")
    return OverviewContextChunk(
        chunk_id=chunk_id,
        section=_optional_string(record.get("section")) or "未标注章节",
        category=category,
        text=text,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取临时工作区自己的 JSON，拒绝缺失或非对象格式。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperOverviewError(f"Cannot read temporary workspace artifact: {path}") from error
    if not isinstance(payload, dict):
        raise PaperOverviewError(f"Temporary workspace JSON must be an object: {path}")
    return payload


def _load_chunk_records(path: Path) -> list[dict[str, Any]]:
    """读取 v0.1 PaperChunk JSONL，保留文件中原有的稳定 chunk 顺序。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PaperOverviewError(f"Cannot read temporary workspace chunks: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PaperOverviewError(f"Invalid chunk JSONL at line {line_number}: {path}") from error
        if not isinstance(record, dict):
            raise PaperOverviewError(f"Chunk JSONL line {line_number} must be an object.")
        records.append(record)
    return records


def _workspace_root(staging_dir: Path, workspace_id: str) -> Path:
    """仅允许通过 staging 根目录中的一个合法 workspace ID 定位输入目录。"""
    if not workspace_id.startswith("staging_") or any(char in workspace_id for char in "/\\"):
        raise PaperOverviewError("Invalid temporary workspace ID.")
    root = staging_dir.resolve()
    workspace = (root / workspace_id).resolve()
    if workspace.parent != root or not workspace.is_dir():
        raise PaperOverviewError("Temporary workspace does not exist inside the staging directory.")
    return workspace


def _record_text(record: dict[str, Any]) -> str:
    """Overview 一律使用 raw_text，避免把 embedding 提示词混入模型阅读上下文。"""
    return _optional_string(record.get("raw_text")) or ""


def _optional_string(value: object) -> str | None:
    """将 JSON 中的可读字符串归一化；其他类型一律视为缺失。"""
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _unique_strings(values: Iterable[str]) -> list[str]:
    """去除空白和重复项，保留首次出现顺序以便前端稳定展示。"""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _optional_string(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _with_overview_output_budget(
    runtime_settings: LLMRuntimeSettings, max_output_tokens: int
) -> LLMRuntimeSettings:
    """只为 Overview 覆盖输出预算，保留同一模型、密钥、超时与温度设置。"""
    return replace(runtime_settings, max_tokens=max_output_tokens)
