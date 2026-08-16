"""从 Temporary Paper Workspace 生成可追溯的单篇论文速览。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
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
_OVERVIEW_ROLES = (
    "research_problem",
    "contributions",
    "main_method",
    "datasets",
    "experimental_setup",
    "main_results",
    "limitations",
)
# 方法和结果是速览中信息密度最高、也最容易被章节前缀偏差遗漏的内容，因此优先入预算。
_ROLE_PRIORITIES = {
    "main_method": 3,
    "main_results": 3,
    "research_problem": 2,
    "contributions": 2,
    "datasets": 2,
    "experimental_setup": 2,
    "limitations": 2,
}


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
    overview_roles: tuple[str, ...]
    token_count: int
    text: str


@dataclass(frozen=True)
class OverviewContext:
    """按章节选择后的 Prompt 数据，不含向量、FAISS 或其他论文内容。"""

    paper_title: str | None
    chunks: tuple[OverviewContextChunk, ...]
    selection_debug: dict[str, Any]

    @property
    def rendered_text(self) -> str:
        """将每个来源明确标为 chunk ID，供模型引用并供本地校验。"""
        return "\n\n".join(
            "[chunk_id: {chunk_id}]\n[category: {category}]\n[section: {section}]\n"
            "[overview_roles: {overview_roles}]\n{content}".format(
                chunk_id=chunk.chunk_id,
                category=chunk.category,
                section=chunk.section or "未标注章节",
                overview_roles=", ".join(chunk.overview_roles),
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
    selection_debug: dict[str, Any]


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
    """从 workspace 构造“字段候选 -> 去重 -> token 预算”的 Overview Context。"""
    root = workspace_root.resolve()
    parsed_record = _load_json_object(root / "parsed" / "parsed_paper.json")
    raw_chunks = _load_chunk_records(root / "chunks" / "chunks.jsonl")
    candidates = [
        record
        for record in raw_chunks
        if record.get("section_type") == "content" and _record_text(record)
    ]
    if not candidates:
        return OverviewContext(
            paper_title=_optional_string(parsed_record.get("paper_title")),
            chunks=(),
            selection_debug={
                "role_candidates": {role: [] for role in _OVERVIEW_ROLES},
                "union_before_budget": [],
                "final_selected_chunks": [],
                "final_token_count": 0,
            },
        )

    # 所有输入来自已落盘的 PaperChunk JSONL：这里不调用 Parser、Embedder 或索引。
    categories_by_chunk_id = _categorize_records(candidates)
    positions_by_chunk_id = _section_positions(candidates)
    role_candidates = _select_role_candidates(
        records=candidates,
        categories_by_chunk_id=categories_by_chunk_id,
        positions_by_chunk_id=positions_by_chunk_id,
        settings=settings,
    )
    selected = _select_union_with_token_budget(
        records=candidates,
        categories_by_chunk_id=categories_by_chunk_id,
        role_candidates=role_candidates,
        settings=settings,
    )

    # 章节名完全失效时才回退到少量正文；不会向该回退路径加入 References。
    if not selected and settings.max_fallback_chunks:
        selected = _select_fallback_chunks(candidates, settings)

    debug = {
        "role_candidates": {
            role: [record["chunk_id"] for record, _score in records]
            for role, records in role_candidates.items()
        },
        "union_before_budget": _union_chunk_ids(role_candidates),
        "final_selected_chunks": [chunk.chunk_id for chunk in selected],
        "final_token_count": sum(chunk.token_count for chunk in selected),
        "token_budget": settings.max_total_context_tokens,
    }

    return OverviewContext(
        paper_title=_optional_string(parsed_record.get("paper_title")),
        chunks=tuple(selected),
        selection_debug=debug,
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
                        "overview_roles": list(chunk.overview_roles),
                        "token_count": chunk.token_count,
                    }
                    for chunk in context.chunks
                ],
                "selection_debug": context.selection_debug,
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
        selection_debug=context.selection_debug,
    )


def _section_category(record: dict[str, Any]) -> str | None:
    """按章节语义而非固定完整标题归类常见中英文论文结构。"""
    section = " ".join(str(record.get("section") or "").casefold().split())
    front_matter_type = str(record.get("front_matter_type") or "").casefold()
    if front_matter_type == "abstract" or "abstract" in section or "摘要" in section:
        return "abstract"
    if any(token in section for token in ("method", "methodology", "approach", "proposed", "framework", "model", "algorithm", "方法", "模型", "框架")):
        return "method"
    if any(token in section for token in ("introduction", "background", "overview", "引言", "背景")):
        return "introduction"
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

    # 很多论文把方法章节直接命名为模型/算法名，例如“3. ESDTW”，并不出现 method；
    # 但其某个子节可能叫“3.3 Algorithm overview”。此时仍要补齐同属第 3 章、尚未
    # 有明确语义的 3.1/3.2，才能保证方法选择器覆盖多个真实模块而非只看到一个子节。
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


def _select_role_candidates(
    *,
    records: list[dict[str, Any]],
    categories_by_chunk_id: dict[str, str],
    positions_by_chunk_id: dict[str, tuple[int, int]],
    settings: PaperOverviewSettings,
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    """为每个 Overview 字段独立挑选高价值候选；此处尚不消耗全局 token 预算。"""
    limits = {
        "research_problem": settings.research_problem_candidate_limit,
        "contributions": settings.contributions_candidate_limit,
        "main_method": settings.main_method_candidate_limit,
        "datasets": settings.datasets_candidate_limit,
        "experimental_setup": settings.experimental_setup_candidate_limit,
        "main_results": settings.main_results_candidate_limit,
        "limitations": settings.limitations_candidate_limit,
    }
    selected: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for role in _OVERVIEW_ROLES:
        scored = [
            (record, _role_score(
                role=role,
                record=record,
                category=categories_by_chunk_id.get(_required_chunk_id(record)),
                position=positions_by_chunk_id.get(_required_chunk_id(record), (0, 1)),
            ))
            for record in records
        ]
        # 分数为零的 chunk 对该字段没有已知证据价值，不能仅因排在前面占用候选名额。
        scored = [(record, score) for record, score in scored if score > 0]
        scored.sort(key=lambda item: (-item[1], _chunk_index(item[0]), _required_chunk_id(item[0])))
        selected[role] = (
            _select_diverse_method_candidates(scored, limits[role])
            if role == "main_method"
            else scored[: limits[role]]
        )
    return selected


def _role_score(
    *, role: str, record: dict[str, Any], category: str | None, position: tuple[int, int]
) -> float:
    """以可解释的章节、位置与文本信号评分，不把评分结果当作论文结论。"""
    section = (_optional_string(record.get("section")) or "").casefold()
    text = _record_text(record).casefold()
    section_position = position[0] / max(position[1] - 1, 1)
    score = 0.0
    if role == "research_problem":
        score += 9 if category == "abstract" else 5 if category == "introduction" else 0
        score += 2 * (1 - section_position) if category == "introduction" else 0
        score += 5 * _signal_count(text, ("problem", "challenge", "limitation", "however", "issue", "difficult", "drawback", "问题", "挑战", "局限", "然而", "不足"))
    elif role == "contributions":
        score += 8 if category == "abstract" else 4 if category == "introduction" else 2 if category == "conclusion" else 0
        score += 3 * section_position if category == "introduction" else 0
        score += 6 * _signal_count(text, ("contribution", "contributions", "we propose", "our work", "in summary", "we present", "提出", "贡献", "本文", "总结"))
    elif role == "main_method":
        # 章节结构已识别为 Method 时应明显优先于背景段落中偶然出现的 “method” 一词。
        score += 20 if category == "method" else 2 if category == "abstract" else 0
        score += 5 * _signal_count(section + " " + text, ("method", "approach", "framework", "algorithm", "model", "proposed", "overview", "methodology", "方法", "框架", "算法", "模型", "提出"))
    elif role == "datasets":
        score += 4 if category == "experiments" else 0
        score += 8 * _signal_count(section + " " + text, ("dataset", "datasets", "data set", "data collection", "benchmark", "corpus", "ucr", "数据集", "数据集", "基准"))
    elif role == "experimental_setup":
        score += 4 if category == "experiments" else 0
        score += 7 * _signal_count(section + " " + text, ("setup", "baseline", "metric", "implementation", "setting", "protocol", "parameter", "evaluation", "experiment", "设置", "基线", "指标", "实现", "参数", "实验"))
    elif role == "main_results":
        score += 5 if category == "experiments" else 4 if category == "conclusion" else 0
        score += 9 * _signal_count(section + " " + text, ("result", "performance", "comparison", "outperform", "improvement", "better than", "accuracy", "rmse", "mae", "mape", "error", "table", "figure", "结果", "性能", "比较", "优于", "提升", "准确率", "误差", "表", "图"))
        score += min(_numeric_density(text) * 8, 6)
    elif role == "limitations":
        score += 5 if category == "conclusion" else 0
        score += 9 * _signal_count(section + " " + text, ("limitation", "limitations", "fail", "failure", "worse", "sensitive", "however", "drawback", "future work", "error analysis", "局限", "失败", "较差", "敏感", "然而", "缺点", "未来工作"))
    return score


def _select_diverse_method_candidates(
    scored: list[tuple[dict[str, Any], float]], limit: int
) -> list[tuple[dict[str, Any], float]]:
    """方法字段先覆盖不同 subsection，再在必要时补充同一 subsection 的高分块。"""
    selected: list[tuple[dict[str, Any], float]] = []
    seen_sections: set[str] = set()
    for record, score in scored:
        section = _optional_string(record.get("section")) or "未标注章节"
        if section in seen_sections:
            continue
        selected.append((record, score))
        seen_sections.add(section)
        if len(selected) == limit:
            return selected
    for record, score in scored:
        if (record, score) in selected:
            continue
        selected.append((record, score))
        if len(selected) == limit:
            break
    return selected


def _select_union_with_token_budget(
    *,
    records: list[dict[str, Any]],
    categories_by_chunk_id: dict[str, str],
    role_candidates: dict[str, list[tuple[dict[str, Any], float]]],
    settings: PaperOverviewSettings,
) -> list[OverviewContextChunk]:
    """合并字段候选、按角色优先级贪心入预算，最后恢复论文原始阅读顺序。"""
    records_by_id = {_required_chunk_id(record): record for record in records}
    roles_by_chunk_id: dict[str, set[str]] = {}
    best_score_by_chunk_id: dict[str, float] = {}
    for role, candidates in role_candidates.items():
        for record, score in candidates:
            chunk_id = _required_chunk_id(record)
            roles_by_chunk_id.setdefault(chunk_id, set()).add(role)
            best_score_by_chunk_id[chunk_id] = max(best_score_by_chunk_id.get(chunk_id, 0.0), score)

    ordered_ids = sorted(
        roles_by_chunk_id,
        key=lambda chunk_id: (
            -max(_ROLE_PRIORITIES[role] for role in roles_by_chunk_id[chunk_id]),
            -best_score_by_chunk_id[chunk_id],
            _chunk_index(records_by_id[chunk_id]),
            chunk_id,
        ),
    )
    selected: list[OverviewContextChunk] = []
    abstract_count = 0
    remaining_tokens = settings.max_total_context_tokens
    for chunk_id in ordered_ids:
        record = records_by_id[chunk_id]
        category = categories_by_chunk_id.get(chunk_id, "fallback")
        if category == "abstract" and abstract_count >= settings.max_abstract_chunks:
            continue
        chunk = _context_chunk(
            record=record,
            category=category,
            roles=tuple(sorted(roles_by_chunk_id[chunk_id])),
            max_tokens=settings.max_tokens_per_chunk,
        )
        if chunk.token_count > remaining_tokens:
            continue
        selected.append(chunk)
        remaining_tokens -= chunk.token_count
        if category == "abstract":
            abstract_count += 1

    return sorted(selected, key=lambda chunk: (_chunk_index(records_by_id[chunk.chunk_id]), chunk.chunk_id))


def _select_fallback_chunks(
    records: list[dict[str, Any]], settings: PaperOverviewSettings
) -> list[OverviewContextChunk]:
    """核心章节均无法识别时，保留最早少量正文作为显式 fallback。"""
    selected: list[OverviewContextChunk] = []
    remaining_tokens = settings.max_total_context_tokens
    for record in records[: settings.max_fallback_chunks]:
        chunk = _context_chunk(
            record=record,
            category="fallback",
            roles=("fallback",),
            max_tokens=settings.max_tokens_per_chunk,
        )
        if chunk.token_count > remaining_tokens:
            continue
        selected.append(chunk)
        remaining_tokens -= chunk.token_count
    return selected


def _union_chunk_ids(role_candidates: dict[str, list[tuple[dict[str, Any], float]]]) -> list[str]:
    """以角色顺序聚合候选 ID，便于 debug 中观察合并前的去重边界。"""
    result: list[str] = []
    seen: set[str] = set()
    for role in _OVERVIEW_ROLES:
        for record, _score in role_candidates[role]:
            chunk_id = _required_chunk_id(record)
            if chunk_id not in seen:
                seen.add(chunk_id)
                result.append(chunk_id)
    return result


def _section_positions(records: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """计算 chunk 在各自 section 内的位置，用于区分引言前部问题与后部贡献。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_optional_string(record.get("section")) or "未标注章节", []).append(record)
    positions: dict[str, tuple[int, int]] = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda record: (_chunk_index(record), _required_chunk_id(record)))
        for index, record in enumerate(ordered):
            positions[_required_chunk_id(record)] = (index, len(ordered))
    return positions


def _context_chunk(
    *, record: dict[str, Any], category: str, roles: tuple[str, ...], max_tokens: int
) -> OverviewContextChunk:
    """以已存 raw_token_count 为预算单位，必要时按比例截断 raw_text。"""
    raw_text = _record_text(record)
    original_tokens = _record_token_count(record)
    token_count = min(original_tokens, max_tokens)
    if original_tokens > max_tokens:
        char_limit = max(1, int(len(raw_text) * max_tokens / original_tokens))
        text = raw_text[:char_limit].rstrip() + "\n[本 chunk 后续内容已截断]"
    else:
        text = raw_text
    return OverviewContextChunk(
        chunk_id=_required_chunk_id(record),
        section=_optional_string(record.get("section")) or "未标注章节",
        category=category,
        overview_roles=roles,
        token_count=token_count,
        text=text,
    )


def _signal_count(text: str, signals: tuple[str, ...]) -> int:
    """统计不同关键词是否出现；重复词不反复加分，降低长段落的长度偏置。"""
    return sum(signal in text for signal in signals)


def _numeric_density(text: str) -> float:
    """用数字、百分号和常见指标的出现比例识别结果密度，不解释这些数字的业务含义。"""
    if not text:
        return 0.0
    return len(re.findall(r"\d+(?:\.\d+)?%?", text)) / max(len(text.split()), 1)


def _record_token_count(record: dict[str, Any]) -> int:
    """优先复用 Chunker 已生成的 token 计数；旧工件缺失时使用保守字符估算。"""
    value = record.get("raw_token_count")
    if isinstance(value, int) and value > 0:
        return value
    return max(1, (len(_record_text(record)) + 3) // 4)


def _chunk_index(record: dict[str, Any]) -> int:
    """读取稳定 chunk_index；缺失时置于同分候选末尾。"""
    value = record.get("chunk_index")
    return value if isinstance(value, int) and value >= 0 else 2**31 - 1


def _required_chunk_id(record: dict[str, Any]) -> str:
    """所有进入 selector 的记录都必须保留 v0.1 PaperChunk 的稳定身份。"""
    chunk_id = _optional_string(record.get("chunk_id"))
    if not chunk_id:
        raise PaperOverviewError("Workspace chunk is missing chunk_id.")
    return chunk_id


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
