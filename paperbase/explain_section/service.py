"""只依赖 Temporary Workspace 持久化结构的 Explain Section 服务。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbase.config import AppSettings, ExplainSectionSettings
from paperbase.llm.client import (
    ChatCompletionClient,
    LLMRequestError,
    LLMRuntimeSettings,
    OpenAICompatibleChatClient,
    load_llm_runtime_settings,
)
from paperbase.prompts.explain_section import (
    EXPLAIN_SECTION_SYSTEM_PROMPT,
    build_explain_section_user_prompt,
)
from paperbase.staging.sections import (
    WorkspaceChunk,
    WorkspaceSection,
    WorkspaceSectionRepository,
    WorkspaceSectionSnapshot,
)


ExplainMode = Literal["section_overview", "section_explanation"]


class ExplainSectionError(RuntimeError):
    """Explain 的 context、模型输出或写入不符合证据边界时抛出。"""


class _ExplainSectionDraft(BaseModel):
    """模型只负责解释和引用；章节身份、标题与 mode 均由程序回填。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    explanation: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list, max_length=10)
    source_chunk_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool


class ExplainSection(BaseModel):
    """对外稳定的 Explain Section 输出。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    section_id: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    mode: ExplainMode
    explanation: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list, max_length=10)
    source_chunk_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool


@dataclass(frozen=True)
class ExplainContextChunk:
    """一次 Explain 请求实际发送给 LLM 的受控证据块。"""

    chunk_id: str
    section_id: str
    section: str
    chunk_index: int
    token_count: int
    text: str


@dataclass(frozen=True)
class ExplainSectionContext:
    """由真实 Section Tree 决定模式、范围和 token 预算的 Prompt 输入。"""

    selected_section: WorkspaceSection
    mode: ExplainMode
    direct_children: tuple[WorkspaceSection, ...]
    chunks: tuple[ExplainContextChunk, ...]
    full_candidate_token_count: int
    final_token_count: int
    target_token_budget: int
    max_token_budget: int
    selection_debug: dict[str, object]

    @property
    def rendered_context(self) -> str:
        """显式标注每块的稳定身份，供模型引用并由程序验证来源。"""
        return "\n\n".join(
            "[chunk_id: {chunk_id}]\n[section_id: {section_id}]\n[section: {section}]\n{content}".format(
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                section=chunk.section or "未标注章节",
                content=chunk.text,
            )
            for chunk in self.chunks
        )

    @property
    def rendered_tree(self) -> str:
        """父章节只展示其自身子树轮廓，帮助模型解释章节职责关系而不提供 sibling 正文。"""
        lines = [
            "[{section_id}] {title}".format(
                section_id=self.selected_section.section_id,
                title=self.selected_section.section_title,
            )
        ]
        for child in self.direct_children:
            lines.append(
                "  └─ [{section_id}] {title}".format(
                    section_id=child.section_id, title=child.section_title
                )
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ExplainSectionOutcome:
    """一次成功 Explain 的结构化结果、证据快照与落盘位置。"""

    explanation: ExplainSection
    context: ExplainSectionContext
    explanation_path: Path
    context_path: Path
    selected_chunk_count: int


def run_explain_section_stage(
    *, settings: AppSettings, workspace_id: str, section_id: str
) -> ExplainSectionOutcome:
    """按项目配置解释当前 temporary workspace 的一个章节。"""
    repository = WorkspaceSectionRepository(settings.storage.staging_dir)
    snapshot = repository.load(workspace_id)
    runtime_settings = load_llm_runtime_settings(settings.config_path.parent / ".env")
    client = OpenAICompatibleChatClient(
        _with_explain_output_budget(runtime_settings, settings.explain_section.max_output_tokens)
    )
    return create_explain_section(
        snapshot=snapshot,
        section_id=section_id,
        settings=settings.explain_section,
        client=client,
    )


def create_explain_section(
    *,
    snapshot: WorkspaceSectionSnapshot,
    section_id: str,
    settings: ExplainSectionSettings,
    client: ChatCompletionClient,
) -> ExplainSectionOutcome:
    """从已保存的 workspace 建 Context、调用一次 LLM，并保存可审计结果。"""
    context = build_explain_section_context(
        snapshot=snapshot, section_id=section_id, settings=settings
    )
    try:
        raw_response = client.complete_json(
            system_prompt=EXPLAIN_SECTION_SYSTEM_PROMPT,
            user_prompt=build_explain_section_user_prompt(
                section_id=context.selected_section.section_id,
                section_title=context.selected_section.section_title,
                mode=context.mode,
                section_tree=context.rendered_tree,
                context=context.rendered_context,
                context_chunk_count=len(context.chunks),
                context_token_count=context.final_token_count,
                minimum_explanation_chars=_minimum_explanation_chars(context),
            ),
            json_schema=_ExplainSectionDraft.model_json_schema(),
            schema_name="paperbase_explain_section",
        )
        explanation = _validate_and_finalize_explanation(
            draft=_ExplainSectionDraft.model_validate_json(raw_response),
            context=context,
            settings=settings,
        )
    except (LLMRequestError, ValidationError, ValueError, TypeError) as error:
        raise ExplainSectionError("Explain Section generation returned an invalid result.") from error
    return _write_explain_artifacts(
        snapshot=snapshot, explanation=explanation, context=context
    )


def build_explain_section_context(
    *,
    snapshot: WorkspaceSectionSnapshot,
    section_id: str,
    settings: ExplainSectionSettings,
) -> ExplainSectionContext:
    """只根据持久化 Section Tree 与精确 chunk.section_id 构造 Explain Context。"""
    selected = snapshot.get_section(section_id)
    children = snapshot.get_children(selected.section_id)
    mode: ExplainMode = "section_overview" if children else "section_explanation"
    direct_chunks = snapshot.get_direct_chunks(selected.section_id)
    if mode == "section_explanation":
        chunks, debug = _build_leaf_context(direct_chunks, settings)
        full_candidate_token_count = sum(chunk.raw_token_count for chunk in direct_chunks)
    else:
        chunks, debug = _build_parent_context(
            snapshot=snapshot,
            selected=selected,
            direct_children=children,
            direct_chunks=direct_chunks,
            settings=settings,
        )
        full_candidate_token_count = sum(
            chunk.raw_token_count for chunk in snapshot.get_subtree_chunks(selected.section_id)
        )
    return ExplainSectionContext(
        selected_section=selected,
        mode=mode,
        direct_children=children,
        chunks=tuple(chunks),
        full_candidate_token_count=full_candidate_token_count,
        final_token_count=sum(chunk.token_count for chunk in chunks),
        target_token_budget=settings.target_context_tokens,
        max_token_budget=settings.max_context_tokens,
        selection_debug=debug,
    )


def _build_leaf_context(
    direct_chunks: tuple[WorkspaceChunk, ...], settings: ExplainSectionSettings
) -> tuple[list[ExplainContextChunk], dict[str, object]]:
    """叶节点只看直属正文；超长时在本节内部保留开头、核心、结尾等代表块。"""
    full_tokens = sum(chunk.raw_token_count for chunk in direct_chunks)
    if full_tokens <= settings.max_context_tokens:
        selected = [_context_chunk(chunk, max_tokens=chunk.raw_token_count) for chunk in direct_chunks]
        return selected, {
            "strategy": "leaf_all_direct_chunks",
            "direct_chunk_ids": [chunk.chunk_id for chunk in direct_chunks],
            "final_selected_chunks": [chunk.chunk_id for chunk in direct_chunks],
            "full_direct_token_count": full_tokens,
            "truncated": False,
        }

    representatives = _representative_chunks(
        direct_chunks, limit=max(settings.max_representative_chunks_per_branch * 3, 5)
    )
    selected = _fit_representatives(
        representatives=representatives,
        max_tokens=settings.max_context_tokens,
        per_chunk_limit=settings.max_tokens_per_chunk,
    )
    return selected, {
        "strategy": "leaf_structured_truncation",
        "direct_chunk_ids": [chunk.chunk_id for chunk in direct_chunks],
        "representative_chunk_ids": [chunk.chunk_id for chunk in representatives],
        "final_selected_chunks": [chunk.chunk_id for chunk in selected],
        "full_direct_token_count": full_tokens,
        "truncated": True,
    }


def _build_parent_context(
    *,
    snapshot: WorkspaceSectionSnapshot,
    selected: WorkspaceSection,
    direct_children: tuple[WorkspaceSection, ...],
    direct_chunks: tuple[WorkspaceChunk, ...],
    settings: ExplainSectionSettings,
) -> tuple[list[ExplainContextChunk], dict[str, object]]:
    """父节点按 direct child 分支取代表块，优先覆盖所有子章节而非只取 subtree 前缀。"""
    branches: list[tuple[str, tuple[WorkspaceChunk, ...]]] = []
    if direct_chunks:
        branches.append((selected.section_id, direct_chunks))
    for child in direct_children:
        # 每一 branch 均包含 child 自身和全部后代，但不会把 sibling 的正文混入。
        branch_chunks = snapshot.get_subtree_chunks(child.section_id)
        if branch_chunks:
            branches.append((child.section_id, branch_chunks))

    representatives_by_branch = {
        branch_id: _representative_chunks(
            chunks, limit=settings.max_representative_chunks_per_branch
        )
        for branch_id, chunks in branches
    }
    selected_chunks: list[ExplainContextChunk] = []
    selected_ids: set[str] = set()
    used_tokens = 0

    # Round-robin 首轮保证每个有正文的 direct child 都获得进入 Context 的机会。
    for round_index in range(settings.max_representative_chunks_per_branch):
        for branch_id, _ in branches:
            representatives = representatives_by_branch[branch_id]
            if round_index >= len(representatives):
                continue
            source = representatives[round_index]
            if source.chunk_id in selected_ids:
                continue
            candidate = _context_chunk(source, max_tokens=settings.max_tokens_per_chunk)
            # target 是“优先目标”；若首轮覆盖需要更多空间，允许安全扩到 max 上限。
            limit = (
                settings.target_context_tokens
                if used_tokens + candidate.token_count <= settings.target_context_tokens
                else settings.max_context_tokens
            )
            if used_tokens + candidate.token_count > limit:
                continue
            selected_chunks.append(candidate)
            selected_ids.add(source.chunk_id)
            used_tokens += candidate.token_count

    selected_chunks.sort(key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))
    covered_branch_ids = {
        branch_id
        for branch_id, representatives in representatives_by_branch.items()
        if any(chunk.chunk_id in {item.chunk_id for item in representatives} for chunk in selected_chunks)
    }
    return selected_chunks, {
        "strategy": "parent_subtree_diverse_representatives",
        "selected_section_direct_chunk_ids": [chunk.chunk_id for chunk in direct_chunks],
        "branches": {
            branch_id: [chunk.chunk_id for chunk in chunks]
            for branch_id, chunks in branches
        },
        "representative_chunk_ids_by_branch": {
            branch_id: [chunk.chunk_id for chunk in chunks]
            for branch_id, chunks in representatives_by_branch.items()
        },
        "covered_branch_ids": sorted(covered_branch_ids),
        "final_selected_chunks": [chunk.chunk_id for chunk in selected_chunks],
    }


def _representative_chunks(
    chunks: tuple[WorkspaceChunk, ...], *, limit: int
) -> tuple[WorkspaceChunk, ...]:
    """从一个 section/subtree 中选开头、中央、结尾和公式/结果信号较强的代表块。"""
    if not chunks or limit < 1:
        return ()
    indexed = list(enumerate(chunks))
    anchor_indexes = [0, len(chunks) // 2, len(chunks) - 1]
    # 先挑三个位置锚点，消除“只保留章节开头”的 prefix bias。
    selected_indexes: list[int] = []
    for index in anchor_indexes:
        if index not in selected_indexes:
            selected_indexes.append(index)
        if len(selected_indexes) == limit:
            return tuple(chunks[index] for index in sorted(selected_indexes))
    scored_indexes = sorted(
        indexed,
        key=lambda item: (-_information_signal(item[1]), item[0], item[1].chunk_id),
    )
    for index, _chunk in scored_indexes:
        if index not in selected_indexes:
            selected_indexes.append(index)
        if len(selected_indexes) == limit:
            break
    return tuple(chunks[index] for index in sorted(selected_indexes))


def _information_signal(chunk: WorkspaceChunk) -> float:
    """只用于选择 evidence：公式符号、数字和结果词提高代表性，不生成任何论文事实。"""
    text = chunk.raw_text.casefold()
    keyword_hits = sum(
        signal in text
        for signal in (
            "equation", "formula", "theorem", "algorithm", "table", "figure",
            "result", "accuracy", "rmse", "mae", "mape", "outperform", "improvement",
        )
    )
    numeric_hits = len(re.findall(r"\d+(?:\.\d+)?%?", text))
    formula_hits = text.count("=") + text.count("∑") + text.count("∈")
    return keyword_hits * 5 + min(numeric_hits, 10) + formula_hits


def _fit_representatives(
    *, representatives: tuple[WorkspaceChunk, ...], max_tokens: int, per_chunk_limit: int
) -> list[ExplainContextChunk]:
    """把已选代表块截断并放入硬 token 预算，最终恢复原始阅读顺序。"""
    selected: list[ExplainContextChunk] = []
    remaining = max_tokens
    for source in representatives:
        if remaining <= 0:
            break
        cap = min(per_chunk_limit, remaining)
        candidate = _context_chunk(source, max_tokens=cap)
        selected.append(candidate)
        remaining -= candidate.token_count
    return sorted(selected, key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))


def _context_chunk(chunk: WorkspaceChunk, *, max_tokens: int) -> ExplainContextChunk:
    """把一个持久化 chunk 变为 Prompt 证据，必要时保留首/中/尾的结构化文本片段。"""
    original_tokens = max(1, chunk.raw_token_count)
    token_count = min(original_tokens, max(1, max_tokens))
    text = (
        chunk.raw_text
        if original_tokens <= token_count
        else _structured_text_crop(chunk.raw_text, original_tokens, token_count)
    )
    return ExplainContextChunk(
        chunk_id=chunk.chunk_id,
        section_id=chunk.section_id or "",
        section=chunk.section or "未标注章节",
        chunk_index=chunk.chunk_index,
        token_count=token_count,
        text=text,
    )


def _structured_text_crop(text: str, original_tokens: int, target_tokens: int) -> str:
    """按字符比例保留开头、中央、结尾，避免传统尾部截断丢失结果或结论。"""
    if not text:
        return text
    keep_chars = max(3, int(len(text) * target_tokens / original_tokens))
    if keep_chars >= len(text):
        return text
    head_size = max(1, int(keep_chars * 0.35))
    middle_size = max(1, int(keep_chars * 0.35))
    tail_size = max(1, keep_chars - head_size - middle_size)
    middle_start = max(head_size, len(text) // 2 - middle_size // 2)
    middle_end = min(len(text) - tail_size, middle_start + middle_size)
    return (
        text[:head_size].rstrip()
        + "\n[中段省略]\n"
        + text[middle_start:middle_end].strip()
        + "\n[中段省略]\n"
        + text[-tail_size:].lstrip()
    )


def _validate_and_finalize_explanation(
    *,
    draft: _ExplainSectionDraft,
    context: ExplainSectionContext,
    settings: ExplainSectionSettings,
) -> ExplainSection:
    """拒绝模型伪造 section 身份、超出 Context 的来源或不一致的证据不足状态。"""
    source_ids = _unique_strings(draft.source_chunk_ids)
    context_ids = {chunk.chunk_id for chunk in context.chunks}
    unknown_ids = set(source_ids).difference(context_ids)
    if unknown_ids:
        raise ValueError(f"Explain references chunks outside the provided context: {unknown_ids}")
    explanation = "\n".join(line.strip() for line in draft.explanation.splitlines() if line.strip())
    key_points = _unique_strings(draft.key_points)
    if len(key_points) > settings.max_key_points:
        raise ValueError("Explain returned more key_points than the configured limit.")
    has_evidence = bool(context.chunks)
    if not has_evidence:
        if not draft.insufficient_evidence or source_ids or key_points:
            raise ValueError("An empty section context must return an uncited insufficient-evidence result.")
    elif draft.insufficient_evidence:
        raise ValueError("A section with supplied body evidence must not be marked insufficient.")
    elif not source_ids:
        raise ValueError("An evidence-grounded explanation must cite at least one context chunk.")
    return ExplainSection(
        section_id=context.selected_section.section_id,
        section_title=context.selected_section.section_title,
        mode=context.mode,
        explanation=explanation,
        key_points=key_points,
        source_chunk_ids=source_ids,
        insufficient_evidence=draft.insufficient_evidence,
    )


def _minimum_explanation_chars(context: ExplainSectionContext) -> int:
    """按本次真实 evidence 密度给 Prompt 一个合理下限，不把长度规则散落在 Prompt 文字里。"""
    if len(context.chunks) >= 6 and context.final_token_count >= 1_800:
        return 900
    if len(context.chunks) >= 3 and context.final_token_count >= 600:
        return 600
    if context.chunks:
        return 350
    return 0


def _write_explain_artifacts(
    *,
    snapshot: WorkspaceSectionSnapshot,
    explanation: ExplainSection,
    context: ExplainSectionContext,
) -> ExplainSectionOutcome:
    """在当前 workspace 内保存结果和本次准确输入的 context 记录。"""
    output_dir = snapshot.root_dir / "explain_sections"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = hashlib.sha256(explanation.section_id.encode("utf-8")).hexdigest()[:16]
    explanation_path = output_dir / f"{artifact_stem}.json"
    context_path = output_dir / f"{artifact_stem}.context.json"
    explanation_path.write_text(
        json.dumps(explanation.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    context_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "section_id": explanation.section_id,
                "section_title": explanation.section_title,
                "mode": explanation.mode,
                "selected_chunks": [asdict(chunk) for chunk in context.chunks],
                "full_candidate_token_count": context.full_candidate_token_count,
                "final_token_count": context.final_token_count,
                "target_token_budget": context.target_token_budget,
                "max_token_budget": context.max_token_budget,
                "selection_debug": context.selection_debug,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ExplainSectionOutcome(
        explanation=explanation,
        context=context,
        explanation_path=explanation_path,
        context_path=context_path,
        selected_chunk_count=len(context.chunks),
    )


def _unique_strings(values: Iterable[object]) -> list[str]:
    """去重、去空白并保留模型给出的首次引用顺序。"""
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _with_explain_output_budget(
    runtime_settings: LLMRuntimeSettings, max_output_tokens: int
) -> LLMRuntimeSettings:
    """仅复制本次 Explain 的运行时配置，不能修改共享 LLM 默认输出预算。"""
    return replace(runtime_settings, max_tokens=max_output_tokens)
