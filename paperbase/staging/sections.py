"""Temporary Workspace 的 Section Tree 与 chunk 只读访问层。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class WorkspaceSectionError(RuntimeError):
    """工作区 ID、Section Tree 或 chunk 持久化数据不符合契约时抛出。"""


@dataclass(frozen=True)
class WorkspaceSection:
    """从 ``parsed_paper.json`` 读取的解析器无关 Section 节点。"""

    section_id: str
    paper_id: str
    section_title: str
    section_number: str | None
    section_level: int
    parent_section_id: str | None
    section_index: int
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class WorkspaceChunk:
    """Explain Section 需要的已落盘 PaperChunk 字段，不读取 embedding 或索引。"""

    chunk_id: str
    chunk_index: int
    raw_text: str
    raw_token_count: int
    section: str | None
    section_id: str | None
    section_type: str
    # 以下字段同样来自既有 PaperChunk JSONL；为 Ask This Paper 保留展示和证据页码。
    paper_title: str | None = None
    content_kind: str = "body"
    front_matter_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class WorkspaceSectionSnapshot:
    """一次工作区读取的稳定快照，所有 hierarchy 查询均以 ``sections`` 为事实来源。"""

    workspace_id: str
    root_dir: Path
    paper_id: str
    sections: tuple[WorkspaceSection, ...]
    chunks: tuple[WorkspaceChunk, ...]
    # 标题只供检索结果展示和回答 Prompt 使用；单篇 BM25 不索引它。
    paper_title: str | None = None

    def get_section_tree(self) -> tuple[WorkspaceSection, ...]:
        """按原始 heading 阅读顺序返回完整 Section Tree 的扁平节点序列。"""
        return self.sections

    def get_section(self, section_id: str) -> WorkspaceSection:
        """返回指定节点；未知 ID 不能静默返回空值。"""
        normalized_id = _required_string(section_id, "section_id")
        for section in self.sections:
            if section.section_id == normalized_id:
                return section
        raise WorkspaceSectionError(
            f"Section does not exist in workspace {self.workspace_id}: {normalized_id}"
        )

    def get_children(self, section_id: str) -> tuple[WorkspaceSection, ...]:
        """返回直属子节点，排序只使用持久化的 section_index。"""
        selected = self.get_section(section_id)
        return tuple(
            section
            for section in self.sections
            if section.parent_section_id == selected.section_id
        )

    def get_descendants(self, section_id: str) -> tuple[WorkspaceSection, ...]:
        """深度优先返回全部后代；不通过标题编号或 section 文本推断关系。"""
        selected = self.get_section(section_id)
        descendants: list[WorkspaceSection] = []

        def visit(parent_id: str) -> None:
            for child in self.get_children(parent_id):
                descendants.append(child)
                visit(child.section_id)

        visit(selected.section_id)
        return tuple(descendants)

    def get_direct_chunks(self, section_id: str) -> tuple[WorkspaceChunk, ...]:
        """只返回 ``chunk.section_id`` 精确等于当前 Section 的正文 chunk。"""
        selected = self.get_section(section_id)
        return tuple(
            chunk
            for chunk in self.chunks
            if chunk.section_type == "content" and chunk.section_id == selected.section_id
        )

    def get_subtree_chunks(self, section_id: str) -> tuple[WorkspaceChunk, ...]:
        """返回当前节点与所有后代的直属正文 chunk，且保持全论文 chunk 阅读顺序。"""
        selected = self.get_section(section_id)
        included_ids = {
            selected.section_id,
            *(section.section_id for section in self.get_descendants(selected.section_id)),
        }
        return tuple(
            chunk
            for chunk in self.chunks
            if chunk.section_type == "content" and chunk.section_id in included_ids
        )


class WorkspaceSectionRepository:
    """以 staging 根目录为边界加载 Temporary Workspace，避免路径穿越到正式 KB。"""

    def __init__(self, staging_dir: Path) -> None:
        self._staging_dir = staging_dir.resolve()

    def load(self, workspace_id: str) -> WorkspaceSectionSnapshot:
        """读取 workspace.json、ParsedPaper sections 与 PaperChunk JSONL，并校验相互身份。"""
        normalized_workspace_id = _validate_workspace_id(workspace_id)
        root = (self._staging_dir / normalized_workspace_id).resolve()
        if root.parent != self._staging_dir or not root.is_dir():
            raise WorkspaceSectionError(
                f"Temporary workspace does not exist inside staging directory: {normalized_workspace_id}"
            )
        manifest = _load_json_object(root / "workspace.json")
        if manifest.get("workspace_id") != normalized_workspace_id:
            raise WorkspaceSectionError("workspace.json does not match its directory name.")
        paper_id = _required_string(manifest.get("paper_id"), "workspace.paper_id")
        parsed = _load_json_object(root / "parsed" / "parsed_paper.json")
        sections = _load_sections(parsed.get("sections"), paper_id=paper_id)
        chunks = _load_chunks(root / "chunks" / "chunks.jsonl", paper_id=paper_id)
        known_section_ids = {section.section_id for section in sections}
        if any(
            chunk.section_id is not None and chunk.section_id not in known_section_ids
            for chunk in chunks
        ):
            raise WorkspaceSectionError(
                "A workspace chunk references a section_id absent from parsed_paper.json."
            )
        return WorkspaceSectionSnapshot(
            workspace_id=normalized_workspace_id,
            root_dir=root,
            paper_id=paper_id,
            sections=sections,
            chunks=chunks,
            paper_title=_optional_string(parsed.get("paper_title")),
        )

    # 下列入口保留题目要求的 workspace_id + section_id 形式；Explain 与未来 UI 可复用。
    def get_section_tree(self, workspace_id: str) -> tuple[WorkspaceSection, ...]:
        return self.load(workspace_id).get_section_tree()

    def get_section(self, workspace_id: str, section_id: str) -> WorkspaceSection:
        return self.load(workspace_id).get_section(section_id)

    def get_children(
        self, workspace_id: str, section_id: str
    ) -> tuple[WorkspaceSection, ...]:
        return self.load(workspace_id).get_children(section_id)

    def get_descendants(
        self, workspace_id: str, section_id: str
    ) -> tuple[WorkspaceSection, ...]:
        return self.load(workspace_id).get_descendants(section_id)

    def get_direct_chunks(
        self, workspace_id: str, section_id: str
    ) -> tuple[WorkspaceChunk, ...]:
        return self.load(workspace_id).get_direct_chunks(section_id)

    def get_subtree_chunks(
        self, workspace_id: str, section_id: str
    ) -> tuple[WorkspaceChunk, ...]:
        return self.load(workspace_id).get_subtree_chunks(section_id)


def _load_sections(value: object, *, paper_id: str) -> tuple[WorkspaceSection, ...]:
    """严格读取持久化 Section Tree，拒绝缺 parent 或重复 ID 的不完整工作区。"""
    if not isinstance(value, list):
        raise WorkspaceSectionError("parsed_paper.json is missing its sections list.")
    sections: list[WorkspaceSection] = []
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise WorkspaceSectionError(f"sections[{index}] must be an object.")
        section = WorkspaceSection(
            section_id=_required_string(raw.get("section_id"), f"sections[{index}].section_id"),
            paper_id=_required_string(raw.get("paper_id"), f"sections[{index}].paper_id"),
            section_title=_required_string(raw.get("section_title"), f"sections[{index}].section_title"),
            section_number=_optional_string(raw.get("section_number")),
            section_level=_required_nonnegative_int(raw.get("section_level"), f"sections[{index}].section_level", minimum=1),
            parent_section_id=_optional_string(raw.get("parent_section_id")),
            section_index=_required_nonnegative_int(raw.get("section_index"), f"sections[{index}].section_index"),
            page_start=_optional_positive_int(raw.get("page_start"), f"sections[{index}].page_start"),
            page_end=_optional_positive_int(raw.get("page_end"), f"sections[{index}].page_end"),
        )
        if section.paper_id != paper_id:
            raise WorkspaceSectionError("Section paper_id does not match workspace paper_id.")
        if section.section_id in seen_ids or section.section_index in seen_indexes:
            raise WorkspaceSectionError("Section IDs and section indexes must be unique per workspace.")
        if section.page_start and section.page_end and section.page_end < section.page_start:
            raise WorkspaceSectionError("Section page_end must not precede page_start.")
        seen_ids.add(section.section_id)
        seen_indexes.add(section.section_index)
        sections.append(section)
    section_ids = {section.section_id for section in sections}
    for section in sections:
        if section.parent_section_id and section.parent_section_id not in section_ids:
            raise WorkspaceSectionError("Section parent_section_id does not exist in the same workspace.")
    return tuple(sorted(sections, key=lambda section: section.section_index))


def _load_chunks(path: Path, *, paper_id: str) -> tuple[WorkspaceChunk, ...]:
    """读取所有 chunk；section_id 只作为精确引用，不从 ``section`` 字符串推断。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkspaceSectionError(f"Cannot read workspace chunks: {path}") from error
    chunks: list[tuple[int, WorkspaceChunk]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkspaceSectionError(f"Invalid chunk JSONL at line {line_number}.") from error
        if not isinstance(raw, dict):
            raise WorkspaceSectionError(f"Chunk JSONL line {line_number} must be an object.")
        chunk_id = _required_string(raw.get("chunk_id"), f"chunks[{line_number}].chunk_id")
        if chunk_id in seen_ids:
            raise WorkspaceSectionError("Chunk IDs must be unique within a workspace.")
        if _required_string(raw.get("paper_id"), f"chunks[{line_number}].paper_id") != paper_id:
            raise WorkspaceSectionError("Chunk paper_id does not match workspace paper_id.")
        raw_text = _required_string(raw.get("raw_text"), f"chunks[{line_number}].raw_text")
        chunk = WorkspaceChunk(
            chunk_id=chunk_id,
            chunk_index=_required_nonnegative_int(raw.get("chunk_index"), f"chunks[{line_number}].chunk_index"),
            raw_text=raw_text,
            raw_token_count=_required_nonnegative_int(raw.get("raw_token_count"), f"chunks[{line_number}].raw_token_count"),
            section=_optional_string(raw.get("section")),
            section_id=_optional_string(raw.get("section_id")),
            section_type=_required_string(raw.get("section_type"), f"chunks[{line_number}].section_type"),
            paper_title=_optional_string(raw.get("paper_title")),
            content_kind=_required_string(
                raw.get("content_kind", "body"), f"chunks[{line_number}].content_kind"
            ),
            front_matter_type=_optional_string(raw.get("front_matter_type")),
            page_start=_optional_positive_int(raw.get("page_start"), f"chunks[{line_number}].page_start"),
            page_end=_optional_positive_int(raw.get("page_end"), f"chunks[{line_number}].page_end"),
        )
        if chunk.page_start and chunk.page_end and chunk.page_end < chunk.page_start:
            raise WorkspaceSectionError("Chunk page_end must not precede page_start.")
        seen_ids.add(chunk_id)
        chunks.append((line_number, chunk))
    return tuple(chunk for _, chunk in sorted(chunks, key=lambda item: (item[1].chunk_index, item[0])))


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取并验证工作区 JSON 对象。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkspaceSectionError(f"Cannot read workspace JSON: {path}") from error
    if not isinstance(value, dict):
        raise WorkspaceSectionError(f"Workspace JSON must be an object: {path}")
    return value


def _validate_workspace_id(value: str) -> str:
    """工作区 ID 只能是 staging 目录的一层子目录名。"""
    workspace_id = _required_string(value, "workspace_id")
    if not workspace_id.startswith("staging_") or any(char in workspace_id for char in "/\\"):
        raise WorkspaceSectionError("Invalid temporary workspace ID.")
    return workspace_id


def _required_string(value: object, field_name: str) -> str:
    normalized = _optional_string(value)
    if normalized is None:
        raise WorkspaceSectionError(f"{field_name} must be a non-empty string.")
    return normalized


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _required_nonnegative_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkspaceSectionError(f"{field_name} must be an integer >= {minimum}.")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _required_nonnegative_int(value, field_name, minimum=1)
