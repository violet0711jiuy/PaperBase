"""正式知识库 Paper Overview 的显式生成与只读仓储。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from paperbase.config import PaperOverviewSettings
from paperbase.database import MetadataDatabase
from paperbase.llm.client import ChatCompletionClient

from .service import (
    PaperOverview,
    PaperOverviewError,
    build_overview_context_from_records,
    generate_paper_overview,
)


@dataclass(frozen=True)
class FormalOverviewArtifact:
    paper_id: str
    overview: PaperOverview
    context: dict[str, Any]
    overview_path: Path
    context_path: Path


class FormalOverviewRepository:
    """按 paper_id 隔离正式 Overview；读取不会触发生成。"""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def get(self, paper_id: str) -> FormalOverviewArtifact | None:
        overview_path, context_path = self._paths(paper_id)
        overview_exists = overview_path.is_file()
        context_exists = context_path.is_file()
        if not overview_exists and not context_exists:
            return None
        if overview_exists != context_exists:
            raise PaperOverviewError("正式 Overview 只存在一半产物，仓储状态不完整。")
        try:
            overview = PaperOverview.model_validate_json(
                overview_path.read_text(encoding="utf-8")
            )
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise PaperOverviewError("正式 Overview 产物损坏或无法读取。") from error
        if not isinstance(context, dict) or context.get("paper_id") != paper_id:
            raise PaperOverviewError("正式 Overview Context 的 paper_id 与目录不一致。")
        return FormalOverviewArtifact(
            paper_id=paper_id,
            overview=overview,
            context=context,
            overview_path=overview_path,
            context_path=context_path,
        )

    def generate(
        self,
        paper_id: str,
        *,
        database: MetadataDatabase,
        settings: PaperOverviewSettings,
        client: ChatCompletionClient,
        overwrite: bool = False,
    ) -> FormalOverviewArtifact:
        """显式从正式数据库生成并持久化 Overview；默认不覆盖既有产物。"""
        current = self.get(paper_id)
        if current is not None and not overwrite:
            return current
        document = database.get_document(paper_id)
        if document is None:
            raise PaperOverviewError(f"正式知识库中不存在论文：{paper_id}")
        raw_chunks = [dict(row) for row in database.list_chunks(paper_id)]
        context = build_overview_context_from_records(
            paper_title=str(document["paper_title"]),
            raw_chunks=raw_chunks,
            settings=settings,
        )
        if not context.chunks:
            raise PaperOverviewError("正式知识库论文没有可用于 Overview 的正文 chunk。")
        overview = generate_paper_overview(context=context, client=client)
        context_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "paper_id": paper_id,
            "paper_title": context.paper_title,
            "source": "formal_knowledge_base",
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
        }
        return self.save(paper_id, overview=overview, context=context_payload)

    def save(
        self,
        paper_id: str,
        *,
        overview: PaperOverview,
        context: dict[str, Any],
    ) -> FormalOverviewArtifact:
        """分别以临时文件原子替换同一论文的两个正式产物。"""
        overview_path, context_path = self._paths(paper_id)
        output_dir = overview_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(context)
        payload["paper_id"] = paper_id
        overview_temp = output_dir / "overview.json.tmp"
        context_temp = output_dir / "context.json.tmp"
        overview_temp.write_text(
            json.dumps(overview.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context_temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        overview_temp.replace(overview_path)
        context_temp.replace(context_path)
        return FormalOverviewArtifact(
            paper_id=paper_id,
            overview=overview,
            context=payload,
            overview_path=overview_path,
            context_path=context_path,
        )

    def _paths(self, paper_id: str) -> tuple[Path, Path]:
        normalized = " ".join(paper_id.split())
        if (
            not normalized
            or normalized != paper_id
            or Path(normalized).name != normalized
            or "/" in normalized
            or "\\" in normalized
        ):
            raise ValueError("paper_id 不能包含空白折叠差异或路径分隔符。")
        output_dir = self._root / normalized
        return output_dir / "overview.json", output_dir / "context.json"
