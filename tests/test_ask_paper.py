"""Ask This Paper 的工作区隔离混合检索、证据边界与生成回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from paperbase.ask_paper.evidence import WorkspaceSectionEvidenceExpander
from paperbase.ask_paper.retriever import WorkspaceHybridRetriever
from paperbase.ask_paper.service import AskThisPaperService
from paperbase.config import (
    AnswerGenerationSettings,
    ContextExpansionSettings,
    RerankingSettings,
    RetrievalSettings,
)
from paperbase.generation.answer_generator import GroundedAnswerGenerator
from paperbase.reranking.base import RerankScore
from paperbase.retrieval.query_rewriter import QueryRewritePlan
from paperbase.staging.bm25 import WorkspaceBM25Index, WorkspaceBM25IndexCache
from paperbase.staging.sections import WorkspaceChunk, WorkspaceSectionSnapshot


class _FakeEmbedder:
    """不加载 Qwen，仅返回与临时 Fake FAISS 对齐的查询向量。"""

    def embed_queries(self, texts: list[str], *, instruction: str) -> np.ndarray:
        _ = instruction
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _FakeVectorIndex:
    """只暴露当前 workspace 的两个 content vector_id。"""

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        _ = top_k
        return (
            np.asarray([[0.91, 0.82] for _ in vectors], dtype=np.float32),
            np.asarray([[2, 1] for _ in vectors], dtype=np.int64),
        )


class _FakeRewriter:
    """稳定覆盖 resolved/semantic Dense、关键词 BM25 与 bibliography 分支。"""

    def plan(
        self, query: str, *, trusted_scope: object = None, conversation_context: object = None
    ) -> QueryRewritePlan:
        _ = trusted_scope, conversation_context
        return QueryRewritePlan(
            original_query=query,
            resolved_query=query,
            semantic_query_en="dynamic time method",
            lexical_keywords_en=("results",),
            search_bibliography=True,
            rewrite_status="success",
        )


class _ContextRecordingRewriter(_FakeRewriter):
    """记录 scope 与上下文，验证两类输入没有混在一起。"""

    def __init__(self) -> None:
        self.contexts: list[tuple[str, ...]] = []
        self.scopes: list[object] = []

    def plan(
        self, query: str, *, trusted_scope: object = None, conversation_context: object = None
    ) -> QueryRewritePlan:
        self.contexts.append(tuple(conversation_context or ()))
        self.scopes.append(trusted_scope)
        return super().plan(
            query, trusted_scope=trusted_scope, conversation_context=conversation_context
        )


class _UnresolvedRewriter:
    """模拟缺少历史对象的追问；Retriever 不得把原句送入任何召回路径。"""

    def plan(
        self, query: str, *, trusted_scope: object = None, conversation_context: object = None
    ) -> QueryRewritePlan:
        _ = trusted_scope, conversation_context
        return QueryRewritePlan(
            original_query=query,
            resolution_status="unresolved",
            rewrite_status="not_run",
            clarification_message="请明确它指的是哪个方法。",
        )


class _FakeReranker:
    backend_id = "fake"
    model_id = "fake-model"

    def rerank(self, query: str, passages: list[str]) -> tuple[RerankScore, ...]:
        _ = query, passages
        # 故意让第二个输入最高，验证 RRF 后确实经过了 Reranker。
        return (RerankScore(input_index=0, score=0.4), RerankScore(input_index=1, score=0.9))


class _FakeAnswerClient:
    """模拟唯一一次结构化回答，不访问远程 LLM。"""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, **kwargs: object) -> str:
        self.calls += 1
        return json.dumps(
            {
                "direct_answer": "该方法先提取极值，再比较形状描述符。[E1]",
                "evidence_explanation": "证据按步骤给出了极值与形状描述符的处理。[E1]",
                "reading_interpretation": "这将序列比较建立在局部形状上。[E1]",
                "citations": ["E1"],
                "insufficient_evidence": False,
            }
        )

    def complete(self, **kwargs: object) -> str:
        raise AssertionError("Ask This Paper 回答必须走结构化输出。")


class AskThisPaperTests(unittest.TestCase):
    """全程仅使用内存 workspace 快照，不创建或读取正式 SQLite。"""

    def test_current_paper_only_hybrid_retrieval_and_bibliography_are_isolated(self) -> None:
        snapshot = _snapshot(Path("/temporary/staging_ask"))
        retriever = _retriever(snapshot)

        result = retriever.retrieve("How does the method work?", result_limit=3)

        self.assertEqual(result.rewrite_plan.rewrite_status, "success")
        self.assertEqual(result.reranking_status, "success")
        self.assertTrue(result.chunks)
        self.assertTrue(all(chunk.paper_id == "paper-current" for chunk in result.chunks))
        self.assertNotIn("other-paper-chunk", [chunk.chunk_id for chunk in result.chunks])
        # References 只能通过 citation intent 的本论文 BM25 直出，绝不进入 FAISS/RRF/Reranker。
        bibliography = [chunk for chunk in result.chunks if chunk.section_type == "bibliography"]
        self.assertEqual([chunk.chunk_id for chunk in bibliography], ["reference-current"])
        self.assertEqual(bibliography[0].source_matches[0].route, "bm25_bibliography")
        content = [chunk for chunk in result.chunks if chunk.section_type == "content"]
        content_routes = {
            item.route for chunk in content for item in chunk.source_matches
        }
        self.assertIn("dense_resolved", content_routes)
        self.assertIn("dense_semantic", content_routes)
        self.assertNotIn("bm25_original", content_routes)  # 旧 resolved_query BM25 路径已移除。

    def test_evidence_expansion_uses_section_id_and_never_crosses_sibling(self) -> None:
        snapshot = _snapshot(Path("/temporary/staging_ask"))
        retrieval = _retriever(snapshot).retrieve("How does the method work?", result_limit=2)

        expansion = WorkspaceSectionEvidenceExpander(
            snapshot=snapshot,
            settings=ContextExpansionSettings(neighbor_window=1, max_total_tokens=500),
        ).expand(retrieval)

        self.assertEqual(len(expansion.content_evidence), 1)
        # method-1 的直属 section_id 是 method；results-1 属于 sibling result，不能被邻居扩展混入。
        self.assertEqual(expansion.content_evidence[0].chunk_ids, ("method-1", "method-2"))
        self.assertNotIn("results-1", expansion.content_evidence[0].chunk_ids)

    def test_service_generates_once_and_writes_only_workspace_result_artifact(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            snapshot = _snapshot(Path(temporary_directory) / "staging_ask")
            snapshot.root_dir.mkdir()
            client = _FakeAnswerClient()
            service = AskThisPaperService(
                snapshot=snapshot,
                retriever=_retriever(snapshot),
                expander=WorkspaceSectionEvidenceExpander(
                    snapshot=snapshot,
                    settings=ContextExpansionSettings(neighbor_window=1, max_total_tokens=500),
                ),
                generator=GroundedAnswerGenerator(
                    settings=AnswerGenerationSettings(), client=client
                ),
            )

            result = service.ask("How does the method work?", retrieval_result_limit=2)

            self.assertEqual(client.calls, 1)
            self.assertEqual(result.answer.status, "success")
            self.assertTrue(result.result_path.is_file())
            self.assertEqual(result.result_path.parent, snapshot.root_dir / "ask_paper")
            self.assertFalse((snapshot.root_dir / "paperbase.sqlite3").exists())

    def test_unresolved_reference_skips_retrieval_and_returns_clarification(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            snapshot = _snapshot(Path(temporary_directory) / "staging_ask")
            snapshot.root_dir.mkdir()
            client = _FakeAnswerClient()
            service = AskThisPaperService(
                snapshot=snapshot,
                retriever=_retriever(snapshot, rewriter=_UnresolvedRewriter()),
                expander=WorkspaceSectionEvidenceExpander(
                    snapshot=snapshot,
                    settings=ContextExpansionSettings(neighbor_window=1, max_total_tokens=500),
                ),
                generator=GroundedAnswerGenerator(
                    settings=AnswerGenerationSettings(), client=client
                ),
            )

            result = service.ask("它和 DTW 有什么区别？")

            self.assertEqual(result.retrieval.reranking_status, "not_run")
            self.assertEqual(result.retrieval.chunks, ())
            self.assertEqual(result.answer.status, "needs_clarification")
            self.assertEqual(result.answer.answer, "请明确它指的是哪个方法。")
            self.assertEqual(client.calls, 0)

    def test_service_passes_current_paper_as_trusted_scope_not_conversation_text(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            snapshot = _snapshot(Path(temporary_directory) / "staging_ask")
            snapshot.root_dir.mkdir()
            rewriter = _ContextRecordingRewriter()
            service = AskThisPaperService(
                snapshot=snapshot,
                retriever=_retriever(snapshot, rewriter=rewriter),
                expander=WorkspaceSectionEvidenceExpander(
                    snapshot=snapshot,
                    settings=ContextExpansionSettings(neighbor_window=1, max_total_tokens=500),
                ),
                generator=GroundedAnswerGenerator(
                    settings=AnswerGenerationSettings(), client=_FakeAnswerClient()
                ),
            )

            service.ask("这篇论文的方法是什么？", conversation_context=("上一轮问题。",))

            self.assertEqual(rewriter.contexts[0][0], "上一轮问题。")
            self.assertEqual(len(rewriter.contexts[0]), 1)
            self.assertEqual(getattr(rewriter.scopes[0], "paper_id"), "paper-current")
            self.assertEqual(getattr(rewriter.scopes[0], "paper_title"), "Current Paper")


def _retriever(
    snapshot: WorkspaceSectionSnapshot, *, rewriter: object | None = None
) -> WorkspaceHybridRetriever:
    """装配完全替身化的 Hybrid Retriever，便于断言当前论文隔离边界。"""
    retrieval_settings = RetrievalSettings(
        backend="hybrid_rrf",
        query_instruction="Retrieve evidence from the currently selected paper only.",
        dense_top_k_per_query=2,
        bm25_keywords_top_k=2,
        fused_top_k=4,
        rrf_k=60,
        dense_resolved_weight=1.0,
        dense_semantic_weight=1.0,
        bm25_keywords_weight=1.0,
    )
    reranking_settings = RerankingSettings(
        backend="fake",
        model_id="fake-model",
        model_path=Path(__file__).resolve(),
        device="cpu",
        batch_size=1,
        candidate_top_k=2,
        final_top_k=2,
        max_length=128,
    )
    cache = WorkspaceBM25IndexCache()
    return WorkspaceHybridRetriever(
        snapshot=snapshot,
        query_embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        vector_index=_FakeVectorIndex(),
        vector_to_chunk_id={1: "method-1", 2: "method-2"},
        content_bm25=cache.get_or_build(snapshot),
        bibliography_bm25=WorkspaceBM25Index.build(snapshot, section_type="bibliography"),
        settings=retrieval_settings,
        query_planner=rewriter or _FakeRewriter(),  # type: ignore[arg-type]
        reranker=_FakeReranker(),  # type: ignore[arg-type]
        reranking_settings=reranking_settings,
    )


def _snapshot(root_dir: Path) -> WorkspaceSectionSnapshot:
    """两段同直属 section + 一段 sibling + 一条 reference 的最小论文。"""
    return WorkspaceSectionSnapshot(
        workspace_id="staging_ask",
        root_dir=root_dir,
        paper_id="paper-current",
        sections=(),
        paper_title="Current Paper",
        chunks=(
            WorkspaceChunk("method-1", 1, "The method extracts local extrema.", 10, "3.1 Method", "method", "content", paper_title="Current Paper"),
            WorkspaceChunk("method-2", 2, "The descriptors compare dynamic time shapes.", 10, "3.1 Method", "method", "content", paper_title="Current Paper"),
            WorkspaceChunk("results-1", 3, "The results improve accuracy.", 10, "4 Results", "results", "content", paper_title="Current Paper"),
            WorkspaceChunk("reference-current", 4, "[1] Dynamic time warping results reference.", 8, "References", None, "bibliography", paper_title="Current Paper"),
        ),
    )
