"""集中配置与解析器工厂的轻量级回归测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml

from paperbase.config import default_config_path, load_settings
from paperbase.parsing import PaperParser
from paperbase.parsing.factory import create_parser


class ConfigTests(unittest.TestCase):
    """验证配置中的路径解析和解析器选择不会依赖当前工作目录。"""

    def test_project_config_resolves_paths_and_creates_docling_parser(self) -> None:
        """默认配置应把相对存储路径解析到项目目录，并创建 Docling 适配器。"""
        settings = load_settings(default_config_path())
        parser = create_parser(settings)

        project_root = default_config_path().parent
        self.assertEqual(settings.project.name, "PaperBase")
        self.assertEqual(settings.storage.papers_dir, project_root / "storage" / "papers")
        self.assertEqual(
            settings.parsing.inspection_output_dir,
            project_root / "storage" / "parsed" / "granite_docling",
        )
        self.assertEqual(settings.chunking.backend, "docling_hybrid")
        self.assertEqual(settings.chunking.max_tokens, 512)
        self.assertEqual(settings.chunking.embedding_metadata_reserve_tokens, 64)
        self.assertEqual(settings.embedding.backend, "qwen_sentence_transformers")
        self.assertEqual(settings.embedding.batch_size, 16)
        self.assertTrue(settings.embedding.normalize_embeddings)
        self.assertEqual(settings.indexing.backend, "faiss_flat_ip")
        self.assertEqual(settings.indexing.index_path.name, "paperbase.faiss")
        self.assertEqual(settings.reranking.backend, "bge_cross_encoder")
        self.assertTrue(settings.reranking.enabled)
        self.assertEqual(settings.reranking.candidate_top_k, 40)
        self.assertEqual(settings.reranking.final_top_k, 5)
        self.assertEqual(settings.reranking.max_length, 1024)
        self.assertTrue(settings.reranking.model_path.is_absolute())
        self.assertEqual(settings.retrieval.backend, "hybrid_rrf")
        self.assertEqual(settings.retrieval.dense_top_k_per_query, 20)
        self.assertEqual(settings.retrieval.bm25_keywords_top_k, 20)
        self.assertTrue(settings.retrieval.query_rewrite.enabled)
        self.assertEqual(settings.answer_generation.max_output_tokens, 4000)
        self.assertFalse(hasattr(settings.agentic_research, "max_output_tokens"))
        self.assertFalse(hasattr(settings.agentic_research, "max_evidence_per_analysis"))
        self.assertFalse(hasattr(settings.agentic_research, "final_evidence_max_per_paper"))
        self.assertFalse(hasattr(settings.agentic_research, "max_no_new_evidence_rounds"))
        self.assertFalse(hasattr(
            settings.agentic_research, "evidence_analysis_max_output_tokens"
        ))
        self.assertFalse(hasattr(
            settings.agentic_research, "evidence_analysis_timeout_seconds"
        ))
        self.assertEqual(settings.agentic_research.node_max_attempts, 2)
        self.assertFalse(hasattr(
            settings.agentic_research, "evidence_analysis_cache_path"
        ))
        self.assertEqual(
            settings.database.path,
            project_root / "storage" / "paperbase.sqlite3",
        )
        self.assertEqual(settings.database.busy_timeout_ms, 5000)
        self.assertEqual(settings.parsing.docling.list_style_heading_min_chars, 80)
        self.assertTrue(settings.parsing.front_matter.enabled)
        self.assertEqual(settings.parsing.front_matter.max_pages, 2)
        self.assertTrue(settings.chunking.tokenizer_path.is_absolute())
        self.assertTrue(settings.parsing.docling.artifacts_path.is_absolute())
        self.assertTrue(settings.parsing.docling.remove_page_furniture)
        self.assertTrue(settings.parsing.docling.remove_peer_review_artifacts)
        self.assertIsInstance(parser, PaperParser)
        self.assertEqual(parser.parser_id, "docling")

    def test_obsolete_agentic_limits_in_existing_yaml_are_ignored(self) -> None:
        raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
        raw.setdefault("agentic_research", {}).update(
            max_evidence_per_analysis=8,
            final_evidence_max_per_paper=1,
            max_no_new_evidence_rounds=10,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            settings = load_settings(path)
        self.assertFalse(hasattr(settings.agentic_research, "max_evidence_per_analysis"))
        self.assertFalse(hasattr(settings.agentic_research, "final_evidence_max_per_paper"))
        self.assertFalse(hasattr(settings.agentic_research, "max_no_new_evidence_rounds"))


if __name__ == "__main__":
    unittest.main()
