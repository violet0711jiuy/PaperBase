"""全量重建的路径隔离与安全发布回归测试。"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from paperbase.rebuild import RebuildPaths, _publish_rebuild


class CleanRebuildPublishTests(unittest.TestCase):
    """不加载 Docling/GPU，仅验证已校验产物的备份与替换契约。"""

    def test_publish_backs_up_every_old_artifact_before_replacement(self) -> None:
        """正式文件和目录都应替换，旧内容必须完整保留在同一次 backup 内。"""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = _paths(root)
            for name, temporary, formal in paths.replacements():
                _write_artifact(temporary, f"new-{name}")
                _write_artifact(formal, f"old-{name}")

            _publish_rebuild(paths)

            for name, _, formal in paths.replacements():
                self.assertEqual(_read_artifact(formal), f"new-{name}")
                self.assertEqual(_read_artifact(paths.backup_root / name), f"old-{name}")
            self.assertIn('"phase": "published"', (paths.backup_root / "publish_journal.json").read_text(encoding="utf-8"))


def _paths(root: Path) -> RebuildPaths:
    """构造文件和目录目标混合的最小路径集，模拟真实正式产物。"""
    temporary_root = root / "staging" / "rebuild_tmp" / "run"
    final_root = root / "storage"
    return RebuildPaths(
        run_id="run",
        temporary_root=temporary_root,
        backup_root=root / "parsed" / "rebuild_backups" / "run",
        temporary_database=temporary_root / "paperbase.sqlite3",
        temporary_embeddings=temporary_root / "embeddings",
        temporary_index=temporary_root / "paperbase.faiss",
        temporary_manifest=temporary_root / "paperbase.faiss.manifest.json",
        temporary_parsed_artifacts=temporary_root / "parsed_artifacts",
        temporary_chunk_artifacts=temporary_root / "chunk_artifacts",
        final_database=final_root / "paperbase.sqlite3",
        final_embeddings=final_root / "staging" / "embeddings",
        final_index=final_root / "paperbase.faiss",
        final_manifest=final_root / "paperbase.faiss.manifest.json",
        final_parsed_artifacts=final_root / "parsed" / "granite_docling",
        final_chunk_artifacts=final_root / "parsed" / "chunks",
    )


def _write_artifact(path: Path, content: str) -> None:
    """按测试路径后缀写一个文件或一个含标记文件的目录。"""
    if path.suffix == ".sqlite3" or path.name == "database":
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE marker (value TEXT NOT NULL)")
            connection.execute("INSERT INTO marker(value) VALUES (?)", (content,))
        return
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker.txt").write_text(content, encoding="utf-8")


def _read_artifact(path: Path) -> str:
    """读取文件或目录中的测试标记。"""
    if path.suffix == ".sqlite3" or path.name == "database":
        with sqlite3.connect(path) as connection:
            return str(connection.execute("SELECT value FROM marker").fetchone()[0])
    return (path if path.is_file() else path / "marker.txt").read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
