"""Phase 4 本地 SQLite checkpoint 与任务生命周期管理。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from threading import RLock

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


class SQLiteCheckpointStore:
    """持有 SqliteSaver 连接，并管理线程保留、压缩和删除。"""

    def __init__(self, *, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection
        self._lock = RLock()
        self.saver = SqliteSaver(
            connection,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_json_modules=(),
                allowed_msgpack_modules=(),
            ),
        )
        self.saver.setup()
        self._initialize_lifecycle_table()

    @classmethod
    def open(cls, path: Path | str) -> "SQLiteCheckpointStore":
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            resolved,
            check_same_thread=False,
            timeout=30.0,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return cls(path=resolved, connection=connection)

    def touch_thread(self, thread_id: str) -> None:
        """记录任务最近活动时间，供启动时清理过期 checkpoint。"""
        normalized = _normalize_thread_id(thread_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO paperbase_agentic_threads(thread_id, updated_at)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (normalized, timestamp),
            )

    def delete_thread(self, thread_id: str) -> None:
        """删除一个明确指定任务的全部 checkpoint 与生命周期记录。"""
        normalized = _normalize_thread_id(thread_id)
        self.saver.delete_thread(normalized)
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM paperbase_agentic_threads WHERE thread_id = ?",
                (normalized,),
            )

    def prune_expired_threads(self, *, retention_days: int) -> tuple[str, ...]:
        """删除超过保留期且长期无活动的任务，返回被清理的 thread_id。"""
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1.")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._lock:
            rows = self._connection.execute(
                "SELECT thread_id FROM paperbase_agentic_threads WHERE updated_at < ?",
                (cutoff,),
            ).fetchall()
        expired = tuple(str(row[0]) for row in rows)
        for thread_id in expired:
            self.delete_thread(thread_id)
        return expired

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize_lifecycle_table(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paperbase_agentic_threads (
                    thread_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL
                )
                """
            )


def _normalize_thread_id(thread_id: str) -> str:
    normalized = " ".join(thread_id.split())
    if not normalized:
        raise ValueError("thread_id must be a non-empty string.")
    return normalized
