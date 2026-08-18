"""Conversation Store：保存最终可见回合和可展示的 Evidence 快照。

Evidence 只保存回答实际使用的用户可见字段，不保存候选集、分数、向量或
其它检索内部状态，因此不会改变 RAG 链路的审计边界。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import json
from typing import Iterator, Literal, Sequence
from uuid import uuid4


ConversationScopeType = Literal["knowledge_base", "workspace"]


class ConversationStoreError(RuntimeError):
    """会话不存在、数据不合法或 SQLite 操作失败时的基础异常。"""


class ConversationScopeError(ConversationStoreError):
    """调用方尝试跨 KB/workspace 使用 conversation_id 时抛出。"""


@dataclass(frozen=True)
class ConversationRecord:
    """会话元数据；scope 是防止不同知识范围串话的第一道边界。"""

    conversation_id: str
    scope_type: ConversationScopeType
    scope_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ConversationTurn:
    """一轮用户问题、最终展示答案和可选 Evidence 快照。

    ``evidence_json`` 是 nullable 的 JSON 文本。旧数据库或旧 turn 没有该值
    时保持 ``None``，页面会正常显示答案但不显示证据入口。
    """

    turn_id: str
    conversation_id: str
    turn_index: int
    user_query: str
    assistant_answer: str
    resolved_query: str | None
    resolution_status: str | None
    audit_path: str | None
    created_at: str
    evidence_json: str | None = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (scope_type IN ('knowledge_base', 'workspace')),
    CHECK (length(trim(conversation_id)) > 0),
    CHECK (length(trim(scope_id)) > 0)
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    user_query TEXT NOT NULL,
    assistant_answer TEXT NOT NULL,
    resolved_query TEXT,
    resolution_status TEXT,
    audit_path TEXT,
    evidence_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    UNIQUE (conversation_id, turn_index),
    CHECK (turn_index >= 0),
    CHECK (length(trim(user_query)) > 0),
    CHECK (length(trim(assistant_answer)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_recent
    ON conversation_turns(conversation_id, turn_index DESC);
"""


class ConversationStore:
    """使用独立 SQLite 文件保存短期多轮对话，可安全跨进程重新打开。"""

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5_000) -> None:
        self._path = Path(path)
        self._busy_timeout_ms = busy_timeout_ms

    @property
    def path(self) -> Path:
        """返回实际会话数据库路径，便于诊断但不暴露连接对象。"""
        return self._path

    def initialize(self) -> None:
        """幂等创建两张会话表；不会触碰正式知识库 SQLite。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(_SCHEMA_SQL)
            # 旧版本的 conversations.sqlite3 没有 evidence_json；使用幂等迁移，
            # 不删除已有聊天记录，也不要求用户手动重建数据库。
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(conversation_turns)")
            }
            if "evidence_json" not in columns:
                connection.execute(
                    "ALTER TABLE conversation_turns ADD COLUMN evidence_json TEXT"
                )

    def create_conversation(
        self, scope_type: ConversationScopeType, scope_id: str
    ) -> ConversationRecord:
        """为指定 KB 或 workspace 创建一条新会话，ID 由程序生成。"""
        _validate_scope(scope_type, scope_id)
        self.initialize()
        now = _utc_now()
        record = ConversationRecord(
            conversation_id=f"conversation_{uuid4().hex}",
            scope_type=scope_type,
            scope_id=scope_id.strip(),
            created_at=now,
            updated_at=now,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, scope_type, scope_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.conversation_id,
                    record.scope_type,
                    record.scope_id,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def get_conversation(
        self,
        conversation_id: str,
        *,
        expected_scope_type: ConversationScopeType | None = None,
        expected_scope_id: str | None = None,
    ) -> ConversationRecord:
        """读取会话，并可同时验证调用范围，防止 KB/workspace 历史混用。"""
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT conversation_id, scope_type, scope_id, created_at, updated_at
                FROM conversations WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise ConversationStoreError(f"Conversation does not exist: {conversation_id}")
        record = _conversation_from_row(row)
        if expected_scope_type is not None and record.scope_type != expected_scope_type:
            raise ConversationScopeError(
                "Conversation scope does not match the current KB or workspace."
            )
        if expected_scope_id is not None and record.scope_id != expected_scope_id:
            raise ConversationScopeError(
                "Conversation scope does not match the current KB or workspace."
            )
        return record

    def list_conversations(
        self,
        scope_type: ConversationScopeType,
        scope_id: str,
    ) -> tuple[ConversationRecord, ...]:
        """列出指定 scope 下的会话，最新有活动的会话排在前面。

        这个查询只返回会话元数据，不读取 turn 正文；调用方如果需要历史，
        再通过 ``get_turns`` 读取，从而保持列表和聊天内容的职责分离。
        """
        _validate_scope(scope_type, scope_id)
        self.initialize()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT conversation_id, scope_type, scope_id, created_at, updated_at
                FROM conversations
                WHERE scope_type = ? AND scope_id = ?
                ORDER BY updated_at DESC, created_at DESC, conversation_id DESC
                """,
                (scope_type, scope_id.strip()),
            ).fetchall()
        return tuple(_conversation_from_row(row) for row in rows)

    def append_turn(
        self,
        conversation_id: str,
        *,
        user_query: str,
        assistant_answer: str,
        resolved_query: str | None = None,
        resolution_status: str | None = None,
        audit_path: str | None = None,
        evidence_json: str | None = None,
    ) -> ConversationTurn:
        """原子追加回合；turn_index 在同一会话中严格单调递增。

        ``evidence_json`` 只接受合法 JSON 文本；空值会以 SQL NULL 保存。
        """
        normalized_user = _required_text(user_query, field_name="user_query")
        normalized_answer = _required_text(assistant_answer, field_name="assistant_answer")
        normalized_evidence = _optional_json_text(evidence_json, field_name="evidence_json")
        self.initialize()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                exists = connection.execute(
                    "SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,)
                ).fetchone()
                if exists is None:
                    raise ConversationStoreError(
                        f"Conversation does not exist: {conversation_id}"
                    )
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_turn_index
                    FROM conversation_turns WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                ).fetchone()
                turn = ConversationTurn(
                    turn_id=f"turn_{uuid4().hex}",
                    conversation_id=conversation_id,
                    turn_index=int(row["next_turn_index"]),
                    user_query=normalized_user,
                    assistant_answer=normalized_answer,
                    resolved_query=_optional_text(resolved_query),
                    resolution_status=_optional_text(resolution_status),
                    audit_path=_optional_text(audit_path),
                    created_at=_utc_now(),
                    evidence_json=normalized_evidence,
                )
                connection.execute(
                    """
                    INSERT INTO conversation_turns (
                        turn_id, conversation_id, turn_index, user_query, assistant_answer,
                        resolved_query, resolution_status, audit_path, evidence_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.turn_id,
                        turn.conversation_id,
                        turn.turn_index,
                        turn.user_query,
                        turn.assistant_answer,
                        turn.resolved_query,
                        turn.resolution_status,
                        turn.audit_path,
                        turn.evidence_json,
                        turn.created_at,
                    ),
                )
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (turn.created_at, conversation_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return turn

    def get_recent_turns(
        self, conversation_id: str, max_turns: int
    ) -> tuple[ConversationTurn, ...]:
        """按真实会话顺序返回最近 N 轮；0 表示不读取任何上下文。"""
        if max_turns < 0:
            raise ValueError("max_turns must not be negative.")
        # 即使 max_turns=0 也检查会话是否存在，避免调用方误把无效 ID 当成空历史。
        self.get_conversation(conversation_id)
        if max_turns == 0:
            return ()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT turn_id, conversation_id, turn_index, user_query, assistant_answer,
                       resolved_query, resolution_status, audit_path, evidence_json, created_at
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (conversation_id, max_turns),
            ).fetchall()
        return tuple(_turn_from_row(row) for row in reversed(rows))

    def get_turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        """按完整会话顺序返回全部历史回合，供聊天历史恢复使用。"""
        # 即使当前没有 turn，也先校验会话存在，避免无效 ID 被误当成空历史。
        self.get_conversation(conversation_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT turn_id, conversation_id, turn_index, user_query, assistant_answer,
                       resolved_query, resolution_status, audit_path, evidence_json, created_at
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_index ASC
                """,
                (conversation_id,),
            ).fetchall()
        return tuple(_turn_from_row(row) for row in rows)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """每次操作短连接，保证进程重启后仍可读取同一 SQLite 数据。"""
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()


def format_turns_as_context(turns: Sequence[ConversationTurn]) -> tuple[str, ...]:
    """将完整 User + Assistant 回合格式化为 Query Resolution 可消费的普通上下文数据。"""
    return tuple(
        f"User: {turn.user_query}\nAssistant: {turn.assistant_answer}" for turn in turns
    )


def _conversation_from_row(row: sqlite3.Row) -> ConversationRecord:
    """集中转换 SQLite 行，避免业务层直接依赖数据库列顺序。"""
    return ConversationRecord(
        conversation_id=str(row["conversation_id"]),
        scope_type=str(row["scope_type"]),  # type: ignore[arg-type]
        scope_id=str(row["scope_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _turn_from_row(row: sqlite3.Row) -> ConversationTurn:
    """集中转换 turn 行，保留可选审计和 Evidence 字段的 null 语义。"""
    try:
        evidence_json = _optional_json_text(row["evidence_json"], field_name="evidence_json")
    except ValueError:
        # 外部旧脚本可能写入了损坏的快照；答案历史仍应可读，只跳过这条 Evidence。
        evidence_json = None
    return ConversationTurn(
        turn_id=str(row["turn_id"]),
        conversation_id=str(row["conversation_id"]),
        turn_index=int(row["turn_index"]),
        user_query=str(row["user_query"]),
        assistant_answer=str(row["assistant_answer"]),
        resolved_query=_optional_text(row["resolved_query"]),
        resolution_status=_optional_text(row["resolution_status"]),
        audit_path=_optional_text(row["audit_path"]),
        created_at=str(row["created_at"]),
        evidence_json=evidence_json,
    )


def _validate_scope(scope_type: str, scope_id: str) -> None:
    """提前拒绝未支持 scope，避免 SQLite CHECK 错误变成不清晰的底层异常。"""
    if scope_type not in {"knowledge_base", "workspace"}:
        raise ValueError(f"Unsupported conversation scope_type: {scope_type}")
    _required_text(scope_id, field_name="scope_id")


def _required_text(value: str, *, field_name: str) -> str:
    """统一保证持久化字段不是空白字符串。"""
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank.")
    return normalized


def _optional_text(value: object) -> str | None:
    """SQLite null 与空白字符串都统一映射为 None。"""
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized or None


def _optional_json_text(value: object, *, field_name: str) -> str | None:
    """规范化 nullable JSON 文本，损坏的数据不会悄悄进入会话库。"""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    try:
        json.loads(normalized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field_name} must contain valid JSON.") from error
    return normalized


def _utc_now() -> str:
    """使用带时区 ISO 时间，跨进程读取时不依赖本地时区。"""
    return datetime.now(timezone.utc).isoformat()
