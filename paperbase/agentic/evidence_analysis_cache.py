"""Evidence Analysis 的独立 SQLite 缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from paperbase.prompts.evidence_analysis import EVIDENCE_ANALYSIS_PROMPT_VERSION

from .contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    EvidenceTarget,
)


EVIDENCE_ANALYSIS_SCHEMA_VERSION = "evidence-analysis-schema-v2"


@dataclass(frozen=True)
class CacheLookup:
    analysis: EvidenceAnalysis | None
    error: bool = False


class SQLiteEvidenceAnalysisCache:
    """按证据、问题、检索目标和版本隔离分析结果。"""

    def __init__(
        self,
        path: Path | str,
        *,
        schema_version: str = EVIDENCE_ANALYSIS_SCHEMA_VERSION,
        prompt_version: str = EVIDENCE_ANALYSIS_PROMPT_VERSION,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_version = schema_version
        self._prompt_version = prompt_version
        self._initialize()

    def lookup(self, *, question: str, record: EvidenceRecord) -> CacheLookup:
        key = self._key(question=question, record=record)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM evidence_analysis_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
            if row is None:
                return CacheLookup(None)
            payload = json.loads(str(row[0]))
            return CacheLookup(_restore_analysis(record, payload))
        except Exception:
            self._delete_quietly(key)
            return CacheLookup(None, error=True)

    def put(
        self,
        *,
        question: str,
        record: EvidenceRecord,
        analysis: EvidenceAnalysis,
    ) -> bool:
        if analysis.status not in {
            EvidenceAnalysisStatus.QUALIFIED,
            EvidenceAnalysisStatus.IRRELEVANT,
        }:
            return False
        key = self._key(question=question, record=record)
        try:
            payload = _analysis_payload(analysis)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO evidence_analysis_cache(
                        cache_key, evidence_fingerprint, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        key,
                        record.fingerprint,
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            return True
        except Exception:
            return False

    def _key(self, *, question: str, record: EvidenceRecord) -> str:
        payload = {
            "fingerprint": record.fingerprint,
            "question": " ".join(question.casefold().split()),
            "targets": sorted(
                target.value for target in record.requested_evidence_targets
            ),
            "missing_information": sorted(
                " ".join(item.casefold().split())
                for item in record.requested_missing_information
            ),
            "schema_version": self._schema_version,
            "prompt_version": self._prompt_version,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_analysis_cache (
                    cache_key TEXT PRIMARY KEY,
                    evidence_fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_analysis_fingerprint
                ON evidence_analysis_cache(evidence_fingerprint)
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _delete_quietly(self, key: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM evidence_analysis_cache WHERE cache_key = ?",
                    (key,),
                )
        except Exception:
            pass


def _analysis_payload(analysis: EvidenceAnalysis) -> dict[str, object]:
    return {
        "status": analysis.status.value,
        "digest": analysis.digest,
        "covered_targets": sorted(
            target.value for target in analysis.covered_targets
        ),
        "addressed_information": list(analysis.addressed_information),
        "claims": [
            {
                "statement": claim.statement,
                "relation": claim.relation.value,
                "ownership": claim.ownership.value,
                "rationale": claim.rationale,
                "source_excerpt": claim.source_excerpt,
            }
            for claim in analysis.claims
        ],
    }


def _restore_analysis(
    record: EvidenceRecord, payload: object
) -> EvidenceAnalysis:
    if not isinstance(payload, dict):
        raise ValueError("缓存 payload 必须是对象。")
    status = EvidenceAnalysisStatus(payload["status"])
    if status not in {
        EvidenceAnalysisStatus.QUALIFIED,
        EvidenceAnalysisStatus.IRRELEVANT,
    }:
        raise ValueError("缓存只允许 qualified 或 irrelevant。")
    covered_targets = frozenset(
        EvidenceTarget(value) for value in payload.get("covered_targets", [])
    )
    raw_addressed = payload.get("addressed_information", [])
    if not isinstance(raw_addressed, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_addressed
    ):
        raise ValueError("缓存 addressed_information 无效。")
    addressed_information = tuple(raw_addressed)
    if not covered_targets.issubset(record.requested_evidence_targets):
        raise ValueError("缓存覆盖目标越界。")
    if not set(addressed_information).issubset(
        set(record.requested_missing_information)
    ):
        raise ValueError("缓存信息缺口越界。")
    claims: list[EvidenceClaim] = []
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("缓存 claims 必须是数组。")
    for index, item in enumerate(raw_claims, start=1):
        if not isinstance(item, dict):
            raise ValueError("缓存 claim 必须是对象。")
        required_text = (
            "statement",
            "rationale",
            "source_excerpt",
        )
        if any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in required_text
        ):
            raise ValueError("缓存 claim 文本字段无效。")
        excerpt = item["source_excerpt"]
        if excerpt not in record.evidence.content:
            raise ValueError("缓存 source_excerpt 已不再对应原文。")
        claims.append(
            EvidenceClaim(
                claim_id=f"{record.evidence_id}:C{index}",
                statement=item["statement"],
                relation=EvidenceRelation(item["relation"]),
                ownership=EvidenceOwnership(item["ownership"]),
                rationale=item["rationale"],
                source_excerpt=excerpt,
            )
        )
    return EvidenceAnalysis(
        evidence_id=record.evidence_id,
        status=status,
        digest=_required_cache_text(payload.get("digest"), "digest"),
        claims=tuple(claims),
        covered_targets=covered_targets,
        addressed_information=addressed_information,
    )


def _required_cache_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缓存 {name} 无效。")
    return value
