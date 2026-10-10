"""Evidence Analysis 的独立 SQLite 缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from paperbase.prompts.evidence_projector import EVIDENCE_PROJECTOR_PROMPT_VERSION

from .contracts import (
    EvidenceAnalysis,
    EvidenceAnalysisStatus,
    EvidenceClaim,
    EvidenceOwnership,
    EvidenceRecord,
    EvidenceRelation,
    ComparisonProjection,
    EnumerationProjection,
    FactItem,
    FactProjection,
    MechanismProjection,
    ProjectedEntity,
    ProjectedEntityRole,
    RequirementSupportAnchor,
)
from .grounding import source_contains
from .projection_semantics import normalize_projection_semantic_text


EVIDENCE_ANALYSIS_SCHEMA_VERSION = "evidence-projection-schema-v22-bounded-span-envelope"


@dataclass(frozen=True)
class CacheLookup:
    analysis: EvidenceAnalysis | None
    error: bool = False


class SQLiteEvidenceAnalysisCache:
    """Reuse by Evidence identity and compiled Projector semantic identity."""

    def __init__(
        self,
        path: Path | str,
        *,
        schema_version: str = EVIDENCE_ANALYSIS_SCHEMA_VERSION,
        prompt_version: str = EVIDENCE_PROJECTOR_PROMPT_VERSION,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_version = schema_version
        self._prompt_version = prompt_version
        self._initialize()

    def lookup(
        self,
        *,
        question: str,
        record: EvidenceRecord,
        projection_context: dict[str, object] | None = None,
    ) -> CacheLookup:
        if not _has_semantic_identity(projection_context):
            return CacheLookup(None)  # never reconstruct/fall back to a legacy key
        key = self._key(
            question=question, record=record, projection_context=projection_context
        )
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM evidence_analysis_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
            if row is None:
                return CacheLookup(None)
            payload = json.loads(str(row[0]))
            allowed = set((projection_context or {}).get("projection_requirement_ids", ()))
            analysis = _restore_analysis(record, payload, allowed_requirement_ids=allowed)
            expected_type = (projection_context or {}).get("projection_type")
            if expected_type and any(p.projection_type.value != expected_type for p in analysis.projections):
                raise ValueError("Cached projection type does not match its group.")
            return CacheLookup(analysis)
        except Exception:
            self._delete_quietly(key)
            return CacheLookup(None, error=True)

    def put(
        self,
        *,
        question: str,
        record: EvidenceRecord,
        analysis: EvidenceAnalysis,
        projection_context: dict[str, object] | None = None,
    ) -> bool:
        if not _has_semantic_identity(projection_context):
            return False  # new producers must compile their semantic identity
        if analysis.status not in {
            EvidenceAnalysisStatus.QUALIFIED,
            EvidenceAnalysisStatus.IRRELEVANT,
        }:
            return False
        if len(analysis.projections) > 1:
            return False  # cache one group result, never an aggregate
        key = self._key(
            question=question, record=record, projection_context=projection_context
        )
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

    def _key(
        self,
        *,
        question: str,
        record: EvidenceRecord,
        projection_context: dict[str, object] | None = None,
    ) -> str:
        payload = {
            "fingerprint": record.fingerprint,
            "question": normalize_projection_semantic_text(question),
            # Only compiled key material; never reconstruct Requirement semantics here.
            "projection_context": {name: projection_context[name] for name in (
                "projection_type", "projection_requirement_ids",
                "semantic_request_version", "semantic_fingerprint",
            ) if name in (projection_context or {})},
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


def _has_semantic_identity(context: dict[str, object] | None) -> bool:
    version = (context or {}).get("semantic_request_version")
    fingerprint = (context or {}).get("semantic_fingerprint")
    return (isinstance(version, str) and bool(version.strip())
            and isinstance(fingerprint, str) and len(fingerprint) == 64
            and all(char in "0123456789abcdef" for char in fingerprint))


def _analysis_payload(analysis: EvidenceAnalysis) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": analysis.status.value,
        "evaluated_requirement_ids": list(analysis.evaluated_requirement_ids),
        "projection_rejections": list(analysis.projection_rejections),
        "projections": [_projection_payload(projection) for projection in analysis.projections],
    }
    if not analysis.projections:
        payload["digest"] = analysis.digest
        payload["claims"] = [
            {
                "statement": claim.statement,
                "relation": claim.relation.value,
                "ownership": claim.ownership.value,
                "rationale": claim.rationale,
                "source_excerpt": claim.source_excerpt,
            }
            for claim in analysis.claims
        ]
    return payload


def _projection_payload(projection: object | None) -> dict[str, object] | None:
    if projection is None:
        return None
    payload: dict[str, object] = {
        "projection_type": projection.projection_type.value,
        "supported_requirement_ids": list(projection.supported_requirement_ids),
    }
    if not isinstance(projection, EnumerationProjection):
        payload["proposed_requirement_ids"] = list(projection.proposed_requirement_ids)
    payload["support_anchors"] = [
        {
            "requirement_id": item.requirement_id,
            "source_excerpt": item.source_excerpt,
            "support_scope": item.support_scope,
            "unresolved_aspects": list(item.unresolved_aspects),
        }
        for item in projection.support_anchors
    ]
    if isinstance(projection, EnumerationProjection):
        payload["entities"] = [
            {
                "canonical_name": item.canonical_name,
                "entity_type": item.entity_type,
                "role": item.role.value,
                "ownership": item.ownership.value,
            }
            for item in projection.entities
        ]
    elif isinstance(projection, ComparisonProjection):
        payload["comparison_dimensions"] = list(projection.comparison_dimensions)
    elif isinstance(projection, FactProjection):
        payload["facts"] = [
            {
                "fact_type": fact.fact_type,
                "value": fact.value,
                "ownership": fact.ownership.value,
                "supported_requirement_ids": list(fact.supported_requirement_ids),
                "proposed_requirement_ids": list(fact.proposed_requirement_ids),
                "support_anchors": [
                    {"requirement_id": anchor.requirement_id,
                     "source_excerpt": anchor.source_excerpt,
                     "support_scope": anchor.support_scope,
                     "unresolved_aspects": list(anchor.unresolved_aspects)}
                    for anchor in fact.support_anchors
                ],
            }
            for fact in projection.facts
        ]
    elif isinstance(projection, MechanismProjection):
        payload.update(
            mechanism=projection.mechanism,
            steps=list(projection.steps),
            inputs=list(projection.inputs),
            outputs=list(projection.outputs),
        )
    return payload


def _restore_analysis(
    record: EvidenceRecord, payload: object, *, allowed_requirement_ids: set[str]
) -> EvidenceAnalysis:
    if not isinstance(payload, dict):
        raise ValueError("缓存 payload 必须是对象。")
    status = EvidenceAnalysisStatus(payload["status"])
    if status not in {
        EvidenceAnalysisStatus.QUALIFIED,
        EvidenceAnalysisStatus.IRRELEVANT,
    }:
        raise ValueError("缓存只允许 qualified 或 irrelevant。")
    raw_rejections = payload.get("projection_rejections", [])
    if not isinstance(raw_rejections, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_rejections
    ):
        raise ValueError("缓存 projection_rejections 无效。")
    if payload.get("addressed_information"):
        raise ValueError("当前投影缓存不能包含旧自由文本缺口。")
    raw_projections = payload.get("projections")
    if raw_projections is None:
        raw_projections = [payload["projection"]] if payload.get("projection") is not None else []
    if not isinstance(raw_projections, list) or len(raw_projections) > 1:
        raise ValueError("Cache payload must contain one projection group, not an aggregate.")
    projections = tuple(_restore_projection(record, value, allowed_requirement_ids=allowed_requirement_ids)
                        for value in raw_projections)
    if any(projection is None for projection in projections):
        raise ValueError("Cached projections cannot contain null.")
    claims: list[EvidenceClaim] = []
    raw_claims = payload.get("claims", []) if not projections else []
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
        digest=_required_cache_text(payload.get("digest"), "digest")
        if not projections and status is EvidenceAnalysisStatus.QUALIFIED else None,
        claims=tuple(claims),
        projections=projections,
        projection_rejections=tuple(raw_rejections),
        evaluated_requirement_ids=tuple(payload.get("evaluated_requirement_ids", ())),
    )


def _restore_projection(
    record: EvidenceRecord, value: object, *, allowed_requirement_ids: set[str]
):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("缓存 projection 必须是对象。")
    supported = tuple(value.get("supported_requirement_ids", []))
    if not set(supported).issubset(allowed_requirement_ids):
        raise ValueError("缓存 Projection Requirement 覆盖越界。")
    evidence_id = record.evidence_id
    projection_type = value.get("projection_type")
    proposed = tuple(value.get(
        "proposed_requirement_ids", supported if projection_type == "enumeration" else []
    ))
    if not set(proposed).issubset(allowed_requirement_ids):
        raise ValueError("缓存 proposed Requirement 覆盖越界。")
    raw_anchors = value.get("support_anchors", [])
    if not isinstance(raw_anchors, list):
        raise ValueError("缓存 support_anchors 无效。")
    anchors = tuple(
        RequirementSupportAnchor(
            item["requirement_id"], item["source_excerpt"], item["support_scope"],
            tuple(item.get("unresolved_aspects", ())),
        )
        for item in raw_anchors
    )
    if any(
        item.requirement_id not in proposed
        or not source_contains(record.evidence.content, item.source_excerpt)
        for item in anchors
    ):
        raise ValueError("缓存 support anchor 无法回溯到当前 Evidence。")
    if not set(supported).issubset(
        {item.requirement_id for item in anchors}
    ):
        raise ValueError("缓存 verified support 缺少 source anchor。")
    if projection_type == "enumeration":
        entities = value.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("缓存 entities 无效。")
        return EnumerationProjection(
            evidence_id,
            tuple(
                ProjectedEntity(
                    item["canonical_name"], item["entity_type"],
                    ProjectedEntityRole(item["role"]), EvidenceOwnership(item["ownership"]),
                )
                for item in entities
                if isinstance(item, dict)
            ),
            supported,
            anchors,
        )
    if projection_type == "comparison":
        return ComparisonProjection(
            evidence_id, tuple(value.get("comparison_dimensions", [])),
            supported, proposed, anchors,
        )
    if projection_type == "fact":
        raw_facts = value.get("facts", [])
        if not isinstance(raw_facts, list) or not raw_facts:
            raise ValueError("缓存 FactProjection 缺少 facts。")
        facts = tuple(
            FactItem(
                item["fact_type"], item["value"], EvidenceOwnership(item["ownership"]),
                tuple(item.get("supported_requirement_ids", ())),
                tuple(item.get("proposed_requirement_ids", ())),
                tuple(RequirementSupportAnchor(
                    anchor["requirement_id"], anchor["source_excerpt"], anchor["support_scope"],
                    tuple(anchor.get("unresolved_aspects", ())),
                ) for anchor in item.get("support_anchors", ())),
            )
            for item in raw_facts
        )
        if (
            tuple(dict.fromkeys(item for fact in facts for item in fact.supported_requirement_ids)) != supported
            or tuple(dict.fromkeys(item for fact in facts for item in fact.proposed_requirement_ids)) != proposed
            or tuple(anchor for fact in facts for anchor in fact.support_anchors) != anchors
        ):
            raise ValueError("缓存 FactProjection 的事实支持与总览不一致。")
        return FactProjection(evidence_id, facts)
    if projection_type == "mechanism":
        return MechanismProjection(
            evidence_id, value["mechanism"], tuple(value.get("steps", [])),
            tuple(value.get("inputs", [])), tuple(value.get("outputs", [])),
            supported, proposed, anchors,
        )
    raise ValueError("未知缓存 projection_type。")


def _required_cache_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缓存 {name} 无效。")
    return value
